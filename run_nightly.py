# ==================================================================================
# NIGHTLY RUN (GitHub Actions) — rule conviction engine + FVG imbalance levels -> Google Sheet
#   Writes: "Rule Dashboard", "Rule History", "Run Info"
#   Env:  GCP_SA_JSON (service-account JSON text) | SHEET_ID | CACHE_DIR | START_DATE | FULL_REBUILD | DRY_RUN
# ==================================================================================
import os, re, sys, json, math, shutil
import numpy as np, pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.expanduser(os.environ.get("CACHE_DIR", "~/fno_cache")))
START     = os.environ.get("START_DATE", "2023-08-01")
FULL      = os.environ.get("FULL_REBUILD", "").lower() in ("1", "true", "yes")
DRY_RUN   = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

def load_engine():
    G = {}
    for f in ("conviction_engine.py", "fvg_lab.py"):
        exec(compile(open(os.path.join(HERE, f)).read(), f, "exec"), G)
    os.makedirs(CACHE_DIR, exist_ok=True); G["CACHE"] = CACHE_DIR
    return G

def seed_cache():
    """If the cache is empty, start from seed/md_raw.parquet + seed/md_members.json (committed in the repo)."""
    for fn in ("md_raw.parquet", "md_members.json"):
        dst = os.path.join(CACHE_DIR, fn); src = os.path.join(HERE, "seed", fn)
        if not os.path.exists(dst) and os.path.exists(src):
            os.makedirs(CACHE_DIR, exist_ok=True); shutil.copy(src, dst); print("seeded", fn)

# ── table building ──
def _vol_mult3(md, days=3, win=20):
    """MAX over last `days` days of (volume / trailing 20d SMA volume). >=2 => a high-volume day in last 3."""
    recs = []
    for sym, g in md.sort_values(["symbol", "date"]).groupby("symbol"):
        vol = pd.to_numeric(g["volume"], errors="coerce").to_numpy(float); n = len(vol)
        if n < days: recs.append({"symbol": sym, "Vol×": ""}); continue
        ratios = []
        for j in range(n - days, n):
            w = vol[max(0, j - win + 1):j + 1]; sma = np.nanmean(w) if len(w) else 0.0
            ratios.append(vol[j] / sma if sma > 0 else 0.0)
        recs.append({"symbol": sym, "Vol×": round(float(max(ratios)), 1)})
    return pd.DataFrame(recs)

def _deg(dash):
    """VWAP %/session slope -> angle in degrees."""
    for col in ("vwap3 slp", "vwap5 slp"):
        if col in dash.columns:
            dash[col] = dash[col].apply(lambda x: round(math.degrees(math.atan(float(x))), 1)
                                        if (x not in ("", None) and pd.notna(x)) else x)
    return dash

def _order(dash):
    strip = [c for c in dash.columns if re.fullmatch(r"\d{2}-\d{2}", str(c))]
    vw    = [c for c in ("vwap3 slp", "vwap5 slp") if c in dash.columns]
    sup   = sorted([c for c in dash.columns if re.fullmatch(r"imb_sup\d+", c)], key=lambda x: int(x[7:]))
    res   = sorted([c for c in dash.columns if re.fullmatch(r"imb_res\d+", c)], key=lambda x: int(x[7:]))
    mid   = [c for c in ("in_imb", "Vol×", "S/R Flag") if c in dash.columns]
    tail  = [c for c in ("Signal (latest)", "Drivers (latest)", "Patterns (latest)") if c in dash.columns]
    return dash[["symbol"] + strip + vw + sup + res + mid + tail]

def build_tables(G, md):
    feat = G["build_features"](md)
    rule_dash, rule_hist = G["build_rule_dashboard"](feat, months=3, last_n=5)
    ni = G["nearest_imbalances"](feat)
    dash = rule_dash.merge(ni, on="symbol", how="left").merge(_vol_mult3(md), on="symbol", how="left")
    return feat, _order(_deg(dash)), rule_hist

# ── Google Sheets ──
import gspread
from google.oauth2.service_account import Credentials

def open_sheet():
    info = json.loads(os.environ["GCP_SA_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return gspread.authorize(creds).open_by_key(os.environ["SHEET_ID"])

def _write(sh, title, df):
    d = df.astype(object).where(pd.notna(df), "")
    rows, cols = len(d) + 10, len(d.columns) + 4
    try:
        ws = sh.worksheet(title); ws.clear(); ws.resize(rows=rows, cols=cols)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
    ws.update(values=[list(d.columns)] + d.values.tolist(), range_name="A1", value_input_option="USER_ENTERED")
    return ws

def _heat(sh, ws, specs):
    """Red-white-green gradient. Clears existing rules first, so call BEFORE _flag_oversold."""
    wid = ws.id
    def rule(c0, c1, lo, hi): return {"addConditionalFormatRule": {"index": 0, "rule": {
        "ranges": [{"sheetId": wid, "startRowIndex": 1, "startColumnIndex": c0, "endColumnIndex": c1}],
        "gradientRule": {"minpoint": {"color": {"red": .80, "green": .15, "blue": .15}, "type": "NUMBER", "value": str(lo)},
                         "midpoint": {"color": {"red": 1, "green": 1, "blue": 1}, "type": "NUMBER", "value": "0"},
                         "maxpoint": {"color": {"red": .13, "green": .55, "blue": .27}, "type": "NUMBER", "value": str(hi)}}}}}
    meta = sh.fetch_sheet_metadata()
    cnt = next((len(s.get("conditionalFormats", [])) for s in meta["sheets"] if s["properties"]["sheetId"] == wid), 0)
    sh.batch_update({"requests": [{"deleteConditionalFormatRule": {"sheetId": wid, "index": 0}} for _ in range(cnt)]
                                 + [rule(*s) for s in specs]})

def _flag_oversold(sh, ws, df, strip_only):
    """Amber flag on scores in the 'Oversold' band (-60 < score <= -40)."""
    if strip_only:
        sc = [i for i, c in enumerate(df.columns) if re.fullmatch(r"\d{2}-\d{2}", str(c))]
        if not sc: return
        c0, c1 = min(sc), max(sc) + 1
    else:
        c0, c1 = 1, len(df.columns)
    tl = gspread.utils.rowcol_to_a1(2, c0 + 1)
    sh.batch_update({"requests": [{"addConditionalFormatRule": {"index": 0, "rule": {
        "ranges": [{"sheetId": ws.id, "startRowIndex": 1, "startColumnIndex": c0, "endColumnIndex": c1}],
        "booleanRule": {"condition": {"type": "CUSTOM_FORMULA", "values": [{"userEnteredValue": f"=AND({tl}>-60,{tl}<=-40)"}]},
                        "format": {"backgroundColor": {"red": 1.0, "green": .75, "blue": 0.0}, "textFormat": {"bold": True}}}}}}]})

def publish(dash, hist, latest, today):
    sh = open_sheet()
    ws = _write(sh, "Rule Dashboard", dash)
    strip = [i for i, c in enumerate(dash.columns) if re.fullmatch(r"\d{2}-\d{2}", str(c))]
    if strip: _heat(sh, ws, [(min(strip), max(strip) + 1, -50, 50)])
    _flag_oversold(sh, ws, dash, strip_only=True)
    wh = _write(sh, "Rule History", hist)
    _heat(sh, wh, [(1, len(hist.columns), -50, 50)]); _flag_oversold(sh, wh, hist, strip_only=False)
    now = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M")
    _write(sh, "Run Info", pd.DataFrame({"item": ["Last run (IST)", "Latest data date", "Symbols"],
                                         "value": [now, latest, len(dash)]}))
    print(f"sheet updated: {len(dash)} symbols, data through {latest}")

def main():
    today = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")
    G = load_engine(); seed_cache()
    print(f"building masterdata {START} -> {today} (full={FULL})")
    md = G["build_masterdata_pit_fast"](START, today, full=FULL)
    latest = str(md["date"].max())
    age = (pd.Timestamp(today) - pd.Timestamp(latest)).days
    if age > 4:   # holidays can make 3-4 days; more means NSE downloads are failing
        sys.exit(f"STALE DATA: latest session {latest} is {age} days old — NSE download problem. Not writing to the sheet.")
    if latest != today: print(f"note: latest session is {latest} (today's bhavcopy not out yet, or market holiday)")
    feat, dash, hist = build_tables(G, md)
    if DRY_RUN:
        dash.to_csv("dash_preview.csv", index=False); print(dash.head(10).to_string()); return
    publish(dash, hist, latest, today)

if __name__ == "__main__":
    main()
