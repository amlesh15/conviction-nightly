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
        if n < days: recs.append({"symbol": sym, "Vol\u00d7": ""}); continue
        ratios = []
        for j in range(n - days, n):
            w = vol[max(0, j - win + 1):j + 1]; sma = np.nanmean(w) if len(w) else 0.0
            ratios.append(vol[j] / sma if sma > 0 else 0.0)
        recs.append({"symbol": sym, "Vol\u00d7": round(float(max(ratios)), 1)})
    return pd.DataFrame(recs)

def _deg(dash):
    """VWAP %/session slope -> angle in degrees."""
    for col in ("vwap3 slp", "vwap5 slp"):
        if col in dash.columns:
            dash[col] = dash[col].apply(lambda x: round(math.degrees(math.atan(float(x))), 1)
                                        if (x not in ("", None) and pd.notna(x)) else x)
    return dash

LIVE_COLS = ["LTP", "Today %", "High %", "Low %", "ATM", "CE \u0394%", "CE bars", "PE \u0394%", "PE bars",
             "Regime", "Eff%", "Exp\u00d7"]   # blank here; filled by the live / ATM jobs during market hours

def _order(dash):
    for c in LIVE_COLS:
        if c not in dash.columns: dash[c] = ""
    strip = [c for c in dash.columns if re.fullmatch(r"\d{2}-\d{2}", str(c))]
    vw    = [c for c in ("vwap3 slp", "vwap5 slp") if c in dash.columns]
    sup   = sorted([c for c in dash.columns if re.fullmatch(r"imb_sup\d+", c)], key=lambda x: int(x[7:]))
    res   = sorted([c for c in dash.columns if re.fullmatch(r"imb_res\d+", c)], key=lambda x: int(x[7:]))
    mid   = [c for c in ("in_imb", "Vol\u00d7", "S/R Flag") if c in dash.columns]
    tail  = [c for c in ("Signal (latest)", "Drivers (latest)", "Patterns (latest)") if c in dash.columns]
    return dash[["symbol"] + strip + LIVE_COLS + vw + sup + res + mid + tail]

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

def _heat_atm(sh, ws, df):
    """CE/PE delta% green(+)/red(-) gradient; CE/PE bars green/red by that side's sign; S/R Flag colours."""
    wid = ws.id; cols = list(df.columns); reqs = []
    def grad(c0, lo, hi): return {"addConditionalFormatRule": {"index": 0, "rule": {
        "ranges": [{"sheetId": wid, "startRowIndex": 1, "startColumnIndex": c0, "endColumnIndex": c0 + 1}],
        "gradientRule": {"minpoint": {"color": {"red": .80, "green": .15, "blue": .15}, "type": "NUMBER", "value": str(lo)},
                         "midpoint": {"color": {"red": 1, "green": 1, "blue": 1}, "type": "NUMBER", "value": "0"},
                         "maxpoint": {"color": {"red": .13, "green": .55, "blue": .27}, "type": "NUMBER", "value": str(hi)}}}}}
    def solid(c0, formula, rgb): return {"addConditionalFormatRule": {"index": 0, "rule": {
        "ranges": [{"sheetId": wid, "startRowIndex": 1, "startColumnIndex": c0, "endColumnIndex": c0 + 1}],
        "booleanRule": {"condition": {"type": "CUSTOM_FORMULA", "values": [{"userEnteredValue": formula}]},
                        "format": {"backgroundColor": {"red": rgb[0], "green": rgb[1], "blue": rgb[2]}}}}}}
    for nm in ("CE \u0394%", "PE \u0394%"):
        if nm in cols: reqs.append(grad(cols.index(nm), -10, 10))
    for barcol, pctcol in (("CE bars", "CE \u0394%"), ("PE bars", "PE \u0394%")):
        if barcol in cols and pctcol in cols:
            bi = cols.index(barcol); pcol = re.match(r"[A-Z]+", gspread.utils.rowcol_to_a1(2, cols.index(pctcol) + 1)).group(0)
            reqs.append(solid(bi, f"=${pcol}2>0", (0.72, 0.88, 0.70))); reqs.append(solid(bi, f"=${pcol}2<0", (0.96, 0.80, 0.80)))
    if "S/R Flag" in cols:
        fi = cols.index("S/R Flag")
        def txt(word, rgb): return {"addConditionalFormatRule": {"index": 0, "rule": {
            "ranges": [{"sheetId": wid, "startRowIndex": 1, "startColumnIndex": fi, "endColumnIndex": fi + 1}],
            "booleanRule": {"condition": {"type": "TEXT_CONTAINS", "values": [{"userEnteredValue": word}]},
                            "format": {"backgroundColor": {"red": rgb[0], "green": rgb[1], "blue": rgb[2]}}}}}}
        reqs += [txt("Bounced", (0.72, 0.88, 0.70)), txt("Rejected", (0.96, 0.80, 0.80)), txt("Broke", (0.99, 0.90, 0.60))]
    if reqs: sh.batch_update({"requests": reqs})

def _numfmt(sh, ws, df):
    """Force decimals on the live columns (a sheet column can otherwise display whole numbers)."""
    cols = list(df.columns); pats = {"LTP": "0.00", "Today %": "0.00", "High %": "0.00", "Low %": "0.00",
                                     "CE \u0394%": "0.0", "PE \u0394%": "0.0"}
    reqs = [{"repeatCell": {"range": {"sheetId": ws.id, "startRowIndex": 1, "endRowIndex": len(df) + 1,
                                      "startColumnIndex": cols.index(nm), "endColumnIndex": cols.index(nm) + 1},
                            "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": pat}}},
                            "fields": "userEnteredFormat.numberFormat"}}
            for nm, pat in pats.items() if nm in cols]
    if reqs: sh.batch_update({"requests": reqs})


# ── "which stocks' imbalance levels changed since the last nightly run" ──────────────────────────
# Pure comparison of last night's imb_sup1..3 / imb_res1..3 with tonight's. It does NOT change how levels are computed.
LV_S = ["imb_sup1", "imb_sup2", "imb_sup3"]; LV_R = ["imb_res1", "imb_res2", "imb_res3"]
SNAP_HDR = ["symbol", "curr_asof", "cs1", "cs2", "cs3", "cr1", "cr2", "cr3", "prev_asof", "ps1", "ps2", "ps3", "pr1", "pr2", "pr3"]

def _f(x):
    try:
        if x in ("", None) or (not isinstance(x, str) and pd.isna(x)): return None
        return round(float(str(x).replace(",", "")), 1)
    except Exception: return None

def _lv(row): return tuple(_f(row.get(c)) for c in LV_S), tuple(_f(row.get(c)) for c in LV_R)

def _fmt(t): return " / ".join("-" if v is None else f"{v:g}" for v in t)

def _diff(old, new, tag):
    add = [v for v in new if v is not None and v not in old]; gone = [v for v in old if v is not None and v not in new]
    return f"{tag}: " + " ".join([f"+{v:g}" for v in add] + [f"\u2212{v:g}" for v in gone])

def compute_changes(dash, snap, latest):
    """snap = {symbol: {c_asof, c, p_asof, p}} (levels as saved by earlier runs). Returns (flags, details, change_rows, new_snap)."""
    flags, details, rows, new_snap = [], [], [], {}
    for _, r in dash.iterrows():
        sym = r["symbol"]; sn, rn = _lv(r); old = snap.get(sym)
        if old is None: prev, prev_asof, cur_prev = None, "", None
        elif old["c_asof"] == latest: prev, prev_asof = (old["p"], old["p_asof"]) if old["p_asof"] else (None, "")   # same-day re-run
        else: prev, prev_asof = old["c"], old["c_asof"]
        new_snap[sym] = {"c_asof": latest, "c": (sn, rn), "p_asof": prev_asof, "p": prev if prev else ((None,) * 3, (None,) * 3)}
        flag = det = ""
        if prev:
            sc, rc = sn != prev[0], rn != prev[1]
            if sc or rc:
                flag = "S+R changed" if (sc and rc) else ("Support changed" if sc else "Resistance changed")
                det = " | ".join(x for x in ((_diff(prev[0], sn, "S") if sc else ""), (_diff(prev[1], rn, "R") if rc else "")) if x)
                rows.append({"symbol": sym, "change": flag, "was (date)": prev_asof, "support was": _fmt(prev[0]), "support now": _fmt(sn),
                             "resistance was": _fmt(prev[1]), "resistance now": _fmt(rn), "detail": det})
        flags.append(flag); details.append(det)
    return flags, details, rows, new_snap

def _snap_rows(new_snap):
    out = []
    for sym, d in new_snap.items():
        c, p = d["c"], d["p"]
        out.append([sym, d["c_asof"], *c[0], *c[1], d["p_asof"], *p[0], *p[1]])
    return pd.DataFrame(out, columns=SNAP_HDR)

def _snap_read(sh):
    try: vals = sh.worksheet("Levels Snapshot").get_all_values()
    except gspread.WorksheetNotFound: vals = []
    snap = {}
    for r in vals[1:]:
        r = r + [""] * (15 - len(r))
        snap[r[0]] = {"c_asof": r[1], "c": (tuple(_f(x) for x in r[2:5]), tuple(_f(x) for x in r[5:8])), "p_asof": r[8],
                      "p": (tuple(_f(x) for x in r[9:12]), tuple(_f(x) for x in r[12:15]))}
    if snap: return snap
    # first ever run: use the levels currently on the dashboard as "last night"
    try:
        v = sh.worksheet("Rule Dashboard").get_all_values(); hdr = [h.strip() for h in v[0]]; ix = {h: i for i, h in enumerate(hdr)}
        asof = ""
        try: asof = {r[0]: r[1] for r in sh.worksheet("Run Info").get_all_values()[1:]}.get("Latest data date", "")
        except Exception: pass
        for r in v[1:]:
            if not r or not r[0].strip(): continue
            g = lambda c: r[ix[c]] if c in ix and ix[c] < len(r) else ""
            snap[r[0].strip()] = {"c_asof": asof, "c": (tuple(_f(g(c)) for c in LV_S), tuple(_f(g(c)) for c in LV_R)), "p_asof": "", "p": ((None,) * 3, (None,) * 3)}
    except Exception as e: print("levels baseline unavailable:", e)
    return snap

def annotate_levels(sh, dash, latest):
    snap = _snap_read(sh)
    flags, details, rows, new_snap = compute_changes(dash, snap, str(latest))
    dash = dash.copy(); dash["Levels \u0394"] = flags; dash["Levels \u0394 detail"] = details
    cols = ["symbol", "change", "was (date)", "support was", "support now", "resistance was", "resistance now", "detail"]
    ch = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame([["(no level changes since last run)"] + [""] * 7], columns=cols)
    wc = _write(sh, "Levels Changes", ch)
    try: wc.freeze(rows=1)
    except Exception: pass
    _write(sh, "Levels Snapshot", _snap_rows(new_snap))
    n = len(rows); print(f"levels changed for {n} of {len(dash)} stocks")
    if n:
        try:
            import notify
            if notify.enabled():
                names = ", ".join(r["symbol"] for r in rows[:60]) + (" ..." if n > 60 else "")
                notify.send(f"\U0001F4D0 Imbalance levels changed for {n} stocks \u2014 redraw: {names}\n(details in the 'Levels Changes' tab)")
        except Exception as e: print("telegram failed:", e)
    return dash

def _flag_levels(sh, ws, dash):
    """Amber fill on non-empty 'Levels \u0394' cells."""
    cols = list(dash.columns)
    if "Levels \u0394" not in cols: return
    i = cols.index("Levels \u0394"); L = gspread.utils.rowcol_to_a1(1, i + 1)[:-1]
    sh.batch_update({"requests": [{"addConditionalFormatRule": {"index": 0, "rule": {
        "ranges": [{"sheetId": ws.id, "startRowIndex": 1, "endRowIndex": len(dash) + 1, "startColumnIndex": i, "endColumnIndex": i + 2}],
        "booleanRule": {"condition": {"type": "CUSTOM_FORMULA", "values": [{"userEnteredValue": f"=LEN(${L}2)>0"}]},
                        "format": {"backgroundColor": {"red": 1.0, "green": 0.9, "blue": 0.6}}}}}}]})

def publish(dash, hist, latest, today):
    sh = open_sheet()
    try: dash = annotate_levels(sh, dash, latest)
    except Exception as e: print("levels-changed step failed (dashboard still updated):", e)
    ws = _write(sh, "Rule Dashboard", dash)
    cols = list(dash.columns); specs = []
    strip = [i for i, c in enumerate(cols) if re.fullmatch(r"\d{2}-\d{2}", str(c))]
    if strip: specs.append((min(strip), max(strip) + 1, -50, 50))
    for nm in ("Today %", "High %", "Low %"):
        if nm in cols: i = cols.index(nm); specs.append((i, i + 1, -5, 5))
    if specs: _heat(sh, ws, specs)                      # clears old rules first
    _flag_oversold(sh, ws, dash, strip_only=True)
    _heat_atm(sh, ws, dash)
    _numfmt(sh, ws, dash)
    try: _flag_levels(sh, ws, dash)
    except Exception as e: print("level highlight skipped:", e)
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
