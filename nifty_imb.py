# ==================================================================================
# NIFTY 50 IMBALANCE SUPPORT / RESISTANCE — the SAME logic as the stock imbalance levels in the conviction engine.
#   Nothing is re-invented: fvg_lab.nearest_imbalances() is called UNCHANGED on Nifty's daily candles (symbol "NIFTY"), exactly as the nightly job
#   calls it for the stocks:
#     bull FVG (support):    high[i-1] < low[i+1]   -> gap [high[i-1] .. low[i+1]],  support LEVEL = top of the gap
#     bear FVG (resistance): low[i-1] > high[i+1]   -> gap [high[i+1] .. low[i-1]],  resistance LEVEL = bottom of the gap
#     a gap is dead once a later daily CLOSE goes through it; the nearest 3 unfilled supports below and 3 resistances above price are kept;
#     in_imb = price is currently inside an unfilled gap (bull / bear).
#   Daily candles: NSE's own index file (ind_close_all_DDMMYYYY.csv, the same archive host as the stock bhavcopy), "Nifty 50" row; Dhan daily as fallback.
#   OUTPUT  tab "NIFTY Imbalance": today's levels + a table of every unfilled gap (edges, date formed, age, distance) + the last 10 closes with the levels
#           as of each close (no look-ahead).   Leaves the old "NIFTY" tab and the stock levels untouched.   Research only.
#   Usage: python nifty_imb.py [--selftest]     env: GCP_SA_JSON, SHEET_ID, (fallback) DHAN_HEADERS_JSON, NIMB_DAYS (calendar days of history, default 200)
# ==================================================================================
import os, sys, io, json, time, datetime as dt
import numpy as np, pandas as pd
import fvg_lab as FL

SELFTEST = "--selftest" in sys.argv
UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{d}.csv"
TRANSPORT = None

# ───────────────────────────── daily candles ─────────────────────────────
def parse_index_csv(text, name="Nifty 50"):
    df = pd.read_csv(io.StringIO(text)); df.columns = [c.strip().lower() for c in df.columns]
    nm = next(c for c in df.columns if "index name" in c); df = df[df[nm].astype(str).str.strip().str.lower() == name.lower()]
    if df.empty: return None
    g = lambda *keys: next(c for c in df.columns if all(k in c for k in keys))
    r = df.iloc[0]; f = lambda c: float(str(r[c]).replace(",", ""))
    return {"open": f(g("open")), "high": f(g("high")), "low": f(g("low")), "close": f(g("clos"))}

def get_day(d):
    if TRANSPORT: return TRANSPORT(d)
    import requests
    try:
        r = requests.get(URL.format(d=d.strftime("%d%m%Y")), headers=UA, timeout=30)
        if r.status_code != 200 or len(r.content) < 200: return None
        return parse_index_csv(r.text)
    except Exception: return None

def daily_from_nse(days):
    from concurrent.futures import ThreadPoolExecutor
    ds = [dt.date.today() - dt.timedelta(days=i) for i in range(days) if (dt.date.today() - dt.timedelta(days=i)).weekday() < 5]
    with ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(get_day, ds))
    rows = [{"date": d.strftime("%Y-%m-%d"), "symbol": "NIFTY", **r} for d, r in zip(ds, res) if r]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()

def daily_from_dhan(days):
    import requests
    h = {"Content-Type": "application/json", "Origin": "https://dext.dhan.co", "Referer": "https://dext.dhan.co/", "User-Agent": UA["User-Agent"]}
    h.update(json.loads(os.environ.get("DHAN_HEADERS_JSON") or "{}")); now = int(time.time())
    r = requests.post("https://ticks.dhan.co/getDataH", json={"EXCH": "NSE", "SEG": "I", "INST": "INDEX", "SEC_ID": 13, "INTERVAL": "D", "START": now - days * 86400, "END": now, "EXPCODE": 0, "SYM": "NIFTY"}, headers=h, timeout=60)
    d = (r.json().get("data") or {}) if r.status_code == 200 else {}
    if not d.get("t"): return pd.DataFrame()
    ts = pd.to_datetime(np.array(d["t"], float), unit="s") + pd.Timedelta(hours=5, minutes=30)
    return pd.DataFrame({"date": ts.strftime("%Y-%m-%d"), "symbol": "NIFTY", "open": d["o"], "high": d["h"], "low": d["l"], "close": d["c"]}).drop_duplicates("date").sort_values("date").reset_index(drop=True)

# ───────────────────────────── levels (the engine's own function) ─────────────────────────────
def levels(feat):
    r = FL.nearest_imbalances(feat); return r.iloc[0].to_dict() if len(r) else {}

def gap_table(feat):
    """Every unfilled gap, using the engine's detect_fvgs and its same 'filled' rule. Read-only extras for display."""
    g = feat.sort_values("date").reset_index(drop=True); h = g["high"].to_numpy(float); l = g["low"].to_numpy(float); c = g["close"].to_numpy(float); n = len(g); px = c[-1]; out = []
    for idx, typ, bot, top in FL.detect_fvgs(h, l):
        if idx >= n: continue
        af = c[idx + 1:]
        if typ == "bull" and len(af) and (af < bot).any(): continue
        if typ == "bear" and len(af) and (af > top).any(): continue
        inside = bot <= px <= top
        level = top if typ == "bull" else bot
        out.append({"type": "support (bull gap)" if typ == "bull" else "resistance (bear gap)", "level": round(level, 1), "gap bottom": round(bot, 1), "gap top": round(top, 1),
                    "formed": g["date"][idx], "age (sessions)": n - 1 - idx, "from last close %": round((level / px - 1) * 100, 2), "price inside": "yes" if inside else ""})
    return pd.DataFrame(out)

def asof_table(feat, k=10):
    rows = []
    for i in range(max(5, len(feat) - k), len(feat)):
        f = feat.iloc[:i + 1]; L = levels(f)
        rows.append([f["date"].iloc[-1], float(f["close"].iloc[-1])] + [L.get(x, "") for x in ("imb_sup1", "imb_sup2", "imb_sup3", "imb_res1", "imb_res2", "imb_res3", "in_imb")])
    return rows[::-1]

# ───────────────────────────── main ─────────────────────────────
def main():
    import live_refresh_local as LR
    days = int(os.environ.get("NIMB_DAYS", "200")); feat = daily_from_nse(days); src = "NSE index file"
    if len(feat) < 30:
        print("NSE index file gave", len(feat), "days; trying Dhan"); feat = daily_from_dhan(days); src = "Dhan daily"
    if len(feat) < 30: raise SystemExit("could not get enough Nifty daily candles from NSE or Dhan")
    print(src, len(feat), "daily candles", feat["date"].iloc[0], "->", feat["date"].iloc[-1])
    L = levels(feat); gt = gap_table(feat); last = float(feat["close"].iloc[-1]); stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    sh = LR.open_sheet()
    import nifty_levels as NL
    top = [["NIFTY imbalance levels (same logic as the stocks)", "", "", "", ""],
           ["last close", last, "as of", feat["date"].iloc[-1], f"source: {src}; run {stamp}"],
           ["", "level", "% from close", "", ""]]
    for k in (3, 2, 1): v = L.get(f"imb_res{k}", ""); top.append([f"imb_res{k}", v, "" if v == "" else round((v / last - 1) * 100, 2), "", ""])
    top.append(["in_imb (price inside a gap)", L.get("in_imb", "") or "no", "", "", ""])
    for k in (1, 2, 3): v = L.get(f"imb_sup{k}", ""); top.append([f"imb_sup{k}", v, "" if v == "" else round((v / last - 1) * 100, 2), "", ""])
    top += [[], ["ALL UNFILLED GAPS (level = the number the stock engine uses; the gap edges are for reference)"], list(gt.columns) if len(gt) else ["none"]]
    body = gt.sort_values("level", ascending=False).values.tolist() if len(gt) else []
    ast = [[], ["LAST 10 CLOSES — levels as of each close (no look-ahead)"], ["date", "close", "imb_sup1", "imb_sup2", "imb_sup3", "imb_res1", "imb_res2", "imb_res3", "in_imb"]] + asof_table(feat)
    rows = top + body + ast; w = max(len(r) for r in rows); rows = [r + [""] * (w - len(r)) for r in rows]
    NL.sheet_put(sh, "NIFTY Imbalance", rows[0], rows[1:])
    print("written"); print({k: v for k, v in L.items()}); print(gt.sort_values("level", ascending=False).to_string(index=False) if len(gt) else "no unfilled gaps")

# ───────────────────────────── self-test ─────────────────────────────
def _fake_feat():
    # 12 days: a bull FVG forms between day 3 (high 105) and day 5 (low 108) -> support level 108 ; a bear gap forms later above
    H = [100, 104, 110, 115, 120, 119, 118, 117, 116, 115, 114, 113.0]; Lw = [95, 99, 105, 111, 108, 112, 113, 112, 111, 110, 109, 108.0]; C = [99, 103, 109, 114, 118, 116, 115, 114, 113, 112, 111, 112.0]
    dates = [(dt.date(2026, 10, 1) + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(12)]
    return pd.DataFrame({"date": dates, "symbol": "NIFTY", "open": C, "high": H, "low": Lw, "close": C})

def selftest():
    csv = "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change\nNifty Next 50,29-10-2026,1,2,3,4,0\nNifty 50,29-10-2026,22900.5,23059.0,22712.3,23048.1,100\n"
    r = parse_index_csv(csv); assert r == {"open": 22900.5, "high": 23059.0, "low": 22712.3, "close": 23048.1}, r
    f = _fake_feat(); L = levels(f); g = gap_table(f); print(L); print(g)
    # the table must agree with the engine's own levels, level for level
    sup = sorted(g[g["type"].str.startswith("support") & (g["price inside"] == "")]["level"], reverse=True)[:3]; res = sorted(g[g["type"].str.startswith("resistance") & (g["price inside"] == "")]["level"])[:3]
    for k, v in enumerate(sup): assert L[f"imb_sup{k+1}"] == v, (L, sup)
    for k, v in enumerate(res): assert L[f"imb_res{k+1}"] == v, (L, res)
    a = asof_table(f, 4); print(a); assert len(a) == 4 and a[0][0] == f["date"].iloc[-1]
    # no look-ahead: the as-of row for an earlier date equals levels() on data cut at that date
    assert a[-1][2:9] == [levels(f.iloc[:len(f) - 3]).get(x, "") for x in ("imb_sup1", "imb_sup2", "imb_sup3", "imb_res1", "imb_res2", "imb_res3", "in_imb")]
    global TRANSPORT
    TRANSPORT = lambda d: {"open": 1, "high": 3, "low": 0.5, "close": 2} if d.weekday() < 5 else None
    D = daily_from_nse(40); assert len(D) >= 25 and (D["symbol"] == "NIFTY").all()
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
