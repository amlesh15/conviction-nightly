# ==================================================================================
# ROLLING VWAP LEVELS (3-session and 5-session) as PRICES — same numbers the conviction engine uses for "vwap3 slp / vwap5 slp",
# but showing the VWAP itself, not only its slope.   The conviction engine is NOT touched; this is a separate, read-only job.
#   VWAP(W) on day t = sum over the last W sessions of [ (high+low+close)/3 * volume ]  /  sum of volume over those W sessions.
#   It ROLLS: every new session drops the oldest day and adds the newest, so it never restarts.  Daily candles = the engine's own masterdata
#   (NSE cash bhavcopy, corporate-action adjusted).
#   OUTPUT tabs: "VWAP Levels" (one row per stock: latest close, vwap3, vwap5, distance of close from each, slopes in %/session AND degrees as on the dashboard)
#                "VWAP History" (last VWAP_HIST sessions per stock: close, vwap3, vwap5 — for charts / checking)
#   Usage: python vwap_levels.py [--selftest]      env: GCP_SA_JSON, SHEET_ID, CACHE_DIR, START_DATE (as run_nightly), VWAP_HIST (default 15)
# ==================================================================================
import os, sys, math
import numpy as np, pandas as pd

SELFTEST = "--selftest" in sys.argv
HIST = int(os.environ.get("VWAP_HIST", "15"))

def vwap_frame(md):
    """Adds vwap3, vwap5 (prices) and the slope columns, exactly as conviction_engine does (shift 2 / shift 4, per symbol)."""
    d = md.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    for c in ("high", "low", "close", "volume"): d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tpv"] = (d["high"] + d["low"] + d["close"]) / 3 * d["volume"]
    for W in (3, 5):
        num = d.groupby("symbol")["tpv"].transform(lambda s: s.rolling(W, min_periods=W).sum())
        den = d.groupby("symbol")["volume"].transform(lambda s: s.rolling(W, min_periods=W).sum())
        d[f"vwap{W}"] = num / den.replace(0, np.nan)
    d["vwap3_prev"] = d.groupby("symbol")["vwap3"].shift(2); d["vwap5_prev"] = d.groupby("symbol")["vwap5"].shift(4)
    d["vwap3_slope"] = ((d["vwap3"] / d["vwap3_prev"] - 1) * 100 / 2).round(3)
    d["vwap5_slope"] = ((d["vwap5"] / d["vwap5_prev"] - 1) * 100 / 4).round(3)
    return d

def deg(x): return round(math.degrees(math.atan(float(x))), 1) if x == x and x is not None else ""

def latest_table(d):
    last = d.sort_values("date").groupby("symbol").tail(1).sort_values("symbol"); rows = []
    for r in last.itertuples():
        f = lambda v, n=2: "" if v != v else round(float(v), n)
        rows.append([r.symbol, str(r.date), f(r.close), f(r.vwap3), f(r.vwap5),
                     "" if r.vwap3 != r.vwap3 else round((r.close / r.vwap3 - 1) * 100, 2), "" if r.vwap5 != r.vwap5 else round((r.close / r.vwap5 - 1) * 100, 2),
                     f(r.vwap3_slope, 3), f(r.vwap5_slope, 3), deg(r.vwap3_slope), deg(r.vwap5_slope), f(r.vwap3_prev), f(r.vwap5_prev)])
    H = ["symbol", "date", "close", "vwap3", "vwap5", "close vs vwap3 %", "close vs vwap5 %", "vwap3 slope %/session", "vwap5 slope %/session",
         "vwap3 slp (deg)", "vwap5 slp (deg)", "vwap3 two sessions ago", "vwap5 four sessions ago"]
    return H, rows

def history_table(d, n=HIST):
    h = d.sort_values(["symbol", "date"]).groupby("symbol").tail(n).sort_values(["date", "symbol"], ascending=[False, True])
    return ["date", "symbol", "close", "vwap3", "vwap5"], [[str(r.date), r.symbol] + ["" if v != v else round(float(v), 2) for v in (r.close, r.vwap3, r.vwap5)] for r in h.itertuples()]

def main():
    import run_nightly as RN, gspread, live_refresh_local as LR
    today = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")
    G = RN.load_engine(); RN.seed_cache(); md = G["build_masterdata_pit_fast"](RN.START, today, full=False)
    d = vwap_frame(md); H, rows = latest_table(d); HH, hr = history_table(d)
    print(f"{len(rows)} stocks; latest session {md['date'].max()}"); print(pd.DataFrame(rows[:8], columns=H).to_string(index=False))
    sh = LR.open_sheet()
    def put(title, header, body):
        vals = [header] + body
        try: ws = sh.worksheet(title); ws.clear()
        except gspread.WorksheetNotFound: ws = sh.add_worksheet(title=title, rows=len(vals) + 20, cols=len(header) + 2)
        if ws.row_count < len(vals) + 5: ws.resize(rows=len(vals) + 20)
        ws.update(values=vals, range_name="A1", value_input_option="USER_ENTERED")
    put("VWAP Levels", H, rows); put("VWAP History", HH, hr); print("tabs written")

def selftest():
    dates = pd.bdate_range("2026-10-01", periods=8).strftime("%Y-%m-%d")
    h = [10, 11, 12, 13, 14, 15, 16, 17.0]; l = [8, 9, 10, 11, 12, 13, 14, 15.0]; c = [9, 10, 11, 12, 13, 14, 15, 16.0]; v = [100, 200, 100, 300, 100, 100, 200, 100.0]
    md = pd.DataFrame({"date": dates, "symbol": "AAA", "high": h, "low": l, "close": c, "volume": v})
    md2 = md.copy(); md2["symbol"] = "BBB"; md2["close"] += 1; md2["high"] += 1; md2["low"] += 1
    d = vwap_frame(pd.concat([md, md2])); a = d[d["symbol"] == "AAA"].reset_index(drop=True)
    tp = [(h[i] + l[i] + c[i]) / 3 for i in range(8)]
    exp3 = lambda t: sum(tp[i] * v[i] for i in range(t - 2, t + 1)) / sum(v[t - 2:t + 1]); exp5 = lambda t: sum(tp[i] * v[i] for i in range(t - 4, t + 1)) / sum(v[t - 4:t + 1])
    assert np.isnan(a["vwap3"].iloc[1]) and abs(a["vwap3"].iloc[2] - exp3(2)) < 1e-9 and abs(a["vwap3"].iloc[7] - exp3(7)) < 1e-9
    assert np.isnan(a["vwap5"].iloc[3]) and abs(a["vwap5"].iloc[4] - exp5(4)) < 1e-9 and abs(a["vwap5"].iloc[7] - exp5(7)) < 1e-9, "must roll (no restart after 3 / 5 sessions)"
    assert abs(a["vwap3"].iloc[5] - exp3(5)) < 1e-9, "day 6 is still a rolling window, not a fresh start"
    s3 = round((exp3(7) / exp3(5) - 1) * 100 / 2, 3); s5 = round((exp5(7) / exp5(3 + 0) - 1) * 100 / 4, 3) if False else None
    assert abs(a["vwap3_slope"].iloc[7] - s3) < 1e-9
    H, rows = latest_table(d); assert len(rows) == 2 and rows[0][0] == "AAA" and rows[0][3] == round(exp3(7), 2)
    assert rows[0][9] == round(math.degrees(math.atan(s3)), 1)
    HH, hr = history_table(d, 3); assert len(hr) == 6 and hr[0][0] == dates[-1]
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
