# ==================================================================================
# NIFTY OI LIVE — feeds the live OI dashboard.
#   Every 5 minutes (market hours) it pulls 1-min price + OI for every Nifty option strike near spot (nearest weekly expiry, or NOI_EXPIRY),
#   turns the whole session so far into 5-minute snapshots, and writes them to the FIRST tab of a small dedicated Google Sheet ("Nifty OI Live").
#   The dashboard page reads that tab. Past sessions work too (workflow input date) — one pass, then it stops, so you can replay any day.
#   Tab columns: session, expiry, lot, updated, time, spot, strike, ce_oi, pe_oi, ce_px, pe_px      (OI in contracts; price = option last price)
#   Usage: python nifty_oi_live.py [--selftest]
#   env: GCP_SA_JSON, OI_SHEET_ID (the dedicated sheet), SENSIBULL cookie (setup_env.py); optional NOI_DATE (YYYY-MM-DD, default today),
#        NOI_EXPIRY, OI_SPAN (strike half-range in points, default 800), OI_EVERY (seconds, default 300), OI_STOP (HH:MM IST, default 15:40)
# ==================================================================================
import os, sys, time, datetime as dt
import numpy as np, pandas as pd

SELFTEST = "--selftest" in sys.argv
import nifty_oi_day as N          # reuses the Sensibull helpers (candles, chain, Day) unchanged
SPAN = int(os.environ.get("OI_SPAN", "800")); EVERY = int(os.environ.get("OI_EVERY", "300")); STOP = os.environ.get("OI_STOP", "15:40")
HDR = ["session", "expiry", "lot", "updated", "time", "spot", "strike", "ce_oi", "pe_oi", "ce_px", "pe_px", "prev_close"]

def ist_now(): return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)

def snapshots(D, lot, expiry, updated, step=5, prev_close=""):
    """5-minute snapshots of the whole session so far -> list of rows (HDR order)."""
    marks = [D.pos("09:15")]; last = len(D.idx) - 1
    first = D.idx[0]; start = first.hour * 60 + first.minute
    m = (start // step + 1) * step
    while m <= 15 * 60 + 30:                                                   # nothing after the 15:30 close (late prints are not trading)
        ts = pd.Timestamp(f"{D.session} {m // 60:02d}:{m % 60:02d}")
        if ts > D.idx[-1]: break
        marks.append(D.pos(f"{m // 60:02d}:{m % 60:02d}")); m += step
    lastok = D.pos("15:30"); last = min(last, lastok)
    if marks[-1] != last: marks.append(last)                                   # always include the newest minute (never later than 15:30)
    seen, rows = set(), []
    for i in marks:
        if i in seen: continue
        seen.add(i); hm = D.hm(i); sp = round(float(D.spot.iloc[i]), 2)
        for k in D.strikes:
            ce = float(D.oi[("CE", k)].iloc[i]) if ("CE", k) in D.oi.columns else 0.0; pe = float(D.oi[("PE", k)].iloc[i]) if ("PE", k) in D.oi.columns else 0.0
            cp = D.px[("CE", k)].iloc[i] if ("CE", k) in D.px.columns else np.nan; pp = D.px[("PE", k)].iloc[i] if ("PE", k) in D.px.columns else np.nan
            rows.append([D.session, expiry, lot, updated, hm, sp, k, int(ce), int(pe), "" if cp != cp else round(float(cp), 2), "" if pp != pp else round(float(pp), 2), prev_close])
    return rows

def one_pass(session, want_expiry):
    spot = N.candles("NIFTY", days=45 if session else 5)
    if spot is None: raise SystemExit("no NIFTY spot candles (cookie expired?)")
    sess = session or spot["session"].max()
    s = spot[spot["session"] == sess]
    if s.empty: print(f"no spot candles for {sess} yet"); return None
    px = float(s["close"].iloc[-1]); c = round(px / 50) * 50; N.LO, N.HI = c - SPAN, c + SPAN
    e, lot, syms, _ = N.chain(sess, want_expiry)
    import concurrent.futures as cf
    frames = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for (t, k), df in zip(syms.keys(), ex.map(lambda s_: N.candles(s_, days=4), syms.values())):
            if df is None: continue
            d = df[["ts", "session", "close", "volume", "oi"]].copy(); d["strike"] = k; d["typ"] = t; frames.append(d)
    if not frames: raise SystemExit("no option candles returned (cookie expired?)")
    panel = pd.concat(frames, ignore_index=True)
    if not (panel["session"] == sess).any(): print(f"no option candles for {sess}"); return None
    D = N.Day(panel, spot, sess)
    pv = spot[spot["session"] < sess]; prev = round(float(pv["close"].iloc[-1]), 2) if len(pv) else ""
    return snapshots(D, lot, e, ist_now().strftime("%Y-%m-%d %H:%M"), 5, prev), sess, e, len(syms)

def write(rows):
    import gspread, json
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(json.loads(os.environ["GCP_SA_JSON"]), scopes=["https://www.googleapis.com/auth/spreadsheets"])
    sh = gspread.authorize(creds).open_by_key(os.environ["OI_SHEET_ID"]); ws = sh.get_worksheet(0)
    vals = [HDR] + rows
    ws.clear()
    if ws.row_count < len(vals) + 5: ws.resize(rows=len(vals) + 10, cols=len(HDR))
    ws.update(values=vals, range_name="A1", value_input_option="RAW")

def main():
    session = os.environ.get("NOI_DATE") or None; want = os.environ.get("NOI_EXPIRY") or None
    today = ist_now().strftime("%Y-%m-%d"); live = (session in (None, today))
    h, m = (int(x) for x in STOP.split(":")); n = 0; t0 = time.time(); maxmin = int(os.environ.get("OI_MAXMIN", "340"))   # GitHub stops a job at 6 h; end cleanly before that
    while True:
        try:
            out = one_pass(session, want)
            if out:
                rows, sess, e, nc = out; write(rows); n += 1
                print(f"{ist_now():%H:%M:%S} wrote {len(rows)} rows: session {sess}, expiry {e}, {nc} contracts, last snapshot {rows[-1][4]}", flush=True)
        except SystemExit as ex_: print("stop:", ex_); 
        except Exception as ex_: print("pass failed:", repr(ex_)[:300], flush=True)
        now = ist_now()
        if not live or (now.hour, now.minute) >= (h, m) or (time.time() - t0) / 60 > maxmin: break
        if (now.hour, now.minute) < (9, 10): time.sleep(60); continue
        time.sleep(EVERY)
    print(f"done, {n} writes")

def selftest():
    ts = pd.date_range("2026-10-30 09:15", periods=60, freq="min"); rows = []
    spot = pd.DataFrame({"ts": ts, "session": "2026-10-30", "open": 100.0, "high": 101 + np.arange(60) * .1, "low": 99 + np.arange(60) * .1, "close": 100 + np.arange(60) * .1, "volume": 1})
    fr = []
    for t in ("CE", "PE"):
        for k in (22900.0, 23000.0, 23100.0):
            fr.append(pd.DataFrame({"ts": ts, "session": "2026-10-30", "close": 50.0 + np.arange(60) * (.1 if t == "PE" else -.1), "volume": 10, "oi": 100000 + np.arange(60) * (1000 if t == "PE" else -500), "strike": k, "typ": t}))
    D = N.Day(pd.concat(fr, ignore_index=True), spot, "2026-10-30")
    r = snapshots(D, 65, "2026-11-03", "2026-10-30 10:15")
    times = sorted({x[4] for x in r}); assert times[0] == "09:15" and times[-1] == "10:14" and "09:20" in times and "10:10" in times, times
    assert len(r) == len(times) * 3 and all(len(x) == len(HDR) for x in r)
    pe = [x for x in r if x[6] == 23000.0]; assert pe[-1][8] > pe[0][8] and pe[-1][7] < pe[0][7]       # puts added, calls shed
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
