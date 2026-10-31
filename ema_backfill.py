# ==================================================================================
# EMA 12/26 + VWAP BACKFILL — replays the NEW alert rule over the last N sessions (default 5) and writes the alerts into "EMA12-26 Triggers",
# fills their performance, and rebuilds the "EMA12-26 Scorecard".
#
#   Rule (identical to the live alert):  the STOCK's hourly EMA12 crosses above (CALL) / below (PUT) EMA26 on a completed hourly candle
#       AND the stock price is above (CALL) / below (PUT) BOTH the rolling 3-session and 5-session VWAP.
#   Each row carries: the stock price at the trigger (the cross candle's close), the 3-day and 5-day VWAP at that moment, and the last 5 daily conviction scores.
#   NO LOOK-AHEAD:
#     * EMAs are causal (a candle only uses itself and earlier candles).
#     * VWAP at the trigger uses ONLY candles that had completed by then (today's session so far + earlier sessions) — later candles are never read.
#     * Conviction scores are the 5 most recent scores dated BEFORE the trigger day (the score dated D is only known after D closes).
#   APPROXIMATIONS (same as the earlier backfill):
#     * "ATM option" = the CURRENT ATM contract of the current expiry (the strike that was ATM at the time is not stored). Its premium at the trigger comes from that
#       contract's own hourly candle; if that candle did not trade (volume 0 — common for a far-month contract) the premium is left blank and no option outcomes are filled.
#     * The price tested against the VWAPs is the cross candle's close (the live alert uses the latest price a few minutes later).
#   Rows already in the tab (same date, candle time, stock, side) are skipped.
#
#   Env: BF_DAYS=5   BF_DRY=true (print only)      Usage: python ema_backfill.py | --selftest
# ==================================================================================
import os, sys, datetime as dt
import concurrent.futures as cf
import numpy as np, pandas as pd
import ema_cross_alert as E
import backtest_triggers as BT

DAYS = int(os.environ.get("BF_DAYS", 5)); DRY = os.environ.get("BF_DRY", "").lower() == "true"

def scores_before(hist_sym, d, n=5):
    ks = sorted(k for k in hist_sym if k < d); return [hist_sym[k] for k in ks[-n:]]

def targets_from(hours, days):
    """Last `days` sessions present in the stock hourly data."""
    ss = sorted({s for h in hours for s in h["session"].unique()}); return set(ss[-days:])

def events_for(h, hist_sym, targets, now):
    """All alerts the live rule would have sent for one stock over the target sessions (CALL and PUT)."""
    d = E.completed_hourly(h, now).reset_index(drop=True)
    if len(d) < E.SLOW + 3: return []
    e12, e26 = E.emas(d["close"]); out = []; per_day = {}
    for i in range(E.SLOW + 2, len(d)):
        sess = d["session"].iloc[i]
        if sess not in targets: continue
        for side in ("CALL", "PUT"):
            hit = (e12[i] > e26[i] and e12[i - 1] <= e26[i - 1]) if side == "CALL" else (e12[i] < e26[i] and e12[i - 1] >= e26[i - 1])
            if not hit: continue
            end = d["end"].iloc[i].to_pydatetime(); px = float(d["close"].iloc[i])
            vw = E.live_vwaps(h, pd.Timestamp(end) - pd.Timedelta(seconds=1))          # candles that had completed by the trigger — the next candle (starts at `end`) is excluded
            if vw[3] is None or vw[5] is None or not E.vwap_ok(side, px, vw[3], vw[5]): continue
            if per_day.get((sess, side), 0) >= E.MAX_PER_DAY: continue
            per_day[(sess, side)] = per_day.get((sess, side), 0) + 1
            out.append({"side": side, "cross": {"end": end, "close": px, "e12": float(e12[i]), "e26": float(e26[i])}, "sc": scores_before(hist_sym, sess),
                        "vwap3": vw[3], "vwap5": vw[5], "spot": px})
    return out

def premium_at(oh, end):
    """Option's hourly close on the candle that ended at `end`; None if that candle did not trade."""
    if oh is None or oh.empty: return None
    d = E.completed_hourly(oh, pd.Timestamp(end) + pd.Timedelta(minutes=1)); m = d[d["end"] == pd.Timestamp(end)]
    if m.empty: return None
    r = m.iloc[-1]; v = r["volume"] if "volume" in m.columns else 1
    return float(r["close"]) if (v == v and v > 0) else None

def main():
    import live_refresh_local as LR, gspread
    now = dt.datetime.now(E.IST).replace(tzinfo=None)
    sh = LR.open_sheet(); syms = sorted(E.read_scores(sh.worksheet("Rule Dashboard")))
    hist_vals = sh.worksheet("Rule History").get_all_values()
    days = [d.date() for d in pd.bdate_range(end=now.date(), periods=90)]; hist = BT.score_history(hist_vals, days)
    print(f"{len(syms)} symbols | score history for {len(hist)} stocks | replaying last {DAYS} sessions")
    def pull(sym):
        try: return sym, E.get_hourly(sym, now)
        except Exception as e: print(f"  {sym}: {type(e).__name__}: {e}"); return sym, None
    stock = {}
    with cf.ThreadPoolExecutor(max_workers=E.WORKERS) as ex:
        for s, h in ex.map(pull, syms):
            if h is not None and not h.empty: stock[s] = h
    tg = targets_from(list(stock.values()), DAYS)
    print(f"stock hourly candles for {len(stock)} stocks | sessions replayed: {sorted(tg)}")
    found = []
    for sym, h in stock.items():
        for ev in events_for(h, hist.get(sym, {}), tg, now): ev["sym"] = sym; found.append(ev)
    print(f"{len(found)} alerts pass the rule; fetching ATM options for them ...")
    atm_c, opt_c = {}, {}
    def opt(sym, side):
        try:
            if sym not in atm_c: atm_c[sym] = E.get_atm(sym)
            a = atm_c[sym]
            if not a: return None
            typ = "CE" if side == "CALL" else "PE"; osym = a[typ]()
            if osym not in opt_c: opt_c[osym] = E.get_hourly(osym, now)
            return osym, a, opt_c[osym]
        except Exception as e: print(f"  {sym} {side}: {type(e).__name__}: {e}"); return None
    try: ws = sh.worksheet(E.TAB); table = ws.get_all_values()
    except gspread.WorksheetNotFound: ws = None; table = [E.HDR]
    ix = {h: i for i, h in enumerate(table[0])}
    have = {(r[ix["date"]], r[ix["time"]], r[ix["symbol"]], r[ix["side"]]) for r in table[1:] if len(r) > ix["side"]}
    rows = []
    for ev in found:
        c = ev["cross"]
        if (c["end"].strftime("%Y-%m-%d"), c["end"].strftime("%H:%M"), ev["sym"], ev["side"]) in have: continue
        o = opt(ev["sym"], ev["side"])
        if o: osym, a, oh = o; ev.update(osym=osym, strike=a["atm"], expiry=str(a["expiry"]), prem=premium_at(oh, c["end"]))
        else: ev.update(osym="", strike="", expiry="", prem=None)
        row = E.event_row(ev, now); row[E.HDR.index("alert time")] = "backfill"; rows.append(row)
    rows.sort(key=lambda x: (x[0], x[1], x[3]))
    df = pd.DataFrame(rows, columns=E.HDR) if rows else pd.DataFrame(columns=E.HDR)
    if len(df):
        print(f"backfill: {len(df)} alerts | with option premium: {int((df['entry prem'] != '').sum())}"); print(df.groupby(["date", "side"]).size().to_string())
        print(df[["date", "time", "symbol", "side", "spot", "vwap3", "vwap5", "s1", "s2", "s3", "s4", "s5", "entry prem"]].head(25).to_string(index=False))
    else: print("backfill: 0 alerts")
    if DRY or not rows: return
    if ws is None:
        ws = sh.add_worksheet(title=E.TAB, rows=2000, cols=len(E.HDR) + 2); ws.update(values=[E.HDR], range_name="A1")
        try: ws.freeze(rows=1)
        except Exception: pass
    ws.append_rows(rows, value_input_option="USER_ENTERED")
    table = ws.get_all_values(); ups = E.fill_outcomes(table, now)
    if ups:
        ws.batch_update([{"range": gspread.utils.rowcol_to_a1(r + 1, c + 1), "values": [[v]]} for (r, c), v in ups.items()], value_input_option="USER_ENTERED")
        for (r, c), v in ups.items(): table[r] = table[r] + [""] * (len(table[0]) - len(table[r])); table[r][c] = v
    E.put_scorecard(sh, table); print(f"written {len(rows)} rows, {len(ups)} outcome cells, scorecard rebuilt")

def selftest():
    hc, endc = E._find_cross_series(+1); hp, endp = E._find_cross_series(-1); now = dt.datetime(2026, 11, 2, 12, 0)
    dc = E.completed_hourly(hc, now); cs = dc[dc["end"] == pd.Timestamp(endc)]["session"].iloc[0]
    hist = {cs - dt.timedelta(days=k): 10.0 * (6 - k) for k in range(1, 7)}
    tgt = set(dc["session"].unique())
    ev = events_for(hc, hist, tgt, now); calls = [e for e in ev if e["side"] == "CALL"]
    print("CALL events:", [(e["cross"]["end"], round(e["spot"], 2), round(e["vwap3"], 2), round(e["vwap5"], 2), e["sc"]) for e in calls])
    for e in ev:                                         # every event obeys the rule, scores are the 5 BEFORE the trigger day and in date order
        assert E.vwap_ok(e["side"], e["spot"], e["vwap3"], e["vwap5"]) and len(e["sc"]) <= 5 and e["sc"] == sorted(e["sc"])
        assert e["sc"] == [hist[k] for k in sorted(hist) if k < e["cross"]["end"].date()][-5:]
    # NO LOOK-AHEAD: wreck every candle after the cross candle -> the alert, its VWAPs and its price must not change
    for h, end, side in ((hc, endc, "CALL"), (hp, endp, "PUT")):
        base = [e for e in events_for(h, hist, tgt, now) if e["side"] == side and e["cross"]["end"] == end]
        w = h.copy(); later = w["ts"] >= pd.Timestamp(end)
        for col in ("open", "high", "low", "close"): w.loc[later, col] = w.loc[later, col] * 7.0
        w.loc[later, "volume"] = w.loc[later, "volume"] * 50
        again = [e for e in events_for(w, hist, tgt, now) if e["side"] == side and e["cross"]["end"] == end]
        assert len(base) == len(again), (side, len(base), len(again))
        for a, b in zip(base, again): assert abs(a["vwap3"] - b["vwap3"]) < 1e-9 and abs(a["vwap5"] - b["vwap5"]) < 1e-9 and abs(a["spot"] - b["spot"]) < 1e-9, "future candle leaked into the VWAP"
    # a stock not in the target sessions gives nothing; the per-day cap holds
    assert events_for(hc, hist, set(), now) == []
    assert all(sum(1 for e in ev if e["side"] == s and e["cross"]["end"].date() == d) <= E.MAX_PER_DAY for s in ("CALL", "PUT") for d in {x["cross"]["end"].date() for x in ev})
    # premium helper: traded candle -> price, zero-volume candle -> None
    oh = E._hourly(closes=[5.0 + 0.01 * i for i in range(42)]); end = oh["ts"].iloc[30] + pd.Timedelta(minutes=60)
    assert abs(premium_at(oh, end.to_pydatetime()) - float(oh["close"].iloc[30])) < 1e-9
    z = oh.copy(); z.loc[z.index[30], "volume"] = 0; assert premium_at(z, end.to_pydatetime()) is None
    print("SELFTEST OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: main()
