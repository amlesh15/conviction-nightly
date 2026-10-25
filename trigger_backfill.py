# ==================================================================================
# TRIGGER BACKFILL — replays the live trigger engine over past sessions and writes the triggers it WOULD
# have fired into the "Triggers" tab (same columns), with spot 15/30/60/EOD outcomes filled in, then rebuilds
# the "Trigger Scorecard" tab.
#
#   Same maths as live: trigger_engine.compute_flow / flow_signal, daily-score gate (previous close, no leak),
#   per-stock daily cap + cooldown, max new per run, 09:30-15:15 window, 5-minute run grid.
#   NOT replayable (needs data that was never stored): live S/R flag + room-to-level gate, the Vol× in-play
#   universe filter, ATM option premium / option outcomes (written as n/a). Rows are marked "BACKFILL" in `note`.
#
#   Days that already have live rows for a symbol are left alone (no duplicates).
#
#   Env:  BF_FROM=2026-10-19   first session to backfill (default 2026-10-19)
#         BF_TO=                last session (default: latest session with data)
#         BF_DRY=true           print only, write nothing
#   Usage: python trigger_backfill.py            (GitHub Action, needs the Sensibull cookie + sheet secrets)
#          python trigger_backfill.py --selftest
# ==================================================================================
import os, sys, datetime as dt
import numpy as np, pandas as pd
import trigger_engine as TE
import backtest_triggers as BT

HDR = TE.HDR; OUT = TE.OUT_COLS
FROM = os.environ.get("BF_FROM", "2026-10-19"); TO = os.environ.get("BF_TO", ""); DRY = os.environ.get("BF_DRY", "").lower() == "true"

def build_rows(G, existing):
    """G = backtest grid with 'score'. existing = set of (date, symbol). Returns list of rows (HDR order)."""
    T = BT.select_triggers(G, use_score=True, use_vwap=True, strong=False)
    f, t = dt.date.fromisoformat(FROM), (dt.date.fromisoformat(TO) if TO else None)
    rows = []
    for r in T.itertuples():
        s = r.session
        if s < f or (t and s > t): continue
        d = s.strftime("%Y-%m-%d")
        if (d, r.symbol) in existing: continue
        sc = r.score; sign = -1.0 if r.side == "PUT" else 1.0
        strong = (r.side == "CALL" and sc >= 30) or (r.side == "PUT" and sc <= -30) if sc == sc else False
        fav = lambda v: "" if v != v else round(float(v) * sign, 2)
        rec = {"date": d, "time": f"{r.m // 60:02d}:{r.m % 60:02d}", "symbol": r.symbol, "side": r.side,
               "tag": "strong-aligned" if strong else "neutral-ok", "spot": round(float(r.spot), 2),
               "imb": round(float(r.imb), 2), "rel_vol": round(float(r.rel_vol), 2), "ret_w%": round(float(r.ret), 2),
               "vs_vwap%": round(float(r.vs_vwap), 2), "score": "" if sc != sc else sc,
               "note": "BACKFILL replay: no S/R, Vol× or option gates",
               "spot15%": fav(r.r15), "spot30%": fav(r.r30), "spot60%": fav(r.r60), "spotEOD%": fav(r.rEOD),
               "opt15%": "n/a", "opt30%": "n/a", "opt60%": "n/a", "optEOD%": "n/a"}
        rows.append([rec.get(h, "") for h in HDR])
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return rows

def main():
    import live_refresh_local as LR, gspread
    sh = LR.open_sheet(); vals = sh.worksheet("Rule Dashboard").get_all_values()
    hdr = [h.strip() for h in vals[0]]; si = hdr.index("symbol") if "symbol" in hdr else 0
    symbols = sorted({r[si].strip().upper() for r in vals[1:] if len(r) > si and r[si].strip()})
    hist_vals = sh.worksheet("Rule History").get_all_values()
    print(f"{len(symbols)} symbols; fetching 1-min bars ...")
    bars = BT.load_bars(symbols)
    if not bars: print("No bars fetched (cookie expired?)"); sys.exit(1)
    sess = sorted({x for d in bars.values() for x in d["session"].unique()}); print("sessions:", sess[0], "..", sess[-1])
    G = BT.build_grid(bars)
    sh_hist = BT.score_history(hist_vals, sess)
    G["score"] = [BT.prev_score(sh_hist, s, d) for s, d in zip(G["symbol"], G["session"])]
    try: ws = sh.worksheet("Triggers"); table = ws.get_all_values()
    except gspread.WorksheetNotFound: ws = None; table = [HDR]
    ix = {h: i for i, h in enumerate(table[0])}
    existing = {(r[ix["date"]], r[ix["symbol"]]) for r in table[1:] if len(r) > ix["symbol"]}
    rows = build_rows(G, existing)
    days = sorted({r[0] for r in rows}); print(f"backfill: {len(rows)} triggers on {len(days)} days: {days}")
    if DRY or not rows: return
    if ws is None: ws = TE._triggers_ws(sh, create=True)
    ws.append_rows(rows, value_input_option="USER_ENTERED")
    full = ws.get_all_values(); sc = TE.scorecard(full)
    if not sc.empty:
        try: wsc = sh.worksheet("Trigger Scorecard"); wsc.clear()
        except gspread.WorksheetNotFound: wsc = sh.add_worksheet(title="Trigger Scorecard", rows=60, cols=14)
        o = sc.astype(object).where(sc.notna(), "")
        wsc.update(values=[list(o.columns)] + o.values.tolist(), range_name="A1", value_input_option="USER_ENTERED")
    print("written to Triggers + Trigger Scorecard")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        FROM = "2026-10-01"
        bars = BT._synth(); G = BT.build_grid(bars); G["score"] = 0.0
        rows = build_rows(G, set()); print("synthetic backfill rows:", len(rows))
        assert all(len(r) == len(HDR) for r in rows)
        if rows:
            df = pd.DataFrame(rows, columns=HDR); print(df[["date","time","symbol","side","spot","spot30%","spotEOD%"]].head(5).to_string(index=False))
            tbl = [HDR] + rows; sc = TE.scorecard(tbl); print(sc[["group","triggers","spot30 avg%","spotEOD avg%"]].to_string(index=False))
        again = build_rows(G, {(r[0], r[2]) for r in rows}); assert len(again) == 0, "dedupe failed"
        print("SELFTEST OK")
    else:
        main()
