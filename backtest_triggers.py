# ==================================================================================
# TRIGGER BACKTEST — replays the live trigger engine over the last ~30 sessions of 1-min bars.
#
#  * Uses trigger_engine.compute_flow()  (the exact live flow maths) at every 5-minute "run" time.
#  * Same rules as live: imbalance / relative volume / price move / VWAP side / daily-score gate /
#    per-stock daily cap + cooldown / max new per run / 09:30-15:15 window.
#  * Judged at STOCK level (15/30/60 min and end of day). Option premiums can't be replayed (no history) —
#    those keep accumulating forward in the "Trigger Scorecard" tab.
#  * Honest comparison: every trigger is compared with
#       (A) the same stock at the same clock time on OTHER days, and
#       (B) all OTHER stocks at the same moment (that day's market drift).
#    "excess" = trigger return minus that control, in the trade direction. Only excess > 0 is an edge.
#  * Time split: first 60% of sessions = "IS" (in-sample), last 40% = "OOS" (out-of-sample). Judge on OOS.
#
#  NOT replayable (needs live data that wasn't stored): live S/R flag gate, room-to-level gate, the
#  Vol× in-play universe filter. So this tests the flow + VWAP + daily-score core.
#
#  Usage:  python backtest_triggers.py            (needs cookie + sheet secrets, run via GitHub Action)
#          python backtest_triggers.py --selftest (synthetic data, checks the machinery)
# ==================================================================================
import os, sys, datetime as dt, re
import concurrent.futures as cf
import numpy as np, pandas as pd
import trigger_engine as TE

CFG = TE.CFG
STEP = int(os.environ.get("BT_STEP", "5"))          # minutes between simulated live runs
WARM = int(os.environ.get("BT_WARM", "5"))          # sessions skipped at the start (need prior sessions to normalise)
DAYS = int(os.environ.get("BT_DAYS", "45"))         # calendar days of 1-min history to request
HZ = (15, 30, 60)
HZ_COLS = ["r15", "r30", "r60", "rEOD"]

VARIANTS = {   # name: (use daily-score gate, use VWAP confirm)
    "ENGINE (as live)":      (True,  True),
    "no score gate":         (False, True),
    "no VWAP confirm":       (True,  False),
}

# ───────────────────────────── 1. scan every stock at every simulated run time ─────────────────────────────
def scan_symbol(args):
    sym, df = args
    sess = sorted(df["session"].unique()); rows = []
    for k in range(WARM, len(sess)):
        s = sess[k]
        sub = df[df["session"].isin(sess[max(0, k - CFG["N_PRIOR"]):k + 1])].reset_index(drop=True)
        cur = sub[sub["session"] == s].reset_index(drop=True)
        if len(cur) < 60: continue
        mod = cur["mod"].to_numpy(); close = cur["close"].to_numpy(float); eod = close[-1]
        for m in range(CFG["START_MIN"], CFG["END_MIN"] + 1, STEP):
            i = int(np.searchsorted(mod, m, side="right")) - 1
            if i < 0 or mod[i] < m - 3: continue
            px = close[i]; rec = {"symbol": sym, "session": s, "m": m, "spot": px}
            for h in HZ:
                j = int(np.searchsorted(mod, mod[i] + h, side="left"))
                rec[f"r{h}"] = (close[j] / px - 1) * 100 if j < len(mod) and mod[j] - mod[i] <= h + 3 else np.nan
            rec["rEOD"] = (eod / px - 1) * 100 if i < len(close) - 1 else np.nan
            f = TE.compute_flow(sub, cur["ts"].iloc[i])
            if f:
                rec.update(imb=f["imb"], ret=f["ret"], vs_vwap=f["vs_vwap"], rel_vol=f["rel_vol"])
            rows.append(rec)
    return pd.DataFrame(rows)

def build_grid(bars):
    jobs = list(bars.items()); parts = []
    workers = max(1, min(os.cpu_count() or 1, 8))
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for n, p in enumerate(ex.map(scan_symbol, jobs, chunksize=4), 1):
            if len(p): parts.append(p)
            if n % 25 == 0: print(f"  scanned {n}/{len(jobs)} stocks")
    G = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for c in ("imb", "ret", "vs_vwap", "rel_vol"):
        if c not in G: G[c] = np.nan
    return G.sort_values(["session", "m", "symbol"]).reset_index(drop=True)

# ───────────────────────────── 2. daily rule score history (previous close, leak-free) ─────────────────────────────
def score_history(vals, sessions):
    hdr = [h.strip() for h in vals[0]]; sym_i = hdr.index("symbol") if "symbol" in hdr else 0
    by_ddmm = {s.strftime("%d-%m"): s for s in sessions}
    cols = [(i, by_ddmm[h]) for i, h in enumerate(hdr) if re.fullmatch(r"\d{2}-\d{2}", h) and h in by_ddmm]
    hist = {}
    for row in vals[1:]:
        if sym_i >= len(row) or not row[sym_i].strip(): continue
        d = {}
        for i, sd in cols:
            v = TE._num(row[i]) if i < len(row) else None
            if v is not None: d[sd] = v
        hist[row[sym_i].strip().upper()] = d
    return hist

def prev_score(hist, sym, session):
    d = hist.get(sym, {}); prior = [x for x in d if x < session]
    return d[max(prior)] if prior else np.nan

# ───────────────────────────── 3. signals + trigger selection (live rules) ─────────────────────────────
def signal_vec(G, use_vwap):
    rel = G["rel_vol"] >= CFG["RELVOL_MIN"]
    vw = G["vs_vwap"]
    up_ok = (vw >= 0) if (use_vwap and CFG["VWAP_CONFIRM"]) else pd.Series(True, index=G.index)
    dn_ok = (vw <= 0) if (use_vwap and CFG["VWAP_CONFIRM"]) else pd.Series(True, index=G.index)
    call = rel & (G["imb"] >= CFG["IMB_MIN"]) & (G["ret"] >= CFG["RET_MIN"]) & up_ok
    put = rel & (G["imb"] <= -CFG["IMB_MIN"]) & (G["ret"] <= -CFG["RET_MIN"]) & dn_ok
    return np.where(call, "CALL", np.where(put, "PUT", ""))

def select_triggers(G, use_score, use_vwap):
    side = signal_vec(G, use_vwap)
    X = G.assign(side=side)
    X = X[X["side"] != ""]
    if use_score:
        sc = X["score"]
        bad = ((X["side"] == "CALL") & (sc < CFG["CALL_MIN_SCORE"])) | ((X["side"] == "PUT") & (sc > CFG["PUT_MAX_SCORE"]))
        X = X[~bad.fillna(False)]
    X = X.assign(strength=X["imb"].abs() * X["rel_vol"]).sort_values(["session", "m", "strength"], ascending=[True, True, False])
    out = []; last = {}; cnt = {}
    for (s, m), g in X.groupby(["session", "m"], sort=True):
        sent = 0
        for idx, r in g.iterrows():
            if sent >= CFG["MAX_NEW_PER_RUN"]: break
            key = (r["symbol"], s)
            if cnt.get(key, 0) >= CFG["MAX_PER_DAY"]: continue
            if key in last and m - last[key] < CFG["COOLDOWN_MIN"]: continue
            cnt[key] = cnt.get(key, 0) + 1; last[key] = m; sent += 1; out.append(idx)
    return X.loc[out] if out else X.iloc[0:0]

# ───────────────────────────── 4. controls ─────────────────────────────
def add_controls(G):
    """Mean raw return of (A) same stock & clock time on other days, (B) other stocks at same session & time."""
    for c in HZ_COLS:
        v = G[c]; ok = v.notna().astype(float); z = v.fillna(0.0)
        for tag, keys in (("A", ["symbol", "m"]), ("B", ["session", "m"])):
            sm = z.groupby([G[k] for k in keys]).transform("sum"); ct = ok.groupby([G[k] for k in keys]).transform("sum")
            G[f"{c}_{tag}"] = np.where(ct - ok > 0, (sm - z) / (ct - ok).replace(0, np.nan), np.nan)
    return G

# ───────────────────────────── 5. report ─────────────────────────────
def summarise(T, variant, sessions):
    cut = sessions[int(len(sessions) * 0.6)] if sessions else None
    T = T.assign(split=np.where(T["session"] < cut, "IS", "OOS")) if len(T) else T.assign(split=[])
    rows = []
    for split, gs in (("ALL", T), ("IS", T[T.split == "IS"]), ("OOS", T[T.split == "OOS"])):
        for side, g in (("BOTH", gs), ("CALL", gs[gs.side == "CALL"]), ("PUT", gs[gs.side == "PUT"])):
            for c, lab in zip(HZ_COLS, ["15m", "30m", "60m", "EOD"]):
                sign = np.where(g["side"] == "PUT", -1.0, 1.0) if len(g) else np.array([])
                fav = g[c].to_numpy(float) * sign if len(g) else np.array([])
                exA = fav - g[f"{c}_A"].to_numpy(float) * sign if len(g) else np.array([])
                exB = fav - g[f"{c}_B"].to_numpy(float) * sign if len(g) else np.array([])
                n = int(np.isfinite(fav).sum())
                if n == 0: continue
                def mean(a): a = a[np.isfinite(a)]; return float(a.mean()) if len(a) else np.nan
                b = exB[np.isfinite(exB)]
                t = float(b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))) if len(b) > 2 and b.std(ddof=1) > 0 else np.nan
                rows.append({"variant": variant, "split": split, "side": side, "horizon": lab, "n": n,
                             "days": int(g["session"].nunique()), "avg fav %": mean(fav),
                             "hit %": float((fav[np.isfinite(fav)] > 0).mean() * 100),
                             "excess vs same-stock-time %": mean(exA), "excess vs market %": mean(exB), "t (vs market)": t})
    return pd.DataFrame(rows)

def run_backtest(bars, score_hist):
    print(f"building grid for {len(bars)} stocks ...")
    G = build_grid(bars)
    if G.empty: print("no data"); return None, None
    G["score"] = [prev_score(score_hist, s, d) for s, d in zip(G["symbol"], G["session"])]
    G = add_controls(G)
    sessions = sorted(G["session"].unique())
    print(f"grid rows {len(G):,} | tested sessions {len(sessions)}: {sessions[0]} .. {sessions[-1]}")
    # sanity: vectorised rule must equal the live flow_signal()
    samp = G.dropna(subset=["imb"]).sample(min(3000, len(G)), random_state=1)
    vec = signal_vec(samp, True)
    live = [TE.flow_signal({"imb": r.imb, "ret": r.ret, "vs_vwap": r.vs_vwap, "rel_vol": r.rel_vol}) or "" for r in samp.itertuples()]
    assert list(vec) == live, "vectorised signal differs from trigger_engine.flow_signal"
    print("signal check OK (matches live engine)")
    summ = []; trig_all = []
    for name, (use_score, use_vwap) in VARIANTS.items():
        T = select_triggers(G, use_score, use_vwap)
        print(f"{name}: {len(T)} triggers on {T['session'].nunique() if len(T) else 0} days")
        summ.append(summarise(T, name, sessions))
        if name.startswith("ENGINE") and len(T):
            keep = ["session", "m", "symbol", "side", "spot", "imb", "rel_vol", "ret", "vs_vwap", "score"] + HZ_COLS
            trig_all = T[keep].copy()
    return pd.concat(summ, ignore_index=True), trig_all

def verdict(S):
    r = S[(S.variant.str.startswith("ENGINE")) & (S.split == "OOS") & (S.side == "BOTH") & (S.horizon == "30m")]
    if r.empty: return "Not enough out-of-sample triggers to judge."
    r = r.iloc[0]
    return (f"ENGINE, out-of-sample, 30 min: {int(r.n)} triggers on {int(r.days)} days, avg {r['avg fav %']:+.2f}%, hit {r['hit %']:.0f}%, "
            f"excess vs market {r['excess vs market %']:+.2f}% (t={r['t (vs market)']:.1f}). "
            "Needs |t| > ~2 AND consistent in-sample/out-of-sample before it means anything; this is a small sample.")

# ───────────────────────────── 6. outputs ─────────────────────────────
def write_sheet(sh, S, T, note):
    import gspread
    def put(title, df, rows=3000):
        try: ws = sh.worksheet(title); ws.clear()
        except gspread.WorksheetNotFound: ws = sh.add_worksheet(title=title, rows=rows, cols=max(14, len(df.columns) + 1))
        out = df.copy().astype(object).where(df.notna(), "")
        ws.update(values=[list(out.columns)] + out.values.tolist(), range_name="A3", value_input_option="USER_ENTERED")
        ws.update(values=[[note]], range_name="A1")
    put("Backtest", S.round(3).assign(**{}))
    if len(T):
        t = T.copy(); t["session"] = t["session"].astype(str); t["time"] = [f"{m // 60:02d}:{m % 60:02d}" for m in t["m"]]
        put("Backtest Triggers", t.drop(columns=["m"]).round(3))

def load_bars(symbols):
    CFG["FETCH_DAYS"] = DAYS; bars = {}
    with cf.ThreadPoolExecutor(max_workers=CFG["WORKERS"]) as ex:
        for s, d in ex.map(TE.fetch_bars, symbols):
            if d is not None and len(d): bars[s] = d
    return bars

def main():
    import live_refresh_local as LR
    sh = LR.open_sheet(); vals = sh.worksheet("Rule Dashboard").get_all_values()
    hdr = [h.strip() for h in vals[0]]; si = hdr.index("symbol") if "symbol" in hdr else 0
    symbols = sorted({r[si].strip().upper() for r in vals[1:] if len(r) > si and r[si].strip()})
    print(f"{len(symbols)} symbols; fetching up to {DAYS} days of 1-min bars ...")
    bars = load_bars(symbols)
    if not bars: print("No bars fetched (cookie expired?)"); sys.exit(1)
    sess = sorted({x for d in bars.values() for x in d["session"].unique()})
    print(f"bars for {len(bars)} stocks | {len(sess)} sessions {sess[0]} .. {sess[-1]}")
    S, T = run_backtest(bars, score_history(vals, sess))
    if S is None: sys.exit(1)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(S[(S.side == "BOTH")].round(3).to_string(index=False))
    v = verdict(S); print("\nVERDICT:", v)
    os.makedirs("out", exist_ok=True)
    S.to_csv("out/backtest_summary.csv", index=False)
    if len(T): T.to_csv("out/backtest_triggers.csv", index=False)
    note = f"Backtest of trigger engine | run {dt.datetime.now():%Y-%m-%d %H:%M} | {len(bars)} stocks, {len(sess)} sessions | stock-level, no option costs | {v}"
    try: write_sheet(sh, S, T, note); print("written to tabs: Backtest, Backtest Triggers")
    except Exception as e: print("sheet write failed:", e)

# ───────────────────────────── self-test ─────────────────────────────
def _synth(n_sym=12, n_sess=22, seed=3):
    rng = np.random.default_rng(seed); bars = {}
    days = [d.date() for d in pd.bdate_range(end="2026-10-16", periods=n_sess)]
    for k in range(n_sym):
        rows = []; p = 500 + 40 * k
        for d in days:
            base = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15)
            for i in range(375):
                o = p; p = p * (1 + rng.normal(0, 0.0007)); hi = max(o, p) * (1 + abs(rng.normal(0, .0003))); lo = min(o, p) * (1 - abs(rng.normal(0, .0003)))
                rows.append({"ts": base + pd.Timedelta(minutes=i), "open": o, "high": hi, "low": lo, "close": p, "volume": float(rng.integers(500, 5000) * (4 if rng.random() < .03 else 1))})
        bars[f"S{k}"] = TE._norm(pd.DataFrame(rows))
    return bars

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        CFG["N_PRIOR"] = 7
        bars = _synth(); S, T = run_backtest(bars, {})
        print(S[(S.side == "BOTH") & (S.split == "ALL") & (S.variant.str.startswith("ENGINE"))].round(3).to_string(index=False))
        print("SELFTEST OK (random data => excess should hover near 0)")
    else:
        main()
