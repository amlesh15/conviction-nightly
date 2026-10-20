# ==================================================================================
# VCP ENGINE — Volatility Contraction Pattern (Minervini) scanner + honest backtest
#
#  A VCP is a stock in an established uptrend that pulls back several times, each pullback SHALLOWER than the one before
#  (e.g. 22% -> 11% -> 5%), volume drying up, then breaks above the last high (the "pivot") on a volume surge.
#
#  DAILY mode    : fetch daily candles for the F&O universe -> find setups -> write the Scanner tab -> backtest
#                  (VCP breakouts vs plain trend-breakouts vs the market, same dates, with day-cluster bootstrap)
#  INTRADAY mode : same detector on 15-minute bars, but only on stocks whose DAILY chart is in an uptrend
#                  ("daily setup, intraday trigger"); also shows live price/volume against each daily pivot.
#
#  No look-ahead: a swing point is only used `sw` bars after it formed; entry = NEXT bar's open; the self-test proves that
#  results at bar t do not change when future bars are removed.
#  Tabs: "VCP Scanner", "VCP Backtest", "VCP Events", "VCP Intraday", "VCP Notes"
#  Usage: python vcp_engine.py [--mode daily|intraday] [--selftest]
# ==================================================================================
import os, sys, time, datetime as dt
import numpy as np, pandas as pd

MODE = (sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else os.environ.get("VCP_MODE", "daily")).lower()
OUT = "out/vcp"; os.makedirs(OUT, exist_ok=True)
WORKERS = int(os.environ.get("VCP_WORKERS", "8"))

PARAMS = {
    # sw=bars each side to confirm a swing | depths as fractions | near = how close under the pivot counts as "set up"
    "D": dict(sw=3, min_c=2, max_c=5, first_min=0.08, first_max=0.45, last_max=0.10, shrink=0.95, low_tol=0.97, base_min=15, base_max=130,
              near=0.06, vwin=50, vdry=0.85, vmult=1.4, max_wait=40, stop_cap=0.12, hz=(5, 10, 20, 40), hold=20),
    "I": dict(sw=3, min_c=2, max_c=5, first_min=0.012, first_max=0.06, last_max=0.012, shrink=0.95, low_tol=0.997, base_min=30, base_max=300,
              near=0.01, vwin=100, vdry=0.85, vmult=1.8, max_wait=80, stop_cap=0.03, hz=(4, 8, 16), hold=16),
}

# ───────────────────────────── detector ─────────────────────────────
def sma(x, n):
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()

def trend_template(c, h, l):
    """Minervini stage-2 template (needs ~200 bars). True where all rules hold."""
    s50, s150, s200 = sma(c, 50), sma(c, 150), sma(c, 200)
    lo252 = pd.Series(l).rolling(252, min_periods=200).min().to_numpy(); hi252 = pd.Series(h).rolling(252, min_periods=200).max().to_numpy()
    rising = s200 > np.r_[np.full(21, np.nan), s200[:-21]]
    with np.errstate(invalid="ignore"):
        return (c > s150) & (c > s200) & (s150 > s200) & rising & (s50 > s150) & (c > s50) & (c >= 1.3 * lo252) & (c >= 0.75 * hi252)

def evaluate(S, t, P, h, v):
    """Given the alternating swing list S=[(idx,price,'H'/'L')...], return the current setup or None."""
    if len(S) < 4: return None
    pairs = []                                                # completed (High, Low) pairs, chronological
    k = len(S) - 1
    pivot = None
    if S[k][2] == "H": pivot = S[k]; k -= 1                   # a fresh high is already the pivot candidate
    while k >= 1:
        if S[k][2] == "L" and S[k - 1][2] == "H": pairs.append((S[k - 1], S[k])); k -= 2
        else: break
    pairs.reverse()
    if len(pairs) < P["min_c"]: return None
    if pivot is None: pivot = pairs[-1][0]                    # recovering from the last low: pivot = the high that started it
    d = [(H[1] - L[1]) / H[1] for H, L in pairs]
    n = len(pairs); start = n - 1                             # longest trailing run of shrinking depths with rising lows
    while start > 0 and d[start] <= d[start - 1] * P["shrink"] and pairs[start][1][1] >= pairs[start - 1][1][1] * P["low_tol"]: start -= 1
    run = list(range(start, n))[-P["max_c"]:]
    if len(run) < P["min_c"]: return None
    dd = [d[i] for i in run]
    if dd[0] < P["first_min"] or dd[0] > P["first_max"] or dd[-1] > P["last_max"]: return None
    base_start = pairs[run[0]][0][0]; blen = t - base_start
    if blen < P["base_min"] or blen > P["base_max"]: return None
    base_hi = h[base_start: t + 1].max()
    if pivot[1] < 0.95 * base_hi: return None                 # pivot must be the right-hand side of the base
    L_last = pairs[run[-1]][1]
    return {"pivot": pivot[1], "stop": L_last[1], "depths": dd, "base_start": base_start, "formed": t, "blen": blen}

def scan(o, h, l, c, v, P):
    """Bar-by-bar, causal. Returns dict of arrays: ready, brk, brk_vol, pivot, stop, ncon, dlast, blen, vr, dry."""
    n = len(c); sw = P["sw"]; S = []; cur = None
    R = {k: np.full(n, np.nan) for k in ("pivot", "stop", "ncon", "dlast", "dfirst", "blen", "vr", "dist")}
    for k in ("ready", "brk", "brk_vol", "dry"): R[k] = np.zeros(n, bool)
    av = sma(v, P["vwin"]); av5 = sma(v, 5)
    for t in range(n):
        j = t - sw
        if j >= sw:
            hi_ = h[j - sw: j + sw + 1]; lo_ = l[j - sw: j + sw + 1]; changed = False
            isH = h[j] >= hi_.max() and h[j] > h[j - 1]; isL = l[j] <= lo_.min() and l[j] < l[j - 1]
            if isH != isL:
                typ = "H" if isH else "L"; px = h[j] if isH else l[j]
                if not S or S[-1][2] != typ: S.append((j, px, typ)); changed = True
                elif (typ == "H" and px > S[-1][1]) or (typ == "L" and px < S[-1][1]): S[-1] = (j, px, typ); changed = True
            if changed: cur = evaluate(S, t, P, h, v)
        if cur is None: continue
        if c[t] < cur["stop"] or t - cur["formed"] > P["max_wait"]: cur = None; continue
        vr = v[t] / av[t] if av[t] and av[t] > 0 else np.nan
        R["pivot"][t], R["stop"][t], R["ncon"][t], R["dlast"][t], R["dfirst"][t] = cur["pivot"], cur["stop"], len(cur["depths"]), cur["depths"][-1], cur["depths"][0]
        R["blen"][t] = t - cur["base_start"]; R["vr"][t] = vr; R["dist"][t] = (cur["pivot"] - c[t]) / cur["pivot"]
        R["dry"][t] = bool(av[t] and av5[t] <= P["vdry"] * av[t])
        if c[t] > cur["pivot"]:
            R["brk"][t] = True; R["brk_vol"][t] = bool(vr == vr and vr >= P["vmult"]); cur = None
        elif R["dist"][t] <= P["near"]: R["ready"][t] = True
    return R

# ───────────────────────────── data ─────────────────────────────
def fetch_daily(P_, sym, days):
    df, info = P_.candles(sym, days=days, interval="1D")
    return df

def pick_depth(P_):
    for probe in ("RELIANCE", "HDFCBANK", "TCS"):
        for days in (2200, 1500, 1000, 600, 400, 300):
            df = fetch_daily(P_, probe, days)
            if df is not None and len(df) > 60:
                print(f"daily history: asked {days}d -> {len(df)} bars ({df['session'].min()} .. {df['session'].max()})"); return days
    return 300

def clean_daily(df):
    """Drop everything before the last split-like gap (>35% overnight) so unadjusted splits cannot fake a pattern."""
    df = df.dropna(subset=["close", "high", "low", "open"]).reset_index(drop=True)
    if len(df) < 2: return df
    jump = (df["close"] / df["close"].shift() - 1).abs() > 0.35
    return df.iloc[int(np.where(jump)[0].max()):].reset_index(drop=True) if jump.any() else df

def to_15m(df):
    d = df.copy(); d["bucket"] = d["ts"].dt.floor("15min")
    vol = d["volume"].fillna(0).to_numpy(float)
    for s, g in d.groupby("session"):
        x = g["volume"].fillna(0).to_numpy(float)
        if len(x) > 30 and (np.diff(x) >= 0).mean() > 0.98 and x[-1] > x[0] * 3: vol[g.index] = np.r_[x[0], np.diff(x)]
    d["volume"] = vol
    a = d.groupby("bucket").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), session=("session", "first"))
    return a.reset_index().rename(columns={"bucket": "ts"})

# ───────────────────────────── per-stock analysis ─────────────────────────────
def analyse(sym, df, key):
    P = PARAMS[key]; df = df.reset_index(drop=True)
    o, h, l, c, v = (df[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    v = np.nan_to_num(v)
    R = scan(o, h, l, c, v, P); R["trend"] = trend_template(c, h, l) if key == "D" else np.zeros(len(c), bool)
    hh = np.r_[np.full(20, np.nan), pd.Series(h).rolling(20).max().shift(1).to_numpy()[20:]]
    av = sma(v, P["vwin"]); vr = np.where(av > 0, v / av, np.nan)
    R["plain"] = (c > hh) & (vr >= P["vmult"]) & ~R["brk"]                  # ordinary 20-bar-high breakout on volume (control)
    R["ret12"] = c / np.r_[np.full(252, np.nan), c[:-252]] - 1 if len(c) > 252 else np.full(len(c), np.nan)
    return df, (o, h, l, c, v), R

def forward(o, h, l, c, i, P):
    """Entry = next bar open. Returns fixed-horizon returns, plus a stop-based R trade."""
    n = len(c)
    if i + 1 >= n: return None
    e = o[i + 1]
    if not e or e != e: return None
    out = {"entry": e}
    for k in P["hz"]: out[f"r{k}"] = (c[i + k] / e - 1) * 100 if i + k < n else np.nan
    return out

def trade_R(o, h, l, c, i, P, stop):
    n = len(c)
    if i + 1 >= n: return np.nan, np.nan
    e = o[i + 1]; stop = max(stop, e * (1 - P["stop_cap"]))
    if stop >= e: return np.nan, np.nan
    risk = e - stop
    for k in range(i + 1, min(i + 1 + P["hold"], n)):
        if o[k] <= stop: return (o[k] - e) / risk, (o[k] / e - 1) * 100
        if l[k] <= stop: return -1.0, (stop / e - 1) * 100
    k = min(i + P["hold"], n - 1); return (c[k] - e) / risk, (c[k] / e - 1) * 100

def collect(sym, df, arrs, R, key, only_trend_days=None):
    """All events for one stock: VCP breakouts, plain breakouts; plus a baseline row per bar (market proxy)."""
    P = PARAMS[key]; o, h, l, c, v = arrs; ev = []; base = []
    for i in range(len(c)):
        for kind, flag in (("VCP", R["brk"][i]), ("PLAIN", R["plain"][i])):
            if not flag: continue
            if only_trend_days is not None and not only_trend_days(df, i): continue
            if key == "D" and not R["trend"][i]: continue
            f = forward(o, h, l, c, i, P)
            if f is None: continue
            stop = R["stop"][i] if kind == "VCP" else l[max(0, i - 10): i + 1].min()
            Rm, rr = trade_R(o, h, l, c, i, P, stop if stop == stop else c[i] * (1 - P["stop_cap"]))
            f.update({"symbol": sym, "kind": kind, "ts": df["ts"].iloc[i], "date": str(df["session"].iloc[i]), "vol_ok": bool(R["brk_vol"][i]) if kind == "VCP" else True,
                      "dry": bool(R["dry"][i]) if kind == "VCP" else False, "ncon": R["ncon"][i], "dfirst": R["dfirst"][i], "dlast": R["dlast"][i], "blen": R["blen"][i],
                      "R": Rm, "trade_ret": rr, "ret12": R["ret12"][i]})
            ev.append(f)
    return ev

def baseline_rows(sym, df, arrs, key):
    P = PARAMS[key]; o, h, l, c, v = arrs; n = len(c); rows = {"date": [str(x) for x in df["session"]], "ts": df["ts"].to_numpy()}
    for k in P["hz"]:
        r = np.full(n, np.nan)
        if n > k + 1: r[: n - k - 1] = (c[k + 1:] / o[1: n - k] - 1) * 100
        rows[f"r{k}"] = r
    return pd.DataFrame(rows).assign(symbol=sym)

# ───────────────────────────── statistics ─────────────────────────────
def week_key(d): return pd.to_datetime(d).strftime("%G-%V")

def boot_ci(df, col, B=1500, seed=3):
    d = df.dropna(subset=[col]);
    if len(d) < 8: return (np.nan,) * 3
    g = d.groupby("wk")[col].agg(["sum", "count"]); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        w = rng.integers(0, len(g), len(g)); n = g["count"].to_numpy()[w].sum()
        if n: bs.append(g["sum"].to_numpy()[w].sum() / n)
    return d[col].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def diff_ci(a, b, col, B=1500, seed=4):
    a = a.dropna(subset=[col]); b = b.dropna(subset=[col])
    if len(a) < 8 or len(b) < 8: return (np.nan,) * 3
    weeks = sorted(set(a["wk"]) | set(b["wk"])); ix = {w: i for i, w in enumerate(weeks)}
    def agg(d):
        g = d.groupby("wk")[col].agg(["sum", "count"]); s = np.zeros(len(weeks)); n = np.zeros(len(weeks))
        for w, r in g.iterrows(): s[ix[w]], n[ix[w]] = r["sum"], r["count"]
        return s, n
    sa, na = agg(a); sb, nb = agg(b); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(weeks), len(weeks)), minlength=len(weeks))
        if (w * na).sum() and (w * nb).sum(): bs.append((w * sa).sum() / (w * na).sum() - (w * sb).sum() / (w * nb).sum())
    return a[col].mean() - b[col].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def summarize(E, B, key, label):
    """E events, B baseline. Rows: variant x horizon with excess vs same-date market, and VCP minus PLAIN."""
    P = PARAMS[key]; E = E.copy(); E["wk"] = E["date"].map(week_key)
    cols = [f"r{k}" for k in P["hz"]]
    mk = B.groupby("date")[cols].mean().add_prefix("mkt_")
    E = E.join(mk, on="date")
    for c in cols: E["x" + c] = E[c] - E["mkt_" + c]
    variants = {"VCP breakout (any volume)": E[E.kind == "VCP"], "VCP breakout + volume surge": E[(E.kind == "VCP") & E.vol_ok],
                "VCP + volume + dried-up volume before": E[(E.kind == "VCP") & E.vol_ok & E.dry], "PLAIN trend breakout (control)": E[E.kind == "PLAIN"]}
    rows = []
    dates = sorted(E["date"].unique()); cut = dates[int(len(dates) * 0.6)] if dates else ""
    for name, S in variants.items():
        for cname in cols:
            xm, xl, xh = boot_ci(S, "x" + cname); m = S[cname].mean(); hit = (S[cname].dropna() > 0).mean() * 100 if S[cname].notna().any() else np.nan
            rows.append({"set": label, "variant": name, "horizon (bars)": cname[1:], "events": int(S[cname].notna().sum()), "stocks": S["symbol"].nunique(),
                         "avg return %": m, "hit %": hit, "excess vs market %": xm, "CI low": xl, "CI high": xh,
                         "excess first 60% of time": S[S.date <= cut]["x" + cname].mean(), "excess last 40% (unseen)": S[S.date > cut]["x" + cname].mean(),
                         "stop-trade avg R": S["R"].mean(), "stop-trade win %": (S["R"].dropna() > 0).mean() * 100 if S["R"].notna().any() else np.nan})
    vc, pl = E[(E.kind == "VCP") & E.vol_ok], E[E.kind == "PLAIN"]
    for cname in cols:
        d, lo, hi = diff_ci(vc, pl, "x" + cname)
        rows.append({"set": label, "variant": ">>> VCP+volume MINUS plain breakout (does the contraction add anything?)", "horizon (bars)": cname[1:], "events": int(vc[cname].notna().sum()),
                     "stocks": vc["symbol"].nunique(), "avg return %": np.nan, "hit %": np.nan, "excess vs market %": d, "CI low": lo, "CI high": hi,
                     "excess first 60% of time": np.nan, "excess last 40% (unseen)": np.nan, "stop-trade avg R": vc["R"].mean() - pl["R"].mean(), "stop-trade win %": np.nan})
    T = pd.DataFrame(rows)
    T["verdict"] = np.where(T["CI low"] > 0, "EDGE (CI above 0)", np.where(T["CI high"] < 0, "NEGATIVE", "no evidence"))
    return T

# ───────────────────────────── runners ─────────────────────────────
def status_row(sym, df, arrs, R, extra=None):
    o, h, l, c, v = arrs; i = len(c) - 1; row = None
    if R["brk"][i]: st = "BREAKOUT TODAY (volume ok)" if R["brk_vol"][i] else "BREAKOUT TODAY (weak volume)"
    elif R["ready"][i]: st = "READY (within %.0f%% of pivot)" % (PARAMS["D"]["near"] * 100) if R["dist"][i] > 0.03 else "READY (within 3% of pivot)"
    else: return None
    # for a breakout bar the setup arrays were reset after firing, so reuse the values stored at that bar
    piv, stop = R["pivot"][i], R["stop"][i]
    dd = None
    return {"symbol": sym, "status": st, "close": c[i], "pivot": piv, "% to pivot": R["dist"][i] * 100 if R["dist"][i] == R["dist"][i] else np.nan,
            "stop (last low)": stop, "risk % (pivot→stop)": (piv - stop) / piv * 100 if piv == piv else np.nan, "contractions": int(R["ncon"][i]) if R["ncon"][i] == R["ncon"][i] else "",
            "first depth %": R["dfirst"][i] * 100, "last depth %": R["dlast"][i] * 100, "base bars": R["blen"][i], "volume dried up": "yes" if R["dry"][i] else "no",
            "volume vs avg": R["vr"][i], "stage-2 uptrend": "yes" if R["trend"][i] else "no", "12m return %": R["ret12"][i] * 100, "last bar": str(df["session"].iloc[i])}

def run_daily():
    import oi_probe as P_
    P_.DEADLINE = time.time() + 55 * 60
    syms = P_.universe_from_sheet(); depth = pick_depth(P_); print(len(syms), "stocks")
    import concurrent.futures as cf
    store = {}
    def one(s):
        try:
            df = fetch_daily(P_, s, depth)
            return s, clean_daily(df) if df is not None else None
        except Exception: return s, None
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s, df in ex.map(one, syms):
            if df is not None and len(df) >= 260: store[s] = df
    print(len(store), "stocks with >=260 daily bars")
    ev, base, scan_rows = [], [], []
    for s, df in store.items():
        df, arrs, R = analyse(s, df, "D"); ev += collect(s, df, arrs, R, "D"); base.append(baseline_rows(s, df, arrs, "D"))
        r = status_row(s, df, arrs, R)
        if r: scan_rows.append(r)
    E = pd.DataFrame(ev); B = pd.concat(base) if base else pd.DataFrame()
    SC = pd.DataFrame(scan_rows)
    if len(SC):
        SC["order"] = SC["status"].map(lambda x: 0 if x.startswith("BREAKOUT TODAY (volume") else 1 if x.startswith("BREAKOUT") else 2) + (SC["stage-2 uptrend"] != "yes") * 10
        SC = SC.sort_values(["order", "% to pivot"]).drop(columns="order")
    T = summarize(E, B, "D", "daily") if len(E) else pd.DataFrame()
    for n, d in (("scanner", SC), ("events", E), ("backtest", T)): d.to_csv(f"{OUT}/vcp_{n}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T.round(3).to_string(index=False) if len(T) else "no events"); print(SC.head(25).round(2).to_string(index=False) if len(SC) else "no candidates")
    pub(SC, T, E, None, depth, store)

def run_intraday():
    import oi_probe as P_
    P_.DEADLINE = time.time() + 55 * 60
    syms = P_.universe_from_sheet(); depth = pick_depth(P_); import concurrent.futures as cf
    daily = {}
    def d1(s):
        try: df = fetch_daily(P_, s, depth); return s, clean_daily(df) if df is not None else None
        except Exception: return s, None
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s, df in ex.map(d1, syms):
            if df is not None and len(df) >= 260: daily[s] = df
    # daily state per stock: stage-2 flag by date, latest pivot if a daily setup is open
    dstate, dpiv = {}, {}
    for s, df in daily.items():
        df2, arrs, R = analyse(s, df, "D"); dstate[s] = dict(zip(df2["session"].astype(str), R["trend"]))
        i = len(df2) - 1
        if R["ready"][i] or R["brk"][i]: dpiv[s] = (R["pivot"][i], R["stop"][i], R["dist"][i], arrs[4][-20:].mean())
    print(len(daily), "daily; setups open:", len(dpiv))
    def m1(s):
        try: df, _ = P_.candles(s, days=60, interval="1M"); return s, to_15m(df) if df is not None else None
        except Exception: return s, None
    intra = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s, df in ex.map(m1, list(daily)):
            if df is not None and len(df) > 400: intra[s] = df
    ev, base, live = [], [], []
    for s, df in intra.items():
        df, arrs, R = analyse(s, df, "I")
        days_sorted = sorted(dstate[s]); idx = {d: k for k, d in enumerate(days_sorted)}
        def ok(df_, i, s=s, ds=days_sorted):
            d = str(df_["session"].iloc[i]); import bisect; k = bisect.bisect_left(ds, d) - 1
            return k >= 0 and dstate[s].get(ds[k], False) and df_["ts"].iloc[i].hour * 60 + df_["ts"].iloc[i].minute <= 14 * 60 + 15
        ev += collect(s, df, arrs, R, "I", only_trend_days=ok); base.append(baseline_rows(s, df, arrs, "I"))
        i = len(df) - 1
        if s in dpiv:
            piv, stop, dist, av20 = dpiv[s]; o, h, l, c, v = arrs; today = df["session"].iloc[-1]; day = df[df["session"] == today]
            elapsed = max((day["ts"].iloc[-1] - day["ts"].iloc[0]).total_seconds() / 60 + 15, 15); pace = day["volume"].sum() / av20 / min(elapsed / 375, 1) if av20 else np.nan
            live.append({"symbol": s, "daily pivot": piv, "daily stop": stop, "price": c[-1], "price vs pivot %": (c[-1] / piv - 1) * 100, "volume pace vs avg day (x)": pace,
                         "15m VCP ready": "yes" if R["ready"][i] else "no", "15m VCP breakout now": "yes" if R["brk"][i] else "no",
                         "TRIGGER": "YES" if (c[-1] > piv and pace == pace and pace >= 1.4) else "", "last bar": str(df["ts"].iloc[-1])})
    E = pd.DataFrame(ev); B = pd.concat(base) if base else pd.DataFrame()
    T = summarize(E, B, "I", "intraday 15m (daily stage-2 stocks only)") if len(E) else pd.DataFrame()
    LV = pd.DataFrame(live).sort_values("price vs pivot %", ascending=False) if live else pd.DataFrame()
    for n, d in (("intra_backtest", T), ("intra_live", LV), ("intra_events", E)): d.to_csv(f"{OUT}/vcp_{n}.csv", index=False)
    print(T.round(3).to_string(index=False) if len(T) else "no intraday events"); print(LV.head(20).round(2).to_string(index=False) if len(LV) else "no live rows")
    pub(None, None, None, (LV, T, E), depth, daily)

NOTES = [["term", "meaning"],
 ["VCP", "uptrend stock whose pullbacks get shallower (e.g. 22% -> 11% -> 5%), volume drying up; buy trigger = close above the pivot (the high that started the last, tightest pullback)"],
 ["stage-2 uptrend", "price > 150 & 200-day averages, 150 > 200, 200-day rising for a month, 50 > 150, price within 25% of the 52-week high and 30%+ above the 52-week low"],
 ["status", "BREAKOUT TODAY = closed above pivot today. READY = tight and within the set distance under the pivot, setup still valid (price above the last low)"],
 ["stop (last low)", "low of the last contraction; risk % = pivot to stop. Backtest caps risk at 12% daily / 3% intraday"],
 ["entry in backtest", "NEXT bar's open after the signal bar (no look-ahead). Stop-trade: exit at stop, or after 20 daily / 16 15-min bars"],
 ["excess vs market", "forward return of the event minus the average of ALL stocks entered on the same date, so a rising market is not credited to the pattern"],
 ["PLAIN breakout control", "ordinary 20-bar-high breakout with volume surge in a stage-2 stock that is NOT a VCP. VCP only has value if it beats this"],
 ["CI", "95% from resampling calendar weeks (events cluster in time)"],
 ["limits", "daily history depth printed in the log; unadjusted splits (>35% gaps) are cut out. Intraday history is short (~60 days), treat intraday results as indicative"]]

def pub(SC, T, E, intra, depth, store):
    import oi_probe as P_
    sh = P_.open_sheet()
    if SC is not None:
        P_.put_tab(sh, "VCP Scanner", list(SC.columns) if len(SC) else ["status"], SC.round(2).values.tolist() if len(SC) else [["no candidates today"]])
        if len(T): P_.put_tab(sh, "VCP Backtest", list(T.columns), T.round(3).values.tolist())
        if E is not None and len(E): P_.put_tab(sh, "VCP Events", [c for c in E.columns], E.sort_values("date", ascending=False).head(3000).astype(str).values.tolist())
    if intra is not None:
        LV, T2, E2 = intra
        if len(LV): P_.put_tab(sh, "VCP Intraday", list(LV.columns), LV.round(2).values.tolist())
        if len(T2): P_.put_tab(sh, "VCP Backtest Intraday", list(T2.columns), T2.round(3).values.tolist())
    nb = [len(v) for v in store.values()]
    P_.put_tab(sh, "VCP Notes", NOTES[0], NOTES[1:] + [["daily bars asked", str(depth)], ["stocks used", str(len(store))], ["median bars", str(int(np.median(nb))) if nb else ""], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]])
    print("sheet tabs written")

# ───────────────────────────── self-test ─────────────────────────────
def synth_vcp(seed=0):
    """Long uptrend, then three shrinking pullbacks (22%, 10%, 4%), volume drying, then a high-volume breakout."""
    rng = np.random.default_rng(seed); px = [100.0]
    def leg(a, b, n):
        for k in range(1, n + 1): px.append(a + (b - a) * k / n + rng.normal(0, 0.15))
    for _ in range(300): px.append(px[-1] * (1 + 0.0035 + rng.normal(0, 0.004)))      # trend
    top = px[-1]; hi1 = top * 1.06; leg(top, hi1, 8)
    lo1 = hi1 * 0.78; leg(hi1, lo1, 12); hi2 = hi1 * 0.995; leg(lo1, hi2, 12)
    lo2 = hi2 * 0.90; leg(hi2, lo2, 8); hi3 = hi2 * 0.99; leg(lo2, hi3, 8)
    lo3 = hi3 * 0.96; leg(hi3, lo3, 6); hi4 = hi3 * 0.995; leg(lo3, hi4, 8)
    for _ in range(6): px.append(hi4 * (0.985 + rng.normal(0, 0.003)))
    px.append(hi4 * 1.03)                                                              # breakout bar
    for _ in range(25): px.append(px[-1] * (1.004 + rng.normal(0, 0.006)))
    c = np.array(px); n = len(c); o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.001, n)); h = np.maximum(o, c) * 1.003; l = np.minimum(o, c) * 0.997
    v = np.full(n, 1e6) * (1 + rng.normal(0, 0.1, n)); v[300 + 8: 300 + 8 + 100] *= 0.7; v[-26] = 3.5e6
    return o, h, l, c, v, n - 26

def selftest():
    P = PARAMS["D"]; o, h, l, c, v, bi = synth_vcp()
    R = scan(o, h, l, c, v, P); tt = trend_template(c, h, l)
    print("planted breakout bar", bi, "| detected breakout bars:", np.where(R["brk"])[0].tolist(), "| with volume:", np.where(R["brk_vol"])[0].tolist(), "| ready bars:", int(R["ready"].sum()))
    assert R["brk_vol"][bi] or R["brk_vol"][bi - 1:bi + 2].any(), "planted VCP breakout not detected"
    j = np.where(R["brk"])[0][0]; print("contractions", R["ncon"][j - 1] if R["ncon"][j - 1] == R["ncon"][j - 1] else R["ncon"][j], "| depths first/last %", round(R["dfirst"][j - 1] * 100, 1), round(R["dlast"][j - 1] * 100, 1), "| trend template on:", bool(tt[bi]))
    # causality: results at bar t must not change when the future is removed
    for t in (bi - 20, bi - 3, bi, bi + 10):
        R2 = scan(o[: t + 1], h[: t + 1], l[: t + 1], c[: t + 1], v[: t + 1], P)
        for k in ("brk", "brk_vol", "ready"): assert R2[k][t] == R[k][t], f"look-ahead in {k} at {t}"
        for k in ("pivot", "stop", "ncon"): assert (R2[k][t] == R[k][t]) or (np.isnan(R2[k][t]) and np.isnan(R[k][t])), f"look-ahead in {k} at {t}"
    print("causality OK")
    # random walks: breakouts should be rare and carry no edge
    rng = np.random.default_rng(1); nb = 0; fw = []
    for s in range(60):
        cc = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, 900)); oo = np.r_[cc[0], cc[:-1]]; hh = np.maximum(oo, cc) * 1.004; ll = np.minimum(oo, cc) * 0.996; vv = rng.uniform(.5, 1.5, 900) * 1e6
        RR = scan(oo, hh, ll, cc, vv, P); tr = trend_template(cc, hh, ll)
        for i in np.where(RR["brk_vol"] & tr)[0]:
            f = forward(oo, hh, ll, cc, i, P)
            if f: nb += 1; fw.append(f["r20"])
    print("random walks: VCP+volume+trend breakouts =", nb, "| mean 20-bar return % =", round(np.nanmean(fw), 2) if fw else None, "(drift only)")
    # end-to-end summary on a tiny synthetic universe
    ev, base = [], []
    for s in range(40):
        o_, h_, l_, c_, v_, _ = synth_vcp(s) if s % 4 == 0 else (lambda r: (r, r * 1.004, r * .996, 100 * np.cumprod(1 + np.random.default_rng(s).normal(0.0005, .014, 420)), np.random.default_rng(s).uniform(.5, 1.5, 420) * 1e6, 0))(np.r_[100, 100 * np.cumprod(1 + np.random.default_rng(s).normal(0.0005, .014, 419))])
        n = len(c_); d = pd.DataFrame({"ts": pd.bdate_range("2025-01-01", periods=n), "open": o_, "high": h_, "low": l_, "close": c_, "volume": v_}); d["session"] = d["ts"].dt.date
        d, arrs, R_ = analyse(f"S{s}", d, "D"); ev += collect(f"S{s}", d, arrs, R_, "D"); base.append(baseline_rows(f"S{s}", d, arrs, "D"))
    E = pd.DataFrame(ev); T = summarize(E, pd.concat(base), "D", "selftest") if len(E) else pd.DataFrame()
    print("events:", len(E), E["kind"].value_counts().to_dict() if len(E) else {}); print(T.head(6).round(2).to_string(index=False) if len(T) else "no table")
    assert len(E) > 0
    print("SELFTEST OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    elif MODE == "intraday": run_intraday()
    else: run_daily()
