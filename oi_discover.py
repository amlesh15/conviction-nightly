# ==================================================================================
# OPEN INTEREST ENGINE — step 3: PATTERN DISCOVERY (the "optimisation")
#
#  Goal : find open-interest situations that predict where the price goes next, without us guessing them one by one.
#  Input: the 1-minute archive saved by oi_universe.py  (out/oi/archive/*.parquet|csv.gz)
#
#  How it searches
#   1. Every stock, every 5 minutes (09:45-14:30) is described by ~90 numbers: price behaviour, OI changes by the four
#      actions (writing / short-cover / long-build / unwind) on calls and puts, near-the-money vs far strikes, put/call ratio,
#      OI walls, option volume ...   Everything is in the stock's own "sigma" units so stocks are comparable.
#   2. Target = what the stock does over the next 30 / 60 minutes / to the close, MINUS what all stocks did at that moment
#      (so we look for stock-specific trend, not just "the market went up").
#   3. A computer-search (gradient-boosted trees) tries all combinations. The objective is the OUT-OF-SAMPLE score:
#      models are always judged on trading days they never saw (leave-days-out).
#   4. The ONLY thing that counts is what OI adds on top of a price-only model of the same kind. Placebo run included.
#   5. Readable scenarios come from shallow decision trees; a scenario is kept only if the same structure re-appears across
#      folds and keeps its sign on unseen days. Confidence intervals resample whole days.
#
#  Tabs: "OI Disc Summary", "OI Disc Features", "OI Disc Scenarios", "OI Disc Notes"
#  Usage: python oi_discover.py [--selftest]       env: OI_ARCH (dir), OI_BAND (10)
# ==================================================================================
import os, sys, glob, time, math, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupKFold

OUT = "out/oi"; os.makedirs(OUT, exist_ok=True)
ARCH = os.environ.get("OI_ARCH", f"{OUT}/archive")
BAND = int(os.environ.get("OI_BAND", "10")); STEP = 5
MIN_OI_SHARE = 0.3; MIN_ACTIVE = 0.5
WINS = {"w15": 15, "w30": 30, "wo": None}         # None = since the open
TARGETS = {"fwd30": "next 30 min", "fwd60": "next 60 min", "fwdEOD": "to the close"}

# ───────────────────────────── loading ─────────────────────────────
def read_any(stem):
    for ext, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        if os.path.exists(stem + ext):
            try: return fn(stem + ext)
            except Exception: pass
    return None

def tsfix(s):
    t = pd.to_datetime(s)
    if getattr(t.dt, "tz", None) is not None: t = t.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return t

def stock_files():
    names = {}
    for f in glob.glob(f"{ARCH}/*_[0-9]*"):
        b = os.path.basename(f); names[b.split("_")[0] if b.count("_") == 1 else b.rsplit("_", 1)[0]] = 1
    return sorted(names)

def pos_fn(mod):
    return lambda m: int(np.searchsorted(mod, m, side="right")) - 1

def stock_sessions(u):
    opt = None
    for f in glob.glob(f"{ARCH}/{u}_[0-9]*"):
        opt = read_any(f.replace(".parquet", "").replace(".csv.gz", "")); break
    spot = read_any(f"{ARCH}/{u}_spot")
    if opt is None or spot is None: return
    spot["ts"] = tsfix(spot["ts"]); opt["ts"] = tsfix(opt["ts"])
    spot["session"] = spot["ts"].dt.date.astype(str); opt["session"] = opt["ts"].dt.date.astype(str)
    opt["typ"] = opt["typ"].astype(str)
    prev = None
    for s, sp in spot.groupby("session"):
        sp = sp.sort_values("ts").reset_index(drop=True)
        d = opt[opt["session"] == s]
        if len(d) and len(sp) > 100: yield s, sp, d, prev
        prev = sp["close"].iloc[-1]

def session_features(u, s, sp, d, prevclose):
    idx = pd.DatetimeIndex(sp["ts"]); ks = sorted(d["strike"].unique())
    if len(ks) < 6: return []
    c = sp["close"].to_numpy(float); hi = np.maximum.accumulate(sp["high"].to_numpy(float)); lo = np.minimum.accumulate(sp["low"].to_numpy(float))
    v = sp["volume"].fillna(0).to_numpy(float); vw = np.cumsum(c * v) / np.maximum(np.cumsum(v), 1)
    i0 = int(np.argmin(np.abs(np.array(ks) - c[0]))); bs = ks[max(0, i0 - BAND): i0 + BAND + 1]
    mats = {}
    for t in ("C", "P"):
        dd = d[(d["typ"] == t) & d["strike"].isin(bs)]
        if dd.empty: return []
        for col in ("oi", "close", "volume"):
            pv = dd.pivot_table(index="ts", columns="strike", values=col, aggfunc="last").sort_index()
            pv = pv.reindex(pv.index.union(idx))
            pv = (pv.ffill().reindex(idx).fillna(0) if col != "volume" else pv.reindex(idx).fillna(0))
            if col == "close": pv = pv.bfill()
            mats[(t, col)] = pv
    cols = sorted(set(mats[("C", "oi")].columns) & set(mats[("P", "oi")].columns))
    if len(cols) < 6: return []
    K = np.array(cols)
    A = {k: v_[cols].to_numpy(float) for k, v_ in mats.items()}
    mod = (idx.hour * 60 + idx.minute).to_numpy(); pos = pos_fn(mod); T = len(c)
    band0 = A[("C", "oi")][0].sum() + A[("P", "oi")][0].sum()
    if band0 <= 0: return []
    ms = [m for m in range(9 * 60 + 45, 14 * 60 + 31, STEP) if pos(m) >= 30 and mod[pos(m)] >= m - 3]
    if not ms: return []
    ii = np.array([pos(m) for m in ms]); mm = np.array(ms)
    F = {"symbol": u, "session": s, "m": mm, "band0": band0}
    P_ = lambda w: np.array([max(pos(m - w), 0) for m in ms])
    # price block
    for w in (5, 15, 30, 60): F[f"ret{w}"] = (c[ii] / c[P_(w)] - 1) * 100
    F["ret_open"] = (c[ii] / c[0] - 1) * 100
    F["gap"] = np.full(len(ii), (c[0] / prevclose - 1) * 100 if prevclose else np.nan)
    rng = np.maximum(hi[ii] - lo[ii], 1e-9)
    F["range_pos"] = (c[ii] - lo[ii]) / rng; F["range_so_far"] = rng / c[ii] * 100
    F["dist_vwap"] = (c[ii] / vw[ii] - 1) * 100
    cv = np.cumsum(v); F["vol_ratio"] = np.array([(cv[i] - cv[max(i - 15, 0)]) / 15 / max(cv[i] / (i + 1), 1e-9) for i in ii])
    F["tod"] = mm
    # OI block
    Coi, Poi, Cpx, Ppx = A[("C", "oi")], A[("P", "oi")], A[("C", "close")], A[("P", "close")]
    atm = np.array([int(np.argmin(np.abs(K - c[i]))) for i in ii]); kk = np.arange(len(K))[None, :]
    near = np.abs(kk - atm[:, None]) <= 2; callotm = kk > atm[:, None] + 2; putotm = kk < atm[:, None] - 2
    for wn, w in WINS.items():
        pw = np.zeros(len(ii), int) if w is None else P_(w)
        res = {}
        for t, OI, PX in (("c", Coi, Cpx), ("p", Poi, Ppx)):
            dO = OI[ii] - OI[pw]; dP = PX[ii] - PX[pw]
            res[t] = {"wr": np.where((dO > 0) & (dP < 0), dO, 0).sum(1), "cv": np.where((dO < 0) & (dP > 0), -dO, 0).sum(1),
                      "lb": np.where((dO > 0) & (dP > 0), dO, 0).sum(1), "un": np.where((dO < 0) & (dP < 0), -dO, 0).sum(1), "net": dO.sum(1), "dO": dO}
            for a in ("wr", "cv", "lb", "un"): F[f"{t}_{a}_{wn}"] = res[t][a] / band0 * 100
            F[f"{t}_net_{wn}"] = res[t]["net"] / band0 * 100
        bull = res["c"]["cv"] + res["c"]["lb"] + res["p"]["wr"] + res["p"]["un"]; bear = res["c"]["wr"] + res["c"]["un"] + res["p"]["cv"] + res["p"]["lb"]
        F[f"pressure_{wn}"] = (bull - bear) / band0 * 100
        F[f"oi_total_{wn}"] = (res["c"]["net"] + res["p"]["net"]) / band0 * 100
        if wn == "w30":
            F["c_near_w30"] = np.where(near, res["c"]["dO"], 0).sum(1) / band0 * 100
            F["c_otm_w30"] = np.where(callotm, res["c"]["dO"], 0).sum(1) / band0 * 100
            F["p_near_w30"] = np.where(near, res["p"]["dO"], 0).sum(1) / band0 * 100
            F["p_otm_w30"] = np.where(putotm, res["p"]["dO"], 0).sum(1) / band0 * 100
    pcr = Poi[ii].sum(1) / np.maximum(Coi[ii].sum(1), 1); pcr0 = Poi[0].sum() / max(Coi[0].sum(), 1)
    F["pcr"] = pcr; F["pcr_chg_open"] = pcr - pcr0; F["pcr_chg_w30"] = pcr - Poi[P_(30)].sum(1) / np.maximum(Coi[P_(30)].sum(1), 1)
    def walls(i, spot_, a):
        cm = np.where(K >= K[a], Coi[i], -1); pm = np.where(K <= K[a], Poi[i], -1)
        return (K[int(np.argmax(cm))] / spot_ - 1) * 100, (K[int(np.argmax(pm))] / spot_ - 1) * 100
    a0 = int(np.argmin(np.abs(K - c[0]))); w0 = walls(0, c[0], a0)
    cw = np.array([walls(i, c[i], a) for i, a in zip(ii, atm)])
    F["call_wall_dist"], F["put_wall_dist"] = cw[:, 0], cw[:, 1]
    F["call_wall_shift"], F["put_wall_shift"] = cw[:, 0] - w0[0] + (c[ii] / c[0] - 1) * 100 * 0, cw[:, 1] - w0[1]
    for t, V in (("c", A[("C", "volume")]), ("p", A[("P", "volume")])):
        cs = np.cumsum(V.sum(1)); F[f"{t}_vol15"] = np.array([(cs[i] - cs[max(i - 15, 0)]) for i in ii]) / band0 * 100
    F["cp_vol15"] = np.log((F["c_vol15"] + 1e-3) / (F["p_vol15"] + 1e-3))
    # targets
    for h in (30, 60):
        j = np.array([int(np.searchsorted(mod, m + h, side="left")) for m in ms])
        ok = (j < T); jj = np.minimum(j, T - 1); ok &= (mod[jj] - (mm + h)) <= 3
        F[f"fwd{h}"] = np.where(ok, (c[jj] / c[ii] - 1) * 100, np.nan)
    F["fwdEOD"] = np.where((mod[-1] >= 15 * 60 + 25) & (ii < T - 1), (c[-1] / c[ii] - 1) * 100, np.nan)
    return pd.DataFrame(F).to_dict("records")

def build_dataset():
    rows = []; t0 = time.time(); names = stock_files()
    for n, u in enumerate(names, 1):
        try:
            for s, sp, d, prev in stock_sessions(u): rows += session_features(u, s, sp, d, prev) or []
        except Exception as e: print("skip", u, type(e).__name__, str(e)[:80])
        if n % 20 == 0: print(f"  features {n}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    if D.empty: raise SystemExit("no archive data found in " + ARCH)
    return D

# ───────────────────────────── preparation ─────────────────────────────
PRICE = ["ret5", "ret15", "ret30", "ret60", "ret_open", "gap", "range_pos", "range_so_far", "dist_vwap", "vol_ratio", "tod"]
SCALE_PRICE = ["ret5", "ret15", "ret30", "ret60", "ret_open", "gap", "range_so_far", "dist_vwap"]

def prepare(D):
    D = D.copy()
    mx = D.groupby("symbol")["band0"].transform("max"); D = D[D["band0"] >= MIN_OI_SHARE * mx]
    oi_cols = [c for c in D.columns if c not in PRICE + ["symbol", "session", "m", "band0"] + list(TARGETS)]
    act = D.assign(a=(D["c_net_w30"] != 0) | (D["p_net_w30"] != 0)).groupby("symbol")["a"].mean()
    D = D[D["symbol"].map(act) >= MIN_ACTIVE].copy()
    skip = {"pcr", "range_pos", "tod", "vol_ratio", "pcr_chg_open", "pcr_chg_w30", "cp_vol15", "call_wall_dist", "put_wall_dist", "call_wall_shift", "put_wall_shift"}
    for c in SCALE_PRICE + [c for c in oi_cols if c not in skip]:      # stock-sigma units (no centring)
        sd = D.groupby("symbol")[c].transform("std").replace(0, np.nan); D[c] = D[c] / sd
    for t in TARGETS:
        sd = D.groupby("symbol")[t].transform("std").replace(0, np.nan); D[f"raw_{t}"] = D[t]; D[t] = D[t] / sd
        D[f"mkt_{t}"] = D.groupby(["session", "m"])[t].transform("mean"); D[f"neu_{t}"] = D[t] - D[f"mkt_{t}"]
    oi_cols = [c for c in oi_cols if c in D.columns]
    return D.reset_index(drop=True), oi_cols

# ───────────────────────────── scoring ─────────────────────────────
def session_ic(df, pred, tgt):
    x = pd.DataFrame({"s": df["session"].values, "p": pred, "t": df[tgt].values}).dropna()
    r = x.groupby("s").apply(lambda g: g["p"].rank().corr(g["t"].rank()) if len(g) > 50 else np.nan).dropna()
    return r

def spread(df, pred, raw):
    x = pd.DataFrame({"s": df["session"].values, "p": pred, "t": df[raw].values}).dropna()
    out = {}
    for s, g in x.groupby("s"):
        if len(g) < 100: continue
        q1, q9 = g["p"].quantile(.1), g["p"].quantile(.9); out[s] = g.loc[g["p"] >= q9, "t"].mean() - g.loc[g["p"] <= q1, "t"].mean()
    return pd.Series(out)

def boot_mean(x, B=2000, seed=1):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 4: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def newmodel(seed=0):
    return HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=0.05, min_samples_leaf=300, l2_regularization=5.0, random_state=seed)

def cv_models(D, oi_cols, tgt, folds, placebo=True):
    base_cols = [c for c in PRICE if c in D.columns]; full_cols = base_cols + oi_cols
    pb, pf, pp = np.full(len(D), np.nan), np.full(len(D), np.nan), np.full(len(D), np.nan)
    imps = {c: [] for c in oi_cols}; rng = np.random.default_rng(3)
    y = D[tgt].to_numpy(); ok = ~np.isnan(y)
    for tr, te in folds:
        tr = tr[ok[tr]]
        mb = newmodel().fit(D.iloc[tr][base_cols].fillna(0), y[tr]); pb[te] = mb.predict(D.iloc[te][base_cols].fillna(0))
        mf = newmodel().fit(D.iloc[tr][full_cols].fillna(0), y[tr]); Xte = D.iloc[te][full_cols].fillna(0); pf[te] = mf.predict(Xte)
        if placebo:                       # OI columns shuffled across rows inside each session: must NOT beat the baseline
            Xs = D.iloc[tr][full_cols].fillna(0).copy(); Xt = Xte.copy()
            for X_, ds in ((Xs, D.iloc[tr]["session"].values), (Xt, D.iloc[te]["session"].values)):
                for s in np.unique(ds):
                    m_ = ds == s; perm = rng.permutation(m_.sum())
                    X_.loc[X_.index[m_], oi_cols] = X_.loc[X_.index[m_], oi_cols].values[perm]
            pp[te] = newmodel().fit(Xs, y[tr]).predict(Xt)
        base_ic = session_ic(D.iloc[te], pf[te], tgt).mean()
        sub = np.arange(len(Xte)); sub = sub if len(sub) < 6000 else rng.choice(sub, 6000, replace=False)
        Xsub, dsub = Xte.iloc[sub], D.iloc[te].iloc[sub]
        ref = session_ic(dsub, mf.predict(Xsub), tgt).mean()
        for c in oi_cols:
            Xp = Xsub.copy(); Xp[c] = rng.permutation(Xp[c].values)
            imps[c].append(ref - session_ic(dsub, mf.predict(Xp), tgt).mean())
    return pb, pf, pp, imps

def summarize(D, oi_cols, folds):
    rows, feat_rows = [], []
    for tgt, lab in TARGETS.items():
        for kind, ytgt in (("stock vs market (neutral)", f"neu_{tgt}"), ("raw direction", tgt)):
            pb, pf, pp, imps = cv_models(D, oi_cols, ytgt, folds, placebo=True)
            ib, if_, ip = (session_ic(D, p, ytgt) for p in (pb, pf, pp))
            diff = (if_ - ib).dropna(); dm, dl, dh = boot_mean(diff.values)
            pdiff = (ip - ib).dropna(); pm, pl, ph = boot_mean(pdiff.values)
            sb, sf = spread(D, pb, f"raw_{tgt}"), spread(D, pf, f"raw_{tgt}")
            sm, sl, sh = boot_mean((sf - sb).dropna().values)
            verdict = "OI ADDS predictive power" if dl > 0 else ("OI hurts" if dh < 0 else "no evidence OI adds anything")
            rows.append({"target": f"{lab} - {kind}", "days tested": len(diff), "IC price-only": ib.mean(), "IC price+OI": if_.mean(),
                         "OI adds (IC)": dm, "CI low": dl, "CI high": dh, "placebo adds": pm, "placebo CI low": pl, "placebo CI high": ph,
                         "top-vs-bottom decile % (price+OI)": sf.mean(), "decile spread OI adds %": sm, "spread CI low": sl, "spread CI high": sh, "verdict": verdict})
            if ytgt == f"neu_fwd60":
                for c, v in imps.items(): feat_rows.append({"feature": c, "importance (IC lost when scrambled)": np.mean(v), "folds positive": int(np.sum(np.array(v) > 0)), "folds": len(v)})
    F = pd.DataFrame(feat_rows).sort_values("importance (IC lost when scrambled)", ascending=False) if feat_rows else pd.DataFrame()
    return pd.DataFrame(rows), F

# ───────────────────────────── readable scenarios ─────────────────────────────
def leaf_conditions(tree, feats):
    t = tree.tree_; out = {}
    def rec(n, conds):
        if t.children_left[n] == -1: out[n] = conds; return
        f, thr = feats[t.feature[n]], t.threshold[n]
        rec(t.children_left[n], conds + [(f, "<=", thr)]); rec(t.children_right[n], conds + [(f, ">", thr)])
    rec(0, []); return out

def mine_scenarios(D, oi_cols, folds, tgt="neu_fwd60", depth=3):
    cols = [c for c in PRICE if c in D.columns] + oi_cols
    y = D[tgt].to_numpy(); ok = ~np.isnan(y); bank = {}
    for k, (tr, te) in enumerate(folds):
        tr = tr[ok[tr]]; te = te[ok[te]]
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(300, len(tr) // 60), random_state=0).fit(D.iloc[tr][cols].fillna(0), y[tr])
        lc = leaf_conditions(tree, cols); leaves = tree.apply(D.iloc[te][cols].fillna(0))
        for leaf, conds in lc.items():
            sel = te[leaves == leaf]
            if len(sel) < 30: continue
            sig = tuple(sorted(set((f, op) for f, op, _ in conds)))
            b = bank.setdefault(sig, {"thr": {}, "folds": [], "rows": []})
            for f, op, thr in conds: b["thr"].setdefault((f, op), []).append(thr)
            b["folds"].append((k, float(np.sign(y[sel].mean())), float(tree.tree_.value[leaf].ravel()[0]))); b["rows"].append(sel)
    out = []; nf = len(folds)
    for sig, b in bank.items():
        if len({k for k, _, _ in b["folds"]}) < max(3, nf // 2): continue
        sel = np.concatenate(b["rows"]); g = D.iloc[sel]
        train_sign = np.sign(np.mean([m for _, _, m in b["folds"]])) or 1
        per_s = (train_sign * g[tgt]).groupby(g["session"]).mean()
        m, lo, hi = boot_mean(per_s.values)
        same = np.mean([np.sign(mm_) == train_sign for _, _, mm_ in b["folds"]])
        txt = " AND ".join(f"{f} {op} {np.median(b['thr'][(f, op)]):.2f}" for f, op in sig)
        out.append({"scenario (stock-sigma units)": txt, "learned direction": "UP (bullish)" if train_sign > 0 else "DOWN (bearish)",
                    "folds seen": len({k for k, _, _ in b["folds"]}), "rows (unseen days)": len(sel), "stocks": g["symbol"].nunique(), "days": g["session"].nunique(),
                    "unseen-day edge (sigma, in direction)": m, "CI low": lo, "CI high": hi,
                    "hit % (direction right)": (np.sign(g[tgt].values) == train_sign).mean() * 100,
                    "raw next-60m % (direction-signed)": (train_sign * g["raw_fwd60"]).mean(),
                    "verdict": "REPEATS on unseen days" if lo > 0 else ("INCONSISTENT" if hi > 0 else "reverses")})
    S = pd.DataFrame(out)
    return (S.sort_values("CI low", ascending=False) if len(S) else S), len(bank)

NOTES = [["term", "meaning"],
 ["what this does", "describes every stock every 5 minutes with ~90 numbers (price + OI actions + walls + volume), then lets a search procedure find which combinations predict the next move"],
 ["sigma units", "a number measured in multiples of that stock's own normal size. 2.0 means twice as big as usual for that stock"],
 ["target", "'stock vs market (neutral)' = stock's next move minus the average stock at that moment: tests stock-specific trend. 'raw direction' also includes the market's move"],
 ["IC", "rank correlation between the model's forecast and what happened, per trading day, averaged. 0 = no skill. ~0.02-0.05 is a useful trading edge when stable"],
 ["OI adds (IC)", "price+OI model minus price-only model, same days, never trained on the day being scored. CI = 95% from resampling days"],
 ["placebo", "same test with the OI columns scrambled inside each day. Must be ~0; if it is not, the test is leaking and results are void"],
 ["scenarios", "shallow-tree rules. Only structures that re-appear in several folds are listed, scored only on days not used to build them. 'REPEATS on unseen days' means CI above zero"],
 ["multiple tests", "many scenarios are examined; a few will look good by luck. Prefer scenarios with many days, high 'folds seen', and economic sense (matches a known OI action). Confirm on new days before trusting"],
 ["sample size", "limited by the history that exists (~15 trading days for the October series). The archive grows every run; re-run weekly"]]

def main(selftest=False):
    t0 = time.time(); D = build_dataset(); print(f"dataset: {len(D):,} marks, {D['symbol'].nunique()} stocks, {D['session'].nunique()} days ({time.time()-t0:.0f}s)")
    if selftest and os.environ.get("PLANT"):
        pass
    D, oi_cols = prepare(D)
    if selftest and os.environ.get("PLANT"):                     # plant a known signal: forward move follows call-short-covering pressure
        z = D["pressure_w30"].fillna(0); D["neu_fwd60"] = D["neu_fwd60"] + 0.6 * z.clip(-3, 3)
    ns = D["session"].nunique(); k = min(6, ns)
    if k < 3: raise SystemExit("need at least 3 trading days of archive")
    folds = list(GroupKFold(n_splits=k).split(D, groups=D["session"]))
    print(f"{len(oi_cols)} OI features, {k} leave-days-out folds"); S, F = summarize(D, oi_cols, folds)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.round(4).to_string(index=False))
    if len(F): print(F.head(12).round(4).to_string(index=False))
    SC, nsig = mine_scenarios(D, oi_cols, folds)
    print(f"{nsig} rule structures examined, {len(SC)} recurring")
    if len(SC): print(SC.head(12).round(3).to_string(index=False))
    for n, df in (("summary", S), ("features", F), ("scenarios", SC)): df.to_csv(f"{OUT}/disc_{n}.csv", index=False)
    if selftest: return S, F, SC
    import oi_probe as P
    sh = P.open_sheet()
    P.put_tab(sh, "OI Disc Summary", list(S.columns), S.round(4).values.tolist())
    if len(F): P.put_tab(sh, "OI Disc Features", list(F.columns), F.round(4).values.tolist())
    P.put_tab(sh, "OI Disc Scenarios", list(SC.columns) if len(SC) else ["scenario"], SC.round(3).values.tolist() if len(SC) else [["none recurred on unseen days"]])
    P.put_tab(sh, "OI Disc Notes", NOTES[0], NOTES[1:] + [["rule structures examined", str(nsig)], ["days", str(ns)], ["marks", str(len(D))]])
    print("sheet tabs written")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n===== SELFTEST", "PLANTED SIGNAL" if plant else "NULL (random data)", "=====")
            S, F, SC = main(selftest=True)
            r = S[S["target"].str.contains("next 60 min - stock vs")].iloc[0]
            if plant: assert r["verdict"] == "OI ADDS predictive power" and len(SC) > 0, "planted signal not detected"
            else: assert r["verdict"] != "OI ADDS predictive power", "false positive on random data"
        print("SELFTEST OK")
    else: main()
