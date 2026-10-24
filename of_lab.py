# ==================================================================================
# LIQUIDITY / ORDER-FLOW LAB — does order flow predict what price does next, and under which conditions?
#  Input: the minute archive from of_archive.py (213 F&O stocks x ~56 sessions; 09:15-15:14 only, official daily candle for day-level work).
#  1-minute data are rolled into 5-minute bars. Everything is measured as an ABNORMAL return: the stock's forward return minus the average of
#  all stocks over the same bars of the same day (market and time-of-day removed). Entry = next bar's open, costs charged (COST_BPS round trip).
#  EVENTS (each with a mirror image and, where possible, a CONTROL with the same price/volume but neutral delta, to show whether delta adds anything):
#   E1/E2  ABSORPTION      heavy selling (buying) on a surge of volume and price does NOT fall (rise)            -> expect reversal
#   E3     DIVERGENCE      new day high with negative delta / new day low with positive delta                       -> expect reversal
#   E4     CONFIRMED BREAK new day high (low) with strong buy (sell) delta and volume                              -> expect continuation
#   E5     STACKED IMBALANCE minutes with >=3 consecutive-level 3:1 imbalances, net buy vs sell                    -> expect continuation
#   E6     HIDDEN ACCUMULATION day delta strongly positive while price sits below vwap (and mirror)                -> expect reversion to the delta side
#   E7     OPENING DELTA   top/bottom decile of first-30-minute delta share across stocks -> rest of day
#   E8     DAY DELTA       top/bottom decile of the day's delta share -> NEXT day open-to-close (official candles)
#  PLUS: rank-IC scan of every order-flow feature (raw and after removing the past 15-min price move), and a decision-tree discovery of
#  CONDITIONS (walk-forward by day, locked holdout = last 20% of sessions, Benjamini-Hochberg FDR 10%).
#  Tabs: "OFL Events", "OFL IC", "OFL Scenarios", "OFL Notes".  Research only - nothing here places orders.
#  Usage: python of_lab.py [--selftest]      env: OF_ARCH (archive dir), COST_BPS (10)
# ==================================================================================
import os, sys, glob, time, warnings, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ARCH = os.environ.get("OF_ARCH", "out/of/archive"); OUT = "out/of_lab"; os.makedirs(OUT, exist_ok=True)
COST = float(os.environ.get("COST_BPS", "10")); SELFTEST = "--selftest" in sys.argv; BAR = 5
COLS = ["symbol", "date", "hhmm", "o", "h", "l", "c", "tot", "buy", "sell", "delta", "cd", "vwap", "bstack", "sstack", "d_o", "d_c", "d_v"]

# ───────────────────────────── load and build 5-minute bars ─────────────────────────────
def read_any(p):
    try: return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    except Exception as e: print("skip", p, type(e).__name__); return None

def load_minutes():
    fr = []
    for f in sorted(glob.glob(f"{ARCH}/*.parquet") + glob.glob(f"{ARCH}/*.csv.gz")):
        if os.path.basename(f).startswith("prof_"): continue
        d = read_any(f)
        if d is None: continue
        for c in COLS:
            if c not in d.columns: d[c] = np.nan
        d = d[COLS]; fr.append(d[(d["hhmm"] >= 915) & (d["hhmm"] < 1515)])
    if not fr: raise SystemExit("no archive files in " + ARCH)
    return pd.concat(fr, ignore_index=True)

def make_bars(M):
    M = M.copy(); M["bar"] = ((M["hhmm"] // 100) * 60 + M["hhmm"] % 100 - 555) // BAR; M["vw"] = M["vwap"] * M["tot"]; M["bs3"] = (M["bstack"] >= 3).astype(int); M["ss3"] = (M["sstack"] >= 3).astype(int)
    B = M.groupby(["symbol", "date", "bar"], sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("tot", "sum"), buy=("buy", "sum"), sell=("sell", "sum"), delta=("delta", "sum"), vw=("vw", "sum"),
                                                          cd=("cd", "last"), bs3=("bs3", "sum"), ss3=("ss3", "sum"), d_o=("d_o", "first"), d_c=("d_c", "first")).reset_index()
    B = B[(B["v"] > 0) & B["o"].notna() & B["c"].notna()].reset_index(drop=True); return B

def expanding_std(s, n=300): return s.expanding(n).std().shift(1)

def features(B):
    g = B.groupby(["symbol", "date"], sort=False); B["dn"] = B["delta"] / B["v"]; B["ret"] = (B["c"] / B["o"] - 1) * 1e4
    B["dz"] = B["dn"] / B.groupby("symbol")["dn"].transform(expanding_std); B["rz"] = B["ret"] / B.groupby("symbol")["ret"].transform(expanding_std)
    B["vr"] = B["v"] / B.groupby(["symbol", "bar"])["v"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    B["cumv"] = g["v"].cumsum(); B["cumd"] = g["delta"].cumsum(); B["cdz"] = B["cumd"] / B["cumv"]; B["cumvw"] = g["vw"].cumsum(); B["dist_vwap"] = (B["c"] / (B["cumvw"] / B["cumv"]) - 1) * 1e4
    B["_cmx"] = g["h"].cummax(); B["_cmn"] = g["l"].cummin(); g = B.groupby(["symbol", "date"], sort=False)
    B["newhigh"] = (B["h"] > g["_cmx"].shift(1)) & (B["bar"] >= 3); B["newlow"] = (B["l"] < g["_cmn"].shift(1)) & (B["bar"] >= 3)
    B["bstk"] = B["bs3"] - B["ss3"]; B["day_ret"] = (B["c"] / g["o"].transform("first") - 1) * 1e4; B["retp3"] = (B["c"] / g["c"].shift(3) - 1) * 1e4
    B["mkt_sofar"] = B.groupby(["date", "bar"])["day_ret"].transform("mean"); B["lq"] = B.groupby("symbol")["v"].transform("mean").rank(pct=True)
    nxt = g["o"].shift(-1); lo, hi = None, None
    for N in (3, 6, 12): B[f"f{N}"] = (g["c"].shift(-N) / nxt - 1) * 1e4
    B["fE"] = (g["c"].transform("last") / nxt - 1) * 1e4
    for f in ("f3", "f6", "f12", "fE"):
        lo, hi = B[f].quantile([.005, .995]); B[f] = B[f].clip(lo, hi); B[f + "x"] = B[f] - B.groupby(["date", "bar"])[f].transform("mean")
    dates = sorted(B["date"].unique()); cut = dates[int(len(dates) * 0.8)]; B["holdout"] = B["date"] >= cut; B["date_id"] = pd.Categorical(B["date"], dates).codes
    return B.drop(columns=["_cmx", "_cmn"]), cut

# ───────────────────────────── statistics ─────────────────────────────
def cmean(v, d, B_=1000, seed=1):
    v = np.asarray(v, float); ok = ~np.isnan(v); v, d = v[ok], np.asarray(d)[ok]
    if len(v) < 30: return np.nan, np.nan, np.nan, np.nan, 0
    u, inv = np.unique(d, return_inverse=True); s = np.bincount(inv, v); n = np.bincount(inv).astype(float); rng = np.random.default_rng(seed); bs = np.empty(B_)
    for b in range(B_): r = rng.integers(0, len(u), len(u)); bs[b] = s[r].sum() / max(n[r].sum(), 1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs <= 0).mean(), len(u)

def cdiff(a, da, b, db, B_=800, seed=3):
    a, b, da, db = np.asarray(a, float), np.asarray(b, float), np.asarray(da), np.asarray(db); oa, ob = ~np.isnan(a), ~np.isnan(b); a, da, b, db = a[oa], da[oa], b[ob], db[ob]
    if len(a) < 30 or len(b) < 30: return np.nan, np.nan, np.nan
    ds = np.unique(np.r_[da, db]); ia, ib = np.searchsorted(ds, da), np.searchsorted(ds, db); sa, na = np.bincount(ia, a, len(ds)), np.bincount(ia, minlength=len(ds)).astype(float); sb, nb = np.bincount(ib, b, len(ds)), np.bincount(ib, minlength=len(ds)).astype(float)
    rng = np.random.default_rng(seed); out = []
    for _ in range(B_):
        w = np.bincount(rng.integers(0, len(ds), len(ds)), minlength=len(ds))
        if (w * na).sum() and (w * nb).sum(): out.append((w * sa).sum() / (w * na).sum() - (w * sb).sum() / (w * nb).sum())
    return a.mean() - b.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)

def bh_fdr(p, q=0.10):
    p = np.asarray(p, float); ok = ~np.isnan(p); out = np.zeros(len(p), bool)
    if ok.sum() == 0: return out
    idx = np.where(ok)[0]; order = idx[np.argsort(p[idx])]; m = len(order); thr = 0
    for rank, i in enumerate(order, 1):
        if p[i] <= q * rank / m: thr = rank
    out[order[:thr]] = True; return out

# ───────────────────────────── events ─────────────────────────────
Z, VR = 1.5, 1.2
def cooldown(B, mask, gap=6):
    idx = np.where(mask.to_numpy())[0]; sym = B["symbol"].to_numpy()[idx]; dat = B["date_id"].to_numpy()[idx]; bar = B["bar"].to_numpy()[idx]; keep = []; last = (None, None, -99)
    for k in range(len(idx)):
        if (sym[k], dat[k]) != (last[0], last[1]) or bar[k] - last[2] >= gap: keep.append(idx[k]); last = (sym[k], dat[k], bar[k])
    return np.array(keep, int)

def definitions(B):
    ok = (B["bar"] >= 3) & (B["bar"] <= 65) & B["dz"].notna() & B["vr"].notna(); up, dn = B["ret"] >= 0, B["ret"] <= 0; neutral = B["dz"].abs() < 0.5
    D = []
    D.append(("E1 absorption of SELLING -> long", +1, ok & (B["dz"] <= -Z) & (B["vr"] >= VR) & up, ok & (B["vr"] >= VR) & up & neutral))
    D.append(("E2 absorption of BUYING -> short", -1, ok & (B["dz"] >= Z) & (B["vr"] >= VR) & dn, ok & (B["vr"] >= VR) & dn & neutral))
    D.append(("E3a divergence: new day HIGH with negative delta -> short", -1, ok & B["newhigh"] & (B["dz"] <= -0.5), ok & B["newhigh"] & (B["dz"] >= 0.5)))
    D.append(("E3b divergence: new day LOW with positive delta -> long", +1, ok & B["newlow"] & (B["dz"] >= 0.5), ok & B["newlow"] & (B["dz"] <= -0.5)))
    D.append(("E4a confirmed break UP (new high, strong buy delta, volume) -> long", +1, ok & B["newhigh"] & (B["dz"] >= Z) & (B["vr"] >= 1.5), ok & B["newhigh"] & (B["vr"] >= 1.5) & (B["dz"] < 0)))
    D.append(("E4b confirmed break DOWN (new low, strong sell delta, volume) -> short", -1, ok & B["newlow"] & (B["dz"] <= -Z) & (B["vr"] >= 1.5), ok & B["newlow"] & (B["vr"] >= 1.5) & (B["dz"] > 0)))
    D.append(("E5a buy-imbalance stacks -> long", +1, ok & (B["bstk"] >= 2), ok & (B["bstk"] == 0) & (B["ret"] > 0) & (B["vr"] >= VR)))
    D.append(("E5b sell-imbalance stacks -> short", -1, ok & (B["bstk"] <= -2), ok & (B["bstk"] == 0) & (B["ret"] < 0) & (B["vr"] >= VR)))
    D.append(("E6a hidden accumulation (day delta strongly +, price below vwap) -> long", +1, ok & (B["cdz"] >= 0.15) & (B["dist_vwap"] <= -20) & (B["bar"] >= 6), ok & (B["cdz"].abs() < 0.05) & (B["dist_vwap"] <= -20) & (B["bar"] >= 6)))
    D.append(("E6b hidden distribution (day delta strongly -, price above vwap) -> short", -1, ok & (B["cdz"] <= -0.15) & (B["dist_vwap"] >= 20) & (B["bar"] >= 6), ok & (B["cdz"].abs() < 0.05) & (B["dist_vwap"] >= 20) & (B["bar"] >= 6)))
    r5 = B["cdz"].where(B["bar"] == 5).groupby(B["date"]).rank(pct=True)
    D.append(("E7a opening delta top decile (first 30 min) -> long rest of day", +1, (r5 >= 0.9), None)); D.append(("E7b opening delta bottom decile -> short rest of day", -1, (r5 <= 0.1) & r5.notna(), None))
    return D

def evaluate(B, defs):
    rows = []; samples = {}
    for name, side, m, ctrl in defs:
        e7 = name.startswith("E7"); hs = ("fE",) if e7 else ("f6", "f12", "fE")
        idx = cooldown(B, m, 6 if not e7 else 99); cidx = cooldown(B, ctrl, 6) if ctrl is not None else None
        for h in hs:
            col = h + "x"; hlab = {"f6": "next 30 min", "f12": "next 60 min", "fE": "to 15:14"}[h]
            for part, sel in (("dev", ~B["holdout"]), ("holdout", B["holdout"])):
                I = idx[sel.to_numpy()[idx]]; v = side * B[col].to_numpy()[I] - COST; d = B["date_id"].to_numpy()[I]; gross = (v + COST)
                mean, lo, hi, p, nd = cmean(v, d); samples[(name, h, part)] = (v, d)
                if part == "dev":
                    r = {"event": name, "horizon": hlab, "dev events": int(np.isfinite(v).sum()), "dev days": nd, "dev gross bps": np.nanmean(gross) if len(gross) else np.nan, "dev net bps": mean, "dev CI low": lo, "dev CI high": hi, "dev p": p,
                         "dev hit % (net>0)": (v[np.isfinite(v)] > 0).mean() * 100 if np.isfinite(v).any() else np.nan}
                    if cidx is not None:
                        C = cidx[sel.to_numpy()[cidx]]; cv = side * B[col].to_numpy()[C] - COST; df, dlo, dhi = cdiff(v, d, cv, B["date_id"].to_numpy()[C]); r.update({"control net bps (same price+volume, neutral delta)": np.nanmean(cv) if len(cv) else np.nan, "delta adds vs control (bps)": df, "adds CI low": dlo, "adds CI high": dhi})
                else: r.update({"holdout events": int(np.isfinite(v).sum()), "holdout net bps": np.nanmean(v) if np.isfinite(v).any() else np.nan, "holdout CI low": lo})
            rows.append(r)
    R = pd.DataFrame(rows); R["survives FDR 10%"] = bh_fdr(R["dev p"].to_numpy())
    R["verdict"] = np.select([(R["dev CI low"] > 0) & R["survives FDR 10%"] & (R["holdout net bps"] > 0) & (R["holdout events"] >= 100), R["dev CI low"] > 0, R["dev CI high"] < 0], ["PASS (dev + FDR + holdout)", "PROMISING (dev only)", "LOSES after costs"], "no edge")
    return R

def event_e8(B):
    D = B.groupby(["symbol", "date"]).agg(vol=("v", "sum"), delta=("delta", "sum"), d_o=("d_o", "first"), d_c=("d_c", "first"), holdout=("holdout", "first"), date_id=("date_id", "first")).reset_index().sort_values(["symbol", "date"])
    D["share"] = D["delta"] / D["vol"]; g = D.groupby("symbol"); D["nx"] = (g["d_c"].shift(-1) / g["d_o"].shift(-1) - 1) * 1e4; D = D.dropna(subset=["nx", "share"]); D["nxx"] = D["nx"] - D.groupby("date")["nx"].transform("mean")
    D["rk"] = D.groupby("date")["share"].rank(pct=True); last_dev = D.loc[~D["holdout"], "date_id"].max(); rows = []
    for nm, side, m in (("E8a day delta top decile -> next day long (open to official close)", +1, D["rk"] >= 0.9), ("E8b day delta bottom decile -> next day short", -1, D["rk"] <= 0.1)):
        for part, sel in (("dev", ~D["holdout"] & (D["date_id"] < last_dev)), ("holdout", D["holdout"])):
            S = D[m & sel]; v = side * S["nxx"].to_numpy() - COST; mean, lo, hi, p, nd = cmean(v, S["date_id"].to_numpy())
            if part == "dev": r = {"event": nm, "horizon": "next day open-close", "dev events": len(v), "dev days": nd, "dev gross bps": v.mean() + COST if len(v) else np.nan, "dev net bps": mean, "dev CI low": lo, "dev CI high": hi, "dev p": p, "dev hit % (net>0)": (v > 0).mean() * 100 if len(v) else np.nan}
            else: r.update({"holdout events": len(v), "holdout net bps": v.mean() if len(v) else np.nan, "holdout CI low": lo})
        rows.append(r)
    return pd.DataFrame(rows)

# ───────────────────────────── rank-IC scan ─────────────────────────────
def gcorr(a, b, key):
    a = a - a.groupby(key).transform("mean"); b = b - b.groupby(key).transform("mean")
    num = (a * b).groupby(key).sum(); den = np.sqrt((a * a).groupby(key).sum() * (b * b).groupby(key).sum()); return num / den

def ic_scan(B):
    feats = {"dn": "bar delta / volume", "dz": "bar delta z-score", "vr": "volume vs usual for this time", "cdz": "day delta share so far", "bstk": "net imbalance stacks (buy-sell)", "dist_vwap": "distance from vwap (bps)", "rz": "bar return z-score (price-only control)", "retp3": "past 15-min return (price-only control)"}
    base = B[(B["bar"] >= 3) & (B["bar"] <= 65)].copy(); key = [base["date"], base["bar"]]; rows = []
    for tgt, tl in (("f6x", "next 30 min"), ("fEx", "to 15:14")):
        for f, fl in feats.items():
            X = base[list(dict.fromkeys([f, tgt, "retp3", "date", "bar", "holdout"]))].dropna(); k = [X["date"], X["bar"]]
            rx = X.groupby(k)[f].rank(pct=True); ry = X.groupby(k)[tgt].rank(pct=True); rz_ = X.groupby(k)["retp3"].rank(pct=True)
            ric = gcorr(rx, ry, k); rxz = gcorr(rx, rz_, k); ryz = gcorr(ry, rz_, k); pic = (ric - rxz * ryz) / np.sqrt((1 - rxz ** 2).clip(lower=1e-6) * (1 - ryz ** 2).clip(lower=1e-6))
            for lab, s in ((("raw", ric),) if f == "retp3" else (("raw", ric), ("after removing past 15-min move", pic))):
                s = s.dropna().reset_index(); s.columns = ["date", "bar", "ic"]; pd_ = s.groupby("date")["ic"].mean(); ho = pd_[pd_.index >= X.loc[X["holdout"], "date"].min()] if X["holdout"].any() else pd_.iloc[:0]; dv = pd_[~pd_.index.isin(ho.index)]
                m, lo, hi, p, nd = cmean(dv.to_numpy(), dv.index.to_numpy())
                rows.append({"feature": f, "meaning": fl, "target": tl, "version": lab, "dev days": nd, "dev mean IC": m, "CI low": lo, "CI high": hi, "p": p, "holdout mean IC": ho.mean() if len(ho) else np.nan})
    R = pd.DataFrame(rows); R["survives FDR 10%"] = bh_fdr(np.minimum(R["p"], 1 - R["p"]) * 2); R["verdict"] = np.select([(R["CI low"] > 0) | (R["CI high"] < 0)], ["non-zero (dev)"], "no signal")
    R.loc[(R["verdict"] == "non-zero (dev)") & (np.sign(R["dev mean IC"]) == np.sign(R["holdout mean IC"])) & R["survives FDR 10%"], "verdict"] = "HOLDS (dev + FDR + holdout same sign)"; return R

# ───────────────────────────── scenario discovery ─────────────────────────────
def scenarios(B, depth=3):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GroupKFold
    feats = ["dz", "dn", "rz", "vr", "cdz", "bstk", "dist_vwap", "day_ret", "mkt_sofar", "retp3", "bar", "lq", "newhigh", "newlow"]
    R = B[(B["bar"] >= 3) & (B["bar"] <= 65)].dropna(subset=feats + ["f6x"]).copy(); R[["newhigh", "newlow"]] = R[["newhigh", "newlow"]].astype(float)
    if len(R) > 300000: R = R.sample(300000, random_state=0)
    dev = R[~R["holdout"]].reset_index(drop=True); ho = R[R["holdout"]].reset_index(drop=True); X = dev[feats].to_numpy(); y = dev["f6x"].clip(-300, 300).to_numpy(); k = min(6, dev["date"].nunique()); bank = {}
    def leaves(tree):
        t = tree.tree_; out = {}
        def rec(n, c):
            if t.children_left[n] == -1: out[n] = c; return
            f, th = feats[t.feature[n]], t.threshold[n]; rec(t.children_left[n], c + [(f, "<=", th)]); rec(t.children_right[n], c + [(f, ">", th)])
        rec(0, []); return out
    for fo, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, groups=dev["date"])):
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(1500, len(tr) // 60), random_state=0).fit(X[tr], y[tr]); lv = leaves(tree); lid = tree.apply(X[te])
        for leaf, conds in lv.items():
            sel = te[lid == leaf]
            if len(sel) < 100: continue
            sig = tuple(sorted(set((f, o) for f, o, _ in conds))); b = bank.setdefault(sig, {"thr": {}, "fold": [], "rows": []})
            for f, o, th in conds: b["thr"].setdefault((f, o), []).append(th)
            b["fold"].append((fo, float(tree.tree_.value[leaf].ravel()[0]))); b["rows"].append(sel)
    out = []
    for sig, b in bank.items():
        if len({f for f, _ in b["fold"]}) < max(3, k // 2): continue
        sgn = np.sign(np.mean([m for _, m in b["fold"]])) or 1; conds = [(f, o, float(np.median(b["thr"][(f, o)]))) for f, o in sig]; sel = np.concatenate(b["rows"])
        v = sgn * dev["f6x"].to_numpy()[sel] - COST; mean, lo, hi, p, nd = cmean(v, dev["date_id"].to_numpy()[sel]); m_ho = np.ones(len(ho), bool)
        for f, o, th in conds: m_ho &= (ho[f].to_numpy() <= th) if o == "<=" else (ho[f].to_numpy() > th)
        hv = sgn * ho["f6x"].to_numpy()[m_ho] - COST
        out.append({"rule (conditions, all must hold)": " AND ".join(f"{f} {o} {th:.2f}" for f, o, th in conds), "trade": "long" if sgn > 0 else "SHORT (reverse)", "folds seen": len({f for f, _ in b["fold"]}), "dev bars (unseen days)": len(sel), "dev days": nd,
                    "dev net bps (30 min)": mean, "dev CI low": lo, "dev CI high": hi, "dev p": p, "holdout bars": len(hv), "holdout net bps": hv.mean() if len(hv) else np.nan})
    S = pd.DataFrame(out)
    if len(S):
        S["survives FDR 10%"] = bh_fdr(S["dev p"].to_numpy()); S["verdict"] = np.select([(S["dev CI low"] > 0) & S["survives FDR 10%"] & (S["holdout net bps"] > 0) & (S["holdout bars"] >= 200), S["dev CI low"] > 0], ["PASS (dev + FDR + holdout)", "PROMISING (dev only)"], "no edge"); S = S.sort_values("dev CI low", ascending=False)
    return S, len(bank)

NOTES = [["term", "meaning"],
 ["abnormal return", "stock's forward return minus the average of all stocks over the same bars of the same day: removes the market and time-of-day. Entry = next bar open"],
 ["net bps", "abnormal return in basis points (1 bp = 0.01%) after the round-trip cost (COST_BPS, default 10)"],
 ["control", "bars with the SAME price and volume behaviour but neutral delta. 'delta adds vs control' is the difference: if ~0 the order flow adds nothing beyond price/volume"],
 ["dz / vr / cdz", "delta z-score (vs that stock's own history), volume vs usual for that time of day, day's cumulative delta as share of volume"],
 ["verdicts", "PASS = dev CI above zero after cost AND survives FDR 10% AND positive on locked holdout (last 20% of sessions, >=100 events). PROMISING = dev only"],
 ["limits", "about 56 sessions, 09:15-15:14 only (last 15 min missing), stocks on one day move together (CIs resample whole days). Treat PASS as a lead to re-test as the archive grows"]]

def main():
    t0 = time.time(); M = load_minutes() if not SELFTEST else synth_minutes(bool(os.environ.get("PLANT"))); print(f"{len(M):,} minutes, {M['symbol'].nunique()} stocks, {M['date'].nunique()} sessions", flush=True)
    B = make_bars(M); B, cut = features(B); print(f"{len(B):,} five-minute bars; holdout from {cut} ({time.time()-t0:.0f}s)", flush=True)
    E = pd.concat([evaluate(B, definitions(B)), event_e8(B)], ignore_index=True); E["survives FDR 10%"] = bh_fdr(E["dev p"].to_numpy())
    E["verdict"] = np.select([(E["dev CI low"] > 0) & E["survives FDR 10%"] & (E["holdout net bps"] > 0) & (E["holdout events"] >= 100), E["dev CI low"] > 0, E["dev CI high"] < 0], ["PASS (dev + FDR + holdout)", "PROMISING (dev only)", "LOSES after costs"], "no edge")
    I = ic_scan(B); S, nb = scenarios(B)
    for n, d in (("events", E), ("ic", I), ("scenarios", S)): d.to_csv(f"{OUT}/ofl_{n}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
    print("\n== EVENTS ==\n", E[["event", "horizon", "dev events", "dev net bps", "dev CI low", "delta adds vs control (bps)", "holdout events", "holdout net bps", "verdict"]].round(2).to_string(index=False))
    print("\n== IC (non-zero) ==\n", I[I["verdict"] != "no signal"].round(4).to_string(index=False)); print("\n== SCENARIOS ==\n", S.head(8).round(2).to_string(index=False) if len(S) else "none recurred")
    if SELFTEST: return B, E, I, S
    import oi_probe as P; sh = P.open_sheet(); rnd = lambda d: d.round(4).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    P.put_tab(sh, "OFL Events", list(E.columns), rnd(E).values.tolist()); P.put_tab(sh, "OFL IC", list(I.columns), rnd(I).values.tolist())
    P.put_tab(sh, "OFL Scenarios", list(S.columns) if len(S) else ["scenario"], rnd(S).values.tolist() if len(S) else [["none recurred"]])
    P.put_tab(sh, "OFL Notes", NOTES[0], NOTES[1:] + [["bars", str(len(B))], ["stocks", str(B["symbol"].nunique())], ["sessions", f"{B['date'].min()} .. {B['date'].max()}"], ["holdout from", cut], ["cost bps", str(COST)], ["tree structures", str(nb)], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]]); print("sheet tabs written")

# ───────────────────────────── self-test: synthetic minutes ─────────────────────────────
def synth_minutes(plant, n_sym=40, n_days=40, seed=5):
    rng = np.random.default_rng(seed); days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-08-03", periods=n_days)]; hh = np.array([(9 * 60 + 15 + m) // 60 * 100 + (9 * 60 + 15 + m) % 60 for m in range(360)]); fr = []
    for k in range(n_sym):
        p = 500.0 + 10 * k; base_v = rng.lognormal(8, 0.5)
        for di, d in enumerate(days):
            n = 360; ret = rng.normal(0, 4e-4, n); vol = rng.lognormal(np.log(base_v), 0.3, n); share = np.full(n, 0.5) + 0.0; drift = np.zeros(n)
            for b in range(0, 66):
                if rng.random() < 0.05:                      # an absorption bar: heavy selling, volume surge, price holds
                    s = b * 5; ret[s: s + 5] = np.abs(ret[s: s + 5]) * 0.3; vol[s: s + 5] *= 3; share[s: s + 5] = 0.15
                    if plant: ret[s + 5: s + 35] += 2.5e-4
            share = np.clip(share + 12 * ret * (share == 0.5) + rng.normal(0, 0.05, n), 0.02, 0.98); c = p * np.cumprod(1 + ret); o = np.r_[p, c[:-1]]; p = c[-1]
            tot = vol; buy = tot * share; sell = tot - buy; dl = buy - sell
            fr.append(pd.DataFrame({"symbol": f"T{k:02d}", "date": d, "hhmm": hh, "o": o, "h": np.maximum(o, c) * 1.0002, "l": np.minimum(o, c) * 0.9998, "c": c, "tot": tot, "buy": buy, "sell": sell, "delta": dl, "cd": np.cumsum(dl),
                                    "vwap": (o + c) / 2, "bstack": rng.poisson(0.3, n), "sstack": rng.poisson(0.3, n), "d_o": o[0], "d_c": c[-1], "d_v": tot.sum()}))
    return pd.concat(fr, ignore_index=True)

if __name__ == "__main__":
    if SELFTEST:
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n" + "=" * 18, "SELFTEST", "PLANTED absorption effect" if plant else "NULL (events exist, no predictive effect)", "=" * 18)
            B, E, I, S = main(); r = E[E["event"].str.startswith("E1 ") & (E["horizon"] == "next 30 min")].iloc[0]
            print("E1 next 30 min:", r[["dev events", "dev net bps", "dev CI low", "holdout net bps", "verdict"]].to_dict())
            if plant: assert r["dev CI low"] > 0, "planted absorption effect not found"
            else: assert not E["verdict"].str.startswith("PASS").any() and not S.get("verdict", pd.Series(dtype=str)).astype(str).str.startswith("PASS").any(), "false PASS in null world"
        print("SELFTEST OK")
    else: main()
