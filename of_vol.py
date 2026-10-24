# ==================================================================================
# LIQUIDITY & VOLATILITY REGIME ENGINE (order flow, non-directional)
#  Question: when and where will a stock MOVE far enough to be worth trading (beyond costs), and does order flow tell us that
#  better than price and volume alone?   Direction is not predicted here - this engine says WHERE and WHEN there is movement.
#  Targets (5-minute bars, from the order-flow archive):
#    fr30  high-low range over the next 30 minutes (6 bars), in bps of price       fE  high-low range from the next bar to 15:14
#  Models (gradient boosting, validated by leaving whole DAYS out, then on a locked holdout = last 20% of sessions):
#    PRICE      : last-bar / last-15-min / day-so-far range, usual range for this stock and time, market-wide range, time of day, liquidity
#    PRICE+FLOW : the above + relative volume, |delta share|, |day delta share|, imbalance stacks, range-per-volume (price impact)
#  Reported: how much FLOW adds (rank IC and R2, with day-cluster confidence interval), predicted-decile table (realised range, share of
#  windows with range >= 4x cost), time-of-day map, stock league table (liquidity, impact, move-to-cost), feature importance.
#  Tabs: "OFV Models", "OFV Deciles", "OFV TimeOfDay", "OFV Stocks", "OFV Features", "OFV Notes".  Research only.
#  Usage: python of_vol.py [--selftest]      env: OF_ARCH, COST_BPS (10)
# ==================================================================================
import os, sys, time, warnings, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import of_lab as L

SELFTEST = "--selftest" in sys.argv; COST = L.COST; OUT = "out/of_vol"; os.makedirs(OUT, exist_ok=True); WORTH = 4 * COST
PRICE = ["rng1", "rng3", "day_rng", "exp_rng", "mkt_rng", "absret1", "bar", "lq"]
FLOW = ["vr", "vr3", "adn", "adz", "acdz", "astk", "imp"]

def prep(B):
    g = B.groupby(["symbol", "date"], sort=False); B["rng"] = (B["h"] / B["l"] - 1) * 1e4; B["rng1"] = B["rng"]
    B["rng3"] = g["rng"].transform(lambda s: s.rolling(3, min_periods=1).mean()); B["day_rng"] = (g["h"].cummax() / g["l"].cummin() - 1) * 1e4
    B["exp_rng"] = B.groupby(["symbol", "bar"])["rng"].transform(lambda s: s.shift(1).expanding(5).mean()); B["mkt_rng"] = B.groupby(["date", "bar"])["rng"].transform("mean")
    B["absret1"] = B["ret"].abs(); B["adn"] = B["dn"].abs(); B["adz"] = B["dz"].abs(); B["acdz"] = B["cdz"].abs(); B["astk"] = B["bstk"].abs(); g = B.groupby(["symbol", "date"], sort=False)
    B["vr3"] = g["vr"].transform(lambda s: s.rolling(3, min_periods=1).mean()); B["imp"] = B["rng"] / B["vr"].clip(lower=0.25)
    hs = np.vstack([g["h"].shift(-k).to_numpy() for k in range(1, 7)]); ls = np.vstack([g["l"].shift(-k).to_numpy() for k in range(1, 7)])
    B["fr30"] = (hs.max(axis=0) / ls.min(axis=0) - 1) * 1e4
    Br = B.iloc[::-1]; rg = Br.groupby(["symbol", "date"], sort=False); B["_rh"] = rg["h"].cummax(); B["_rl"] = rg["l"].cummin(); g = B.groupby(["symbol", "date"], sort=False)
    B["frE"] = (g["_rh"].shift(-1) / g["_rl"].shift(-1) - 1) * 1e4; B["y30"] = np.log1p(B["fr30"]); B["yE"] = np.log1p(B["frE"])
    return B.drop(columns=["_rh", "_rl"])

def spearman_by(df, a, b, key):
    ra = df.groupby(key)[a].rank(pct=True); rb = df.groupby(key)[b].rank(pct=True); return L.gcorr(ra, rb, [df[k] for k in key] if isinstance(key, list) else df[key])

def model_pass(R, target, feats, label):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    dev = R[~R["holdout"]].reset_index(drop=True); ho = R[R["holdout"]].reset_index(drop=True); oof = np.full(len(dev), np.nan); k = min(5, dev["date"].nunique())
    mk = lambda: HistGradientBoostingRegressor(max_depth=4, max_iter=120, learning_rate=0.08, random_state=0)
    for tr, te in GroupKFold(n_splits=k).split(dev, groups=dev["date"]): oof[te] = mk().fit(dev.loc[tr, feats], dev.loc[tr, target]).predict(dev.loc[te, feats])
    pho = mk().fit(dev[feats], dev[target]).predict(ho[feats]) if len(ho) else np.array([])
    return dev.assign(pred=oof), ho.assign(pred=pho)

def score(D, target):
    """overall rank IC, per-date cross-sectional rank IC (stock selection), R2 of the log target"""
    ic_all = D[["pred", target]].corr(method="spearman").iloc[0, 1]; xd = spearman_by(D, "pred", target, ["date", "bar"])
    r2 = 1 - ((D[target] - D["pred"]) ** 2).sum() / ((D[target] - D[target].mean()) ** 2).sum(); return ic_all, xd, r2

def run_models(B):
    rows = []; keep = {}; need = list(dict.fromkeys(PRICE + FLOW + ["y30", "yE"])); R0 = B[(B["bar"] >= 3) & (B["bar"] <= 65)].dropna(subset=need + ["fr30", "frE"])
    if len(R0) > 280000: R0 = R0.sample(280000, random_state=0)
    for tgt, ylab, fcol in (("y30", "next 30 min range", "fr30"), ("yE", "range to 15:14", "frE")):
        P_dev, P_ho = model_pass(R0, tgt, PRICE, "price"); F_dev, F_ho = model_pass(R0, tgt, PRICE + FLOW, "flow")
        for part, P, Fm in (("dev (days left out)", P_dev, F_dev), ("holdout", P_ho, F_ho)):
            if not len(P): continue
            ia, xa, r2a = score(P, tgt); ib, xb, r2b = score(Fm, tgt); dd = (xb - xa).dropna(); m, lo, hi, p, nd = L.cmean(dd.to_numpy(), dd.index.get_level_values(0).to_numpy())
            rows.append({"target": ylab, "sample": part, "rows": len(P), "days": P["date"].nunique(), "rank IC price-only": ia, "rank IC price+flow": ib, "R2 price-only": r2a, "R2 price+flow": r2b, "flow adds to R2": r2b - r2a,
                         "per-bar cross-section IC price-only": xa.mean(), "per-bar cross-section IC price+flow": xb.mean(), "flow adds to cross-section IC": m, "CI low": lo, "CI high": hi,
                         "verdict": "FLOW ADDS" if lo > 0 else ("flow does not add" if hi >= 0 else "flow hurts")})
        keep[fcol] = (F_dev, F_ho)
    return pd.DataFrame(rows), keep

def deciles(keep):
    rows = []
    for fcol, (Dv, Ho) in keep.items():
        for part, D in (("dev (days left out)", Dv), ("holdout", Ho)):
            if not len(D): continue
            D = D.copy(); D["dec"] = pd.qcut(D["pred"].rank(method="first"), 10, labels=False) + 1; base = D[fcol].mean(); wb = (D[fcol] >= WORTH).mean()
            for d, g in D.groupby("dec"): rows.append({"target": "next 30 min range" if fcol == "fr30" else "range to 15:14", "sample": part, "predicted decile (10 = most movement)": d, "windows": len(g), "realised range bps": g[fcol].mean(), "x average": g[fcol].mean() / base,
                                                           f"% windows with range >= {WORTH:.0f} bps (4x cost)": (g[fcol] >= WORTH).mean() * 100, "x base rate": (g[fcol] >= WORTH).mean() / wb if wb > 0 else np.nan})
    return pd.DataFrame(rows)

def time_of_day(B):
    X = B[(B["bar"] <= 65)].dropna(subset=["fr30"]); T = X.groupby("bar").agg(range_30m=("fr30", "mean"), share_worth=("fr30", lambda s: (s >= WORTH).mean() * 100), vol_rel=("vr", "mean"), abs_delta_share=("adn", "mean")).reset_index()
    T.insert(0, "time (bar start)", [f"{(555 + b * 5) // 60:02d}:{(555 + b * 5) % 60:02d}" for b in T["bar"]]); T = T.rename(columns={"range_30m": "avg next-30m range bps", "share_worth": f"% windows >= {WORTH:.0f} bps", "vol_rel": "avg relative volume", "abs_delta_share": "avg |delta share|"}); return T.drop(columns=["bar"])

def stock_table(B):
    X = B[(B["bar"] <= 65)].dropna(subset=["fr30"]); d = B.groupby(["symbol", "date"]).apply(lambda g: (g["v"] * g["c"]).sum()).groupby("symbol").mean() / 1e7
    S = X.groupby("symbol").agg(avg_5m_range=("rng", "mean"), avg_30m_range=("fr30", "mean"), share_worth=("fr30", lambda s: (s >= WORTH).mean() * 100), impact=("imp", "median"), flow_noise=("adn", "mean")).reset_index()
    S["avg daily value (Rs crore)"] = S["symbol"].map(d); S["move-to-cost (30m range / cost)"] = S["avg_30m_range"] / COST
    S = S.rename(columns={"avg_5m_range": "avg 5-min range bps", "avg_30m_range": "avg next-30m range bps", "share_worth": f"% 30m windows >= {WORTH:.0f} bps", "impact": "range per unit relative volume (median)", "flow_noise": "avg |delta share|"}); return S.sort_values("move-to-cost (30m range / cost)", ascending=False)

def importance(keep):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    rows = []
    for fcol, (Dv, Ho) in keep.items():
        tgt = "y30" if fcol == "fr30" else "yE"; feats = PRICE + FLOW; m = HistGradientBoostingRegressor(max_depth=4, max_iter=120, learning_rate=0.08, random_state=0).fit(Dv[feats], Dv[tgt])
        T = Ho if len(Ho) > 2000 else Dv; T = T.sample(min(20000, len(T)), random_state=0); pi = permutation_importance(m, T[feats], T[tgt], n_repeats=3, random_state=0, n_jobs=1)
        for f, v in zip(feats, pi.importances_mean): rows.append({"target": "next 30 min range" if fcol == "fr30" else "range to 15:14", "feature": f, "group": "FLOW" if f in FLOW else "price", "importance (R2 lost when scrambled)": v})
    return pd.DataFrame(rows).sort_values(["target", "importance (R2 lost when scrambled)"], ascending=[True, False])

NOTES = [["term", "meaning"], ["range", "high minus low over the window as bps of price (1 bp = 0.01%). A window is 'worth trading' when range >= 4x the round-trip cost"],
 ["price-only vs +flow", "same model, flow features withheld or included. 'flow adds' = improvement in rank IC / R2, measured on days the model never saw, CI from resampling whole days"],
 ["cross-section IC", "within each 5-minute bar, how well the model ranks stocks by how much they will move - the stock-selection ability"],
 ["deciles", "bars sorted by predicted movement: if the engine works the top decile has several times the realised range of the bottom decile"],
 ["use", "this engine is a FILTER and a timing map (where/when to expect movement; which stocks have cheap impact). It does not say which direction"],
 ["limits", "about 56 sessions, 09:15-15:14 only; range uses 5-minute bars"]]

def main():
    t0 = time.time(); M = L.load_minutes() if not SELFTEST else synth_vol(bool(os.environ.get("PLANT"))); B = L.make_bars(M); B, cut = L.features(B); B = prep(B); print(f"{len(B):,} bars, holdout from {cut} ({time.time()-t0:.0f}s)", flush=True)
    Mo, keep = run_models(B); De = deciles(keep); Tm = time_of_day(B); St = stock_table(B); Fi = importance(keep)
    for n, d in (("models", Mo), ("deciles", De), ("timeofday", Tm), ("stocks", St), ("features", Fi)): d.to_csv(f"{OUT}/ofv_{n}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n== MODELS ==\n", Mo.round(4).T.to_string()); print("\n== DECILES (holdout, next 30 min) ==\n", De[(De["sample"] == "holdout") & (De["target"] == "next 30 min range")].round(2).to_string(index=False)); print("\n== FEATURES ==\n", Fi.head(10).round(4).to_string(index=False))
    if SELFTEST: return Mo, De
    import oi_probe as P; sh = P.open_sheet(); rnd = lambda d: d.round(4).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    for n, d in (("OFV Models", Mo), ("OFV Deciles", De), ("OFV TimeOfDay", Tm), ("OFV Stocks", St), ("OFV Features", Fi)): P.put_tab(sh, n, list(d.columns), rnd(d).values.tolist())
    P.put_tab(sh, "OFV Notes", NOTES[0], NOTES[1:] + [["bars", str(len(B))], ["stocks", str(B["symbol"].nunique())], ["holdout from", cut], ["cost bps", str(COST)], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]]); print("sheet tabs written")

def synth_vol(plant, n_sym=25, n_days=30, seed=7):
    rng = np.random.default_rng(seed); days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-08-03", periods=n_days)]; hh = np.array([(555 + m) // 60 * 100 + (555 + m) % 60 for m in range(360)]); fr = []
    for k in range(n_sym):
        p = 400.0 + 20 * k; bv = rng.lognormal(8, 0.4); z = 0.0
        for d in days:
            n = 360; zs = np.empty(n); sig = np.empty(n)
            for i in range(n): z = 0.995 * z + 0.1 * rng.normal(); zs[i] = z
            sig = 3e-4 * np.exp(0.7 * zs); ret = sig * rng.normal(size=n); vol = bv * np.exp((0.6 * zs if plant else 0) + 0.35 * rng.normal(size=n)); mag = (0.04 + 0.25 * np.clip(np.exp(0.6 * zs) - 0.8, 0, 2)) if plant else np.full(n, 0.12)
            share = np.clip(0.5 + mag * rng.choice([-1, 1], n) * rng.uniform(0.2, 1, n) + 8 * ret, 0.02, 0.98); c = p * np.cumprod(1 + ret); o = np.r_[p, c[:-1]]; p = c[-1]; buy = vol * share; sell = vol - buy
            fr.append(pd.DataFrame({"symbol": f"T{k:02d}", "date": d, "hhmm": hh, "o": o, "h": np.maximum(o, c) * 1.0002, "l": np.minimum(o, c) * 0.9998, "c": c, "tot": vol, "buy": buy, "sell": sell, "delta": buy - sell, "cd": np.cumsum(buy - sell),
                                    "vwap": (o + c) / 2, "bstack": rng.poisson(0.3, n), "sstack": rng.poisson(0.3, n), "d_o": o[0], "d_c": c[-1], "d_v": vol.sum()}))
    return pd.concat(fr, ignore_index=True)

if __name__ == "__main__":
    if SELFTEST:
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n" + "=" * 18, "SELFTEST", "PLANTED: flow reveals hidden volatility" if plant else "NULL: flow unrelated to volatility", "=" * 18)
            Mo, De = main(); r = Mo[(Mo["target"] == "next 30 min range") & (Mo["sample"].str.startswith("dev"))].iloc[0]; print("flow adds (30m, dev):", r[["flow adds to cross-section IC", "CI low", "CI high", "flow adds to R2", "verdict"]].to_dict())
            if plant: assert r["verdict"] == "FLOW ADDS" and r["flow adds to R2"] > 0.01, "planted flow information not detected"
            else: assert r["verdict"] != "FLOW ADDS" or r["flow adds to R2"] < 0.01, "false positive on null"
        print("SELFTEST OK")
    else: main()
