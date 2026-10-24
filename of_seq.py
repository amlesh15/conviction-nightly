# ==================================================================================
# SEQUENCE ENGINE  ("how did the fight go, session after session")  -  research only, no orders
#  1. FIGHT CARD per stock-day from the order-flow archive: who won (day delta share, early vs late), where did the close land,
#     were buyers/sellers TRAPPED (buying done above the final price / selling done below it), effort vs result, break of the 10-day range.
#  2. NAMED PATTERNS (your examples, both directions, signed so "up" and "down" are mirror images):
#     P1 trend -> counter-attempt fails -> trend re-breaks      P2 coil/sideways -> FIRST break
#     P3 break -> pause -> second leg (ride)                    P4 opponents trapped on 2 of last 3 days -> push
#     P5 break against the trend that flow ABSORBS (spring / upthrust) -> reversal
#     each shown 3 ways: flow CONFIRMS / price shape only (control) / flow CONTRADICTS.  If flow adds, 'confirms' beats 'price only' beats 'contradicts'.
#  3. DISCOVERY: gradient-boosting on the last-5-session sequence (price-only vs price+flow), day-left-out validation, locked holdout (last 20% of sessions),
#     long-top-decile / short-bottom-decile spread, plus a small readable decision tree whose leaves are checked on later days.
#  Trade = enter NEXT open after the signal close, hold 1 / 3 / 5 sessions, abnormal return (minus same-day average of all stocks), cost 10 bps round trip.
#  Tabs: "OFS Patterns", "OFS Discovery", "OFS Rules", "OFS Fight Today", "OFS Notes".     Usage: python of_seq.py [--selftest]     env: OF_ARCH, COST_BPS
# ==================================================================================
import os, sys, time, warnings, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import of_lab as L

SELFTEST = "--selftest" in sys.argv; COST = L.COST; OUT = "out/of_seq"; os.makedirs(OUT, exist_ok=True); HZ = (1, 3, 5)

# ────────────────────────── 1. fight card ──────────────────────────
def fight_cards(M):
    M = M.copy(); M["date"] = M["date"].astype(str); g = M.groupby(["symbol", "date"], sort=False); M["clf"] = g["c"].transform("last")
    M["dpos"] = M["delta"].clip(lower=0); M["dneg"] = (-M["delta"]).clip(lower=0); M["tb"] = np.where(M["c"] > M["clf"], M["dpos"], 0.0); M["ts"] = np.where(M["c"] < M["clf"], M["dneg"], 0.0)
    M["ed"] = np.where(M["hhmm"] < 1030, M["delta"], 0.0); M["ev"] = np.where(M["hhmm"] < 1030, M["tot"], 0.0); M["ld"] = np.where(M["hhmm"] >= 1345, M["delta"], 0.0); M["lv"] = np.where(M["hhmm"] >= 1345, M["tot"], 0.0); M["vw"] = M["vwap"] * M["tot"]
    P = M.groupby(["symbol", "date"], sort=True).agg(hi=("h", "max"), lo=("l", "min"), op=("d_o", "first"), cl=("d_c", "first"), v=("tot", "sum"), d=("delta", "sum"), tb=("tb", "sum"), ts=("ts", "sum"), ed=("ed", "sum"), ev=("ev", "sum"), ld=("ld", "sum"), lv=("lv", "sum"), vw=("vw", "sum")).reset_index()
    P = P[(P["v"] > 0) & (P["hi"] > P["lo"])].sort_values(["symbol", "date"]).reset_index(drop=True); g = P.groupby("symbol", sort=False)
    P["dds"] = P["d"] / P["v"]; P["early"] = P["ed"] / P["ev"].replace(0, np.nan); P["late"] = P["ld"] / P["lv"].replace(0, np.nan); P["trapnet"] = (P["ts"] - P["tb"]) / P["v"]; P["cvw"] = (P["cl"] / (P["vw"] / P["v"]) - 1) * 100
    P["pc"] = g["cl"].shift(1); P["ret"] = (P["cl"] / P["pc"] - 1) * 100; P["rng"] = (P["hi"] - P["lo"]) / P["pc"] * 100; P["loc"] = ((P["cl"] - P["lo"]) / (P["hi"] - P["lo"])).clip(0, 1)
    P["h10"] = g["hi"].transform(lambda s: s.shift(1).rolling(10, min_periods=7).max()); P["l10"] = g["lo"].transform(lambda s: s.shift(1).rolling(10, min_periods=7).min())
    P["brk_up"] = (P["cl"] > P["h10"]).astype(float).where(P["h10"].notna()); P["brk_dn"] = (P["cl"] < P["l10"]).astype(float).where(P["l10"].notna()); P["pos"] = (P["cl"] - P["l10"]) / (P["h10"] - P["l10"])
    P["t10"] = (P["pc"] / g["cl"].shift(11) - 1) * 100; P["comp"] = g["rng"].transform(lambda s: s.rolling(5).mean()) / g["rng"].transform(lambda s: s.shift(5).rolling(5).mean()); P["relv"] = P["v"] / g["v"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    P["effort"] = P["relv"] / (P["rng"] / g["rng"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())).clip(lower=0.2)          # high volume, little range = effort without result
    P["div"] = P.groupby("date")["dds"].rank(pct=True) - P.groupby("date")["ret"].rank(pct=True)                                                   # + = buying heavier than price shows (absorption)
    P["tn_r"] = P.groupby("date")["trapnet"].rank(pct=True) - 0.5
    for h in HZ: P[f"r{h}"] = (g["cl"].shift(-h) / g["op"].shift(-1) - 1) * 1e4; P[f"r{h}x"] = P[f"r{h}"] - P.groupby("date")[f"r{h}"].transform("mean")
    dates = sorted(P["date"].unique()); cut = dates[int(len(dates) * 0.8)]; P["holdout"] = P["date"] >= cut; return P, cut

def story(r):
    """one-line plain-language read of a day"""
    s = []; s.append("closed UP" if r["ret"] > 0.3 else "closed DOWN" if r["ret"] < -0.3 else "flat")
    s.append("buyers won the flow" if r["dds"] > 0.06 else "sellers won the flow" if r["dds"] < -0.06 else "flow balanced")
    if r["dds"] > 0.06 and r["ret"] < 0: s.append("ABSORPTION: heavy buying but price fell (buyers may be trapped)")
    if r["dds"] < -0.06 and r["ret"] > 0: s.append("ABSORPTION: heavy selling but price rose (sellers may be trapped)")
    if r["trapnet"] > 0.12: s.append("sellers trapped below close")
    if r["trapnet"] < -0.12: s.append("buyers trapped above close")
    if r["brk_up"] == 1: s.append("closed ABOVE 10-day high")
    if r["brk_dn"] == 1: s.append("closed BELOW 10-day low")
    if r["effort"] > 2: s.append("big effort, little result")
    return "; ".join(s)

# ────────────────────────── 2. named patterns ──────────────────────────
def signed_view(P, s):
    g = P.groupby("symbol", sort=False); V = pd.DataFrame({"symbol": P["symbol"], "date": P["date"], "holdout": P["holdout"]})
    V["ret_s"] = s * P["ret"]; V["dds_s"] = s * P["dds"]; V["loc_s"] = P["loc"] if s > 0 else 1 - P["loc"]; V["tn_s"] = s * P["tn_r"]; V["t10_s"] = s * P["t10"]
    V["brk_s"] = (P["brk_up"] if s > 0 else P["brk_dn"]); V["brk_o"] = (P["brk_dn"] if s > 0 else P["brk_up"]); V["comp"] = P["comp"]; V["t10"] = P["t10"]; V["cls"] = s * P["cl"]
    gv = V.groupby("symbol", sort=False)
    counter = gv["ret_s"].transform(lambda x: x.shift(1).rolling(7, min_periods=5).min()) <= -1.0
    first_brk = gv["brk_s"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).max()) == 0
    recent_brk = gv["brk_s"].transform(lambda x: x.shift(2).rolling(4, min_periods=3).max()) == 1
    beyond = V["cls"] > gv["cls"].transform(lambda x: x.shift(1).rolling(4, min_periods=3).max())
    trap2 = gv["tn_s"].transform(lambda x: (x > 0.3).astype(float).rolling(3, min_periods=3).sum().shift(0)) >= 2
    conf = (V["dds_s"] > 0.05) & (V["loc_s"] > 0.55); bad = (V["dds_s"] < -0.05)
    pat = {"P1 trend, failed counter-attempt, re-break": ((V["t10_s"] > 3) & counter & (V["brk_s"] == 1), conf, bad),
           "P2 coil/sideways then FIRST break": ((V["comp"] < 0.85) & (V["t10"].abs() < 4) & (V["brk_s"] == 1) & first_brk, conf, bad),
           "P3 break, pause, second leg": (recent_brk & beyond & (V["ret_s"] > 0), conf, bad),
           "P4 opponents trapped 2 of last 3 days, push": ((V["ret_s"] > 0) & beyond, trap2, ~trap2 & (V["tn_s"] < -0.1)),
           "P5 break against trend absorbed by flow (reversal)": ((V["brk_o"] == 1) & (V["t10_s"] < -2), conf, bad)}
    return V, pat

def pattern_table(P):
    rows = []
    for s, sl in ((1, "UP / long"), (-1, "DOWN / short"), (0, "both")):
        parts = [(1, signed_view(P, 1)), (-1, signed_view(P, -1))] if s == 0 else [(s, signed_view(P, s))]
        for name in signed_view(P, 1)[1]:
            for var, k in (("flow CONFIRMS", 1), ("price shape only (control)", 0), ("flow CONTRADICTS", 2)):
                for h in HZ:
                    vals = []; dts = []; ho = []
                    for sg, (V, pat) in parts:
                        pm, cf, bd = pat[name]; m = pm & cf if k == 1 else pm if k == 0 else pm & bd; y = sg * P[f"r{h}x"]; m = m.fillna(False) & y.notna()
                        vals.append(y[m].to_numpy()); dts.append(P["date"][m].to_numpy()); ho.append(P["holdout"][m].to_numpy())
                    v = np.concatenate(vals); d = np.concatenate(dts); hh = np.concatenate(ho)
                    for part, sel in (("dev", ~hh), ("holdout", hh)):
                        if sel.sum() < 15: continue
                        m_, lo, hi, p, nd = L.cmean(v[sel], d[sel])
                        rows.append({"pattern": name, "direction": sl, "flow": var, "hold (sessions)": h, "sample": part, "signals": int(sel.sum()), "days": nd, "gross bps": m_, "net bps (after cost)": m_ - COST, "net CI low": lo - COST, "net CI high": hi - COST, "hit %": (v[sel] > 0).mean() * 100, "p (gross>0)": p})
    R = pd.DataFrame(rows)
    if len(R):
        R["read"] = np.where(R["net CI low"] > 0, "NET POSITIVE", np.where(R["net CI high"] < 0, "net negative", "not distinguishable from 0"))
        dv = R[(R["sample"] == "dev") & (R["flow"] == "flow CONFIRMS")]; q = bh(dv["p (gross>0)"].to_numpy()); R["FDR pass (dev, confirms)"] = ""; R.loc[dv.index, "FDR pass (dev, confirms)"] = np.where(q <= 0.10, "yes", "no")
    return R

def bh(p, q=0.10):
    p = np.where(np.isnan(p), 1.0, p); n = len(p); o = np.argsort(p); adj = np.empty(n); mn = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]): mn = min(mn, p[i] * n / rank); adj[i] = mn
    return adj

# ────────────────────────── 3. discovery ──────────────────────────
PRICE_F = ["ret", "loc", "rng", "relv", "effort", "brk_up", "brk_dn", "pos"]; FLOW_F = ["dds", "early", "late", "trapnet", "div", "cvw"]; CTX = ["t10", "comp", "pos"]
def seq_frame(P, K=5):
    g = P.groupby("symbol", sort=False); cols = {}
    for f in dict.fromkeys(PRICE_F + FLOW_F):
        for k in range(K): cols[f"{f}_l{k}"] = g[f].shift(k) if k else P[f]
    X = pd.concat([P[["symbol", "date", "holdout"] + CTX + [f"r{h}x" for h in HZ]], pd.DataFrame(cols)], axis=1)
    pf = [f"{f}_l{k}" for f in PRICE_F for k in range(K)] + CTX; ff = [f"{f}_l{k}" for f in FLOW_F for k in range(K)]; return X, pf, ff

def discovery(P):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X, pf, ff = seq_frame(P); rows = []; keep = {}
    for h in (3, 5):
        y = f"r{h}x"; D = X.dropna(subset=[y]).reset_index(drop=True); D[y] = D[y].clip(D[y].quantile(0.01), D[y].quantile(0.99)); dev = D[~D["holdout"]].reset_index(drop=True); ho = D[D["holdout"]].reset_index(drop=True)
        out = {}
        for lab, feats in (("price-only", pf), ("price+flow", pf + ff)):
            oof = np.full(len(dev), np.nan); k = min(5, dev["date"].nunique()); mk = lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=80, learning_rate=0.06, min_samples_leaf=80, l2_regularization=2.0, random_state=0)
            for tr, te in GroupKFold(n_splits=k).split(dev, groups=dev["date"]): oof[te] = mk().fit(dev.loc[tr, feats], dev.loc[tr, y]).predict(dev.loc[te, feats])
            pho = mk().fit(dev[feats], dev[y]).predict(ho[feats]) if len(ho) else np.array([]); out[lab] = (oof, pho)
        for part, DD, idx in (("dev (days left out)", dev, 0), ("holdout", ho, 1)):
            if not len(DD): continue
            res = {}
            for lab in out:
                T = DD[["date", y]].copy(); T["pred"] = out[lab][idx]; T["rp"] = T.groupby("date")["pred"].rank(pct=True) - 0.5; T["ry"] = T.groupby("date")[y].rank(pct=True) - 0.5; ic = L.gcorr(T["rp"], T["ry"], T["date"])
                T["dec"] = pd.qcut(T["pred"].rank(method="first"), 10, labels=False); top = T[T["dec"] == 9]; bot = T[T["dec"] == 0]; res[lab] = (ic, top, bot)
            ica, ta, ba = res["price-only"]; icb, tb_, bb = res["price+flow"]; d = (icb - ica).dropna(); m, lo, hi, n = cm(d.to_numpy()); mb, lob, hib, _ = cm(icb.to_numpy()); ma, loa, hia, _ = cm(ica.to_numpy())
            for lab, (ic, top, bot) in res.items():
                lsp = (top[y].mean() - bot[y].mean()) / 2 - COST; mi, l_, h_, nn = cm(ic.to_numpy())
                rows.append({"hold (sessions)": h, "sample": part, "model": lab, "rows": len(DD), "days": DD["date"].nunique(), "mean cross-section rank IC": mi, "IC CI low": l_, "IC CI high": h_, "top decile abnormal bps": top[y].mean(), "bottom decile abnormal bps": bot[y].mean(),
                             "long-top/short-bottom net bps per leg": lsp, "flow adds to IC": (m if lab == "price+flow" else np.nan), "flow-add CI low": (lo if lab == "price+flow" else np.nan), "flow-add CI high": (hi if lab == "price+flow" else np.nan),
                             "verdict": ("FLOW ADDS" if lo > 0 else "flow does not add" if hi >= 0 else "flow hurts") if lab == "price+flow" else ""})
        keep[h] = (dev, pf, ff)
    return pd.DataFrame(rows), X, pf, ff

def cm(v, B_=2000, seed=5):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 6: return np.nan, np.nan, np.nan, len(v)
    r = np.random.default_rng(seed); bs = v[r.integers(0, len(v), (B_, len(v)))].mean(axis=1); return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(v)

def readable_rules(X, pf, ff, h=3):
    from sklearn.tree import DecisionTreeRegressor
    y = f"r{h}x"; feats = [f for f in pf + ff if f.endswith("_l0") or f.endswith("_l1") or f in CTX]; D = X.dropna(subset=[y] + feats).reset_index(drop=True); dev = D[~D["holdout"]]; ho = D[D["holdout"]]
    if len(dev) < 2000: return pd.DataFrame()
    dates = sorted(dev["date"].unique()); mid = dates[len(dates) // 2]; t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(200, len(dev) // 60), random_state=0).fit(dev[feats], dev[y]); tr = t.tree_
    leaf_dev = t.apply(dev[feats]); leaf_ho = t.apply(ho[feats]) if len(ho) else np.array([]); rows = []
    def walk(node, conds):
        if tr.children_left[node] == -1:
            m = leaf_dev == node; e = dev[y].to_numpy()[m]; a = dev["date"].to_numpy()[m]; mh = leaf_ho == node
            rows.append({"rule (all must hold)": " AND ".join(conds) or "(all)", "dev signals": int(m.sum()), "dev mean abnormal bps": e.mean(), "dev first half": e[a < mid].mean() if (a < mid).any() else np.nan, "dev second half": e[a >= mid].mean() if (a >= mid).any() else np.nan,
                         "holdout signals": int(mh.sum()), "holdout mean abnormal bps": ho[y].to_numpy()[mh].mean() if mh.sum() else np.nan}); return
        f = feats[tr.feature[node]]; th = tr.threshold[node]; walk(tr.children_left[node], conds + [f"{f} <= {th:.3g}"]); walk(tr.children_right[node], conds + [f"{f} > {th:.3g}"])
    walk(0, []); R = pd.DataFrame(rows); R["hold (sessions)"] = h
    R["stable?"] = np.where((np.sign(R["dev first half"]) == np.sign(R["dev second half"])) & (np.sign(R["dev mean abnormal bps"]) == np.sign(R["holdout mean abnormal bps"])) & (R["dev mean abnormal bps"].abs() > COST), "same sign in both halves and holdout, bigger than cost", "not stable / below cost")
    return R.sort_values("dev mean abnormal bps")

NOTES = [["term", "meaning"], ["fight card", "per stock-day: day delta share (buyers vs sellers), early/late flow, close location, trapped buyers/sellers, effort vs result, 10-day range break"],
 ["trapnet", "(selling done BELOW the closing price - buying done ABOVE it) / volume. Positive = sellers stuck underwater (bullish squeeze fuel), negative = buyers stuck (bearish)"],
 ["signed patterns", "UP and DOWN are mirror images of the same rule, so 'flow CONFIRMS' always means flow in the direction of the trade"],
 ["3 versions", "CONFIRMS vs price-shape-only vs CONTRADICTS. Flow only earns its place if CONFIRMS > price-only > CONTRADICTS on dev AND holdout"],
 ["trade", "enter next open after the signal close, exit close of day 1/3/5, abnormal vs all stocks that day, minus 10 bps"],
 ["FDR pass", "Benjamini-Hochberg 10% across all confirm-patterns on dev - guards against picking the lucky one out of many"],
 ["discovery", "model sees the last 5 sessions; 'flow adds' = IC gain over the price-only model, on days it never saw, CI from resampling days"],
 ["limits", "~55 sessions of order flow -> few independent patterns are testable; treat a single lucky cell as noise, trust only repeats across dev, holdout and 3/5-day holds"]]

def main():
    t0 = time.time(); M = L.load_minutes() if not SELFTEST else synth_seq(bool(os.environ.get("PLANT"))); P, cut = fight_cards(M); print(f"{len(P):,} stock-days, {P['symbol'].nunique()} stocks, holdout from {cut} ({time.time()-t0:.0f}s)", flush=True)
    Pt = pattern_table(P); Di, X, pf, ff = discovery(P); Ru = pd.concat([readable_rules(X, pf, ff, 3), readable_rules(X, pf, ff, 5)], ignore_index=True) if not SELFTEST else readable_rules(X, pf, ff, 3)
    last = P["date"].max(); Td = P[P["date"] == last].copy(); Td["story"] = Td.apply(story, axis=1); Td = Td[["symbol", "date", "ret", "dds", "early", "late", "trapnet", "loc", "relv", "effort", "t10", "pos", "brk_up", "brk_dn", "story"]].sort_values("trapnet", key=abs, ascending=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90)
    for n, d in (("patterns", Pt), ("discovery", Di), ("rules", Ru), ("fight_today", Td)): d.to_csv(f"{OUT}/ofs_{n}.csv", index=False)
    if len(Pt): print("\n== PATTERNS (net, flow CONFIRMS, 3-day hold) ==\n", Pt[(Pt["flow"] == "flow CONFIRMS") & (Pt["hold (sessions)"] == 3)].drop(columns=["p (gross>0)"]).round(2).to_string(index=False))
    print("\n== DISCOVERY ==\n", Di.round(4).T.to_string());
    if len(Ru): print("\n== RULES ==\n", Ru.round(2).to_string(index=False))
    if SELFTEST: return Pt, Di, Ru
    import oi_probe as P_; sh = P_.open_sheet(); rnd = lambda d: d.round(4).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    for n, d in (("OFS Patterns", Pt), ("OFS Discovery", Di), ("OFS Rules", Ru), ("OFS Fight Today", Td)):
        if len(d): P_.put_tab(sh, n, list(d.columns), rnd(d).values.tolist())
    P_.put_tab(sh, "OFS Notes", NOTES[0], NOTES[1:] + [["stock-days", str(len(P))], ["holdout from", cut], ["cost bps", str(COST)], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]]); print("sheet tabs written")

# ────────────────────────── self-test data ──────────────────────────
def synth_seq(plant, n_sym=60, n_days=50, seed=21):
    """planted: when buyers are absorbed-then-trapped-sellers pattern appears for 2 days (sell-heavy flow while price holds), the next 3 days drift up"""
    rng = np.random.default_rng(seed); days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-08-03", periods=n_days)]; hh = np.array([(555 + m) // 60 * 100 + (555 + m) % 60 for m in range(360)]); fr = []
    for k in range(n_sym):
        p = 500.0; sym = f"T{k:02d}"; z = np.zeros(n_days); flow = rng.normal(size=n_days)
        for di, d in enumerate(days):
            f1 = flow[di - 1] if di else 0; f2 = flow[di - 2] if di > 1 else 0; sig = (f1 < -0.5) & (f2 < -0.5); drift = (150e-4 / 360 if (plant and sig) else 0.0)
            ret = 3e-4 * rng.normal(size=360) + drift; c = p * np.cumprod(1 + ret); o = np.r_[p, c[:-1]]; p = c[-1]; vol = rng.lognormal(7, 0.3, 360)
            share = np.clip(0.5 + 0.04 * np.tanh(flow[di]) + 0.01 * rng.normal(size=360), 0.05, 0.95); buy = vol * share; sell = vol - buy
            fr.append(pd.DataFrame({"symbol": sym, "date": d, "hhmm": hh, "o": o, "h": np.maximum(o, c) * 1.0002, "l": np.minimum(o, c) * 0.9998, "c": c, "tot": vol, "buy": buy, "sell": sell, "delta": buy - sell, "cd": np.cumsum(buy - sell), "vwap": (o + c) / 2,
                                    "bstack": 0, "sstack": 0, "d_o": o[0], "d_c": c[-1], "d_v": vol.sum()}))
    return pd.concat(fr, ignore_index=True)

if __name__ == "__main__":
    if SELFTEST:
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n" + "=" * 18, "SELFTEST", "PLANTED: two sell-heavy days that price shrugs off -> up" if plant else "NULL", "=" * 18)
            Pt, Di, Ru = main(); r = Di[(Di["hold (sessions)"] == 3) & (Di["sample"].str.startswith("dev")) & (Di["model"] == "price+flow")].iloc[0]; print("flow adds (3d, dev):", r[["flow adds to IC", "flow-add CI low", "flow-add CI high", "verdict"]].to_dict())
            if plant: assert r["verdict"] == "FLOW ADDS", "planted sequence not detected"
            else: assert r["verdict"] != "FLOW ADDS", "false positive on null"
        print("SELFTEST OK")
    else: main()
