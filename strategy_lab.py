# ==================================================================================
# STRATEGY LAB — build and honestly test strategies on the CONVICTION scores
#
#  Inputs (all point-in-time, built from the same masterdata as the Conviction Engine, history from 2023-08):
#    rule conviction score + its 8 components + score-pattern notes, volume / delivery, futures OI build-ups,
#    PCR, price trend, daily imbalance (FVG) support/resistance distance, market breadth.
#  Rules of the game
#    * signal known at the CLOSE of day t  ->  entry = OPEN of day t+1  ->  exit after h days (3 / 5 / 10)
#    * costs charged (COST_BPS round trip, default 10 bps)
#    * returns measured against the average stock entered the same day ("excess"), so a rising market earns nothing
#    * the LAST 120 trading days are a HOLDOUT: never used to choose anything; strategies are judged on the earlier
#      "development" period first, and only then looked at on the holdout
#    * with dozens of strategies tried, some win by luck: a false-discovery-rate (Benjamini-Hochberg) filter is applied
#  Three layers
#    1 LIBRARY    ~100 explicit strategies (score levels / shapes, components, volume, OI, imbalance, patterns, combos)
#    2 DISCOVERY  walk-forward machine-learning on every feature: which FAMILIES of information add predictive power,
#                 recurring decision-tree "scenarios", and a model-ranked long/short book
#    3 PORTFOLIO  daily equity curve of every strategy that passes
#  Tabs: "SL Strategies", "SL Portfolio", "SL Families", "SL Scenarios", "SL Today", "SL Notes"
#  Usage: python strategy_lab.py [--selftest]       env: COST_BPS (10), START_DATE (2023-08-01), HOLDOUT_DAYS (120)
# ==================================================================================
import os, sys, time, warnings, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

OUT = "out/strategy"; os.makedirs(OUT, exist_ok=True)
COST = float(os.environ.get("COST_BPS", "10")) / 100.0            # round-trip cost in %
HOLDOUT = int(os.environ.get("HOLDOUT_DAYS", "120"))
HZ = (3, 5, 10); EMBARGO = 10
SELFTEST = "--selftest" in sys.argv

# ───────────────────────────── 1. panel ─────────────────────────────
def real_panel():
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    import run_nightly as RN
    G = RN.load_engine(); RN.seed_cache()
    today = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")
    print(f"building masterdata {RN.START} -> {today}", flush=True)
    md = G["build_masterdata_pit_fast"](RN.START, today, full=False)
    feat = G["build_features"](md); cv = G["cv_score_all"](feat)
    P = feat.merge(cv[["date", "symbol", "score", "patterns", "comps"]], on=["date", "symbol"], how="inner")
    for k in ("BM", "TA", "OR", "BT", "DC", "PS", "TG", "VW"):
        P["c_" + k] = [(c or {}).get(k, np.nan) if isinstance(c, dict) else np.nan for c in P["comps"]]
    return P.drop(columns=["comps"])

def fvg_levels(o, h, l, c):
    """Point-in-time unfilled daily FVG support/resistance (same definition as fvg_lab.py, evaluated at every close)."""
    n = len(c); sup = np.full(n, -np.inf); res = np.full(n, np.inf); inb = np.zeros(n, bool); ine = np.zeros(n, bool)
    for i in range(1, n - 1):
        for typ, bot, top in (("bull", h[i - 1], l[i + 1]) if h[i - 1] < l[i + 1] else (None, 0, 0), ("bear", h[i + 1], l[i - 1]) if l[i - 1] > h[i + 1] else (None, 0, 0)):
            if typ is None: continue
            t0 = i + 1; fill = n
            seg = c[t0 + 1:]
            if len(seg):
                bad = np.where(seg < bot)[0] if typ == "bull" else np.where(seg > top)[0]
                if len(bad): fill = t0 + 1 + bad[0]
            tt = slice(t0, fill); cc = c[tt]
            if typ == "bull":
                sup[tt] = np.maximum(sup[tt], np.where(top < cc, top, -np.inf)); inb[tt] |= (cc >= bot) & (cc <= top)
            else:
                res[tt] = np.minimum(res[tt], np.where(bot > cc, bot, np.inf)); ine[tt] |= (cc >= bot) & (cc <= top)
    sup[~np.isfinite(sup)] = np.nan; res[~np.isfinite(res)] = np.nan
    return sup, res, inb, ine

def derive(P):
    P = P.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    P["date"] = P["date"].astype(str); P["pos"] = P.groupby("symbol").cumcount()
    out = []
    for sym, g in P.groupby("symbol", sort=False):
        g = g.copy(); o, h, l, c = (g[x].to_numpy(float) for x in ("open", "high", "low", "close")); s = g["score"].to_numpy(float)
        sh = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]] if k < len(a) else np.full(len(a), np.nan)
        for k in (1, 5, 10, 20): g[f"ret{k}"] = (c / sh(c, k) - 1) * 100
        tr = np.maximum.reduce([h - l, np.abs(h - sh(c, 1)), np.abs(l - sh(c, 1))]); g["atr_pct"] = pd.Series(tr).rolling(14, min_periods=5).mean().to_numpy() / c * 100
        g["dist_ema20"] = (c / g["ema20"].to_numpy(float) - 1) * 100; g["ema_spread"] = (g["ema20"].to_numpy(float) / g["ema50"].to_numpy(float) - 1) * 100
        g["gap"] = (o / sh(c, 1) - 1) * 100
        mn = pd.Series(c).rolling(20, min_periods=10).min().to_numpy(); mx = pd.Series(c).rolling(20, min_periods=10).max().to_numpy(); g["rng20pos"] = (c - mn) / np.where(mx > mn, mx - mn, np.nan)
        s1, s2, s3, s4 = sh(s, 1), sh(s, 2), sh(s, 3), sh(s, 4)
        g["s1"], g["d1"], g["d3"] = s1, s - s1, s - s3; g["slope5"] = (2 * s + s1 - s3 - 2 * s4) / 10; g["accel"] = (s - s1) - (s1 - s2)
        g["max5"] = pd.Series(s).rolling(5, min_periods=4).max().to_numpy(); g["min5"] = pd.Series(s).rolling(5, min_periods=4).min().to_numpy()
        ru = np.zeros(len(s)); rd = np.zeros(len(s))
        for i in range(1, len(s)):
            if s[i] > s[i - 1]: ru[i] = ru[i - 1] + 1
            if s[i] < s[i - 1]: rd[i] = rd[i - 1] + 1
        g["run_up"], g["run_down"] = ru, rd
        g["cross0"] = ((s > 0) & (np.nan_to_num(g["min5"].to_numpy(), nan=1) < 0)).astype(float); g["cross0_dn"] = ((s < 0) & (np.nan_to_num(g["max5"].to_numpy(), nan=-1) > 0)).astype(float)
        g["cross30"] = ((s >= 30) & (s1 < 30)).astype(float); g["cross_m30"] = ((s <= -30) & (s1 > -30)).astype(float)
        vx = g["volx"].to_numpy(float); g["vol3_min"] = pd.Series(vx).rolling(3, min_periods=3).min().to_numpy()
        oi = g["oi"].to_numpy(float); g["oi5"] = (oi / sh(oi, 5) - 1) * 100
        sup, res, inb, ine = fvg_levels(o, h, l, c)
        g["sup_dist"] = (c - sup) / c * 100; g["res_dist"] = (res - c) / c * 100
        g["sup_dist_atr"] = g["sup_dist"] / g["atr_pct"]; g["res_dist_atr"] = g["res_dist"] / g["atr_pct"]; g["in_bull"], g["in_bear"] = inb.astype(float), ine.astype(float)
        o1 = np.r_[o[1:], np.nan]
        for k in HZ: g[f"er{k}"] = (np.r_[c[k:], np.full(k, np.nan)] / o1 - 1) * 100
        g["oc_next"] = np.r_[(c[1:] / o[1:] - 1), np.nan]
        out.append(g)
    P = pd.concat(out, ignore_index=True)
    for b in ("lng_bld", "sht_cov", "sht_bld", "lng_unv"): P["bu_" + b] = (P["buildup"].astype(str).str.strip() == b).astype(float)
    d = P.groupby("date")
    P["rank_pct"] = d["score"].rank(pct=True); P["mkt_ret5"] = d["ret5"].transform("mean"); P["mkt_score"] = d["score"].transform("mean")
    P["breadth"] = d["score"].transform(lambda x: (x > 0).mean()); P["rel_ret5"] = P["ret5"] - P["mkt_ret5"]
    for k in HZ: P[f"xr{k}"] = P[f"er{k}"] - P.groupby("date")[f"er{k}"].transform("mean")
    dates = sorted(P["date"].unique()); P["holdout"] = P["date"] >= dates[-HOLDOUT]; P["near_cut"] = (P["date"] >= dates[-HOLDOUT - EMBARGO]) & ~P["holdout"]
    pats = P["patterns"].fillna("").astype(str)
    toks = pd.Series([t.strip() for x in pats for t in x.split(",") if t.strip()]).value_counts()
    P.attrs["pat_tokens"] = [t for t, n in toks.items() if n >= 300]
    for t in P.attrs["pat_tokens"]: P["pat_" + t] = pats.str.contains(t, regex=False).astype(float)
    return P

# ───────────────────────────── 2. statistics ─────────────────────────────
def wk(d): return pd.to_datetime(d).strftime("%G-%V")

def pick(mask, P, h):
    """First signal per stock, then at least h trading days apart (no overlapping trades in one stock)."""
    idx = np.where(mask)[0]; sym = P["symbol"].to_numpy()[idx]; pos = P["pos"].to_numpy()[idx]; keep = []; last = {}
    for k, s, p in zip(idx, sym, pos):
        if s in last and p - last[s] < h: continue
        last[s] = p; keep.append(k)
    return np.array(keep, int)

def cluster_boot(vals, weeks, B=1000, seed=1):
    v = np.asarray(vals, float); ok = ~np.isnan(v); v, w = v[ok], np.asarray(weeks)[ok]
    if len(v) < 15: return np.nan, np.nan, np.nan, np.nan
    u, inv = np.unique(w, return_inverse=True); s = np.bincount(inv, v); n = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed); bs = np.empty(B)
    for b in range(B):
        r = rng.integers(0, len(u), len(u)); bs[b] = s[r].sum() / max(n[r].sum(), 1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs <= 0).mean()

def evaluate(P, mask, side, h, name, fam):
    ex = f"xr{h}"; er = f"er{h}"; rows = []
    for part, sel in (("dev", ~P["holdout"] & ~P["near_cut"]), ("holdout", P["holdout"])):
        m = np.asarray(mask) & sel.to_numpy() & P[ex].notna().to_numpy()
        ev = pick(m, P, h)
        if len(ev) == 0: rows.append((part, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)); continue
        net = side * P[ex].to_numpy()[ev] - COST; gross = side * P[er].to_numpy()[ev]
        mean, lo, hi, p = cluster_boot(net, [wk(d) for d in P["date"].to_numpy()[ev]])
        rows.append((part, len(ev), net.mean(), lo, hi, p, (gross > COST).mean() * 100, gross.mean(), P["symbol"].to_numpy()[ev].tolist().__len__() and len(set(P["symbol"].to_numpy()[ev])), np.nan))
    d, ho = rows
    # consistency: share of calendar half-years in dev where the net excess was positive
    ev = pick(np.asarray(mask) & (~P["holdout"] & ~P["near_cut"]).to_numpy() & P[ex].notna().to_numpy(), P, h)
    cons = np.nan
    if len(ev) >= 30:
        hy = pd.to_datetime(P["date"].to_numpy()[ev]).strftime("%Y") + pd.Series(pd.to_datetime(P["date"].to_numpy()[ev]).month).map(lambda m: "H1" if m <= 6 else "H2").to_numpy()
        t = pd.DataFrame({"hy": hy, "v": side * P[ex].to_numpy()[ev] - COST}).groupby("hy")["v"].agg(["mean", "count"]); t = t[t["count"] >= 8]
        cons = (t["mean"] > 0).mean() * 100 if len(t) else np.nan
    return {"family": fam, "strategy": name, "side": "LONG" if side > 0 else "SHORT", "hold (days)": h,
            "dev events": d[1], "dev stocks": d[8], "dev net excess %": d[2], "dev CI low": d[3], "dev CI high": d[4], "dev p": d[5], "dev hit % (after cost)": d[6],
            "dev gross avg %": d[7], "dev half-years positive %": cons, "holdout events": ho[1], "holdout net excess %": ho[2], "holdout CI low": ho[3], "holdout CI high": ho[4]}

def bh_fdr(p, q=0.10):
    p = np.asarray(p, float); ok = ~np.isnan(p); out = np.zeros(len(p), bool)
    if ok.sum() == 0: return out
    idx = np.where(ok)[0]; order = idx[np.argsort(p[idx])]; m = len(order); thr = 0
    for rank, i in enumerate(order, 1):
        if p[i] <= q * rank / m: thr = rank
    out[order[:thr]] = True; return out

# ───────────────────────────── 3. strategy library ─────────────────────────────
def library(P):
    S = []; A = lambda fam, name, side, m: S.append((fam, name, side, np.asarray(m, bool)))
    s = P["score"]; v = lambda c: P[c].fillna(np.nan)
    for t in (30, 40, 55, 70):
        A("score level", f"score >= {t}", 1, s >= t); A("score level", f"score <= -{t}", -1, s <= -t)
        A("score level (contrarian)", f"score >= {t} -> SHORT", -1, s >= t); A("score level (contrarian)", f"score <= -{t} -> LONG", 1, s <= -t)
    A("score rank", "top 5% score each day", 1, P["rank_pct"] >= 0.95); A("score rank", "bottom 5% score each day", -1, P["rank_pct"] <= 0.05)
    A("score rank (contrarian)", "bottom 5% score -> LONG", 1, P["rank_pct"] <= 0.05); A("score rank (contrarian)", "top 5% score -> SHORT", -1, P["rank_pct"] >= 0.95)
    A("score shape", "fresh bull flip (crosses above 0, score>=10)", 1, (P["cross0"] == 1) & (s >= 10)); A("score shape", "fresh bear flip", -1, (P["cross0_dn"] == 1) & (s <= -10))
    A("score shape", "crosses above +30", 1, P["cross30"] == 1); A("score shape", "crosses below -30", -1, P["cross_m30"] == 1)
    A("score shape", "strong & accelerating (>=50, rising)", 1, (s >= 50) & (P["accel"] > 0) & (P["d1"] > 0)); A("score shape", "weak & accelerating down (<=-50)", -1, (s <= -50) & (P["accel"] < 0) & (P["d1"] < 0))
    A("score shape", "strong but stalling (>=60, falling) -> SHORT", -1, (s >= 60) & (P["d1"] < 0)); A("score shape", "deep weak turning up (<=-40, rising) -> LONG", 1, (s <= -40) & (P["d1"] > 0))
    A("score shape", "3+ days rising into >=40", 1, (P["run_up"] >= 3) & (s >= 40)); A("score shape", "3+ days falling into <=-40", -1, (P["run_down"] >= 3) & (s <= -40))
    A("score shape", "V-recovery (min5<=-20, now>=20)", 1, (P["min5"] <= -20) & (s >= 20)); A("score shape", "inverted-V (max5>=20, now<=-20)", -1, (P["max5"] >= 20) & (s <= -20))
    A("score shape", "steep climb (slope5>=8, score>0)", 1, (P["slope5"] >= 8) & (s > 0)); A("score shape", "steep fall (slope5<=-8, score<0)", -1, (P["slope5"] <= -8) & (s < 0))
    for k in ("BM", "TA", "OR", "BT", "DC", "PS", "TG", "VW"):
        c = P["c_" + k]
        A("component", f"{k} >= 0.7", 1, c >= 0.7); A("component", f"{k} <= -0.7", -1, c <= -0.7)
        A("component (contrarian)", f"{k} >= 0.7 -> SHORT", -1, c >= 0.7); A("component (contrarian)", f"{k} <= -0.7 -> LONG", 1, c <= -0.7)
    for t in P.attrs.get("pat_tokens", []):
        A("score pattern note", f"note '{t}' -> LONG", 1, P["pat_" + t] == 1); A("score pattern note", f"note '{t}' -> SHORT", -1, P["pat_" + t] == 1)
    for b, sd in (("lng_bld", 1), ("sht_cov", 1), ("sht_bld", -1), ("lng_unv", -1)):
        A("OI build-up (day)", f"{b}", sd, P["bu_" + b] == 1); A("OI build-up (contrarian)", f"{b} -> reversed", -sd, P["bu_" + b] == 1)
    A("volume", "score>=30 & 3-day volume >=2x avg", 1, (s >= 30) & (P["vol3_min"] >= 2)); A("volume", "score<=-30 & 3-day volume >=2x avg", -1, (s <= -30) & (P["vol3_min"] >= 2))
    A("volume", "score>=30 & today volume >=2x", 1, (s >= 30) & (P["volx"] >= 2)); A("volume", "score<=-30 & today volume >=2x", -1, (s <= -30) & (P["volx"] >= 2))
    A("volume", "score>=30 & volume DRY (<0.7x)", 1, (s >= 30) & (P["volx"] <= 0.7)); A("volume", "score<=-30 & volume DRY (<0.7x)", -1, (s <= -30) & (P["volx"] <= 0.7))
    A("delivery", "score>=30 & delivery >=1.3x avg", 1, (s >= 30) & (P["delivx"] >= 1.3)); A("delivery", "score<=-30 & delivery >=1.3x avg", -1, (s <= -30) & (P["delivx"] >= 1.3))
    A("volume (climax)", "volume >=3x & big up day & score<=0 -> LONG", 1, (P["volx"] >= 3) & (P["pxChg"] >= 3) & (s <= 0)); A("volume (climax)", "volume >=3x & big down day & score>=0 -> SHORT", -1, (P["volx"] >= 3) & (P["pxChg"] <= -3) & (s >= 0))
    for d in (1.0, 2.0):
        A("imbalance", f"score>=30 & price within {d} ATR above support gap", 1, (s >= 30) & (P["sup_dist_atr"] <= d) & (P["sup_dist_atr"] >= 0))
        A("imbalance", f"score<=-30 & price within {d} ATR under resistance gap", -1, (s <= -30) & (P["res_dist_atr"] <= d) & (P["res_dist_atr"] >= 0))
        A("imbalance (contrarian)", f"score<=-30 & within {d} ATR above support -> LONG bounce", 1, (s <= -30) & (P["sup_dist_atr"] <= d) & (P["sup_dist_atr"] >= 0))
        A("imbalance (contrarian)", f"score>=30 & within {d} ATR under resistance -> SHORT fade", -1, (s >= 30) & (P["res_dist_atr"] <= d) & (P["res_dist_atr"] >= 0))
    A("imbalance", "inside bull gap & score>=0", 1, (P["in_bull"] == 1) & (s >= 0)); A("imbalance", "inside bear gap & score<=0", -1, (P["in_bear"] == 1) & (s <= 0))
    A("imbalance", "no support gap below & score<=-30", -1, P["sup_dist"].isna() & (s <= -30)); A("imbalance", "no resistance gap above & score>=30", 1, P["res_dist"].isna() & (s >= 30))
    A("market context", "score>=40 when breadth>60% (broad strength)", 1, (s >= 40) & (P["breadth"] > 0.6)); A("market context", "score>=40 when breadth<40% (lone strength)", 1, (s >= 40) & (P["breadth"] < 0.4))
    A("market context", "score<=-40 when breadth<40% (broad weakness)", -1, (s <= -40) & (P["breadth"] < 0.4)); A("market context", "score<=-40 when breadth>60% (lone weakness)", -1, (s <= -40) & (P["breadth"] > 0.6))
    A("trend filter", "score>=40 & price above EMA20>EMA50", 1, (s >= 40) & (P["dist_ema20"] > 0) & (P["ema_spread"] > 0)); A("trend filter", "score<=-40 & price below EMA20<EMA50", -1, (s <= -40) & (P["dist_ema20"] < 0) & (P["ema_spread"] < 0))
    A("trend filter (counter)", "score>=40 but price BELOW EMA20 (dip in strength)", 1, (s >= 40) & (P["dist_ema20"] < 0)); A("trend filter (counter)", "score<=-40 but price ABOVE EMA20 (bounce in weakness)", -1, (s <= -40) & (P["dist_ema20"] > 0))
    A("combo", "score>=40 & volume>=1.5x & above support gap & trend up", 1, (s >= 40) & (P["volx"] >= 1.5) & P["sup_dist"].notna() & (P["ema_spread"] > 0))
    A("combo", "score<=-40 & volume>=1.5x & under resistance gap & trend down", -1, (s <= -40) & (P["volx"] >= 1.5) & P["res_dist"].notna() & (P["ema_spread"] < 0))
    A("combo", "long build-up + delivery>=1.3x + score>=20", 1, (P["bu_lng_bld"] == 1) & (P["delivx"] >= 1.3) & (s >= 20)); A("combo", "short build-up + delivery>=1.3x + score<=-20", -1, (P["bu_sht_bld"] == 1) & (P["delivx"] >= 1.3) & (s <= -20))
    return S

def run_library(P):
    rows = []
    for fam, name, side, m in library(P):
        if m.sum() < 30: continue
        for h in HZ: rows.append(evaluate(P, m, side, h, name, fam))
    T = pd.DataFrame(rows)
    T["survives FDR 10% (dev)"] = bh_fdr(T["dev p"].to_numpy())
    pas = (T["dev CI low"] > 0) & T["survives FDR 10% (dev)"] & (T["holdout net excess %"] > 0) & (T["holdout events"] >= 20)
    prom = (T["dev CI low"] > 0) & ~pas
    T["verdict"] = np.select([pas, prom, T["dev CI high"] < 0], ["PASS (dev + FDR + holdout)", "PROMISING (dev only, not confirmed)", "LOSES MONEY"], "no edge")
    return T.sort_values(["verdict", "dev net excess %"], ascending=[True, False]), len(T)

# ───────────────────────────── 4. portfolio curves ─────────────────────────────
def portfolio(P, mask, side, h, label):
    dates = sorted(P["date"].unique()); di = {d: i for i, d in enumerate(dates)}; syms = sorted(P["symbol"].unique()); si = {s: i for i, s in enumerate(syms)}
    R1 = np.full((len(dates), len(syms)), np.nan); R2 = R1.copy()
    a, b = P["date"].map(di).to_numpy(), P["symbol"].map(si).to_numpy(); R1[a, b] = P["oc_next"].shift(1).where(P["symbol"].eq(P["symbol"].shift(1))).to_numpy()
    cl = P["close"].to_numpy(float); cc = np.where(P["symbol"].eq(P["symbol"].shift(1)), cl / np.r_[np.nan, cl[:-1]] - 1, np.nan); R2[a, b] = cc
    ev = pick(np.asarray(mask), P, h); pnl = np.zeros(len(dates)); cnt = np.zeros(len(dates)); ent = np.zeros(len(dates))
    for k in ev:
        d0, s0 = di[P["date"].iat[k]], si[P["symbol"].iat[k]]; ent[min(d0 + 1, len(dates) - 1)] += 1
        for j in range(1, h + 1):
            d = d0 + j
            if d >= len(dates): break
            r = R1[d, s0] if j == 1 else R2[d, s0]
            if r == r: pnl[d] += side * r * 100; cnt[d] += 1
    daily = np.where(cnt > 0, pnl / np.maximum(cnt, 1), 0.0) - np.where(cnt > 0, COST * ent / np.maximum(cnt, 1), 0.0)
    out = []
    cut = len(dates) - HOLDOUT
    for part, sl in (("dev", slice(0, cut - EMBARGO)), ("holdout", slice(cut, None)), ("all", slice(0, None))):
        x = daily[sl]; act = cnt[sl] > 0
        if act.sum() < 20: continue
        eq = np.cumsum(x); dd = (eq - np.maximum.accumulate(eq)).min()
        out.append({"strategy": label, "period": part, "days": len(x), "days invested %": act.mean() * 100, "avg positions": cnt[sl][act].mean(),
                    "return % p.a. (simple)": x.mean() * 252, "Sharpe": (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan,
                    "max drawdown % (cum. sum)": dd, "worst day %": x.min(), "best day %": x.max()})
    return out, pd.Series(daily, index=dates)

# ───────────────────────────── 5. discovery (walk-forward ML) ─────────────────────────────
FAMILIES = {
 "price & trend": ["ret1", "ret5", "ret10", "ret20", "atr_pct", "dist_ema20", "ema_spread", "gap", "rng20pos", "rel_ret5"],
 "market context": ["mkt_ret5", "mkt_score", "breadth"],
 "conviction score": ["score", "s1", "d1", "d3", "slope5", "accel", "max5", "min5", "run_up", "run_down", "cross0", "cross30", "rank_pct"],
 "score components": ["c_BM", "c_TA", "c_OR", "c_BT", "c_DC", "c_PS", "c_TG", "c_VW"],
 "volume & delivery": ["volx", "delivx", "delivPct", "close_pos", "impulse", "vol3_min"],
 "OI & PCR": ["oiChg", "oi5", "pcr", "pcrChg", "oi_px_ratio", "bu_lng_bld", "bu_sht_cov", "bu_sht_bld", "bu_lng_unv"],
 "imbalance levels": ["sup_dist", "res_dist", "sup_dist_atr", "res_dist_atr", "in_bull", "in_bear"]}

def newgbm(seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, max_iter=140, learning_rate=0.05, min_samples_leaf=400, l2_regularization=10.0, random_state=seed)

def date_ic(dates, pred, y):
    d = pd.DataFrame({"d": dates, "p": pred, "y": y}).dropna()
    return d.groupby("d").apply(lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) > 30 else np.nan).dropna()

def folds_for(P, first=250, step=60):
    dates = sorted(P["date"].unique()); out = []; a = first
    while a < len(dates):
        b = min(a + step, len(dates)); out.append((dates[a - EMBARGO - 1] if a - EMBARGO - 1 >= 0 else dates[0], dates[a], dates[b - 1])); a = b
    return out

def walk_forward(P, cols, target="xr5", clip=15):
    y = P[target].clip(-clip, clip).to_numpy(); d = P["date"].to_numpy(); pred = np.full(len(P), np.nan); X = P[cols].fillna(0.0).to_numpy()
    for tr_end, te0, te1 in folds_for(P):
        tr = (d <= tr_end) & ~np.isnan(y); te = (d >= te0) & (d <= te1)
        if tr.sum() < 5000 or te.sum() == 0: continue
        pred[te] = newgbm().fit(X[tr], y[tr]).predict(X[te])
    return pred

def discovery(P):
    rows = []; allcols = sum(FAMILIES.values(), []); allcols = [c for c in allcols if c in P.columns]
    base = [c for g in ("price & trend", "market context") for c in FAMILIES[g]]
    sets = {"BASE: price & market only": base}
    for g in ("conviction score", "score components", "volume & delivery", "OI & PCR", "imbalance levels"): sets[f"base + {g}"] = base + FAMILIES[g]
    sets["base + conviction score + components"] = base + FAMILIES["conviction score"] + FAMILIES["score components"]
    sets["EVERYTHING"] = allcols
    preds = {}
    for name, cols in sets.items(): preds[name] = walk_forward(P, cols)
    hold = P["holdout"].to_numpy(); dates = P["date"].to_numpy(); y = P["xr5"].to_numpy()
    ic = {}
    for name, p in preds.items(): ic[name] = date_ic(dates, p, y)
    b = ic["BASE: price & market only"]
    for name in sets:
        s = ic[name]; part = lambda ser, ho: ser[[(d >= sorted(P['date'].unique())[-HOLDOUT]) == ho for d in ser.index]]
        res = {"model inputs": name, "dates tested": len(s)}
        for lab, ho in (("dev", False), ("holdout", True)):
            x, bb = part(s, ho), part(b, ho)
            res[f"{lab} IC"] = x.mean() if len(x) else np.nan
            if name != "BASE: price & market only" and len(x) > 10:
                dlt = (x - bb.reindex(x.index)).dropna(); m, lo, hi, _ = cluster_boot(dlt.values, [wk(d) for d in dlt.index])
                res[f"{lab} IC gain vs base"], res[f"{lab} gain CI low"], res[f"{lab} gain CI high"] = m, lo, hi
        g = res.get("dev gain CI low", np.nan)
        res["verdict"] = "" if name.startswith("BASE") else ("ADDS information" if g > 0 else ("no evidence" if res.get("dev gain CI high", 1) >= 0 else "HURTS"))
        rows.append(res)
    return pd.DataFrame(rows), preds["EVERYTHING"], ic

def model_book(P, pred):
    P = P.copy(); P["pred"] = pred; ok = P["pred"].notna()
    rk = P.groupby("date")["pred"].rank(pct=True)
    m_long = ok & (rk >= 0.95); m_short = ok & (rk <= 0.05)
    return m_long.to_numpy(), m_short.to_numpy()

def scenarios(P, cols=None, depth=3, target="xr5"):
    from sklearn.tree import DecisionTreeRegressor
    cols = cols or [c for c in sum(FAMILIES.values(), []) if c in P.columns]; y = P[target].clip(-15, 15).to_numpy(); d = P["date"].to_numpy(); X = P[cols].fillna(0.0).to_numpy()
    cut = sorted(P["date"].unique())[-HOLDOUT]; bank = {}; fl = [f for f in folds_for(P) if f[2] < cut]
    def leaves(tree):
        t = tree.tree_; out = {}
        def rec(n, c):
            if t.children_left[n] == -1: out[n] = c; return
            f, th = cols[t.feature[n]], t.threshold[n]; rec(t.children_left[n], c + [(f, "<=", th)]); rec(t.children_right[n], c + [(f, ">", th)])
        rec(0, []); return out
    for k, (tr_end, te0, te1) in enumerate(fl):
        tr = (d <= tr_end) & ~np.isnan(y); te = (d >= te0) & (d <= te1) & ~np.isnan(y)
        if tr.sum() < 5000: continue
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(400, tr.sum() // 80), random_state=0).fit(X[tr], y[tr]); lv = leaves(tree); lid = tree.apply(X[te]); tei = np.where(te)[0]
        for leaf, conds in lv.items():
            sel = tei[lid == leaf]
            if len(sel) < 30: continue
            sig = tuple(sorted(set((f, o) for f, o, _ in conds))); b = bank.setdefault(sig, {"thr": {}, "fold": [], "rows": []})
            for f, o, th in conds: b["thr"].setdefault((f, o), []).append(th)
            b["fold"].append((k, float(tree.tree_.value[leaf].ravel()[0]))); b["rows"].append(sel)
    out = []
    for sig, b in bank.items():
        if len({k for k, _ in b["fold"]}) < max(3, len(fl) // 2): continue
        sgn = np.sign(np.mean([m for _, m in b["fold"]])) or 1
        conds = [(f, o, float(np.median(b["thr"][(f, o)]))) for f, o in sig]
        def mask_of(rows_mask):
            m = rows_mask.copy()
            for f, o, th in conds: m &= (P[f].fillna(0.0).to_numpy() <= th) if o == "<=" else (P[f].fillna(0.0).to_numpy() > th)
            return m
        txt = " AND ".join(f"{f} {o} {th:.2f}" for f, o, th in conds)
        sel = np.concatenate(b["rows"]); v = sgn * P["xr5"].to_numpy()[sel] - COST
        mean, lo, hi, p = cluster_boot(v, [wk(x) for x in d[sel]])
        hm = mask_of(P["holdout"].to_numpy()); hv = sgn * P["xr5"].to_numpy()[hm] - COST; hv = hv[~np.isnan(hv)]
        out.append({"scenario": txt, "direction": "LONG" if sgn > 0 else "SHORT", "folds seen": len({k for k, _ in b["fold"]}), "dev events (unseen days)": len(sel), "dev net excess 5d %": mean,
                    "dev CI low": lo, "dev CI high": hi, "dev p": p, "holdout events": len(hv), "holdout net excess 5d %": hv.mean() if len(hv) else np.nan})
    S = pd.DataFrame(out)
    if len(S):
        S["survives FDR 10%"] = bh_fdr(S["dev p"].to_numpy()); S = S.sort_values("dev CI low", ascending=False)
        S["verdict"] = np.where((S["dev CI low"] > 0) & S["survives FDR 10%"] & (S["holdout net excess 5d %"] > 0), "PASS", np.where(S["dev CI low"] > 0, "PROMISING", "no edge"))
    return S, len(bank)

# ───────────────────────────── 6. outputs ─────────────────────────────
NOTES = [["term", "meaning"],
 ["timing", "signal from day t's close; entry at day t+1's OPEN; exit after 3 / 5 / 10 days; first signal per stock, then at least h days apart"],
 ["net excess %", "average of (return - average stock's return entered the same day) in the trade direction, MINUS costs (COST_BPS round trip)"],
 ["dev / holdout", "dev = everything except the last 120 trading days (and the 10 days before them). Holdout is only looked at after a strategy is picked on dev"],
 ["CI / p", "95% range and p-value from resampling calendar weeks (stocks on the same day are not independent)"],
 ["FDR 10%", "Benjamini-Hochberg: with ~300 tests tried, this keeps the expected share of lucky winners among the survivors at about 10%"],
 ["verdict PASS", "dev CI above zero AND survives FDR AND positive on the untouched holdout with >=20 trades. PROMISING = dev only. Everything else = no edge"],
 ["contrarian families", "the opposite trade of an obvious signal. Included so we find out if the textbook direction is actually backwards"],
 ["families (discovery)", "walk-forward machine learning predicting each stock's next-5-day excess return. IC = rank correlation (0 = no skill; 0.02-0.05 is useful). 'gain vs base' = what the new information adds beyond price & market"],
 ["scenarios", "shallow decision-tree rules that recur in several walk-forward folds; thresholds are medians across folds; evaluated on unseen days, then on the holdout"],
 ["portfolio", "equal-weight over all open trades each day, costs charged on entry; simple (non-compounded) sums"],
 ["limits", "about 3 years of data = one market cycle; costs are an assumption; no position sizing; futures margin and tax not modelled"]]

def main():
    t0 = time.time(); P = synth_panel() if SELFTEST else real_panel(); print(f"panel: {len(P):,} rows, {P['symbol'].nunique()} stocks, {P['date'].min()}..{P['date'].max()} ({time.time()-t0:.0f}s)", flush=True)
    P = derive(P); P.to_parquet(f"{OUT}/panel.parquet", index=False) if not SELFTEST else None
    print(f"derived ({time.time()-t0:.0f}s) | holdout from {sorted(P['date'].unique())[-HOLDOUT]}", flush=True)
    T, ntests = run_library(P); print(f"library: {ntests} tests ({time.time()-t0:.0f}s)")
    F, mpred, ic = discovery(P); print(f"discovery ({time.time()-t0:.0f}s)")
    ml, ms = model_book(P, mpred)
    mrows = []
    for nm, m, sd in (("MODEL top 5% (walk-forward)", ml, 1), ("MODEL bottom 5% (walk-forward)", ms, -1)):
        for h in HZ: mrows.append(evaluate(P, m & P["xr%d" % h].notna().to_numpy(), sd, h, nm, "model"))
    M = pd.DataFrame(mrows)
    T = pd.concat([T, M], ignore_index=True)
    SC, nsig = scenarios(P)
    # portfolios for every PASS / PROMISING strategy (plus the model book)
    port, curves = [], {}
    lib = {(f, n): (sd, m) for f, n, sd, m in library(P)}
    for r in T[T["verdict"].str.startswith(("PASS", "PROMISING")) | (T["family"] == "model")].head(40).itertuples():
        if r.family == "model": sd, m = (1, ml) if "top" in r.strategy else (-1, ms)
        else: sd, m = lib[(r.family, r.strategy)]
        rows, ser = portfolio(P, m, sd, int(r._4), f"{r.strategy} [{r._4}d]"); port += rows; curves[f"{r.strategy} [{r._4}d]"] = ser
    PF = pd.DataFrame(port)
    # today's signals
    last = sorted(P["date"].unique())[-1]; today = []
    for r in T[T["verdict"].str.startswith(("PASS", "PROMISING"))].drop_duplicates("strategy").itertuples():
        if (r.family, r.strategy) not in lib: continue
        sd, m = lib[(r.family, r.strategy)]; sel = P[(P["date"] == last) & m]
        for q in sel.itertuples(): today.append({"date": last, "symbol": q.symbol, "strategy": r.strategy, "side": "LONG" if sd > 0 else "SHORT", "verdict": r.verdict, "hold (days)": r._4, "score": q.score, "close": q.close})
    TD = pd.DataFrame(today)
    for n, d in (("strategies", T), ("portfolio", PF), ("families", F), ("scenarios", SC), ("today", TD)): d.to_csv(f"{OUT}/sl_{n}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70)
    print("\n== VERDICT COUNTS ==\n", T["verdict"].value_counts().to_string())
    print("\n== TOP STRATEGIES ==\n", T.head(15)[["family", "strategy", "side", "hold (days)", "dev events", "dev net excess %", "dev CI low", "holdout events", "holdout net excess %", "verdict"]].round(3).to_string(index=False))
    print("\n== FAMILIES (what information adds predictive power) ==\n", F.round(4).to_string(index=False))
    if len(SC): print("\n== SCENARIOS ==\n", SC.head(10).round(3).to_string(index=False))
    if len(PF): print("\n== PORTFOLIOS ==\n", PF.round(2).head(20).to_string(index=False))
    if SELFTEST: return T, F, SC, PF
    import oi_probe as P_
    sh = P_.open_sheet(); rnd = lambda d: d.round(4).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    P_.put_tab(sh, "SL Strategies", list(T.columns), rnd(T).values.tolist())
    P_.put_tab(sh, "SL Families", list(F.columns), rnd(F).values.tolist())
    P_.put_tab(sh, "SL Scenarios", list(SC.columns) if len(SC) else ["scenario"], rnd(SC).values.tolist() if len(SC) else [["none recurred"]])
    P_.put_tab(sh, "SL Portfolio", list(PF.columns) if len(PF) else ["portfolio"], rnd(PF).values.tolist() if len(PF) else [["no strategy passed or looked promising"]])
    P_.put_tab(sh, "SL Today", list(TD.columns) if len(TD) else ["today"], rnd(TD).values.tolist() if len(TD) else [["no PASS/PROMISING signal today"]])
    P_.put_tab(sh, "SL Notes", NOTES[0], NOTES[1:] + [["tests in library", str(ntests)], ["rule structures examined", str(nsig)], ["rows", str(len(P))], ["stocks", str(P["symbol"].nunique())], ["data", f"{P['date'].min()} .. {P['date'].max()}"], ["cost bps", str(COST * 100)], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]])
    print("sheet tabs written")

# ───────────────────────────── self-test (synthetic world) ─────────────────────────────
def synth_panel(n_sym=45, n_days=760, seed=11):
    plant = bool(os.environ.get("PLANT")); rng = np.random.default_rng(seed); dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-08-01", periods=n_days)]
    mkt = rng.normal(0.0003, 0.008, n_days); rows = []
    for k in range(n_sym):
        a = np.zeros(n_days)
        for t in range(1, n_days): a[t] = 0.9 * a[t - 1] + rng.normal(0, 0.45)          # latent "information" state
        score = np.clip(45 * a + rng.normal(0, 14, n_days), -100, 100)
        ret = mkt + rng.normal(0, 0.014, n_days)
        if plant: ret[1:] += 0.0016 * np.clip(score[:-1] / 40, -2, 2) * (np.abs(score[:-1]) > 35)    # bigger score -> drift next days (only beyond +/-35)
        c = 100 * np.cumprod(1 + ret); o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n_days)); h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n_days))); l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        vol = rng.lognormal(13, 0.3, n_days); ema = lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
        bu = rng.choice(["lng_bld", "sht_cov", "sht_bld", "lng_unv", ""], n_days)
        comps = {kk: np.clip(score / 60 + rng.normal(0, 0.5, n_days), -1, 1) for kk in ("BM", "TA", "OR", "BT", "DC", "PS", "TG", "VW")}
        notes = np.where(rng.random(n_days) < 0.25, "call-writing-pressure", np.where(rng.random(n_days) < 0.2, "delivery-spike", ""))
        for t in range(n_days):
            rows.append(dict(date=dates[t], symbol=f"S{k:02d}", open=o[t], high=h[t], low=l[t], close=c[t], volume=vol[t], delivPct=40 + rng.normal(0, 8), oi=1e6 * (1 + 0.1 * rng.normal()), oiChg=rng.normal(0, 3), pcr=1 + rng.normal(0, .2),
                             pcrChg=rng.normal(0, .05), ema20=ema(c, 20)[t], ema50=ema(c, 50)[t], pxChg=(c[t] / c[t - 1] - 1) * 100 if t else 0, buildup=bu[t], score=score[t], patterns=notes[t], **{"c_" + kk: v[t] for kk, v in comps.items()}))
    D = pd.DataFrame(rows); g = D.groupby("symbol")
    D["volx"] = (D["volume"] / g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())).clip(0, 10); D["delivx"] = (D["delivPct"] / g["delivPct"].transform(lambda s: s.rolling(20, min_periods=5).mean())).clip(0, 5)
    D["close_pos"] = ((D["close"] - D["low"]) / (D["high"] - D["low"]).replace(0, np.nan)).clip(0, 1); D["impulse"] = D["volx"].clip(0, 3) / 3; D["oi_px_ratio"] = (D["oiChg"] / (D["pxChg"].abs() + 0.3)).clip(-20, 20)
    return D

if __name__ == "__main__":
    if SELFTEST:
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n" + "=" * 20, "SELFTEST", "PLANTED EDGE" if plant else "NULL WORLD", "=" * 20)
            T, F, SC, PF = main(); n_pass = int(T["verdict"].str.startswith("PASS").sum())
            print("PASS count:", n_pass)
            if plant: assert n_pass > 0 and (F.loc[F["model inputs"] == "base + conviction score", "verdict"] == "ADDS information").any(), "planted edge not found"
            else: assert n_pass == 0, "false PASS in null world"
        print("SELFTEST OK")
    else: main()
