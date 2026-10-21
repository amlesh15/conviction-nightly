# ==================================================================================
# STRIKE REACTION ENGINE — what does price do when it meets an option strike, and under which conditions does that matter?
#
#  Reads the 1-minute archive saved by oi_universe.py (spot candles + per-strike OI / premium / volume), then:
#   1  ENCOUNTERS   finds every time price arrives at a strike (from below or above) - REAL strikes - and at PHANTOM levels
#                   (the midpoint between two strikes: same grid spacing, no option attached) as the control
#   2  PATTERNS     classifies what price did in the next 30 minutes (distances in "strike spacings"):
#                     PIN              stays within 0.3 spacing of the strike the whole time
#                     INSTANT_BOUNCE   leaves the strike fast (<=5 min dwell) and retreats >= 0.35 spacing, never closes through it
#                     DWELL_BOUNCE     sits on the strike >= 6 min, then retreats >= 0.25 spacing
#                     FAKEOUT          pokes through (>= 0.1) and is back on the original side (>= 0.1) at +30 min
#                     BREAK_RUN        closes >= 0.25 through and is still >= 0.15 through at +30 min, never came back
#                     BREAK_RETEST     breaks, comes back to the strike, then goes through again
#                     FAILED_BREAK     breaks >= 0.25 then falls back to the original side
#                   What the OI did at that strike BEFORE (known) and AFTER (explanatory) is recorded for each.
#   3  CATALOGUE    which conditions (defending-side OI wall, writing vs unwinding at the strike, repeat touch, time of day, days to
#                   expiry, speed of approach, volume, trend ...) make each pattern more or less frequent, with day-cluster CIs
#   4  EDGES        tradeable rules: FADE (counter-trend entry at the touch, stop 0.3 spacing through the strike, target 0.5 back,
#                   30 min) and BREAK (entry after a 0.2-spacing close-through, stop 0.15 back inside, target 0.6, 30 min), costs
#                   charged. Judged against phantom levels, then conditional rules are DISCOVERED walk-forward (leave-days-out)
#                   and checked on a locked holdout of the last 20% of sessions, with FDR control.
#  Tabs: "SK Overview", "SK Patterns", "SK Edges", "SK Events", "SK Now", "SK Notes"
#  Usage: python strike_engine.py [--selftest]       env: OI_ARCH (archive dir), SK_EXPIRY (2026-10-27), COST_BPS (10)
# ==================================================================================
import os, sys, glob, time, datetime as dt, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import oi_discover as DS
import strategy_lab as SL

OUT = "out/strike"; os.makedirs(OUT, exist_ok=True)
EXPIRY = os.environ.get("SK_EXPIRY", "2026-10-27"); COST_BPS = float(os.environ.get("COST_BPS", "10"))
SELFTEST = "--selftest" in sys.argv
TOL, FAR = 0.10, 0.30                  # touch tolerance and "came from at least this far", in strike spacings
CLASSES = ["PIN", "INSTANT_BOUNCE", "DWELL_BOUNCE", "FAKEOUT", "BREAK_RUN", "BREAK_RETEST", "FAILED_BREAK", "OTHER"]

# ───────────────────────────── encounter detection ─────────────────────────────
def find_touches(c, h, l, mod, K, s):
    tol, far = TOL * s, FAR * s; n = len(c); out = []; i = 31
    while i < n - 61:
        if mod[i] > 14 * 60 + 30: break
        lo30, hi30 = l[i - 30: i].min(), h[i - 30: i].max()
        fb = c[i - 1] < K - tol and lo30 <= K - far and h[i] >= K - tol
        fa = c[i - 1] > K + tol and hi30 >= K + far and l[i] <= K + tol
        if fb ^ fa: out.append((i, 1 if fb else -1)); i += 30
        else: i += 1
    return out

def classify(beyond):
    """beyond[t-1] = dirn*(close[t0+t]-K)/s for t=1..30 : + = through the strike, - = back on the side price came from."""
    mx, mn, fin = beyond.max(), beyond.min(), beyond[-1]; dwell = int((np.abs(beyond) <= 0.1).sum())
    if np.abs(beyond).max() <= 0.3: return "PIN"
    if mx < 0.1 and mn <= -0.35 and dwell <= 5: return "INSTANT_BOUNCE"
    if mx < 0.1 and dwell >= 6 and fin <= -0.25: return "DWELL_BOUNCE"
    if mx >= 0.25:
        tb = int(np.argmax(beyond >= 0.25)); returned = beyond[tb:].min() <= 0.05
        if returned and fin >= 0.15: return "BREAK_RETEST"
        if returned and fin <= -0.1: return "FAILED_BREAK"
        if fin >= 0.15: return "BREAK_RUN"
        return "OTHER"
    if mx >= 0.1 and fin <= -0.1: return "FAKEOUT"
    return "OTHER"

def cost_px(e): return e * COST_BPS / 1e4

def sim_fade(c, h, l, i, K, s, dirn):
    e = c[i]; stop = K + dirn * 0.3 * s; risk = dirn * (stop - e)
    if risk < 0.1 * s: return np.nan
    tgt = e - dirn * 0.5 * s
    for t in range(1, 31):
        j = i + t; hs = h[j] >= stop if dirn > 0 else l[j] <= stop; ht = l[j] <= tgt if dirn > 0 else h[j] >= tgt
        if hs: return (-risk - cost_px(e)) / risk
        if ht: return (0.5 * s - cost_px(e)) / risk
    return (-dirn * (c[i + 30] - e) - cost_px(e)) / risk

def sim_break(c, h, l, i, K, s, dirn):
    for t in range(1, 16):
        if dirn * (c[i + t] - K) >= 0.2 * s: break
    else: return np.nan
    j0 = i + t; e = c[j0]; stop = K - dirn * 0.15 * s; risk = dirn * (e - stop); tgt = e + dirn * 0.6 * s
    if risk <= 0.05 * s: return np.nan
    for u in range(1, 31):
        j = j0 + u
        if j >= len(c): return np.nan
        hs = l[j] <= stop if dirn > 0 else h[j] >= stop; ht = h[j] >= tgt if dirn > 0 else l[j] <= tgt
        if hs: return (-risk - cost_px(e)) / risk
        if ht: return (0.6 * s - cost_px(e)) / risk
    return (dirn * (c[min(j0 + 30, len(c) - 1)] - e) - cost_px(e)) / risk

def side_feats(oi, px, col, i, band0, side_mask_idx, pre):
    """Features of one option side (CE or PE) at strike column `col` at minute i."""
    out = {}
    v = oi[:, col]; cur = v[i]; sub = oi[i, side_mask_idx]
    out[pre + "oi_share"] = cur / max(oi[i].sum(), 1); out[pre + "rank"] = float((sub <= cur).mean()) if len(sub) else np.nan
    out[pre + "wall"] = float(len(sub) > 0 and cur >= sub.max() and cur > 0)
    nb = [v[i] for v in () ]                                  # placeholder (neighbours handled by caller)
    out[pre + "d30"] = (cur - v[i - 30]) / band0 * 100; out[pre + "dopen"] = (cur - v[0]) / band0 * 100
    p0, p1 = px[i - 30, col], px[i, col]; out[pre + "prem30"] = (p1 / p0 - 1) * 100 if p0 > 0 else np.nan
    return out

def session_events(u, s, sp, d, expiry):
    idx = pd.DatetimeIndex(sp["ts"]); ks = sorted(d["strike"].unique())
    if len(ks) < 6: return []
    c = sp["close"].to_numpy(float); h = sp["high"].to_numpy(float); l = sp["low"].to_numpy(float); v = sp["volume"].fillna(0).to_numpy(float)
    mod = (idx.hour * 60 + idx.minute).to_numpy(); T = len(c)
    if T < 300: return []
    mats = {}
    for t in ("C", "P"):
        dd = d[d["typ"] == t]
        if dd.empty: return []
        for colname in ("oi", "close"):
            pv = dd.pivot_table(index="ts", columns="strike", values=colname, aggfunc="last").sort_index(); pv = pv.reindex(pv.index.union(idx)).ffill().reindex(idx)
            mats[(t, colname)] = pv.bfill() if colname == "close" else pv.fillna(0)
    cols = sorted(set(mats[("C", "oi")].columns) & set(mats[("P", "oi")].columns))
    if len(cols) < 6: return []
    Kc = np.array(cols, float); A = {k: m[cols].to_numpy(float) for k, m in mats.items()}; CE, PE, CP, PP = A[("C", "oi")], A[("P", "oi")], A[("C", "close")], A[("P", "close")]
    band0 = CE[0].sum() + PE[0].sum()
    if band0 <= 0: return []
    sp_ = float(np.median(np.diff(Kc))); lo_, hi_ = l.min(), h.max()
    cand = [(k, "real", j) for j, k in enumerate(Kc) if lo_ - 0.5 * sp_ <= k <= hi_ + 0.5 * sp_]
    cand += [((Kc[j] + Kc[j + 1]) / 2, "phantom", j) for j in range(len(Kc) - 1) if lo_ - 0.5 * sp_ <= (Kc[j] + Kc[j + 1]) / 2 <= hi_ + 0.5 * sp_]
    try: dte = int(np.busday_count(str(s), expiry))
    except Exception: dte = np.nan
    vmean = pd.Series(v).expanding().mean().to_numpy(); out = []; touches_today = {}
    for K, kind, j in cand:
        for i, dirn in find_touches(c, h, l, mod, K, sp_):
            beyond = dirn * (c[i + 1: i + 31] - K) / sp_
            if len(beyond) < 30: continue
            ev = {"symbol": u, "session": str(s), "kind": kind, "K": float(K), "spacing": sp_, "spacing_bps": sp_ / c[i] * 1e4, "tod": int(mod[i]), "dirn": dirn, "dte": dte, "band0": band0,
                  "cls": classify(beyond), "dwell30": int((np.abs(beyond) <= 0.1).sum()), "max_through": float(beyond.max()), "min_back": float(beyond.min()), "fin30": float(beyond[-1])}
            for hz in (5, 15, 30, 60): ev[f"back{hz}"] = -dirn * (c[i + hz] - c[i]) / sp_ if i + hz < T else np.nan
            ev["Rfade"] = sim_fade(c, h, l, i, K, sp_, dirn); ev["Rbreak"] = sim_break(c, h, l, i, K, sp_, dirn)
            ev["approach_speed"] = dirn * (c[i] - c[i - 10]) / sp_; ev["vol_surge"] = v[i - 5: i + 1].mean() / vmean[i] if vmean[i] > 0 else np.nan
            ev["day_ret"] = (c[i] / c[0] - 1) * 100; ev["trend_with"] = float(np.sign(c[i] - c[0]) == dirn); ev["range_sofar"] = (h[: i + 1].max() - l[: i + 1].min()) / c[i] * 100
            ev["dist_atm"] = (K - c[0]) / sp_; ev["prior_touches"] = touches_today.get(round(K, 4), 0); touches_today[round(K, 4)] = ev["prior_touches"] + 1
            ev["i"] = i
            if kind == "real":
                dside = (CE, CP, np.where(Kc >= c[i])[0], PE, PP, np.where(Kc <= c[i])[0]) if dirn > 0 else (PE, PP, np.where(Kc <= c[i])[0], CE, CP, np.where(Kc >= c[i])[0])
                for pre, (oi, px, msk) in (("def_", dside[0:3]), ("att_", dside[3:6])):
                    f = side_feats(oi, px, j, i, band0, msk, pre); ev.update(f)
                    nb = [oi[i, jj] for jj in (j - 1, j + 1) if 0 <= jj < len(Kc)]; ev[pre + "peak"] = oi[i, j] / max(np.mean(nb), 1) if nb else np.nan
                    ev[pre + "d_after30"] = (oi[i + 30, j] - oi[i, j]) / band0 * 100; ev[pre + "prem_after30"] = (px[i + 30, j] / px[i, j] - 1) * 100 if px[i, j] > 0 else np.nan
            out.append(ev)
    return out

def build_events():
    expiry = EXPIRY; rows = []; t0 = time.time(); names = DS.stock_files()
    for n, u in enumerate(names, 1):
        try:
            for s, sp, d, prev in DS.stock_sessions(u): rows += session_events(u, s, sp, d, expiry)
        except Exception as e: print("skip", u, type(e).__name__, str(e)[:90])
        if n % 20 == 0: print(f"  {n}/{len(names)} stocks, {len(rows):,} encounters ({time.time()-t0:.0f}s)", flush=True)
    E = pd.DataFrame(rows)
    if E.empty: raise SystemExit("no encounters found in " + DS.ARCH)
    mx = E.groupby("symbol")["band0"].transform("max"); E["thin"] = E["band0"] < DS.MIN_OI_SHARE * mx
    ds = sorted(E["session"].unique()); cut = ds[int(len(ds) * 0.8)]; E["holdout"] = E["session"] >= cut
    E["tod_bucket"] = pd.cut(E["tod"], [0, 11 * 60, 13 * 60, 24 * 60], labels=["morning (<11:00)", "midday", "afternoon (>=13:00)"]).astype(str)
    E["dte_bucket"] = np.where(E["dte"] == 0, "expiry day", np.where(E["dte"] <= 2, "1-2 days to expiry", "3+ days to expiry"))
    return E

# ───────────────────────────── statistics helpers ─────────────────────────────
def cmean(vals, sessions):
    m, lo, hi, p = SL.cluster_boot(np.asarray(vals, float), np.asarray(sessions)); return m, lo, hi, p

def cdiff(a, sa, b, sb, B=1000, seed=3):
    a, sa, b, sb = (np.asarray(x) for x in (a, sa, b, sb)); oka, okb = ~pd.isna(a), ~pd.isna(b); a, sa, b, sb = a[oka].astype(float), sa[oka], b[okb].astype(float), sb[okb]
    if len(a) < 15 or len(b) < 15: return (np.nan,) * 3
    ss = sorted(set(sa) | set(sb)); ix = {x: k for k, x in enumerate(ss)}
    def agg(v, s_):
        sm = np.zeros(len(ss)); n = np.zeros(len(ss))
        for x, vv in zip(s_, v): sm[ix[x]] += vv; n[ix[x]] += 1
        return sm, n
    sa_, na = agg(a, sa); sb_, nb = agg(b, sb); rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ss), len(ss)), minlength=len(ss))
        if (w * na).sum() and (w * nb).sum(): bs.append((w * sa_).sum() / (w * na).sum() - (w * sb_).sum() / (w * nb).sum())
    return a.mean() - b.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ───────────────────────────── 1. overview: real strikes vs phantom levels ─────────────────────────────
def overview(E):
    rows = []
    def add(label, R, Ph):
        if len(R) < 30 or len(Ph) < 30: return
        r = {"split": label, "real encounters": len(R), "phantom encounters": len(Ph), "days": R["session"].nunique()}
        for c in CLASSES[:-1]:
            a, b = (R["cls"] == c).astype(float), (Ph["cls"] == c).astype(float); d, lo, hi = cdiff(a, R["session"], b, Ph["session"])
            r[f"{c} % real"] = a.mean() * 100; r[f"{c} % phantom"] = b.mean() * 100; r[f"{c} diff (pp)"] = d * 100 if d == d else np.nan; r[f"{c} diff CI low"] = lo * 100 if lo == lo else np.nan; r[f"{c} diff CI high"] = hi * 100 if hi == hi else np.nan
        for m, lab in (("back30", "bounce back at +30m (spacings)"), ("Rfade", "FADE trade R"), ("Rbreak", "BREAK trade R")):
            d, lo, hi = cdiff(R[m], R["session"], Ph[m], Ph["session"]); r[lab + " real"] = R[m].mean(); r[lab + " phantom"] = Ph[m].mean(); r[lab + " real-phantom"] = d; r[lab + " CI low"] = lo; r[lab + " CI high"] = hi
        rows.append(r)
    R, Ph = E[E.kind == "real"], E[E.kind == "phantom"]; add("ALL", R, Ph)
    for col in ("dirn", "dte_bucket", "tod_bucket"):
        for v in sorted(E[col].unique()): add(f"{col} = {v}" + (" (price came from below: strike is a CEILING)" if col == "dirn" and v == 1 else (" (came from above: strike is a FLOOR)" if col == "dirn" else "")), R[R[col] == v], Ph[Ph[col] == v])
    return pd.DataFrame(rows)

# ───────────────────────────── 2. pattern catalogue ─────────────────────────────
def conditions(R):
    q = lambda s, p: s.quantile(p)
    C = {}
    if "def_rank" not in R: return C
    C["defender is the chain's biggest OI wall"] = R["def_wall"] == 1; C["defender OI in top 10% of its side"] = R["def_rank"] >= 0.9; C["defender OI is a local peak (>=1.3x neighbours)"] = R["def_peak"] >= 1.3
    hi, lo = q(R["def_d30"], .75), q(R["def_d30"], .25)
    C["defender WRITING at strike (OI up big, premium falling, last 30m)"] = (R["def_d30"] >= hi) & (R["def_prem30"] < 0); C["defender UNWINDING at strike (OI down big, premium rising)"] = (R["def_d30"] <= lo) & (R["def_prem30"] > 0)
    C["attacker side WRITING at strike"] = (R["att_d30"] >= q(R["att_d30"], .75)) & (R["att_prem30"] < 0); C["attacker side UNWINDING at strike"] = (R["att_d30"] <= q(R["att_d30"], .25)) & (R["att_prem30"] > 0)
    C["first touch of this strike today"] = R["prior_touches"] == 0; C["repeat touch (2nd+)"] = R["prior_touches"] >= 1
    C["expiry day"] = R["dte"] == 0; C["1-2 days to expiry"] = R["dte"].between(1, 2); C["morning (<11:00)"] = R["tod"] < 660; C["afternoon (>=13:00)"] = R["tod"] >= 780
    C["fast approach (top quartile)"] = R["approach_speed"] >= q(R["approach_speed"], .75); C["slow approach (bottom quartile)"] = R["approach_speed"] <= q(R["approach_speed"], .25)
    C["volume surge on arrival (>=1.5x)"] = R["vol_surge"] >= 1.5; C["price came WITH the day's trend"] = R["trend_with"] == 1; C["price came AGAINST the day's trend"] = R["trend_with"] == 0
    C["near ATM (within 1 spacing of open)"] = R["dist_atm"].abs() <= 1; C["far from ATM (>=3 spacings)"] = R["dist_atm"].abs() >= 3
    return C

def catalogue(E):
    R = E[(E.kind == "real") & ~E.thin].copy(); C = conditions(R); rows = []
    for cname, m in C.items():
        m = m.fillna(False).to_numpy()
        if m.sum() < 40 or (~m).sum() < 40: continue
        for cls in CLASSES[:-1]:
            y = (R["cls"] == cls).astype(float).to_numpy(); base = y[~m].mean()
            d, lo, hi = cdiff(y[m], R["session"].to_numpy()[m], y[~m], R["session"].to_numpy()[~m])
            rows.append({"pattern": cls, "condition": cname, "events with condition": int(m.sum()), "share with condition %": y[m].mean() * 100, "share without %": base * 100, "difference (pp)": d * 100, "CI low": lo * 100, "CI high": hi * 100,
                         "lift (x)": y[m].mean() / base if base > 0 else np.nan, "days": R["session"][m].nunique()})
    T = pd.DataFrame(rows)
    if len(T):
        T["p-ish"] = np.where(T["CI low"] > 0, 0.01, np.where(T["CI high"] < 0, 0.01, 0.5)); T["verdict"] = np.where(T["CI low"] > 0, "MORE frequent", np.where(T["CI high"] < 0, "LESS frequent", ""))
        T = T.sort_values("difference (pp)", key=abs, ascending=False)
    # after-state: what the OI did next, by pattern (descriptive)
    A = R.groupby("cls").agg(events=("cls", "size"), def_oi_change_after_pctband0=("def_d_after30", "mean"), att_oi_change_after_pctband0=("att_d_after30", "mean"),
                              def_premium_change_after_pct=("def_prem_after30", "mean"), share_defender_wall=("def_wall", "mean"), share_first_touch=("prior_touches", lambda x: (x == 0).mean())).reset_index().rename(columns={"cls": "pattern"})
    return T, A

# ───────────────────────────── 3. edges ─────────────────────────────
def edge_rows(E, rules, target, label):
    rows = []
    R = E[(E.kind == "real") & ~E.thin]; Ph = E[E.kind == "phantom"]
    for name, m in rules:
        m = np.asarray(m.reindex(E.index) if hasattr(m, "reindex") else m)[: len(E)] if False else m
        sel = R[m.reindex(R.index).fillna(False)] if hasattr(m, "reindex") else R[m]
        for part, S in (("dev", sel[~sel.holdout]), ("holdout", sel[sel.holdout])):
            v = S[target].dropna()
            if len(v) < 10: rows.append((name, part, len(v), np.nan, np.nan, np.nan, np.nan, np.nan, 0)); continue
            mean, lo, hi, p = cmean(v.values, S.loc[v.index, "session"].values)
            rows.append((name, part, len(v), mean, lo, hi, p, (v > 0).mean() * 100, S.loc[v.index, "session"].nunique()))
    out = []
    for k in range(0, len(rows), 2):
        d, h = rows[k], rows[k + 1]
        out.append({"trade": label, "rule": d[0], "dev events": d[2], "dev days": d[8], "dev avg R": d[3], "dev CI low": d[4], "dev CI high": d[5], "dev p": d[6], "dev win %": d[7], "holdout events": h[2], "holdout avg R": h[3]})
    return out

def predefined_rules(R):
    C = conditions(R); rules = [("ALL real strikes", pd.Series(True, index=R.index))]
    for k, v in C.items(): rules.append((k, v.fillna(False)))
    for d in (1, -1):
        rules.append((("strike as CEILING (came from below)" if d == 1 else "strike as FLOOR (came from above)"), (R["dirn"] == d)))
    if "def_wall" in R:
        w = (R["def_wall"] == 1); wr = R["def_rank"] >= 0.9
        rules += [("wall & first touch", w & (R["prior_touches"] == 0)), ("wall & defender WRITING", w & C["defender WRITING at strike (OI up big, premium falling, last 30m)"]), ("wall & defender UNWINDING", w & C["defender UNWINDING at strike (OI down big, premium rising)"]),
                  ("wall & morning", w & (R["tod"] < 660)), ("top-10% OI & expiry day", wr & (R["dte"] == 0)), ("wall & fast approach", w & C["fast approach (top quartile)"]), ("wall & volume surge", w & C["volume surge on arrival (>=1.5x)"])]
    return rules

def scenarios(E, target, label, depth=3):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GroupKFold
    R = E[(E.kind == "real") & ~E.thin].copy(); R = R.dropna(subset=[target]); feats = ["def_rank", "def_wall", "def_peak", "def_d30", "def_dopen", "def_prem30", "att_rank", "att_d30", "att_prem30", "prior_touches", "dte", "tod", "dirn", "approach_speed", "vol_surge", "trend_with", "day_ret", "range_sofar", "dist_atm", "spacing_bps"]
    feats = [f for f in feats if f in R.columns]; dev = R[~R.holdout].reset_index(drop=True); ho = R[R.holdout].reset_index(drop=True)
    if len(dev) < 400 or dev["session"].nunique() < 6: return pd.DataFrame(), 0
    X = dev[feats].fillna(0.0).to_numpy(); y = dev[target].clip(-3, 3).to_numpy(); bank = {}; k = min(6, dev["session"].nunique())
    def leaves(tree):
        t = tree.tree_; out = {}
        def rec(n, c):
            if t.children_left[n] == -1: out[n] = c; return
            f, th = feats[t.feature[n]], t.threshold[n]; rec(t.children_left[n], c + [(f, "<=", th)]); rec(t.children_right[n], c + [(f, ">", th)])
        rec(0, []); return out
    for fo, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, groups=dev["session"])):
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(60, len(tr) // 40), random_state=0).fit(X[tr], y[tr]); lv = leaves(tree); lid = tree.apply(X[te])
        for leaf, conds in lv.items():
            sel = te[lid == leaf]
            if len(sel) < 20: continue
            sig = tuple(sorted(set((f, o) for f, o, _ in conds))); b = bank.setdefault(sig, {"thr": {}, "fold": [], "rows": []})
            for f, o, th in conds: b["thr"].setdefault((f, o), []).append(th)
            b["fold"].append((fo, float(tree.tree_.value[leaf].ravel()[0]))); b["rows"].append(sel)
    out = []
    for sig, b in bank.items():
        if len({f for f, _ in b["fold"]}) < max(3, k // 2): continue
        sgn = np.sign(np.mean([m for _, m in b["fold"]])) or 1; conds = [(f, o, float(np.median(b["thr"][(f, o)]))) for f, o in sig]
        sel = np.concatenate(b["rows"]); v = sgn * dev[target].to_numpy()[sel]; mean, lo, hi, p = cmean(v, dev["session"].to_numpy()[sel])
        m_ho = np.ones(len(ho), bool)
        for f, o, th in conds: m_ho &= (ho[f].fillna(0.0).to_numpy() <= th) if o == "<=" else (ho[f].fillna(0.0).to_numpy() > th)
        hv = (sgn * ho[target].to_numpy()[m_ho]); hv = hv[~np.isnan(hv)]
        out.append({"trade": label, "rule": " AND ".join(f"{f} {o} {th:.2f}" for f, o, th in conds), "direction": "as simulated" if sgn > 0 else "REVERSE the trade", "folds seen": len({f for f, _ in b["fold"]}), "dev events (unseen days)": len(sel),
                    "dev days": dev["session"].iloc[sel].nunique(), "dev avg R": mean, "dev CI low": lo, "dev CI high": hi, "dev p": p, "holdout events": len(hv), "holdout avg R": hv.mean() if len(hv) else np.nan})
    S = pd.DataFrame(out)
    if len(S): S["survives FDR 10%"] = SL.bh_fdr(S["dev p"].to_numpy())
    return S, len(bank)

def edges(E):
    R = E[(E.kind == "real") & ~E.thin]; rules = predefined_rules(R); rows = []
    for tgt, lab in (("Rfade", "FADE (counter-trend at the touch)"), ("Rbreak", "BREAK (follow a close-through)")):
        rows += edge_rows(E, rules, tgt, lab)
    T = pd.DataFrame(rows); T["survives FDR 10%"] = SL.bh_fdr(T["dev p"].to_numpy())
    T["verdict"] = np.select([(T["dev CI low"] > 0) & T["survives FDR 10%"] & (T["holdout avg R"] > 0) & (T["holdout events"] >= 20), T["dev CI low"] > 0, T["dev CI high"] < 0], ["PASS (dev + FDR + holdout)", "PROMISING (dev only)", "LOSES"], "no edge")
    Sc, n = [], 0
    for tgt, lab in (("Rfade", "FADE"), ("Rbreak", "BREAK")):
        s_, nn = scenarios(E, tgt, lab); n += nn
        if len(s_): Sc.append(s_)
    S = pd.concat(Sc) if Sc else pd.DataFrame()
    if len(S):
        S["verdict"] = np.select([(S["dev CI low"] > 0) & S["survives FDR 10%"] & (S["holdout avg R"] > 0) & (S["holdout events"] >= 15), S["dev CI low"] > 0], ["PASS (dev + FDR + holdout)", "PROMISING (dev only)"], "no edge"); S = S.sort_values("dev CI low", ascending=False)
    return T.sort_values(["verdict", "dev avg R"], ascending=[True, False]), S, len(rules) * 2, n

def now_table(E_last):
    return E_last

NOTES = [["term", "meaning"],
 ["spacing", "distance between two neighbouring strikes. Every distance here is in spacings so a Rs 5-strike stock and a Rs 50-strike stock are comparable"],
 ["real vs phantom", "real = an actual strike; phantom = midpoint between two strikes (same grid, no option). Differences between them = the part of the behaviour that is about OPTIONS and not just round-number support/resistance"],
 ["ceiling / floor", "price arriving from below meets the strike as a ceiling (call writers defend); arriving from above it is a floor (put writers defend). 'defender' = the call side at a ceiling, the put side at a floor; 'attacker' = the other side"],
 ["patterns", "classified on closes during the 30 minutes after the first touch; see the header of strike_engine.py for exact thresholds"],
 ["BEFORE vs AFTER", "def_d30/prem30 etc. are measured before the touch (usable as conditions). *_after30 are measured after it and are only there to describe what happens next to OI inside each pattern"],
 ["FADE / BREAK trades", "FADE: enter against the approach at the touch bar's close, stop 0.3 spacing through the strike, target 0.5 spacing back, 30 min. BREAK: enter after a close >= 0.2 through, stop 0.15 back inside, target 0.6, 30 min. R = profit / risk after costs"],
 ["tests", "CIs resample whole trading days. Holdout = last 20% of sessions, never used to build rules. Benjamini-Hochberg FDR over all rules tried. 'thin' sessions (far-month, low OI) excluded from OI conditions"],
 ["limits", "about 14-20 usable sessions of the current series; stocks on one day move together; expiry-day sample is small. Treat PASS as a lead to re-test as the archive grows"]]

def main():
    t0 = time.time(); E = synth_events_via_archive() if SELFTEST else build_events()
    print(f"{len(E):,} encounters: real {int((E.kind=='real').sum()):,}, phantom {int((E.kind=='phantom').sum()):,}, sessions {E['session'].nunique()}, stocks {E['symbol'].nunique()} ({time.time()-t0:.0f}s)", flush=True)
    E.to_parquet(f"{OUT}/encounters.parquet", index=False) if not SELFTEST else None
    O = overview(E); C, A = catalogue(E); T, S, nrules, nsc = edges(E)
    for n, d in (("overview", O), ("patterns", C), ("after_state", A), ("edges", T), ("scenarios", S)): d.to_csv(f"{OUT}/sk_{n}.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
    print("\n== REAL vs PHANTOM (all) ==\n", O.head(1).T.round(3).to_string()); print("\n== PATTERN CATALOGUE (largest effects) ==\n", C.head(12)[["pattern", "condition", "events with condition", "share with condition %", "share without %", "difference (pp)", "CI low", "CI high", "verdict"]].round(2).to_string(index=False))
    print("\n== AFTER-STATE by pattern ==\n", A.round(3).to_string(index=False)); print("\n== EDGES ==\n", T["verdict"].value_counts().to_string()); print(T.head(10)[["trade", "rule", "dev events", "dev avg R", "dev CI low", "holdout events", "holdout avg R", "verdict"]].round(3).to_string(index=False))
    if len(S): print("\n== DISCOVERED SCENARIOS ==\n", S.head(8)[["trade", "rule", "folds seen", "dev events (unseen days)", "dev avg R", "dev CI low", "holdout events", "holdout avg R", "verdict"]].round(3).to_string(index=False))
    if SELFTEST: return E, O, C, T, S
    import oi_probe as P_
    sh = P_.open_sheet(); rnd = lambda d: d.round(4).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    P_.put_tab(sh, "SK Overview", list(O.columns), rnd(O).values.tolist()); P_.put_tab(sh, "SK Patterns", list(C.columns), rnd(C).values.tolist())
    P_.put_tab(sh, "SK Patterns After-State", list(A.columns), rnd(A).values.tolist()); P_.put_tab(sh, "SK Edges", list(T.columns), rnd(T).values.tolist())
    P_.put_tab(sh, "SK Scenarios", list(S.columns) if len(S) else ["scenario"], rnd(S).values.tolist() if len(S) else [["none recurred"]])
    ev = E[(E.kind == "real") & ~E.thin].sort_values("session", ascending=False).head(3000)[["symbol", "session", "tod", "K", "dirn", "cls", "dwell30", "back30", "Rfade", "Rbreak", "def_rank", "def_wall", "def_d30", "def_prem30", "def_d_after30", "prior_touches", "dte"]]
    P_.put_tab(sh, "SK Events", list(ev.columns), rnd(ev).values.tolist())
    P_.put_tab(sh, "SK Notes", NOTES[0], NOTES[1:] + [["encounters", str(len(E))], ["rules tried (predefined x2 trades)", str(nrules)], ["tree structures examined", str(nsc)], ["sessions", f"{E['session'].min()} .. {E['session'].max()}"], ["cost bps", str(COST_BPS)], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]])
    print("sheet tabs written")

# ───────────────────────────── self-test: synthetic archive ─────────────────────────────
def write_synth_archive(dirpath, plant, n_sym=30, n_days=24, seed=4):
    rng = np.random.default_rng(seed); os.makedirs(dirpath, exist_ok=True); days = pd.bdate_range("2026-09-21", periods=n_days)
    for k in range(n_sym):
        u = f"T{k:02d}"; px0 = 200.0; ks = np.arange(140.0, 261.0, 5.0); sp_rows, op_rows = [], []; tol = 0.5
        for di, d in enumerate(days):
            base = {t: rng.lognormal(10, 0.4, len(ks)) for t in ("C", "P")}; wallC = rng.integers(len(ks) // 2, len(ks) - 2); wallP = rng.integers(2, len(ks) // 2)
            base["C"][wallC] *= 6; base["P"][wallP] *= 6; n = 376; c = np.empty(n); p = px0; kick = 0.0; kdir = 0
            for i in range(n):
                p *= 1 + rng.normal(0, 0.0007) + (kick * kdir / 1000 if kick > 0 else 0)
                if kick > 0: kick -= 1
                if plant:
                    for wk_, w_dir in ((ks[wallC], -1), (ks[wallP], 1)):
                        if kick <= 0 and abs(p - wk_) <= 0.5 and ((w_dir == -1 and p <= wk_) or (w_dir == 1 and p >= wk_)) and rng.random() < 0.25: kick, kdir = 8, w_dir * 1.2
                c[i] = p
            px0 = c[-1]; vol = rng.lognormal(7, 0.5, n); o = np.r_[c[0], c[:-1]]
            for i in range(n):
                ts = (d + pd.Timedelta(hours=9, minutes=15 + i)).strftime("%Y-%m-%d %H:%M:%S"); sp_rows.append((ts, o[i], max(o[i], c[i]) * 1.0003, min(o[i], c[i]) * 0.9997, c[i], vol[i]))
            for t in ("C", "P"):
                oi = base[t].copy()
                for i in range(n):
                    if i % 5 == 0: oi = oi * (1 + rng.normal(0, 0.003, len(ks)))
                    ts = (d + pd.Timedelta(hours=9, minutes=15 + i)).strftime("%Y-%m-%d %H:%M:%S")
                    for j in range(len(ks)):
                        intr = max(c[i] - ks[j], 0) if t == "C" else max(ks[j] - c[i], 0); op_rows.append((ts, intr + 2.0, rng.integers(0, 50), oi[j], ks[j], t))
        pd.DataFrame(sp_rows, columns=["ts", "open", "high", "low", "close", "volume"]).to_csv(f"{dirpath}/{u}_spot.csv.gz", index=False)
        pd.DataFrame(op_rows, columns=["ts", "close", "volume", "oi", "strike", "typ"]).to_csv(f"{dirpath}/{u}_26OCT.csv.gz", index=False)

def synth_events_via_archive():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="sk_arch_"); write_synth_archive(tmp, bool(os.environ.get("PLANT"))); DS.ARCH = tmp
    return build_events()

if __name__ == "__main__":
    if SELFTEST:
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n" + "=" * 20, "SELFTEST", "PLANTED WALL BOUNCE" if plant else "NULL (no strike effect)", "=" * 20)
            E, O, C, T, S = main(); r = T[(T["rule"] == "defender is the chain's biggest OI wall") & (T["trade"].str.startswith("FADE"))]
            print("fade at wall:", r[["dev events", "dev avg R", "dev CI low", "holdout avg R", "verdict"]].round(3).to_string(index=False))
            if plant: assert len(r) and r["dev CI low"].iloc[0] > 0, "planted wall bounce not found"
            else: assert not T["verdict"].str.startswith("PASS").any(), "false PASS in null world"
        print("SELFTEST OK")
    else: main()
