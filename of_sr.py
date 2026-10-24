# ==================================================================================
# SUPPORT / RESISTANCE FROM VOLUME PROFILE  +  ORDER-FLOW READ AT THE LEVEL      (research only, no orders)
#  1. LEVEL MAP built BEFORE the session from earlier sessions only (no peeking):
#       POC10 / VAH10 / VAL10 / HVN10 = point of control, value-area high/low, heavy-volume nodes of the last 10 sessions' volume profile
#       PDH / PDL = previous day high/low        PDPOC / PDVAH / PDVAL = previous day's own profile
#  2. TOUCH = the first time price comes back to a level during the session (within a tolerance that scales with the stock's usual range).
#     We watch the next 5 minutes of footprint and tag what we see (fixed rules, written before testing):
#       ABSORB   heavy volume (>=1.5x usual for that time) + aggressive flow INTO the level (>=10% net delta) but price holds the level
#       SWEEP    price pokes through the level then closes back on the holding side (stop run / liquidity grab)
#       EFFORT   heavy volume but a tiny range (effort without result)
#       EXHAUST  aggression into the level fades in the last 2 minutes of the window
#       RSTACK   stacked imbalance in the RESPONSIVE direction (buyers at support / sellers at resistance)
#       DIVERG   price makes a new local extreme but cumulative delta does not
#  3. TRADE = fade the level (long at support, short at resistance), entered at the open after the 5-minute window, exits at +15/+30/+60 min or 15:14, cost 10 bps.
#  4. CONTROL = PHANTOM levels: the same distance from the open on the OTHER side of price (a random price in the same place at the same time of day).
#     Levels only matter if real levels beat phantom levels; a flag only matters if it beats the same flag at phantom levels.  Last 20% of sessions held out.
#  Tabs: "OSR Level Map" (tomorrow's S/R for every stock), "OSR Levels vs Random", "OSR Flags", "OSR Absorption Detail", "OSR Terms", "OSR Notes".
#  Usage: python of_sr.py [--selftest]      env: OF_ARCH, COST_BPS (10)
# ==================================================================================
import os, sys, glob, time, warnings, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import of_lab as L

SELFTEST = "--selftest" in sys.argv; COST = L.COST; ARCH = L.ARCH; OUT = "out/of_sr"; os.makedirs(OUT, exist_ok=True)
NH = 10; MINP = 5; WIN = 5; HZS = (15, 30, 60)
PRIO = ["POC10", "VAH10", "VAL10", "PDH", "PDL", "HVN10", "PDPOC", "PDVAH", "PDVAL"]; FLAGS = ["absorb", "sweep", "effort", "exhaust", "rstack", "diverg"]
FLAGNAME = {"absorb": "ABSORPTION", "sweep": "SWEEP / stop run", "effort": "EFFORT without result", "exhaust": "EXHAUSTION of aggression", "rstack": "RESPONSIVE stacked imbalance", "diverg": "DELTA DIVERGENCE"}

# ────────────────────────── profile -> levels ──────────────────────────
def load_hist(G):
    """profile frame (date, per, bin, w, buy, sell) -> common price grid; returns wg and {date: (k, vol)}"""
    wg = float(G["w"].max()); c = (G["bin"].to_numpy() + 0.5) * G["w"].to_numpy(); k = np.floor(c / wg).astype(np.int64); v = (G["buy"] + G["sell"]).to_numpy(float); H = {}
    for d, ix in G.groupby("date").indices.items(): H[d] = (k[ix], v[ix])
    return wg, H

def composite(H, dates):
    kk = np.concatenate([H[d][0] for d in dates]); vv = np.concatenate([H[d][1] for d in dates]); uk, inv = np.unique(kk, return_inverse=True); sv = np.bincount(inv, vv)
    dense = np.zeros(uk[-1] - uk[0] + 1); dense[uk - uk[0]] = sv; return int(uk[0]), dense

def value_area(dn, frac=0.70):
    p = int(np.argmax(dn)); lo = hi = p; acc = dn[p]; tot = dn.sum()
    while acc < frac * tot and (lo > 0 or hi < len(dn) - 1):
        a = dn[lo - 1] if lo > 0 else -1.0; b = dn[hi + 1] if hi < len(dn) - 1 else -1.0
        if a >= b: lo -= 1; acc += a
        else: hi += 1; acc += b
    return p, lo, hi

def levels_from(H, dates, wg, with_hvn):
    from scipy.signal import find_peaks
    k0, dn = composite(H, dates)
    if dn.sum() <= 0: return {}
    sm = np.convolve(dn, [0.25, 0.5, 0.25], mode="same"); p, lo, hi = value_area(sm); out = {"POC": (k0 + p + 0.5) * wg, "VAH": (k0 + hi + 1) * wg, "VAL": (k0 + lo) * wg}
    if with_hvn and len(sm) > 12:
        ref = out["POC"]; dist = max(1, int(round(0.004 * ref / wg))); pk, pr = find_peaks(sm, prominence=0.15 * sm.max(), distance=dist)
        pk = [x for x in pk[np.argsort(-sm[pk])] if abs((k0 + x + 0.5) * wg - ref) > 0.003 * ref][:3]; out["HVN"] = [(k0 + x + 0.5) * wg for x in pk]
    return out

def day_levels(dates_prior, H, wg, prev_hl):
    """list of (type, price) from sessions strictly before the trading day"""
    if len(dates_prior) < MINP: return []
    use = dates_prior[-NH:]; a = levels_from(H, use, wg, True); b = levels_from(H, [dates_prior[-1]], wg, False); lv = []
    if a: lv += [("POC10", a["POC"]), ("VAH10", a["VAH"]), ("VAL10", a["VAL"])] + [("HVN10", x) for x in a.get("HVN", [])]
    if b: lv += [("PDPOC", b["POC"]), ("PDVAH", b["VAH"]), ("PDVAL", b["VAL"])]
    if prev_hl is not None: lv += [("PDH", prev_hl[0]), ("PDL", prev_hl[1])]
    return lv

def dedupe(lv, tol):
    lv = sorted(lv, key=lambda t: PRIO.index(t[0])); kept = []
    for t, p in lv:
        near = [q for _, q in lv if abs(q - p) <= 2 * tol and q != p]
        if all(abs(p - q) > tol for _, q, _c in kept): kept.append((t, p, 1 + len(near)))
    return kept

# ────────────────────────── per-day arrays ──────────────────────────
def day_arrays(S):
    S = S.copy(); S["m"] = (S["hhmm"] // 100) * 60 + S["hhmm"] % 100 - 555; S = S[(S["m"] >= 0) & (S["m"] < 360)]; D = {}
    for d, g in S.groupby("date", sort=True):
        if len(g) < 330: continue
        A = {}; mm = g["m"].to_numpy()
        for k, col in (("o", "o"), ("h", "h"), ("l", "l"), ("c", "c"), ("v", "tot"), ("dl", "delta"), ("bs", "bstack"), ("ss", "sstack"), ("cd", "cd")):
            a = np.full(360, np.nan); a[mm] = g[col].to_numpy(float); A[k] = a
        c = pd.Series(A["c"]).ffill().bfill().to_numpy(); A["c"] = c
        for k in ("o", "h", "l"): A[k] = np.where(np.isnan(A[k]), c, A[k])
        A["h"] = np.maximum(A["h"], np.maximum(A["o"], c)); A["l"] = np.minimum(A["l"], np.minimum(A["o"], c))
        for k in ("v", "dl", "bs", "ss"): A[k] = np.nan_to_num(A[k]); A["cd"] = pd.Series(A["cd"]).ffill().fillna(0).to_numpy()
        D[d] = A
    return D

# ────────────────────────── touch evaluation ──────────────────────────
def evaluate(A, i, s, Lp, tol, U, relbase):
    e = i + WIN
    if e > 344: return None
    sl = slice(i, e); v = A["v"][sl]; vs = v.sum()
    if vs <= 0: return None
    dsh = -s * A["dl"][sl].sum() / vs; base = relbase[sl].sum(); relv = vs / base if base > 0 else np.nan
    X = A["l"] if s > 0 else A["h"]; pierce = float(np.max(s * (Lp - X[sl]))); hold = s * (A["c"][e - 1] - Lp) >= 0; held = bool(hold and pierce <= 0.5 * tol)
    rng = A["h"][sl].max() - A["l"][sl].min(); big = (relv >= 1.5) if relv == relv else False
    v1 = max(v[:3].sum(), 1e-9); v2 = max(v[3:].sum(), 1e-9); d1 = -s * A["dl"][i:i + 3].sum() / v1; d2 = -s * A["dl"][i + 3:e].sum() / v2
    st = A["bs" if s > 0 else "ss"][sl].sum(); dv = False
    if i >= 10:
        lb = slice(max(0, i - 30), i)
        if s > 0: dv = bool(A["l"][sl].min() < A["l"][lb].min() and A["cd"][e - 1] > A["cd"][lb].min())
        else: dv = bool(A["h"][sl].max() > A["h"][lb].max() and A["cd"][e - 1] < A["cd"][lb].max())
    r = {"relv": relv, "dsh_into": dsh, "pierce_tol": pierce / tol, "held": held, "absorb": bool(held and big and dsh >= 0.10), "sweep": bool(hold and pierce >= 0.5 * tol), "effort": bool(big and rng <= 2 * tol),
         "exhaust": bool(hold and d1 >= 0.10 and d2 <= 0.5 * d1), "rstack": bool(hold and st >= 1), "diverg": bool(hold and dv)}
    ent = A["o"][e]; thr = ent * 0.25 * U / 1e4
    for h in HZS: j = e + h - 1; r[f"ret{h}"] = s * (A["c"][j] / ent - 1) * 1e4 - COST if j <= 359 else np.nan
    r["retE"] = s * (A["c"][359] / ent - 1) * 1e4 - COST; sh = A["h"][e:e + 60]; sl_ = A["l"][e:e + 60]
    up = (sh >= ent + thr) if s > 0 else (sl_ <= ent - thr); dn = (sl_ <= ent - thr) if s > 0 else (sh >= ent + thr); fu = np.argmax(up) if up.any() else 999; fd = np.argmax(dn) if dn.any() else 999
    r["bounce_first"] = np.nan if (fu == 999 and fd == 999) else float(fu < fd); return r

def first_touch(A, Lp, tol):
    o = A["o"][0]
    if abs(o - Lp) <= 2 * tol: return None, 0
    s = 1 if o > Lp else -1; hit = np.nonzero(A["l"] <= Lp + tol)[0] if s > 0 else np.nonzero(A["h"] >= Lp - tol)[0]
    return (int(hit[0]), s) if len(hit) else (None, s)

def stock_events(sym, D, H, wg, seed=0):
    dates = sorted(D); rows = []; rng = np.random.default_rng(abs(hash(sym)) % (2 ** 32) + seed); volm = np.vstack([D[d]["v"] for d in dates]); rngd = np.array([(D[d]["h"].max() - D[d]["l"].min()) / D[d]["o"][0] * 1e4 for d in dates])
    for di, d in enumerate(dates):
        prior = [x for x in dates[:di] if x in H]
        if len(prior) < MINP or di < MINP: continue
        A = D[d]; pa = D[dates[di - 1]]; U = float(np.mean(rngd[max(0, di - NH):di])); tolb = float(np.clip(0.08 * U, 4, 20)); tol = A["o"][0] * tolb / 1e4; relbase = volm[max(0, di - NH):di].mean(axis=0)
        lv = dedupe(day_levels(prior, H, wg, (pa["h"].max(), pa["l"].min())), tol); real_prices = [p for _, p, _ in lv]
        for t, p, conf in lv:
            for kind in ("real", "phantom", "phantom", "phantom"):
                Lp = p if kind == "real" else A["o"][0] - (p - A["o"][0]) * rng.uniform(0.85, 1.15)
                if kind == "phantom" and (Lp <= 0 or any(abs(Lp - q) <= 2 * tol for q in real_prices)): continue
                i, s = first_touch(A, Lp, tol)
                if i is None: continue
                r = evaluate(A, i, s, Lp, tol, U, relbase)
                if r is None: continue
                r.update({"symbol": sym, "date": d, "kind": kind, "ltype": t, "conf": conf, "side": s, "i": i, "dist_bps": abs(p - A["o"][0]) / A["o"][0] * 1e4, "U": U, "tol_bps": tolb}); rows.append(r)
    return rows

# ────────────────────────── statistics ──────────────────────────
def st(df, col):
    v = df[col].to_numpy(float); m, lo, hi, p, nd = L.cmean(v, df["date"].to_numpy()); return m, lo, hi, nd, (np.nanmean(v > 0) * 100 if len(v) else np.nan)

def diff(a, b, col):
    return L.cdiff(a[col].to_numpy(float), a["date"].to_numpy(), b[col].to_numpy(float), b["date"].to_numpy())

def tables(E):
    dates = sorted(E["date"].unique()); cut = dates[int(len(dates) * 0.8)]; E["holdout"] = E["date"] >= cut; E["ALL"] = True; E["none"] = ~E[FLAGS].any(axis=1)
    sets = [("ALL touches", "ALL")] + [(FLAGNAME[f], f) for f in FLAGS] + [("held, no flag required", "held"), ("none of the flags", "none")]; rows = []
    for nm, col in sets:
        for part, sel in (("dev", ~E["holdout"]), ("holdout", E["holdout"])):
            R = E[sel & (E["kind"] == "real") & E[col]]; Pm = E[sel & (E["kind"] == "phantom") & E[col]]; Rn = E[sel & (E["kind"] == "real") & ~E[col]] if col not in ("ALL",) else None
            for h in (*HZS, "E"):
                c = f"ret{h}"
                if len(R) < 30: continue
                m, lo, hi, nd, hit = st(R, c); pm = st(Pm, c)[0] if len(Pm) >= 30 else np.nan; d, dlo, dhi = diff(R, Pm, c) if len(Pm) >= 30 else (np.nan,) * 3; dn_ = diff(R, Rn, c) if (Rn is not None and len(Rn) >= 30) else (np.nan,) * 3
                rows.append({"tag": nm, "sample": part, "hold": ("to 15:14" if h == "E" else f"{h} min"), "real touches": len(R), "days": nd, "REAL net bps": m, "CI low": lo, "CI high": hi, "real hit %": hit, "real bounce-first %": np.nanmean(R["bounce_first"]) * 100,
                             "PHANTOM net bps": pm, "phantom touches": len(Pm), "real - phantom": d, "diff CI low": dlo, "diff CI high": dhi, "tag vs untagged real": dn_[0], "vs untagged CI low": dn_[1], "vs untagged CI high": dn_[2]})
    F = pd.DataFrame(rows)
    if len(F):
        F["read"] = np.where((F["CI low"] > 0) & (F["diff CI low"] > 0), "REAL > ZERO and > PHANTOM", np.where(F["CI high"] < 0, "loses after cost", "not distinguishable from 0"))
        piv = F[F["hold"] == "30 min"].pivot_table(index="tag", columns="sample", values=["REAL net bps", "diff CI low"], aggfunc="first"); ok = []
        for tag in piv.index:
            try: ok.append(piv.loc[tag, ("diff CI low", "dev")] > 0 and piv.loc[tag, ("REAL net bps", "holdout")] > 0)
            except Exception: pass
        F["survives (dev CI beats phantom AND holdout positive, 30 min)"] = F["tag"].map(lambda t: "YES" if (t in piv.index and piv.loc[t, ("diff CI low", "dev")] > 0 and piv.loc[t, ("REAL net bps", "holdout")] > 0) else "no")
    LV = []
    for lt in ["ALL levels"] + PRIO:
        for part, sel in (("dev", ~E["holdout"]), ("holdout", E["holdout"])):
            R = E[sel & (E["kind"] == "real")]; Pm = E[sel & (E["kind"] == "phantom")]
            if lt != "ALL levels": R = R[R["ltype"] == lt]; Pm = Pm[Pm["ltype"] == lt]
            for h in HZS:
                c = f"ret{h}"
                if len(R) < 30 or len(Pm) < 30: continue
                m, lo, hi, nd, hit = st(R, c); pm, plo, phi, _, phit = st(Pm, c); d, dlo, dhi = diff(R, Pm, c)
                LV.append({"level type": lt, "sample": part, "hold": f"{h} min", "real touches": len(R), "days": nd, "REAL fade net bps": m, "CI low": lo, "CI high": hi, "real bounce-first %": np.nanmean(R["bounce_first"]) * 100, "PHANTOM fade net bps": pm, "phantom bounce-first %": np.nanmean(Pm["bounce_first"]) * 100,
                           "real - phantom": d, "diff CI low": dlo, "diff CI high": dhi, "read": "levels HOLD more than random prices" if dlo > 0 else ("levels are WORSE than random" if dhi < 0 else "same as random prices")})
    AB = []; A = E[(E["kind"] == "real") & E["absorb"]].copy(); A["conf2"] = np.where(A["conf"] >= 2, "confluence (2+ levels within range)", "single level"); A["relv band"] = pd.cut(A["relv"], [0, 2, 3, 5, 1e9], labels=["1.5-2x", "2-3x", "3-5x", ">5x"])
    for gname, gcol in (("level type", "ltype"), ("confluence", "conf2"), ("volume surge", "relv band"), ("side", "side")):
        for k, g in A.groupby(gcol, observed=True):
            if len(g) < 20: continue
            m, lo, hi, nd, hit = st(g, "ret30"); AB.append({"split": gname, "group": str(k), "absorption touches": len(g), "days": nd, "net bps @30 min": m, "CI low": lo, "CI high": hi, "hit %": hit, "bounce-first %": np.nanmean(g["bounce_first"]) * 100})
    return F, pd.DataFrame(LV), pd.DataFrame(AB), cut

def tomorrow_map(profiles, days_all):
    rows = []
    for sym, (H, wg) in profiles.items():
        D = days_all.get(sym)
        if not D: continue
        dates = sorted(D); prior = [x for x in dates if x in H]
        if len(prior) < MINP: continue
        pa = D[dates[-1]]; U = float(np.mean([(D[d]["h"].max() - D[d]["l"].min()) / D[d]["o"][0] * 1e4 for d in dates[-NH:]])); close = float(pa["c"][-1]); tol = close * float(np.clip(0.08 * U, 4, 20)) / 1e4
        lv = dedupe(day_levels(prior, H, wg, (pa["h"].max(), pa["l"].min())), tol); sup = sorted([(p, t, c) for t, p, c in lv if p < close - tol], reverse=True)[:3]; res = sorted([(p, t, c) for t, p, c in lv if p > close + tol])[:3]
        f = lambda L_: [f"{t} {p:.2f} ({(p / close - 1) * 100:+.1f}%){' x' + str(c) if c > 1 else ''}" for p, t, c in L_]; row = {"symbol": sym, "last close (15:14)": round(close, 2), "usual daily range %": round(U / 100, 2), "latest session": dates[-1]}
        for n in range(3): row[f"support {n + 1}"] = f(sup)[n] if n < len(sup) else ""
        for n in range(3): row[f"resistance {n + 1}"] = f(res)[n] if n < len(res) else ""
        rows.append(row)
    return pd.DataFrame(rows)

TERMS = [["term", "plain meaning", "how this engine detects it (touch window = 5 minutes after price first returns to a level)"],
 ["Volume profile", "volume traded at each price, not each time", "from the saved 30-minute volume-at-price tables"], ["POC", "price with the most volume = where the market agreed on value", "peak of the last-10-session profile (POC10) or of yesterday alone (PDPOC)"],
 ["VAH / VAL", "top / bottom of the band holding 70% of the volume", "expanded from the POC; VAH10/VAL10 and PDVAH/PDVAL"], ["HVN", "a secondary heavy-volume price: price lingers / reacts there", "up to 3 profile peaks, at least 0.3% away from the POC"],
 ["LVN", "thin-volume price: price tends to slice through", "not traded as a level; the gaps between HVNs"], ["PDH / PDL", "previous day's high and low", "from the minute feed (matches the official daily high/low)"],
 ["Confluence", "two or more levels stacked within a small band = stronger", "'conf' column and the 'x2' in the level map"], ["Absorption", "aggressors keep hitting but the price does not give way: passive orders are soaking it up", "volume >=1.5x usual AND >=10% net delta into the level AND price held (pierce <= half tolerance, closes on the holding side)"],
 ["Sweep / stop run", "a quick poke beyond the level to trigger stops, then back inside", "pierce >= half tolerance but the window closes back on the holding side"], ["Effort without result", "lots of volume, price barely moves = a fight is happening", "volume >=1.5x usual and window range <= 2 tolerances"],
 ["Exhaustion", "the aggressive side runs out of steam", "net delta into the level in minutes 1-3, less than half of that in minutes 4-5"], ["Stacked imbalance", "3+ adjacent prices where one side dominates: strong aggression", "bstack/sstack count >=1 on the responsive side (buyers at support, sellers at resistance)"],
 ["Delta divergence", "price makes a new low/high but cumulative delta does not confirm", "new local extreme vs the prior 30 minutes while cumulative delta stays above its low / below its high"], ["Initiative vs responsive", "initiative pushes price away from value; responsive pushes it back", "absorption and responsive stacks are responsive; sweeps that follow through are initiative"],
 ["Phantom level", "a fake level at the same distance on the other side of price", "the control: real levels must beat these"]]

def main():
    t0 = time.time(); M, Gs = (L.load_minutes(), None) if not SELFTEST else synth_sr(bool(os.environ.get("PLANT")))[:2]; M["date"] = M["date"].astype(str)
    if SELFTEST: pass
    else:
        Gs = {}
        for f in glob.glob(f"{ARCH}/prof_*.parquet") + glob.glob(f"{ARCH}/prof_*.csv.gz"):
            sym = os.path.basename(f).split("_", 1)[1].split(".")[0]; Gs[sym] = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        if not Gs: raise SystemExit("no prof_*.parquet in the archive - run of-archive (v5) first")
    print(f"{M['symbol'].nunique()} stocks with minutes, {len(Gs)} with profiles ({time.time()-t0:.0f}s)", flush=True)
    rows = []; profiles = {}; days_all = {}
    for sym, S in M.groupby("symbol", sort=False):
        if sym not in Gs or Gs[sym].empty: continue
        S = S.sort_values(["date", "hhmm"]); D = day_arrays(S); wg, H = load_hist(Gs[sym]); profiles[sym] = (H, wg); days_all[sym] = D
        if len(D) >= 12: rows += stock_events(sym, D, H, wg)
    E = pd.DataFrame(rows); print(f"{len(E):,} touches ({(E['kind']=='real').sum():,} real, {(E['kind']=='phantom').sum():,} phantom), {time.time()-t0:.0f}s", flush=True)
    if len(E) < 200: raise SystemExit("too few touches")
    F, LV, AB, cut = tables(E); MP = tomorrow_map(profiles, days_all); pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("\n== LEVELS vs RANDOM (30 min) ==\n", LV[LV["hold"] == "30 min"].round(2).to_string(index=False)); print("\n== FLAGS (30 min) ==\n", F[F["hold"] == "30 min"].drop(columns=["vs untagged CI low", "vs untagged CI high"]).round(2).to_string(index=False))
    print("\n== ABSORPTION DETAIL ==\n", AB.round(2).to_string(index=False))
    for n, d in (("flags", F), ("levels_vs_random", LV), ("absorption_detail", AB), ("level_map", MP)): d.to_csv(f"{OUT}/osr_{n}.csv", index=False)
    E.drop(columns=["ALL"]).to_csv(f"{OUT}/osr_touches.csv.gz", index=False)
    if SELFTEST: return F, LV
    import oi_probe as P; sh = P.open_sheet(); rnd = lambda d: d.round(3).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    for n, d in (("OSR Level Map", MP), ("OSR Levels vs Random", LV), ("OSR Flags", F), ("OSR Absorption Detail", AB)):
        if len(d): P.put_tab(sh, n, list(d.columns), rnd(d).values.tolist())
    P.put_tab(sh, "OSR Terms", TERMS[0], TERMS[1:]); P.put_tab(sh, "OSR Notes", ["item", "value"], [["touches real", str((E["kind"] == "real").sum())], ["touches phantom", str((E["kind"] == "phantom").sum())], ["stocks", str(E["symbol"].nunique())], ["sessions", str(E["date"].nunique())], ["holdout from", cut], ["cost bps (round trip)", str(COST)],
        ["lookback sessions for profile", str(NH)], ["trade", "fade the level: long at support / short at resistance, enter after the 5-minute window, net of cost"], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]]); print("sheet tabs written")

# ────────────────────────── self-test ──────────────────────────
def synth_sr(plant, n_sym=30, n_days=34, seed=3):
    rng = np.random.default_rng(seed); days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-08-03", periods=n_days)]; hh = np.array([(555 + m) // 60 * 100 + (555 + m) % 60 for m in range(360)]); fr = []; Gs = {}
    for k in range(n_sym):
        p = 400.0 + 10 * k; sym = f"T{k:02d}"; nodes = p * (1 + np.array([-0.01, 0.0, 0.012]))                       # heavy-volume prices that act as levels
        for d in days:
            ret = 6e-4 * rng.normal(size=360); c = p * np.cumprod(1 + ret)
            for nd in nodes: c = c + 0.0 * nd
            o = np.r_[p, c[:-1]]; p = c[-1]; near = np.min(np.abs(c[:, None] - nodes[None, :]) / c[:, None], axis=1); vol = rng.lognormal(7, 0.3, 360) * (1 + 3 * np.exp(-(near / 0.002) ** 2)); share = np.clip(0.5 + 0.02 * rng.normal(size=360), 0.05, 0.95)
            buy = vol * share; sell = vol - buy; fr.append(pd.DataFrame({"symbol": sym, "date": d, "hhmm": hh, "o": o, "h": np.maximum(o, c) * 1.0003, "l": np.minimum(o, c) * 0.9997, "c": c, "tot": vol, "buy": buy, "sell": sell, "delta": buy - sell, "cd": np.cumsum(buy - sell),
                                    "vwap": (o + c) / 2, "bstack": rng.poisson(0.2, 360), "sstack": rng.poisson(0.2, 360), "d_o": o[0], "d_c": c[-1], "d_v": vol.sum()}))
    M = pd.concat(fr, ignore_index=True)
    def prof(S):
        w = max(0.05, round(0.0005 * float(S["c"].median()) / 0.05) * 0.05); X = S.copy(); X["per"] = ((X["hhmm"] // 100) * 60 + X["hhmm"] % 100 - 555) // 30; X["bin"] = np.floor(X["c"] / w + 1e-9).astype(int)
        G = X.groupby(["date", "per", "bin"], as_index=False)[["buy", "sell"]].sum(); G["w"] = w; return G
    Gs = {s: prof(S) for s, S in M.groupby("symbol")}
    if plant:                                                                                                          # force absorption + reversal after real-level touches
        M = M.reset_index(drop=True); E0 = []
        for sym, S in M.groupby("symbol", sort=False):
            S = S.sort_values(["date", "hhmm"]); D = day_arrays(S); wg, H = load_hist(Gs[sym]); E0 += [(sym, r) for r in stock_events_raw(sym, D, H, wg)]
        idx = M.groupby(["symbol", "date"]).indices; n = 0
        for sym, r in E0:
            if r["kind"] != "real" or not r["held"]: continue
            ix = idx[(sym, r["date"])]; ix = ix[np.argsort(M["hhmm"].to_numpy()[ix])]; i = r["i"]; s = r["side"]; e = i + WIN
            if e > 340 or len(ix) < 360: continue
            w = ix[i:e]; M.loc[w, "tot"] *= 3.0; M.loc[w, "delta"] = -s * 0.3 * M.loc[w, "tot"].to_numpy(); M.loc[w, "buy"] = (M.loc[w, "tot"] + M.loc[w, "delta"]) / 2; M.loc[w, "sell"] = (M.loc[w, "tot"] - M.loc[w, "delta"]) / 2
            ramp = np.minimum(1.0, (np.arange(e, 360) - e + 1) / 60.0) * s * 0.0060; f = 1 + ramp; ww = ix[e:]
            for c_ in ("o", "h", "l", "c", "vwap"): M.loc[ww, c_] = M.loc[ww, c_].to_numpy() * f
            n += 1
        print("planted reversal after", n, "real touches")
    return M, Gs, None

def stock_events_raw(sym, D, H, wg):
    return stock_events(sym, D, H, wg)

if __name__ == "__main__":
    if SELFTEST:
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n" + "=" * 18, "SELFTEST", "PLANTED: absorption at levels reverses" if plant else "NULL", "=" * 18)
            F, LV = main(); r = F[(F["tag"] == "ABSORPTION") & (F["sample"] == "dev") & (F["hold"] == "30 min")]; a = LV[(LV["level type"] == "ALL levels") & (LV["sample"] == "dev") & (LV["hold"] == "30 min")]
            print(r[["REAL net bps", "CI low", "CI high", "PHANTOM net bps", "real - phantom", "diff CI low"]].round(2).to_string(index=False))
            if plant: assert len(r) and r.iloc[0]["CI low"] > 0 and (r.iloc[0]["diff CI low"] > 0 or r.iloc[0]["vs untagged CI low"] > 0), "planted absorption reversal not detected"
            else: assert (not len(r)) or not (r.iloc[0]["CI low"] > 0 and r.iloc[0]["diff CI low"] > 0), "false positive on null"
        print("SELFTEST OK")
    else: main()
