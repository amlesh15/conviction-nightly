# ==================================================================================
# FLOW CONFIRMATION BOARD + PAPER JOURNAL  (sits BESIDE the conviction engine - it only READS "Rule History"; nothing there is changed)
#  For every stock whose conviction score is strong (|score| >= 30) it asks what a trader would ask: does the REAL order flow agree with the story?
#   reads (each +1 agrees / -1 disagrees with the score's direction; thresholds fixed in advance):
#     5-session delta   net buy-minus-sell volume over the last 5 sessions            > +/-3% of volume
#     day delta         today's net aggressor volume                                    > +/-5%
#     trapped volume    measured: buyers who bought above the close / sellers who sold below it      > +/-10%
#     location          close vs the previous-10-session value area; +1 only if value was ACCEPTED (>=20% of the day's volume traded beyond it), -1 if the score fights value
#     value migration   today's point of control vs yesterday's (moved by more than one tolerance)
#     sweep             -1 failed breakout in the score's direction (poked the 10-day extreme, closed back inside); +1 spring (poked the other side, closed back inside)
#   state: CONFIRMED (net >= +3) / UNCONFIRMED (-1..+2) / CONTRADICTED (net <= -2, or a failed breakout while our side is the trapped side)
#  PLAN (mechanical, same rules every time): pullback entry at the nearest volume-profile level against the move, stop beyond it (never tighter than 0.25 of the usual daily range),
#   target 1 = next level that pays at least 1R, target 2 = the one after.  "Do not chase" shown when buying/selling at the close pays under 1.2R.
#  JOURNAL: every setup (CONFIRMED, UNCONFIRMED and CONTRADICTED - the last two are the control group) is logged once with its plan frozen, then scored in R after 5 sessions:
#     pullback entry (limit fills only if price comes back) and open entry (next open).  Stop is checked before target on the same day; cost 10 bps converted to R.
#     source = backfill (past sessions, plan rules fixed before the test) or live (logged the day it was produced).  A flow filter is only believed if CONFIRMED beats UNCONFIRMED/CONTRADICTED
#     after 60+ setups per group, on live rows too.
#  Tabs: "FC Board", "FC Detail", "FC Journal", "FC Scorecard", "FC Notes".    Usage: python of_flow.py [--selftest]      env: OF_ARCH, COST_BPS
# ==================================================================================
import os, sys, re, time, glob, warnings, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import of_lab as L, of_seq as S, of_sr as R

SELFTEST = "--selftest" in sys.argv; COST = L.COST; OUT = "out/of_flow"; os.makedirs(OUT, exist_ok=True); ARCH = L.ARCH
MIN_SCORE = 30; HOLD = 5; MINRISK_U = 0.25

# ────────────────────────── feature table (stock x date) ──────────────────────────
def build_features(P, profiles):
    rows = []
    for sym, g in P.groupby("symbol", sort=False):
        if sym not in profiles: continue
        H, wg = profiles[sym]; g = g.sort_values("date").reset_index(drop=True); dates = list(g["date"]); own = {}
        for d in dates:
            if d in H:
                lv = R.levels_from(H, [d], wg, False)
                if lv: own[d] = lv
        for j in range(len(g)):
            d = dates[j]; r = g.iloc[j]; prior = [x for x in dates[:j] if x in H]
            if len(prior) < R.MINP or j < 5 or d not in H: continue
            lv = R.levels_from(H, prior[-R.NH:], wg, False)
            if not lv: continue
            U = float(g["rng"].iloc[max(0, j - 9):j + 1].mean() * 100); tolb = float(np.clip(0.08 * U, 4, 20)); close = float(r["cl"]); tol = close * tolb / 1e4
            kk, vv = H[d]; px = (kk + 0.5) * wg; tot = vv.sum(); fa = float(vv[px > lv["VAH"]].sum() / tot) if tot > 0 else 0.0; fb = float(vv[px < lv["VAL"]].sum() / tot) if tot > 0 else 0.0
            loc = 1 if close > lv["VAH"] else (-1 if close < lv["VAL"] else 0); acc = (loc == 1 and fa >= 0.20) or (loc == -1 and fb >= 0.20)
            mig = 0
            if d in own and dates[j - 1] in own:
                dp = own[d]["POC"] - own[dates[j - 1]]["POC"]; mig = int(np.sign(dp)) if abs(dp) > tol else 0
            w5 = g.iloc[j - 4:j + 1]; d5 = float(w5["d"].sum() / w5["v"].sum()) if w5["v"].sum() > 0 else 0.0
            h10 = r["h10"]; l10 = r["l10"]; swp_up = bool(pd.notna(h10) and r["hi"] > h10 and close < h10); swp_dn = bool(pd.notna(l10) and r["lo"] < l10 and close > l10)
            rows.append({"symbol": sym, "date": d, "close": close, "hi": float(r["hi"]), "lo": float(r["lo"]), "U": U, "tol_bps": tolb, "dds": float(r["dds"]), "trapnet": float(r["trapnet"]), "d5": d5, "loc": loc, "accepted": bool(acc), "frac_above": fa, "frac_below": fb,
                         "mig": mig, "swp_up": swp_up, "swp_dn": swp_dn, "poc10": lv["POC"], "vah10": lv["VAH"], "val10": lv["VAL"], "v": float(r["v"]), "ret": float(r["ret"]) if pd.notna(r["ret"]) else 0.0})
    return pd.DataFrame(rows)

def read_points(f, s):
    """f: feature row; s: +1 long / -1 short.  returns net, list of (label, point)"""
    pts = []
    x = s * f["d5"]; pts.append(("5-session delta " + ("with" if x > 0.03 else "against" if x < -0.03 else "flat"), 1 if x > 0.03 else -1 if x < -0.03 else 0))
    x = s * f["dds"]; pts.append(("day delta " + ("with" if x > 0.05 else "against" if x < -0.05 else "flat"), 1 if x > 0.05 else -1 if x < -0.05 else 0))
    x = s * f["trapnet"]; pts.append(("opponents trapped" if x > 0.10 else "OUR side trapped" if x < -0.10 else "no clear trap", 1 if x > 0.10 else -1 if x < -0.10 else 0))
    lp = 1 if (s * f["loc"] == 1 and f["accepted"]) else (-1 if s * f["loc"] == -1 else 0)
    pts.append(("close beyond value, accepted" if lp == 1 else "score fights value (close on the wrong side of the value area)" if lp == -1 else "close inside / not accepted", lp))
    m = s * f["mig"]; pts.append(("value migrating with" if m == 1 else "value migrating against" if m == -1 else "value flat", int(m)))
    failed = (s > 0 and f["swp_up"]) or (s < 0 and f["swp_dn"]); spring = (s > 0 and f["swp_dn"]) or (s < 0 and f["swp_up"])
    pts.append(("FAILED breakout in score direction" if failed else "spring / sweep reclaimed against score" if spring else "no sweep", -1 if failed else 1 if spring else 0))
    net = sum(p for _, p in pts); veto = failed and (s * f["trapnet"] < -0.10)
    state = "CONTRADICTED" if (net <= -2 or veto) else ("CONFIRMED" if net >= 3 else "UNCONFIRMED"); return net, pts, state

# ────────────────────────── plan ──────────────────────────
def make_plan(f, s, lv_list):
    """lv_list: deduped [(type, price, conf)] as of the end of the signal session.  returns dict of frozen plan numbers"""
    c = f["close"]; U = f["U"]; tol = c * f["tol_bps"] / 1e4; minrisk = MINRISK_U * U / 1e4 * c
    if s > 0: cand = sorted([(p, t) for t, p, _ in lv_list if p < c - tol and p > c * (1 - 0.6 * U / 1e4)], reverse=True)
    else: cand = sorted([(p, t) for t, p, _ in lv_list if p > c + tol and p < c * (1 + 0.6 * U / 1e4)])
    if cand: lvl, lt = cand[0]; entry = lvl + s * 0.5 * tol; edge = lvl - s * 0.5 * tol; basis = lt
    else: entry = c - s * 0.3 * U / 1e4 * c; edge = entry - s * minrisk; basis = "0.3 range pullback"
    risk = max(s * (entry - edge), minrisk); stop = entry - s * risk
    tg = sorted([p for t, p, _ in lv_list if s * (p - entry) >= 1.0 * risk], key=lambda p: s * (p - entry))
    t1 = tg[0] if tg else entry + s * 1.5 * risk; t2 = tg[1] if len(tg) > 1 else t1 + s * risk
    r1 = s * (t1 - entry) / risk; r2 = s * (t2 - entry) / risk; chase_r = s * (t1 - c) / max(s * (c - stop), 1e-9)
    return {"entry": entry, "zone_lo": min(entry, entry - s * 0.5 * tol), "zone_hi": max(entry, entry - s * 0.5 * tol), "stop": stop, "t1": t1, "t2": t2, "risk_bps": risk / entry * 1e4, "R_t1": r1, "R_t2": r2, "R_t1_at_close": chase_r, "entry_basis": basis}

def levels_end_of(sym, d, g, profiles):
    H, wg = profiles[sym]; dates = list(g["date"]); j = dates.index(d); prior = [x for x in dates[:j + 1] if x in H]; r = g.iloc[j]; c = float(r["cl"]); U = float(g["rng"].iloc[max(0, j - 9):j + 1].mean() * 100); tolb = float(np.clip(0.08 * U, 4, 20))
    return R.dedupe(R.day_levels(prior, H, wg, (float(r["hi"]), float(r["lo"]))), c * tolb / 1e4)

# ────────────────────────── trade simulation (daily bars, conservative) ──────────────────────────
def sim_trade(op, hi, lo, cl, s, entry, stop, t1, mode):
    """returns (status, R_net, filled). bars are the sessions AFTER the signal session (up to HOLD).
    pullback: limit order fills when price trades back to the entry; R unit = planned risk (entry-stop); on the FILL day only the stop is checked (the target may have printed before the fill);
              a gap open through the stop = skipped.
    open: enter at the next open; R unit = actual risk (open-stop); skipped if gapped past the stop or if less than 1R is left to target 1 (the 'do not chase' rule)."""
    n = min(len(op), HOLD)
    if n == 0: return "open", np.nan, False
    risk0 = s * (entry - stop)
    if risk0 <= 0: return "bad plan", np.nan, False
    k0 = None; ep = None
    if mode == "open":
        k0 = 0; ep = op[0]; risk = s * (ep - stop)
        if risk < 0.2 * risk0: return "skipped (gapped past stop)", np.nan, False
        if s * (t1 - ep) < 1.0 * risk: return "skipped (no chase: under 1R to target)", np.nan, False
    else:
        for k in range(n):
            if (s > 0 and lo[k] <= entry) or (s < 0 and hi[k] >= entry): k0 = k; ep = (min(op[k], entry) if s > 0 else max(op[k], entry)); break
        if k0 is None: return ("no fill" if len(op) >= HOLD else "open"), np.nan, False
        if s * (ep - stop) <= 0.2 * risk0: return "skipped (gapped past stop)", np.nan, False
        risk = risk0
    cost_R = COST * 1e-4 * ep / risk
    for k in range(k0, n):
        o, h, l = op[k], hi[k], lo[k]
        if k > k0 or (mode == "open" and k > 0):
            if (s > 0 and o <= stop) or (s < 0 and o >= stop): return "stopped", s * (o - ep) / risk - cost_R, True
            if (s > 0 and o >= t1) or (s < 0 and o <= t1): return "target", s * (o - ep) / risk - cost_R, True
        if (s > 0 and l <= stop) or (s < 0 and h >= stop): return "stopped", s * (stop - ep) / risk - cost_R, True
        if k == k0 and mode == "pullback": continue
        if (s > 0 and h >= t1) or (s < 0 and l <= t1): return "target", s * (t1 - ep) / risk - cost_R, True
    if len(op) >= HOLD: return "time exit", s * (cl[n - 1] - ep) / risk - cost_R, True
    return "open", np.nan, True

def outcomes(J, P):
    byd = {sym: g.sort_values("date").reset_index(drop=True) for sym, g in P.groupby("symbol")}; res = []
    for _, r in J.iterrows():
        g = byd.get(r["symbol"]); out = {"status_pullback": "", "R_pullback": np.nan, "status_open": "", "R_open": np.nan}
        if g is not None:
            nxt = g[g["date"] > r["date"]].head(HOLD); op, hi, lo, cl = (nxt[c].to_numpy(float) for c in ("op", "hi", "lo", "cl"))
            for mode in ("pullback", "open"): st, rr, _ = sim_trade(op, hi, lo, cl, int(r["bias"]), float(r["entry"]), float(r["stop"]), float(r["t1"]), mode); out[f"status_{mode}"] = st; out[f"R_{mode}"] = rr
        res.append(out)
    return pd.concat([J.reset_index(drop=True), pd.DataFrame(res)], axis=1)

# ────────────────────────── scorecard ──────────────────────────
def scorecard(J):
    rows = []; J = J.copy(); J["group"] = J["state"]
    allr = J.copy(); allr["group"] = "ALL strong scores (no flow filter)"; X = pd.concat([J, allr])
    for srcname, sel in (("all rows", X["source"].notna()), ("live only", X["source"] == "live"), ("backfill only", X["source"] == "backfill")):
        for grp in ("CONFIRMED", "UNCONFIRMED", "CONTRADICTED", "ALL strong scores (no flow filter)"):
            for mode in ("pullback", "open"):
                G = X[sel & (X["group"] == grp)]; st = G[f"status_{mode}"]; done = G[st.isin(["stopped", "target", "time exit"])]; v = done[f"R_{mode}"].to_numpy(float)
                if len(G) == 0: continue
                m, lo, hi, p, nd = L.cmean(v, done["date"].to_numpy()) if len(v) >= 30 else (np.nanmean(v) if len(v) else np.nan, np.nan, np.nan, np.nan, done["date"].nunique())
                rows.append({"rows": srcname, "group": grp, "entry": "pullback limit" if mode == "pullback" else "next open", "setups": len(G), "filled/triggered": int(st.isin(["stopped", "target", "time exit", "open"]).sum()), "fill %": st.isin(["stopped", "target", "time exit", "open"]).mean() * 100, "resolved": len(done), "mean R (net of cost)": m, "CI low": lo, "CI high": hi, "hit %": (v > 0).mean() * 100 if len(v) else np.nan,
                             "target %": (done[f"status_{mode}"] == "target").mean() * 100 if len(done) else np.nan, "stopped %": (done[f"status_{mode}"] == "stopped").mean() * 100 if len(done) else np.nan,
                             "read": ("too few (<60 resolved)" if len(done) < 60 else ("positive" if (not np.isnan(lo) and lo > 0) else "negative" if (not np.isnan(hi) and hi < 0) else "not distinguishable from 0"))})
    SC = pd.DataFrame(rows)
    d = []
    for srcname, sel in (("all rows", J["source"].notna()), ("live only", J["source"] == "live")):
        for mode in ("pullback", "open"):
            A = J[sel & (J["state"] == "CONFIRMED") & J[f"status_{mode}"].isin(["stopped", "target", "time exit"])]; B = J[sel & (J["state"].isin(["CONTRADICTED", "UNCONFIRMED"])) & J[f"status_{mode}"].isin(["stopped", "target", "time exit"])]
            if len(A) >= 30 and len(B) >= 30:
                m, lo, hi = L.cdiff(A[f"R_{mode}"].to_numpy(float), A["date"].to_numpy(), B[f"R_{mode}"].to_numpy(float), B["date"].to_numpy())
                d.append({"rows": srcname, "entry": "pullback limit" if mode == "pullback" else "next open", "CONFIRMED setups": len(A), "others": len(B), "CONFIRMED minus others (R)": m, "CI low": lo, "CI high": hi, "read": "flow filter HELPS" if lo > 0 else ("flow filter HURTS" if hi < 0 else "no proven difference")})
    return SC, pd.DataFrame(d)

# ────────────────────────── board text ──────────────────────────
def fmt(x): return f"{x:,.2f}" if x < 1000 else f"{x:,.1f}"

def plan_text(f, s, plan, state, pts):
    side = "LONG" if s > 0 else "SHORT"
    if state == "CONTRADICTED": return f"SKIP - flow disagrees ({', '.join(l for l, p in pts if p < 0) or 'net negative'})"
    t = f"{side}: {'buy' if s > 0 else 'sell'} pullback {fmt(plan['zone_lo'])}-{fmt(plan['zone_hi'])} ({plan['entry_basis']}); stop {fmt(plan['stop'])}; T1 {fmt(plan['t1'])} ({plan['R_t1']:.1f}R), T2 {fmt(plan['t2'])} ({plan['R_t2']:.1f}R)"
    if plan["R_t1_at_close"] < 1.2: t += f". DO NOT CHASE at {fmt(f['close'])}: only {plan['R_t1_at_close']:.1f}R to T1"
    if state == "UNCONFIRMED": t += ". Flow not confirming: half size or wait for the delta to turn"
    return t

# ────────────────────────── main ──────────────────────────
def load_all():
    M = L.load_minutes(); M["date"] = M["date"].astype(str); M["d_c"] = M["d_c"].fillna(M.groupby(["symbol", "date"])["c"].transform("last")); Gs = {}
    for f in glob.glob(f"{ARCH}/prof_*.parquet") + glob.glob(f"{ARCH}/prof_*.csv.gz"):
        sym = os.path.basename(f).split("_", 1)[1].split(".")[0]; Gs[sym] = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
    if not Gs: raise SystemExit("no prof_*.parquet in the archive - run of-archive (v5) first")
    return M, Gs

def run_pipeline(M, Gs, hist, J_old, run_date):
    P, cut = S.fight_cards(M); profiles = {s: R.load_hist(G)[::-1] for s, G in Gs.items() if len(G)}; F = build_features(P, profiles); latest = P["date"].max()
    F["score"] = [hist.get(s, {}).get(d[5:10], np.nan) for s, d in zip(F["symbol"], F["date"])]; Fs = F[F["score"].abs() >= MIN_SCORE].copy()
    byd = {sym: g.sort_values("date").reset_index(drop=True) for sym, g in P.groupby("symbol")}; have = set(J_old["setup_id"]) if len(J_old) else set(); new = []
    for _, f in Fs.iterrows():
        sid = f"{f['date']}_{f['symbol']}"
        if sid in have: continue
        s = 1 if f["score"] > 0 else -1; net, pts, state = read_points(f, s); lv = levels_end_of(f["symbol"], f["date"], byd[f["symbol"]], profiles); plan = make_plan(f, s, lv)
        new.append({"setup_id": sid, "source": "live" if f["date"] == latest else "backfill", "date": f["date"], "symbol": f["symbol"], "bias": s, "score": f["score"], "state": state, "net": net, "reads": "; ".join(f"{l} ({'+' if p > 0 else ''}{p})" for l, p in pts), "close": f["close"],
                    **{k: plan[k] for k in ("entry", "stop", "t1", "t2", "risk_bps", "R_t1", "R_t2", "R_t1_at_close", "entry_basis")}, "first_logged": run_date})
    J = pd.concat([J_old, pd.DataFrame(new)], ignore_index=True) if len(new) else J_old.copy()
    for c in ("bias", "score", "net", "close", "entry", "stop", "t1", "t2", "risk_bps", "R_t1", "R_t2", "R_t1_at_close"): J[c] = pd.to_numeric(J[c], errors="coerce")
    J = J.drop(columns=[c for c in ("status_pullback", "R_pullback", "status_open", "R_open") if c in J.columns]); J = outcomes(J, P).sort_values(["date", "symbol"], ascending=[False, True]).reset_index(drop=True)
    # board for the latest session
    B = []
    for _, f in F[F["date"] == latest].iterrows():
        sc = f["score"]
        if not (abs(sc) >= MIN_SCORE): continue
        s = 1 if sc > 0 else -1; net, pts, state = read_points(f, s); sid = f"{latest}_{f['symbol']}"; jr = J[J["setup_id"] == sid]
        if len(jr): j = jr.iloc[0]; plan = {k: j[k] for k in ("entry", "stop", "t1", "t2", "risk_bps", "R_t1", "R_t2", "R_t1_at_close", "entry_basis")}; plan["zone_lo"] = min(j["entry"], j["entry"] - s * 0.5 * f["close"] * f["tol_bps"] / 1e4); plan["zone_hi"] = max(j["entry"], j["entry"] - s * 0.5 * f["close"] * f["tol_bps"] / 1e4)
        else: continue
        B.append({"symbol": f["symbol"], "bias": "LONG" if s > 0 else "SHORT", "score": round(sc, 1), "flow state": state, "net reads": net, "plan": plan_text(f, s, plan, state, pts), "reads": "; ".join(f"{l} ({'+' if p > 0 else ''}{p})" for l, p in pts), "close": round(f["close"], 2),
                  "entry": round(plan["entry"], 2), "stop": round(plan["stop"], 2), "T1": round(plan["t1"], 2), "T2": round(plan["t2"], 2), "R to T1": round(plan["R_t1"], 2), "R at close": round(plan["R_t1_at_close"], 2), "risk bps": round(plan["risk_bps"], 1),
                  "usual daily range %": round(f["U"] / 100, 2), "cost as % of range": round(COST / f["U"] * 100, 1), "5d delta %": round(f["d5"] * 100, 1), "day delta %": round(f["dds"] * 100, 1), "trapped %": round(f["trapnet"] * 100, 1), "session": latest})
    Bd = pd.DataFrame(B)
    if len(Bd): Bd["_o"] = Bd["flow state"].map({"CONFIRMED": 0, "UNCONFIRMED": 1, "CONTRADICTED": 2}); Bd = Bd.sort_values(["_o", "score"], key=lambda c: c if c.name == "_o" else -c.abs()).drop(columns="_o")
    Dt = F[F["date"] == latest].drop(columns=["hi", "lo"]).round(4); return J, Bd, Dt, latest, P

NOTES = [["item", "meaning"], ["what this is", "a separate engine beside the conviction engine. It reads Rule History scores only and never changes them"], ["states", "CONFIRMED: real order flow agrees with the score (net reads >= +3). UNCONFIRMED: mixed. CONTRADICTED: flow disagrees (net <= -2 or failed breakout with our side trapped)"],
 ["reads", "5-session delta, day delta, measured trapped volume, location vs the previous 10-session value area (accepted or not), value migration, sweep/failed breakout - thresholds fixed in advance (see code header)"],
 ["plan", "pullback limit at the nearest profile level against the move; stop beyond it but never tighter than 0.25 of the usual daily range; T1 = next level paying >= 1R; T2 after that"], ["R", "one unit of risk = distance from entry to stop. +2R = made twice what was risked. Cost (10 bps) is deducted"],
 ["scoring", "5 sessions after the signal; daily bars; if stop and target are both touched on a day the stop is assumed first; gap through the stop exits at the open"], ["backfill vs live", "backfill = past sessions replayed with the same fixed rules (not a forward test). live = logged the day it was produced. Only live rows are real evidence"],
 ["judging", "trust the flow filter only if CONFIRMED beats the others by a clear margin after 60+ resolved setups per group, and live rows agree with backfill"], ["not a signal to trade blindly", "paper-trading research; earlier tests found no statistically proven flow edge on 57 sessions"]]

def main():
    t0 = time.time()
    if SELFTEST: M, Gs, hist = synth_flow(); J_old = pd.DataFrame(); sh = None
    else:
        M, Gs = load_all(); import oi_probe as P_; sh = P_.open_sheet(); rh = sh.worksheet("Rule History").get_all_values(); hdr = [h.strip() for h in rh[0]]; si = hdr.index("symbol") if "symbol" in hdr else 0
        cols = [(i, h) for i, h in enumerate(hdr) if re.fullmatch(r"\d{2}-\d{2}", h)]; hist = {}
        for r in rh[1:]:
            if si < len(r) and r[si].strip():
                d = {}
                for i, h in cols:
                    if i < len(r) and re.fullmatch(r"-?\d+(\.\d+)?", r[i].strip()): d[h] = float(r[i])
                hist[r[si].strip().upper()] = d
        try:
            v = sh.worksheet("FC Journal").get_all_values(); J_old = pd.DataFrame(v[1:], columns=v[0]) if len(v) > 1 and "setup_id" in v[0] else pd.DataFrame()
        except Exception: J_old = pd.DataFrame()
    run_date = dt.datetime.now().strftime("%Y-%m-%d"); print(f"{M['symbol'].nunique()} stocks, {len(Gs)} profiles ({time.time()-t0:.0f}s); journal rows before: {len(J_old)}", flush=True)
    J, Bd, Dt, latest, P = run_pipeline(M, Gs, hist, J_old, run_date); SC, DF = scorecard(J)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 120)
    print(f"\nlatest session {latest}: {len(Bd)} strong-score names on the board; journal {len(J)} rows")
    if len(Bd): print(Bd[["symbol", "bias", "score", "flow state", "net reads", "plan"]].head(15).to_string(index=False))
    print("\n== SCORECARD ==\n", SC[SC["rows"] == "all rows"].round(2).to_string(index=False)); print("\n== CONFIRMED vs others ==\n", DF.round(2).to_string(index=False) if len(DF) else "(too few resolved setups)")
    for n, d in (("board", Bd), ("journal", J), ("scorecard", SC), ("detail", Dt)): d.to_csv(f"{OUT}/fc_{n}.csv", index=False)
    if SELFTEST: return J, Bd, SC
    rnd = lambda d: d.round(4).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    for n, d in (("FC Board", Bd), ("FC Detail", Dt), ("FC Journal", J), ("FC Scorecard", pd.concat([SC, pd.DataFrame([{}]), DF], ignore_index=True))):
        if len(d): P_.put_tab(sh, n, list(d.columns), rnd(d).values.tolist())
    P_.put_tab(sh, "FC Notes", NOTES[0], NOTES[1:] + [["latest session", latest], ["journal rows", str(len(J))], ["cost bps", str(COST)], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]]); print("sheet tabs written")

# ────────────────────────── self-test ──────────────────────────
def unit_tests():
    op = np.array([100., 101, 99, 102, 104]); hi = np.array([102., 103, 101, 105, 106]); lo = np.array([99., 100, 97, 101, 103]); cl = np.array([101., 102, 100, 104, 105])
    st, r, f = sim_trade(op, hi, lo, cl, 1, 99.5, 96.0, 105.0, "pullback"); assert st == "target" and f, (st, r)                  # fills day0 low 99<=99.5, target 105 day3 high 105
    st, r, f = sim_trade(op, hi, lo, cl, 1, 99.5, 99.0, 110.0, "pullback"); assert st == "stopped" and r < -0.9, (st, r)         # day0 low 99 touches stop 99 right after fill
    st, r, f = sim_trade(op, hi, lo, cl, 1, 90.0, 85.0, 110.0, "pullback"); assert st == "no fill", st
    st, r, f = sim_trade(op, hi, lo, cl, -1, 104.0, 106.5, 98.0, "open"); assert st.startswith("skipped"), st
    st, r, f = sim_trade(op, hi, lo, cl, 1, 100.0, 90.0, 120.0, "open"); assert st == "time exit" and abs(r - ((105 - 100) / 10 - COST * 1e-4 * 100 / 10)) < 1e-6, (st, r)
    f = {"d5": 0.06, "dds": 0.10, "trapnet": 0.2, "loc": 1, "accepted": True, "mig": 1, "swp_up": False, "swp_dn": False}
    net, pts, state = read_points(f, 1); assert state == "CONFIRMED" and net == 5, (net, state)
    net, pts, state = read_points(f, -1); assert state == "CONTRADICTED", (net, state)
    f2 = dict(f, swp_up=True, trapnet=-0.2, d5=0.0, dds=0.0, loc=0, mig=0); net, pts, state = read_points(f2, 1); assert state == "CONTRADICTED", (net, state)   # failed breakout with our side trapped = veto
    st, r, f = sim_trade(np.array([100., 100, 100]), np.array([103., 100, 100]), np.array([96., 100, 100]), np.array([100., 100, 100]), 1, 98.0, 94.0, 102.0, "pullback"); assert st == "time exit" or st == "open", st   # fill day: target ignored
    st, r, f = sim_trade(np.array([90., 100, 100]), np.array([91., 100, 100]), np.array([89., 100, 100]), np.array([90., 100, 100]), 1, 98.0, 85.0, 120.0, "pullback"); assert abs(r - ((120 - 90) / 13)) < 5 or st != "stopped", (st, r)
    st, r, f = sim_trade(np.array([84., 100, 100]), np.array([86., 100, 100]), np.array([83., 100, 100]), np.array([84., 100, 100]), 1, 98.0, 85.0, 120.0, "pullback"); assert st.startswith("skipped"), st   # gap through stop
    print("unit tests ok")

def synth_flow(plant=False):
    M, Gs, _ = R.synth_sr(False, n_sym=20, n_days=40, seed=9); dates = sorted(M["date"].unique()); rng = np.random.default_rng(1); hist = {}
    for s in M["symbol"].unique(): hist[s] = {d[5:10]: float(rng.choice([-60, -40, 0, 40, 60])) for d in dates}
    return M, Gs, hist

if __name__ == "__main__":
    if SELFTEST:
        unit_tests(); J, Bd, SC = main()
        assert len(J) > 50 and {"UNCONFIRMED", "CONTRADICTED"} <= set(J["state"]), "journal looks wrong"
        ok = J[J["status_pullback"].isin(["stopped", "target", "time exit"])]; assert ok["R_pullback"].between(-6, 12).all(), "R out of range"
        assert (J[["entry", "stop"]].notna().all().all()) and (J["bias"].abs() == 1).all(); print("SELFTEST OK")
    else: main()
