# ==================================================================================
# CONVICTION x ORDER FLOW  (research only - no orders)
#  Part A  DAILY : the conviction engine's signed score for a stock on day D  +  that day's order flow (from the archive)
#          -> does flow ADD to the score for the NEXT day?  Does it help when it AGREES with the score and warn when it CONFLICTS?
#          Next-day trade = enter at next open, exit next close (official daily candle); abnormal = minus same-day average of all stocks; cost 10 bps.
#  Part B  TRIGGERS : each intraday "Backtest Triggers" entry + the order flow in the 15 minutes BEFORE the trigger
#          -> does trading only triggers whose flow agrees beat trading all of them?  (return = side-signed r15/r30/r60/rEOD minus cost)
#  Statistics: day-cluster bootstrap, locked holdout (last 20% of sessions), results shown for dev AND holdout, null/planted self-tests.
#  Tabs written: "OFC Daily IC", "OFC Daily Buckets", "OFC Trigger Filter", "OFC Notes".
#  Usage: python of_conv.py [--selftest]      env: OF_ARCH, COST_BPS (10)
# ==================================================================================
import os, sys, re, time, warnings, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import of_lab as L

SELFTEST = "--selftest" in sys.argv; COST = L.COST; OUT = "out/of_conv"; os.makedirs(OUT, exist_ok=True); STRONG = 30

def cm(v, B_=2000, seed=5):
    """bootstrap mean over independent units (days); needs >= 6 units"""
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) < 6: return np.nan, np.nan, np.nan, len(v)
    r = np.random.default_rng(seed); bs = v[r.integers(0, len(v), (B_, len(v)))].mean(axis=1); return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(v)

# ────────────────────────── daily panel ──────────────────────────
def day_panel(M):
    """one row per stock-day from the minute archive: official open/close, flow summaries (09:15-15:14)"""
    M = M.copy(); M["late"] = M["hhmm"] >= 1345
    g = M.groupby(["symbol", "date"], sort=True)
    P = g.agg(d_o=("d_o", "first"), d_c=("d_c", "first"), d_v=("d_v", "first"), v=("tot", "sum"), delta=("delta", "sum"), o1=("o", "first"), cl=("c", "last")).reset_index()
    L_ = M[M["late"]].groupby(["symbol", "date"]).agg(ld=("delta", "sum"), lv=("tot", "sum")).reset_index(); P = P.merge(L_, on=["symbol", "date"], how="left")
    P["dds"] = P["delta"] / P["v"].replace(0, np.nan); P["late_s"] = P["ld"] / P["lv"].replace(0, np.nan); P = P.sort_values(["symbol", "date"]).reset_index(drop=True)
    gs = P.groupby("symbol", sort=False); P["n_o"] = gs["d_o"].shift(-1); P["n_c"] = gs["d_c"].shift(-1); P["oc"] = (P["n_c"] / P["n_o"] - 1) * 1e4; P["cc"] = (P["n_c"] / P["d_c"] - 1) * 1e4
    P["ret_d"] = (P["d_c"] / gs["d_c"].shift(1) - 1) * 1e4
    for c in ("oc", "cc"): P[c + "x"] = P[c] - P.groupby("date")[c].transform("mean")
    P["dds_r"] = P.groupby("date")["dds"].rank(pct=True) - 0.5; P["late_r"] = P.groupby("date")["late_s"].rank(pct=True) - 0.5
    return P

def attach_score(P, hist):
    """hist: symbol -> {MM-DD: score}; map to archive dates by month-day"""
    P = P.copy(); md = P["date"].str[5:10]; P["score"] = [hist.get(s, {}).get(m, np.nan) for s, m in zip(P["symbol"], md)]; return P

def zr(df, col): return df.groupby("date")[col].rank(pct=True) - 0.5

def daily_ic(P):
    D = P.dropna(subset=["score", "dds", "ocx"]).copy(); dates = sorted(D["date"].unique()); cut = dates[int(len(dates) * 0.8)]; D["holdout"] = D["date"] >= cut
    D["r_score"] = zr(D, "score"); D["r_flow"] = D["dds_r"]; D["r_late"] = D["late_r"].fillna(0)
    D["combo"] = D["r_score"] + D["r_flow"]; D["combo_late"] = D["r_score"] + 0.5 * D["r_flow"] + 0.5 * D["r_late"]
    rows = []; per = {}
    for tgt, tl in (("ocx", "next-day open->close"), ("ccx", "next-day close->close")):
        D["r_y"] = D.groupby("date")[tgt].rank(pct=True) - 0.5
        for f in ("r_score", "r_flow", "r_late", "combo", "combo_late"):
            per[(tgt, f)] = L.gcorr(D[f], D["r_y"], D["date"])
        # partial IC of flow given score
        a = per[(tgt, "r_flow")]; b = per[(tgt, "r_score")]; rsf = L.gcorr(D["r_flow"], D["r_score"], D["date"]); per[(tgt, "partial flow|score")] = (a - rsf * b) / np.sqrt((1 - rsf ** 2).clip(lower=1e-6) * (1 - b ** 2).clip(lower=1e-6))
        for part, sel in (("dev", lambda i: i < cut), ("holdout", lambda i: i >= cut)):
            for name in ("r_score", "r_flow", "r_late", "combo", "combo_late", "partial flow|score"):
                s = per[(tgt, name)]; s = s[sel(s.index)]; m, lo, hi, n = cm(s.to_numpy()); rows.append({"target": tl, "sample": part, "signal": name, "mean rank IC": m, "CI low": lo, "CI high": hi, "days": n})
            s = (per[(tgt, "combo")] - per[(tgt, "r_score")]); s = s[sel(s.index)]; m, lo, hi, n = cm(s.to_numpy())
            rows.append({"target": tl, "sample": part, "signal": "combo minus score alone", "mean rank IC": m, "CI low": lo, "CI high": hi, "days": n})
    R = pd.DataFrame(rows); R["read"] = np.where(R["CI low"] > 0, "positive", np.where(R["CI high"] < 0, "negative", "not distinguishable from 0")); return R, D

def daily_buckets(D):
    rows = []; S = D[D["score"].abs() >= STRONG].copy(); S["dirn"] = np.sign(S["score"]); S["fl"] = np.sign(S["dds_r"]) * S["dirn"]
    S["flow vs score"] = np.where(S["dds_r"].abs() < 0.15, "flow neutral", np.where(S["fl"] > 0, "flow AGREES", "flow CONFLICTS"))
    for tgt, tl in (("ocx", "next-day open->close"), ("ccx", "next-day close->close")):
        S["tr"] = S["dirn"] * S[tgt]
        for part, G in (("dev", S[~S["holdout"]]), ("holdout", S[S["holdout"]]), ("all", S)):
            for lab, side in (("all strong", None), ("CALL side (score>=30)", 1), ("PUT side (score<=-30)", -1)):
                H = G if side is None else G[G["dirn"] == side]
                for k in ("all", "flow AGREES", "flow CONFLICTS", "flow neutral"):
                    X = H if k == "all" else H[H["flow vs score"] == k]
                    if len(X) < 10: continue
                    m, lo, hi, p, nd = L.cmean(X["tr"].to_numpy(), X["date"].to_numpy()); rows.append({"target": tl, "sample": part, "side": lab, "flow": k, "trades": len(X), "days": nd, "gross bps": m, "net bps (after cost)": m - COST, "net CI low": lo - COST, "net CI high": hi - COST, "hit %": (X["tr"] > 0).mean() * 100})
    B = pd.DataFrame(rows); B["read"] = np.where(B["net CI low"] > 0, "NET POSITIVE", np.where(B["net CI high"] < 0, "net negative", "not distinguishable from 0")); return B

# ────────────────────────── triggers ──────────────────────────
def bar_flow(M):
    """running flow state at every 5-min bar end: last-3-bar delta share, day delta share so far, relative volume"""
    B = L.make_bars(M); B["date"] = B["date"].astype(str); g = B.groupby(["symbol", "date"], sort=False)
    B["d3"] = g["delta"].transform(lambda s: s.rolling(3, min_periods=1).sum()) / g["v"].transform(lambda s: s.rolling(3, min_periods=1).sum()).replace(0, np.nan)
    B["dday"] = g["delta"].cumsum() / g["v"].cumsum().replace(0, np.nan); B["endmin"] = 555 + (B["bar"] + 1) * 5; return B[["symbol", "date", "endmin", "d3", "dday"]]

def parse_trig(vals):
    h = next(i for i, r in enumerate(vals) if r and r[0].strip().lower() == "session"); hdr = [x.strip() for x in vals[h]]; T = pd.DataFrame([r + [""] * (len(hdr) - len(r)) for r in vals[h + 1:] if r and r[0].strip()], columns=hdr)
    for c in ("score", "r15", "r30", "r60", "rEOD", "imb", "rel_vol", "ret", "vs_vwap"):
        if c in T.columns: T[c] = pd.to_numeric(T[c].astype(str).str.replace(",", ""), errors="coerce")
    T["session"] = T["session"].astype(str).str[:10]; T["m"] = [int(t[:2]) * 60 + int(t[3:5]) if isinstance(t, str) and ":" in t else np.nan for t in T["time"]]; return T

def trigger_filter(T, BF):
    T = T.copy(); BF = BF.sort_values("endmin"); out = []
    for s, g in T.groupby("symbol"):
        b = BF[BF["symbol"] == s]
        if b.empty: continue
        # per-session as-of join (as-of must stay inside the same session)
        for sess, gg in g.groupby("session"):
            bb = b[b["date"] == sess].sort_values("endmin")
            if bb.empty: continue
            gg = gg.sort_values("m").copy(); idx = np.searchsorted(bb["endmin"].to_numpy(), gg["m"].to_numpy(), side="right") - 1; ok = idx >= 0
            gg["d3"] = np.where(ok, bb["d3"].to_numpy()[np.clip(idx, 0, None)], np.nan); gg["dday"] = np.where(ok, bb["dday"].to_numpy()[np.clip(idx, 0, None)], np.nan); out.append(gg)
    if not out: return pd.DataFrame()
    X = pd.concat(out, ignore_index=True); sgn = np.where(X["side"] == "PUT", -1.0, 1.0); X["a3"] = sgn * X["d3"]; X["aday"] = sgn * X["dday"]; X["flow15"] = np.where(X["a3"] > 0.05, "flow AGREES (last 15m)", np.where(X["a3"] < -0.05, "flow CONFLICTS (last 15m)", "flow neutral"))
    X["flowday"] = np.where(X["aday"] > 0.03, "day flow AGREES", np.where(X["aday"] < -0.03, "day flow CONFLICTS", "day flow neutral")); dates = sorted(X["session"].unique()); cut = dates[int(len(dates) * 0.8)]; X["holdout"] = X["session"] >= cut; return X

def trigger_table(X):
    rows = []
    for hz in ("r15", "r30", "r60", "rEOD"):
        if hz not in X.columns: continue
        X["tr"] = np.where(X["side"] == "PUT", -1.0, 1.0) * X[hz] * 100          # bps, side-signed (r columns are % raw spot return)
        for part, G in (("dev", X[~X["holdout"]]), ("holdout", X[X["holdout"]]), ("all", X)):
            for fcol in ("flow15", "flowday"):
                for k in (["all"] if fcol == "flow15" else []) + sorted(G[fcol].dropna().unique()):
                    H = G if k == "all" else G[G[fcol] == k]
                    if len(H) < 15: continue
                    m, lo, hi, p, nd = L.cmean(H["tr"].to_numpy(), H["session"].to_numpy()); rows.append({"horizon": hz, "sample": part, "flow measure": fcol if k != "all" else "none", "group": k, "triggers": len(H), "sessions": nd, "gross bps": m, "net bps (after cost)": m - COST, "net CI low": lo - COST, "net CI high": hi - COST, "hit %": (H["tr"] > 0).mean() * 100})
    R = pd.DataFrame(rows)
    if len(R):
        base = R[R["group"] == "all"].set_index(["horizon", "sample"])["net bps (after cost)"]; R["vs all triggers (bps)"] = [r["net bps (after cost)"] - base.get((r["horizon"], r["sample"]), np.nan) for _, r in R.iterrows()]
        R["read"] = np.where(R["net CI low"] > 0, "NET POSITIVE", np.where(R["net CI high"] < 0, "net negative", "not distinguishable from 0"))
    return R

NOTES = [["term", "meaning"], ["score", "conviction engine signed score (-100..+100) from the daily Rule History for that stock-day (known at the close)"],
 ["flow (daily)", "day delta share = (buy - sell volume)/volume over 09:15-15:14, ranked across stocks the same day; late = last 90 minutes"],
 ["flow AGREES", "flow points the same way as the score (buying with a bullish score, selling with a bearish one)"],
 ["abnormal", "stock's return minus the average of all stocks that day; the next-day trade is open->close with official candles"],
 ["IC", "rank correlation between the signal and next-day abnormal return, one number per day, then averaged with a day-resampled CI"],
 ["partial flow|score", "what flow adds AFTER removing what the score already says"],
 ["trigger filter", "return of trading only triggers whose last-15-minute / day flow agrees with the trigger side; r columns are raw spot returns, sign by side, cost deducted"],
 ["limits", "about 55 sessions of flow; holdout is the last 20% of sessions; read dev AND holdout"]]

def main():
    t0 = time.time(); M = L.load_minutes() if not SELFTEST else synth_all(bool(os.environ.get("PLANT")))[0]; M["date"] = M["date"].astype(str); P = day_panel(M); print(f"{len(P):,} stock-days ({time.time()-t0:.0f}s)", flush=True)
    if SELFTEST: _, hist, T = synth_all(bool(os.environ.get("PLANT"))); sh = None
    else:
        import oi_probe as P_; sh = P_.open_sheet(); rh = sh.worksheet("Rule History").get_all_values(); hdr = [h.strip() for h in rh[0]]; si = hdr.index("symbol") if "symbol" in hdr else 0
        cols = [(i, h) for i, h in enumerate(hdr) if re.fullmatch(r"\d{2}-\d{2}", h)]; hist = {}
        for r in rh[1:]:
            if si < len(r) and r[si].strip(): d = {}; [d.__setitem__(h, float(r[i])) for i, h in cols if i < len(r) and r[i].strip() not in ("", "-") and re.fullmatch(r"-?\d+(\.\d+)?", r[i].strip())]; hist[r[si].strip().upper()] = d
        try: T = parse_trig(sh.worksheet("Backtest Triggers").get_all_values())
        except Exception as e: print("no Backtest Triggers tab:", e); T = pd.DataFrame()
    P = attach_score(P, hist); print(f"stock-days with score+next day: {P.dropna(subset=['score','ocx']).shape[0]:,}")
    IC, D = daily_ic(P); BK = daily_buckets(D); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n== DAILY IC ==\n", IC.round(4).to_string(index=False)); print("\n== DAILY BUCKETS (net, strong score) ==\n", BK[(BK["side"] == "all strong") & (BK["target"].str.contains("open"))].round(2).to_string(index=False))
    TF = pd.DataFrame()
    if len(T):
        BF = bar_flow(M); X = trigger_filter(T, BF); TF = trigger_table(X) if len(X) else pd.DataFrame(); print(f"\ntriggers matched to flow: {len(X):,} of {len(T):,}")
        if len(TF): print("\n== TRIGGER FILTER ==\n", TF[TF["sample"] != "all"].round(2).to_string(index=False))
    for n, d in (("daily_ic", IC), ("daily_buckets", BK), ("trigger_filter", TF)): d.to_csv(f"{OUT}/ofc_{n}.csv", index=False)
    if SELFTEST: return IC, BK, TF
    rnd = lambda d: d.round(4).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    for n, d in (("OFC Daily IC", IC), ("OFC Daily Buckets", BK), ("OFC Trigger Filter", TF)):
        if len(d): P_.put_tab(sh, n, list(d.columns), rnd(d).values.tolist())
    P_.put_tab(sh, "OFC Notes", NOTES[0], NOTES[1:] + [["stock-days", str(len(D))], ["triggers", str(len(T))], ["cost bps", str(COST)], ["run", dt.datetime.now().strftime("%Y-%m-%d %H:%M")]]); print("sheet tabs written")

# ────────────────────────── self-test data ──────────────────────────
def synth_all(plant, n_sym=40, n_days=40, seed=11):
    rng = np.random.default_rng(seed); days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-08-03", periods=n_days)]; hh = np.array([(555 + m) // 60 * 100 + (555 + m) % 60 for m in range(360)])
    fr = []; hist = {}; trig = []; lat = {(k, d): rng.normal() for k in range(n_sym) for d in days}
    for k in range(n_sym):
        p = 500.0; sym = f"T{k:02d}"; hist[sym] = {}
        for di, d in enumerate(days):
            g = lat[(k, d)]; gp = lat[(k, days[di - 1])] if di else 0.0; drift = (6.0 * gp if plant else 0.0) * 1e-4 / 360 * 2       # next-day drift driven by yesterday's latent (flow + score both see it)
            ret = 3e-4 * rng.normal(size=360) + drift; c = p * np.cumprod(1 + ret); o = np.r_[p, c[:-1]]; p = c[-1]; vol = rng.lognormal(7, 0.3, 360)
            share = np.clip(0.5 + (0.05 * g if True else 0) + 0.01 * rng.normal(size=360), 0.05, 0.95); buy = vol * share; sell = vol - buy
            fr.append(pd.DataFrame({"symbol": sym, "date": d, "hhmm": hh, "o": o, "h": np.maximum(o, c) * 1.0002, "l": np.minimum(o, c) * 0.9998, "c": c, "tot": vol, "buy": buy, "sell": sell, "delta": buy - sell, "cd": np.cumsum(buy - sell), "vwap": (o + c) / 2,
                                    "bstack": 0, "sstack": 0, "d_o": o[0], "d_c": c[-1], "d_v": vol.sum()}))
            hist[sym][d[5:10]] = float(np.clip(25 * g + 15 * rng.normal(), -80, 80))
            if rng.random() < 0.25:
                m = 600 + 5 * int(rng.integers(0, 40)); side = "CALL" if hist[sym][d[5:10]] > 0 else "PUT"; trig.append({"session": d, "time": f"{m // 60:02d}:{m % 60:02d}", "symbol": sym, "side": side, "score": hist[sym][d[5:10]], "r15": 0.0, "r30": 0.0, "r60": 0.0, "rEOD": 0.0, "g": g})
    T = pd.DataFrame(trig); T["m"] = [int(t[:2]) * 60 + int(t[3:]) for t in T["time"]]
    for hz in ("r15", "r30", "r60", "rEOD"): T[hz] = rng.normal(0, 0.4, len(T)) + (0.02 * np.where(T["side"] == "CALL", 1, -1) * T["g"] if plant else 0)
    return pd.concat(fr, ignore_index=True), hist, T.drop(columns=["g"])

if __name__ == "__main__":
    if SELFTEST:
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n" + "=" * 18, "SELFTEST", "PLANTED: flow+score carry next-day information" if plant else "NULL: nothing predicts", "=" * 18)
            IC, BK, TF = main(); r = IC[(IC["sample"] == "dev") & (IC["target"].str.contains("open")) & (IC["signal"] == "partial flow|score")].iloc[0]; r2 = IC[(IC["sample"] == "dev") & (IC["target"].str.contains("open")) & (IC["signal"] == "r_score")].iloc[0]
            print("partial flow|score:", r[["mean rank IC", "CI low", "CI high", "read"]].to_dict(), "| score:", r2[["mean rank IC", "read"]].to_dict())
            if plant: assert r2["read"] == "positive", "planted score info not found"
            else: assert r["read"] != "positive" and r2["read"] != "positive", "false positive on null"
        print("SELFTEST OK")
    else: main()
