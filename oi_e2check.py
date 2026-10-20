# ==================================================================================
# OPEN INTEREST ENGINE — E2 CHECK: "writers on both sides while price is flat -> bigger move afterwards?"
#
#  The first study said yes (range next 60m: 0.93% vs 0.67%), but its comparison windows were not matched, and 58% of the
#  events sat in the first days of the new series, over half before 11:00. This re-tests it fairly:
#   * compare each flat window ONLY against other flat windows of the SAME stock, SAME time-of-day, SAME day-of-series
#     (regression with stock + time-of-day + day fixed effects)
#   * control for how wild the stock already was in the last 60 minutes (volatility is sticky)
#   * variants: all days / after the series roll-over / roll-over excluded AND after 11:00
#   * confidence from resampling whole days
#  Reads the archive saved by oi_universe.py (same input as oi_discover.py).   Tab: "OI E2 Check"
#  Usage: python oi_e2check.py [--selftest]       env: OI_ARCH, OI_ROLL_END (default 2026-10-01, last day treated as roll-over)
# ==================================================================================
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import oi_discover as DS

ROLL_END = os.environ.get("OI_ROLL_END", "2026-10-01")
Z_BOTH, FLAT = 1.5, 0.3

def add_context(u, s, sp, recs):
    """forward and backward 60-minute ranges for each record (from the spot tape)."""
    idx = pd.DatetimeIndex(sp["ts"]); mod = (idx.hour * 60 + idx.minute).to_numpy(); c = sp["close"].to_numpy(float); T = len(c)
    pos = lambda m: int(np.searchsorted(mod, m, side="right")) - 1
    for r in recs:
        i = pos(r["m"]); j = min(pos(r["m"] + 60), T - 1); b = max(pos(r["m"] - 60), 0)
        fw, bw = c[i: j + 1], c[b: i + 1]
        ok = mod[j] >= r["m"] + 57
        r["rng60"] = (fw.max() - fw.min()) / c[i] * 100 if ok and len(fw) > 10 else np.nan
        r["abs60"] = abs(c[j] / c[i] - 1) * 100 if ok else np.nan
        r["rng_prev60"] = (bw.max() - bw.min()) / c[i] * 100 if len(bw) > 10 else np.nan
    return recs

def build():
    rows = []; names = DS.stock_files(); t0 = time.time()
    for n, u in enumerate(names, 1):
        try:
            for s, sp, d, prev in DS.stock_sessions(u):
                rows += add_context(u, s, sp, DS.session_features(u, s, sp, d, prev) or [])
        except Exception as e: print("skip", u, type(e).__name__, str(e)[:80])
        if n % 25 == 0: print(f"  {n}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    if D.empty: raise SystemExit("no archive data in " + DS.ARCH)
    return D

def prep(D):
    mx = D.groupby("symbol")["band0"].transform("max"); D = D[D["band0"] >= DS.MIN_OI_SHARE * mx].copy()
    act = D.assign(a=(D["c_net_w30"] != 0) | (D["p_net_w30"] != 0)).groupby("symbol")["a"].mean(); D = D[D["symbol"].map(act) >= DS.MIN_ACTIVE].copy()
    for c in ("c_net_w30", "p_net_w30"): D["z_" + c] = D[c] / D.groupby("symbol")[c].transform("std").replace(0, np.nan)
    D["flat"] = D["ret30"].abs() <= FLAT
    D["event"] = (D["z_c_net_w30"] >= Z_BOTH) & (D["z_p_net_w30"] >= Z_BOTH) & D["flat"]
    D = D[(D["m"] <= 14 * 60 + 30)].copy()
    D["tod"] = (D["m"] // 30).astype(int)
    return D

def fe_effect(S, y, label):
    """effect of event on log(y) after stock + time-of-day + day fixed effects and prior-volatility controls."""
    X = S[["symbol", "tod", "session"]].astype(str); W = pd.get_dummies(X, drop_first=True, dtype=float)
    W["lprev"] = np.log(S["rng_prev60"].clip(lower=0.02)); W["lso"] = np.log(S["range_so_far"].clip(lower=0.02)); W["const"] = 1.0
    ly = np.log(S[y].clip(lower=0.02)).to_numpy(); e = S["event"].astype(float).to_numpy()
    A = W.to_numpy();
    def resid(v): b, *_ = np.linalg.lstsq(A, v, rcond=None); return v - A @ b
    ry, re = resid(ly), resid(e); den = (re ** 2).sum()
    if den < 1e-9 or e.sum() < 20: return None
    b = (re * ry).sum() / den
    g = pd.DataFrame({"s": S["session"].values, "n": re * ry, "d": re ** 2}).groupby("s").sum(); rng = np.random.default_rng(7); bs = []
    for _ in range(1000):
        w = rng.integers(0, len(g), len(g)); dd = g["d"].to_numpy()[w].sum()
        if dd > 0: bs.append(g["n"].to_numpy()[w].sum() / dd)
    lo, hi = np.percentile(bs, [2.5, 97.5]); pct = lambda x: (np.exp(x) - 1) * 100
    raw = S.loc[S["event"], y].mean() / S.loc[~S["event"], y].mean() - 1
    verdict = "MORE volatile after (holds)" if lo > 0 else ("QUIETER after" if hi < 0 else "no evidence")
    return {"sample": label, "outcome": {"rng60": "high-low range next 60m", "abs60": "net move next 60m (abs)"}[y], "flat windows": len(S), "events": int(e.sum()),
            "stocks with events": S.loc[S["event"], "symbol"].nunique(), "days with events": S.loc[S["event"], "session"].nunique(),
            "unmatched ratio (old method) %": raw * 100, "matched effect %": pct(b), "CI low %": pct(lo), "CI high %": pct(hi), "verdict": verdict}

def study(D):
    F = D[D["flat"]].dropna(subset=["rng60", "abs60", "rng_prev60", "range_so_far"]).copy()
    variants = [("all days", F), (f"after roll-over (> {ROLL_END})", F[F["session"] > ROLL_END]),
                (f"after roll-over, and after 11:00", F[(F["session"] > ROLL_END) & (F["m"] >= 11 * 60)]),
                (f"after roll-over, before 11:00", F[(F["session"] > ROLL_END) & (F["m"] < 11 * 60)])]
    out = []
    for lab, S in variants:
        for y in ("rng60", "abs60"):
            r = fe_effect(S, y, lab)
            if r: out.append(r)
    return pd.DataFrame(out)

NOTES = [["term", "meaning"],
 ["event", "30-min call OI AND put OI both up >= 1.5x normal (that stock's own), price within +/-0.3% over the 30 min"],
 ["matched effect %", "how much bigger (+) or smaller (-) the next-60-min move is after an event than after a flat window with NO both-side writing, for the same stock, same half-hour of the day, same trading day, and same prior-hour volatility"],
 ["unmatched ratio", "the naive comparison from the first study, for reference"],
 ["CI", "95% range from resampling whole trading days (stocks on one day are not independent)"],
 ["reading it", "if matched effect stays clearly positive after roll-over and after 11:00, the 'both-side writing -> bigger move' flag is real. If it shrinks to ~0, the first result was an early-series/morning effect"]]

def main(selftest=False):
    D = prep(build()); print(f"{len(D):,} marks, flat windows {int(D['flat'].sum()):,}, events {int(D['event'].sum())}")
    if selftest:      # simulated OI has events on whole days only; draw events at random so the test is meaningful
        D["event"] = D["flat"] & (np.random.default_rng(1).random(len(D)) < 0.2)
    if selftest and os.environ.get("PLANT"):
        k = D["event"]; D.loc[k, "rng60"] = D.loc[k, "rng60"] * 1.5 + 0.1
    R = study(D); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); print(R.round(2).to_string(index=False))
    R.to_csv(f"{DS.OUT}/e2check.csv", index=False)
    if selftest: return R
    import oi_probe as P
    sh = P.open_sheet(); P.put_tab(sh, "OI E2 Check", list(R.columns), R.round(3).values.tolist())
    P.put_tab(sh, "OI E2 Notes", NOTES[0], NOTES[1:]); print("sheet tabs written")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        global_roll = os.environ.setdefault("OI_ROLL_END", "2026-09-30")
        for plant in ("", "1"):
            os.environ["PLANT"] = plant; print("\n== SELFTEST", "PLANTED volatility effect" if plant else "NULL ==")
            R = main(True); r = R[(R["sample"] == "all days") & R["outcome"].str.contains("range")].iloc[0]
            if plant: assert r["verdict"].startswith("MORE"), "planted effect not detected"
            else: assert not r["verdict"].startswith("MORE"), "false positive on random data"
        print("SELFTEST OK")
    else: main()
