# ==================================================================================
# INTRADAY REGIME — classify a stock's day-type from its intraday candles:
#     Trend ↑ / Trend ↓ — clean one-directional day (high efficiency)
#     Fired ↑ / Fired ↓ — a second leg breaking out of a base (range then expansion)
#     Coiling           — volatility compressed, based (energy stored)
#     Chop              — low efficiency, no compression, rotational
#
#   Drivers: efficiency ratio (net move / path travelled), range compression
#   (fast range vs baseline), and expansion + break of the base extremes.
#   compute_regime(h,l,c) is pure numpy — used live (in live_fvg) and in backtest().
#
#   Backtest on the 1-min archive:  cd ~/live-refresh && python intraday_regime.py
# ==================================================================================
import numpy as np

# ── knobs ──
MIN_BARS  = 15     # bars into the session before classifying (15 x 1m = 15 min)
FAST      = 5      # fast range window (bars)
BASE      = 20     # baseline / consolidation window (bars)
ER_TREND  = 0.45   # efficiency ratio >= this => trend day
COMP_LOW  = 0.60   # fast/baseline range < this => coiling (compressed)
EXP_HIGH  = 1.5    # last bar range > this x baseline => expansion (fired candidate)

def compute_regime(h, l, c):
    """Returns (tag, eff_pct, exp_ratio). tag='' until MIN_BARS bars exist."""
    c=np.asarray(c,float); h=np.asarray(h,float); l=np.asarray(l,float); n=len(c)
    if n < MIN_BARS: return "", "", ""
    net=abs(c[-1]-c[0]); path=np.abs(np.diff(c)).sum()
    er = net/path if path>0 else 0.0
    rng=h-l; b0=max(0, n-BASE)
    fast = rng[-FAST:].mean()
    base = rng[b0:-1].mean() if (n-1) > b0 else rng[:-1].mean()
    exp_ratio = (rng[-1]/base) if base>0 else 1.0
    comp      = (fast/base)    if base>0 else 1.0
    base_hi = h[b0:-1].max() if (n-1) > b0 else h[:-1].max()
    base_lo = l[b0:-1].min() if (n-1) > b0 else l[:-1].min()
    fired_up = exp_ratio > EXP_HIGH and c[-1] > base_hi
    fired_dn = exp_ratio > EXP_HIGH and c[-1] < base_lo
    if fired_up: tag = "Fired ↑"                             # expansion breaking the base = the release
    elif fired_dn: tag = "Fired ↓"
    elif comp < COMP_LOW: tag = "Coiling"                     # recent bars compressed = based right now
    elif er >= ER_TREND: tag = "Trend ↑" if c[-1] >= c[0] else "Trend ↓"   # clean expanded trend
    else: tag = "Chop"
    return tag, round(er*100), round(float(exp_ratio),1)

# ── backtest: does each tag lead to the move its playbook expects? ──
def backtest(df, H=15):
    """df = 1-min bars (symbol, ts, high, low, close, session) from archive_1m.load_1m()."""
    import pandas as pd
    rows=[]
    for (sym,sess),g in df.sort_values(["symbol","ts"]).groupby(["symbol","session"]):
        g=g.reset_index(drop=True); n=len(g)
        if n < MIN_BARS+H+1: continue
        h=g["high"].to_numpy(float); l=g["low"].to_numpy(float); c=g["close"].to_numpy(float)
        for i in range(MIN_BARS, n-H, 3):                    # step 3 to reduce overlap
            tag,_,_ = compute_regime(h[:i+1], l[:i+1], c[:i+1])
            if not tag: continue
            fwd=(c[i+H]-c[i])/c[i]*100
            d = fwd if "↑" in tag else (-fwd if "↓" in tag else np.nan)   # oriented in tag direction
            rows.append({"tag":tag, "fwd":fwd, "dir":d, "absfwd":abs(fwd)})
    R=pd.DataFrame(rows)
    if R.empty: print("no data (run the archiver first)"); return R
    print(f"intraday regime backtest — forward {H}-min move after each tag\n")
    print(f"{'tag':10s} {'n':>7s} {'dir_move%':>10s} {'win%':>7s} {'|move|%':>8s}")
    for t,x in R.groupby("tag"):
        dm=x["dir"].mean(); win=(x["dir"]>0).mean()*100 if x["dir"].notna().any() else float('nan')
        print(f"{t:10s} {len(x):>7,d} {dm:>10.3f} {win:>7.1f} {x['absfwd'].mean():>8.3f}")
    print("\n(dir_move% = forward move oriented in the tag's direction; Trend/Fired should be >0,")
    print(" Chop ~0, Coiling's |move|% should be high = it precedes a big move either way)")
    return R

if __name__ == "__main__":
    import archive_1m as A
    df = A.load_1m("cash")
    if len(df): backtest(df, H=15)
