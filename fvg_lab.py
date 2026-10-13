# ==================================================================================
# FVG LAB — Fair Value Gaps (price imbalances) on the daily candles.
#   Bull FVG (support): high[i-1] < low[i+1]  -> gap zone [high[i-1] .. low[i+1]]
#   Bear FVG (resistance): low[i-1] > high[i+1] -> gap zone [high[i+1] .. low[i-1]]
#
#   nearest_imbalances(feat)  -> per-stock latest: nearest unfilled support/resistance gap + "inside" flag (dashboard)
#   backtest_fvg(feat)        -> distance-controlled test: when price REACHES a gap, does it REJECT vs random zones?
#
#   Runs on the conviction masterdata's daily OHLC (cash). Needs only `feat`.
# ==================================================================================
import numpy as np, pandas as pd

def detect_fvgs(h, l):
    """3-candle FVGs. Returns (confirm_idx, type, bottom, top). confirm_idx = candle i+1 (leak-free from there)."""
    out=[]
    for i in range(1, len(h)-1):
        if h[i-1] < l[i+1]: out.append((i+1,"bull",float(h[i-1]),float(l[i+1])))   # support gap below
        if l[i-1] > h[i+1]: out.append((i+1,"bear",float(h[i+1]),float(l[i-1])))   # resistance gap above
    return out

# ── dashboard: nearest N UNFILLED imbalances per stock, as of the latest date ──
N_IMB = 3      # how many stacked support/resistance levels to output: imb_sup1..N / imb_res1..N
REACT_LOOKBACK = 2   # sessions to look back for a bounce/rejection at the nearest S/R level

def nearest_imbalances(feat, n_levels=N_IMB):
    d=feat.sort_values(["symbol","date"]); recs=[]
    for sym,g in d.groupby("symbol"):
        g=g.reset_index(drop=True); n=len(g)
        if n<5: continue
        h=g["high"].to_numpy(float); l=g["low"].to_numpy(float); c=g["close"].to_numpy(float); px=c[-1]
        sup=[]; res=[]; inside=""
        for idx,typ,bot,top in detect_fvgs(h,l):
            if idx>=n: continue
            af=c[idx+1:]
            if typ=="bull":
                if len(af) and (af<bot).any(): continue         # support already closed through -> filled
                if bot<=px<=top: inside="bull"
                elif top<px: sup.append(top)                    # unfilled support below
            else:
                if len(af) and (af>top).any(): continue         # resistance already closed through
                if bot<=px<=top: inside="bear"
                elif bot>px: res.append(bot)                    # unfilled resistance above
        sup=sorted(set(sup), reverse=True)                      # 1 = nearest support (highest below price)
        res=sorted(set(res))                                    # 1 = nearest resistance (lowest above price)
        rec={"symbol":sym}
        for k in range(n_levels):                               # imb_sup1, imb_res1, imb_sup2, imb_res2, ...
            rec[f"imb_sup{k+1}"]=round(sup[k],1) if k<len(sup) else ""
            rec[f"imb_res{k+1}"]=round(res[k],1) if k<len(res) else ""
        rec["in_imb"]=inside
        # S/R Flag is now filled LIVE by live_fvg.py (5-min, off today's 1-min bars vs these levels).
        # Kept here only so the column/header is created nightly in the dashboard; left empty.
        rec["S/R Flag"]=""
        recs.append(rec)
    return pd.DataFrame(recs)

# ── backtest: do FVGs hold when price reaches them? (distance-matched control) ──
def _atr(h,l,c,n=14):
    tr=np.zeros(len(c))
    for i in range(1,len(c)): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    return pd.Series(tr).rolling(n,min_periods=3).mean().to_numpy()

def _ctrl_curve(ctrl, step=0.25, min_bin=10):
    gm=float(ctrl["rej"].mean()) if len(ctrl) else np.nan
    cc=ctrl.assign(fb=np.floor(ctrl["dist"]/step)*step); grp=cc.groupby("fb")["rej"].agg(["mean","size"]); grp=grp[grp["size"]>=min_bin]
    xs=grp.index.values.astype(float)+step/2; ys=grp["mean"].values
    if len(xs)==0: return (lambda d: np.full(np.shape(d), gm))
    return (lambda d: np.interp(d, xs, ys, left=ys[0], right=ys[-1]))

def backtest_fvg(feat, H=5, maxfwd=20, frac=0.70, n_ctrl=3):
    rng=np.random.default_rng(7); rows=[]
    for sym,g in feat.sort_values(["symbol","date"]).groupby("symbol"):
        g=g.reset_index(drop=True); n=len(g)
        if n<30: continue
        h=g["high"].to_numpy(float); l=g["low"].to_numpy(float); c=g["close"].to_numpy(float); dts=g["date"].to_numpy(); atr=_atr(h,l,c)
        for idx,typ,bot,top in detect_fvgs(h,l):
            if idx>=n-1 or not (atr[idx]>0): continue
            ref=c[idx]; edge = top if typ=="bull" else bot; w=abs(top-bot)
            reach=next((j for j in range(idx+1,min(n,idx+1+maxfwd)) if (typ=="bull" and l[j]<=top) or (typ=="bear" and h[j]>=bot)), None)
            if reach is not None:
                wc=c[reach:min(n,reach+H+1)]; broke=(wc<bot).any() if typ=="bull" else (wc>top).any()
                rows.append({"date":dts[reach],"kind":"fvg","side":typ,"dist":round(abs(ref-edge)/atr[idx],2),"rej":0 if broke else 1})
            for _ in range(n_ctrl):                                     # distance-matched random zones (same side/width)
                d0=float(rng.uniform(0.3,3.0))
                if typ=="bull": cbot=ref-d0*atr[idx]-w; ctop=cbot+w; cedge=ctop
                else: ctop=ref+d0*atr[idx]+w; cbot=ctop-w; cedge=cbot
                cr=next((j for j in range(idx+1,min(n,idx+1+maxfwd)) if (typ=="bull" and l[j]<=ctop) or (typ=="bear" and h[j]>=cbot)), None)
                if cr is None: continue
                wc=c[cr:min(n,cr+H+1)]; cbroke=(wc<cbot).any() if typ=="bull" else (wc>ctop).any()
                rows.append({"date":dts[cr],"kind":"control","side":typ,"dist":round(abs(ref-cedge)/atr[idx],2),"rej":0 if cbroke else 1})
    R=pd.DataFrame(rows)
    if R.empty: print("no FVG reactions"); return R,{}
    dates=sorted(R["date"].unique()); cut=dates[int(len(dates)*frac)]; te=R[R.date>=cut]
    ctrl=te[te.kind=="control"]; real=te[te.kind=="fvg"].copy()
    curve=_ctrl_curve(ctrl); real["exp"]=curve(real["dist"].values)
    br=float(real["rej"].mean()); be=float(real["exp"].mean())
    print(f"FVG reactions: real {int((R.kind=='fvg').sum()):,} / control {int((R.kind=='control').sum()):,} | OOS reached: {len(real):,}")
    print(f"\nreject-rate when reached (OOS): FVG {round(br,3)}  vs  distance-matched control {round(be,3)}  ->  lift {round(br/be,2) if be>0 else 'na'}")
    print("(lift ~1.0 = a gap holds no better than a random zone that far away; >1 = imbalances are real S&R)")
    for sd in ("bull","bear"):
        rs=real[real.side==sd]
        if len(rs): e=rs["exp"].mean(); print(f"  {sd:4s} FVG: n={len(rs):,}  reject={round(rs['rej'].mean(),3)}  lift={round(rs['rej'].mean()/e,2) if e>0 else 'na'}")
    return R, {"lift": round(br/be,2) if be>0 else None}

print("fvg_lab loaded | nearest_imbalances(feat) for dashboards | backtest_fvg(feat) to test if they hold")
