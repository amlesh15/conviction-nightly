# ==================================================================================
# INTRADAY FVG — Fair Value Gaps on Sensibull 1-minute candles (cash AND futures).
#
#   Endpoint (cookie-authed):  POST /v1/compute/{seg}/candles/{SYMBOL}
#     body {from_date, to_date, interval:"1M", skip_last_ts:false}
#     cash = seg 1. Futures uses a different symbol/seg — set FUT_SEG / pass the fut symbol.
#
#   fetch_1m(symbol, days, seg)       -> 1-min DataFrame (ts,open,high,low,close,volume,oi,session)
#   depth_probe(symbol)               -> how far back 1M actually serves (run this FIRST)
#   intraday_imbalances(symbol)       -> live: nearest UNFILLED intraday gap above/below spot + inside flag
#   backtest_intraday_fvg(symbols,..) -> distance-matched: when price reaches an intraday gap, does it REJECT?
#
#   Cookie is read from ~/live-refresh/cookie.txt (same one the live refresher / scanner use).
# ==================================================================================
import os, requests, datetime as dt
import numpy as np, pandas as pd
import concurrent.futures as cf

BASE       = "https://oxide.sensibull.com/v1/compute/{seg}/candles/{sym}"
COOKIE_PATH= os.path.expanduser("~/live-refresh/cookie.txt")
CASH_SEG   = 1
FUT_SEG    = 1          # <-- set once we confirm the futures segment from the network tab

def _cookie():
    return open(COOKIE_PATH).read().strip()

def _hdr():
    return {"cookie":_cookie(),"content-type":"application/json",
            "origin":"https://web.sensibull.com","referer":"https://web.sensibull.com/",
            "user-agent":"Mozilla/5.0"}

def fetch_1m(symbol, days=30, seg=CASH_SEG, end=None):
    """1-minute candles for `symbol` over ~`days` calendar days back. Empty DF if none."""
    end = end or dt.date.today()
    frm = (end - dt.timedelta(days=days)).isoformat()
    body= {"from_date":frm, "to_date":(end+dt.timedelta(days=1)).isoformat(),
           "interval":"1M", "skip_last_ts":False}
    r = requests.post(BASE.format(seg=seg, sym=symbol), json=body, headers=_hdr(), timeout=30)
    c = ((r.json() or {}).get("payload") or {}).get("candles") or []
    df= pd.DataFrame(c)
    if df.empty: return df
    df["ts"]=pd.to_datetime(df["ts"]); df["session"]=df["ts"].dt.date
    for col in ("open","high","low","close","volume","oi"):
        if col in df: df[col]=pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)

def depth_probe(symbol="RELIANCE", seg=CASH_SEG):
    """Run FIRST. Tells us how much 1-min history the endpoint will serve -> sizes the backtest."""
    for d in (7,30,60,90,180,365):
        df=fetch_1m(symbol, days=d, seg=seg)
        if df.empty: print(f"{d:4d}d -> 0 candles"); continue
        print(f"{d:4d}d -> {len(df):7,} candles | {df['session'].min()} .. {df['session'].max()} "
              f"| {df['session'].nunique()} sessions")

# ── detection (3-candle FVG, leak-free from confirm bar i+1) ──
def _detect(h, l):
    out=[]
    for i in range(1, len(h)-1):
        if h[i-1] < l[i+1]: out.append((i+1,"bull",float(h[i-1]),float(l[i+1])))   # support gap
        if l[i-1] > h[i+1]: out.append((i+1,"bear",float(h[i+1]),float(l[i-1])))   # resistance gap
    return out

def _atr(h,l,c,n=30):
    tr=np.zeros(len(c))
    for i in range(1,len(c)): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    return pd.Series(tr).rolling(n,min_periods=5).mean().to_numpy()

# ── live: nearest unfilled intraday gap for the CURRENT session ──
def intraday_imbalances(symbol, seg=CASH_SEG):
    df=fetch_1m(symbol, days=2, seg=seg)
    if df.empty: return {"symbol":symbol,"error":"no data"}
    g=df[df.session==df.session.max()].reset_index(drop=True); n=len(g)
    if n<5: return {"symbol":symbol,"error":"thin session"}
    h=g["high"].to_numpy(float); l=g["low"].to_numpy(float); c=g["close"].to_numpy(float); px=c[-1]
    sup=[]; res=[]; inside=""
    for idx,typ,bot,top in _detect(h,l):
        if idx>=n: continue
        af=c[idx+1:]
        if typ=="bull":
            if len(af) and (af<bot).any(): continue           # filled (closed through support)
            if bot<=px<=top: inside="bull"
            elif top<px: sup.append(top)
        else:
            if len(af) and (af>top).any(): continue           # filled (closed through resistance)
            if bot<=px<=top: inside="bear"
            elif bot>px: res.append(bot)
    s=max(sup) if sup else None; r=min(res) if res else None
    return {"symbol":symbol,"spot":round(px,2),
            "imb_sup": round(s,2) if s is not None else "",
            "sup_dist%": round((px-s)/px*100,2) if s is not None else "",
            "imb_res": round(r,2) if r is not None else "",
            "res_dist%": round((r-px)/px*100,2) if r is not None else "",
            "in_imb": inside}

# ── backtest: do intraday FVGs hold when reached? (per-session, distance-matched control) ──
def _ctrl_curve(ctrl, step=0.25, min_bin=15):
    gm=float(ctrl["rej"].mean()) if len(ctrl) else np.nan
    cc=ctrl.assign(fb=np.floor(ctrl["dist"]/step)*step); grp=cc.groupby("fb")["rej"].agg(["mean","size"]); grp=grp[grp["size"]>=min_bin]
    xs=grp.index.values.astype(float)+step/2; ys=grp["mean"].values
    if len(xs)==0: return (lambda d: np.full(np.shape(d), gm))
    return (lambda d: np.interp(d, xs, ys, left=ys[0], right=ys[-1]))

def backtest_intraday_fvg(symbols, days=30, seg=CASH_SEG, H=15, maxfwd=60, frac=0.70, n_ctrl=3, workers=8):
    """
    H, maxfwd in MINUTES. Detects FVGs per session, tests reach->reject vs random same-side zones
    at matched distance. OOS split by session date. symbols = list of cash (or fut) symbols.
    """
    rng=np.random.default_rng(7)
    def _sym_rows(sym):
        try: df=fetch_1m(sym, days=days, seg=seg)
        except Exception as e: print(sym,"fetch err",e); return []
        if df.empty: return []
        rows=[]
        for sess,g in df.groupby("session"):
            g=g.reset_index(drop=True); n=len(g)
            if n<40: continue
            h=g["high"].to_numpy(float); l=g["low"].to_numpy(float); c=g["close"].to_numpy(float); atr=_atr(h,l,c)
            for idx,typ,bot,top in _detect(h,l):
                if idx>=n-1 or not (atr[idx]>0): continue
                ref=c[idx]; edge=top if typ=="bull" else bot; w=abs(top-bot)
                reach=next((j for j in range(idx+1,min(n,idx+1+maxfwd))
                            if (typ=="bull" and l[j]<=top) or (typ=="bear" and h[j]>=bot)), None)
                if reach is not None:
                    wc=c[reach:min(n,reach+H+1)]; broke=(wc<bot).any() if typ=="bull" else (wc>top).any()
                    rows.append({"session":sess,"kind":"fvg","side":typ,"dist":round(abs(ref-edge)/atr[idx],2),"rej":0 if broke else 1})
                for _ in range(n_ctrl):
                    d0=float(rng.uniform(0.3,3.0))
                    if typ=="bull": cbot=ref-d0*atr[idx]-w; ctop=cbot+w; cedge=ctop
                    else: ctop=ref+d0*atr[idx]+w; cbot=ctop-w; cedge=cbot
                    cr=next((j for j in range(idx+1,min(n,idx+1+maxfwd))
                             if (typ=="bull" and l[j]<=ctop) or (typ=="bear" and h[j]>=cbot)), None)
                    if cr is None: continue
                    wc=c[cr:min(n,cr+H+1)]; cbroke=(wc<cbot).any() if typ=="bull" else (wc>ctop).any()
                    rows.append({"session":sess,"kind":"control","side":typ,"dist":round(abs(ref-cedge)/atr[idx],2),"rej":0 if cbroke else 1})
        return rows
    allrows=[]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for rr in ex.map(_sym_rows, list(symbols)): allrows+=rr
    R=pd.DataFrame(allrows)
    if R.empty: print("no intraday FVG reactions"); return R,{}
    sess=sorted(R["session"].unique()); cut=sess[int(len(sess)*frac)]; te=R[R.session>=cut]
    ctrl=te[te.kind=="control"]; real=te[te.kind=="fvg"].copy()
    curve=_ctrl_curve(ctrl); real["exp"]=curve(real["dist"].values)
    br=float(real["rej"].mean()); be=float(real["exp"].mean())
    print(f"intraday FVG reactions: real {int((R.kind=='fvg').sum()):,} / control {int((R.kind=='control').sum()):,} | OOS reached: {len(real):,}")
    print(f"\nreject-rate when reached (OOS, {H}m hold): FVG {round(br,3)}  vs  matched control {round(be,3)}  ->  lift {round(br/be,2) if be>0 else 'na'}")
    print("(lift ~1.0 = holds no better than a random zone; >1 = intraday imbalances are real S&R)")
    for sd in ("bull","bear"):
        rs=real[real.side==sd]
        if len(rs): e=rs["exp"].mean(); print(f"  {sd:4s}: n={len(rs):,}  reject={round(rs['rej'].mean(),3)}  lift={round(rs['rej'].mean()/e,2) if e>0 else 'na'}")
    return R, {"lift": round(br/be,2) if be>0 else None}

# ── crossover test: does BREAKING an intraday FVG lead to continuation? ──
def backtest_crossover(df, H=15, maxfwd=60, frac=0.70, n_ctrl=3):
    """
    Event = price CLOSES THROUGH the far edge of an intraday FVG (a crossover).
      bull gap broken (close<bottom) -> break direction = DOWN
      bear gap broken (close>top)    -> break direction = UP
    Measures forward return over H minutes IN THE BREAK DIRECTION, vs a matched
    control: random bars in the same session pushed the SAME direction.
    Positive EDGE = crossovers give real continuation beyond a random directional bar.
    df = 1-min bars with columns symbol, ts, high, low, close, session
         (load from the archive:  archive_1m.load_1m('cash')).
    """
    if df is None or len(df)==0: print("no data"); return pd.DataFrame(), {}
    rng=np.random.default_rng(11); rows=[]
    df=df.sort_values(["symbol","ts"])
    for (sym,sess),g in df.groupby(["symbol","session"]):
        g=g.reset_index(drop=True); n=len(g)
        if n<60: continue
        h=g["high"].to_numpy(float); l=g["low"].to_numpy(float); c=g["close"].to_numpy(float)
        for idx,typ,bot,top in _detect(h,l):
            if idx>=n-1: continue
            b=None                                            # break bar: first close through far edge
            for j in range(idx+1, min(n, idx+1+maxfwd)):
                if (typ=="bull" and c[j]<bot) or (typ=="bear" and c[j]>top): b=j; break
            if b is None or b+H>=n: continue
            d = -1.0 if typ=="bull" else 1.0                  # bull-break->down, bear-break->up
            rows.append({"session":str(sess),"side":typ,"kind":"break","fwd":d*(c[b+H]-c[b])/c[b]})
            for _ in range(n_ctrl):
                r=int(rng.integers(30, n-H-1))
                rows.append({"session":str(sess),"side":typ,"kind":"control","fwd":d*(c[r+H]-c[r])/c[r]})
    R=pd.DataFrame(rows)
    if R.empty: print("no crossover events"); return R, {}
    sess=sorted(R["session"].unique()); cut=sess[int(len(sess)*frac)]; te=R[R.session>=cut]
    ev=te[te.kind=="break"]; ct=te[te.kind=="control"]
    st=lambda x:(x["fwd"].mean()*1e4, float((x["fwd"]>0).mean()))
    em,ew=st(ev); cm,cw=st(ct)
    print(f"crossover events OOS: {len(ev):,} | controls: {len(ct):,}  (H={H}m, forward return in break direction, bps)")
    print(f"\n  break  : mean {em:+.1f} bps | win {ew:.3f}")
    print(f"  control: mean {cm:+.1f} bps | win {cw:.3f}")
    print(f"  EDGE   : {em-cm:+.1f} bps | win +{(ew-cw):+.3f}   ( >0 = crossover continues beyond random; <0 = it tends to fade )")
    for sd in ("bull","bear"):
        e=ev[ev.side==sd]; cc=ct[ct.side==sd]
        if len(e) and len(cc):
            em2,cm2=e["fwd"].mean()*1e4, cc["fwd"].mean()*1e4
            ew2,cw2=float((e["fwd"]>0).mean()), float((cc["fwd"]>0).mean())
            lab="bull-break(down)" if sd=="bull" else "bear-break(up)"
            print(f"  {lab:16s}: n={len(e):,}  edge {em2-cm2:+.1f} bps | win +{(ew2-cw2):+.3f}")
    return R, {"edge_bps": round(em-cm,1)}

print("intraday_fvg loaded | depth_probe() | intraday_imbalances(sym) | backtest_intraday_fvg(syms) | backtest_crossover(df)")
