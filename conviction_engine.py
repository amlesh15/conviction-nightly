# ==================================================================================
# CONVICTION ENGINE v2 — SELF-CONTAINED (data + point-in-time universe + calibrated dual engine)
# Rule-based-but-data-calibrated. Reads what happened (price<->OI character, aggressiveness,
# buildup sequence). NSE bhavcopy only. Point-in-time F&O universe. 7-day continuation target.
#
# FRESH-START USAGE (Colab):
#   from google.colab import drive; drive.mount('/content/drive')
#   !pip install -q scikit-learn
#   exec(open('/content/drive/MyDrive/conviction_engine.py').read())
#   TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")
#   md   = build_masterdata_pit("2023-08-01", TODAY)   # first build ~15-20 min, cached to Drive
#   feat = build_labels(build_features(md))
#   oc, coef_call, F = calibrate(feat, "up_label")
#   op, coef_put, _  = calibrate(feat, "dn_label")
#   print(evaluate(oc,"up_label")); print(evaluate(op,"dn_label"))
# ==================================================================================
import io, os, zipfile, math, time, requests
import numpy as np, pandas as pd

CACHE = "/content/drive/MyDrive/fno_cache"
ROLL_BUF, CA_GAP = 2, 0.20
UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
INDEX_NAMES = {"NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY","NIFTYNXT50","NIFTYIT"}

# ─────────────────────────── DATA LAYER (NSE bhavcopy, Drive-cached) ───────────────────────────
def load_universe():
    inst = pd.read_csv("https://api.kite.trade/instruments")
    fut  = inst[(inst["exchange"]=="NFO") & (inst["instrument_type"]=="FUT")]
    return sorted(fut[~fut["name"].isin(INDEX_NAMES)]["name"].unique())

_cash_bhav, _fo_bhav = {}, {}
def get_cash_bhav(D):
    if D in _cash_bhav: return _cash_bhav[D]
    fp = f"{CACHE}/cash/{D}.parquet"
    if os.path.exists(fp): _cash_bhav[D] = pd.read_parquet(fp); return _cash_bhav[D]
    dt = pd.to_datetime(D); df = None
    try:
        r = requests.get(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{dt.strftime('%d%m%Y')}.csv", headers=UA, timeout=30)
        if r.status_code==200 and r.text.strip():
            df = pd.read_csv(io.StringIO(r.text)); df.columns = [c.strip() for c in df.columns]
            df["SERIES"] = df["SERIES"].str.strip(); df = df[df["SERIES"]=="EQ"].copy()
            for c in ["PREV_CLOSE","OPEN_PRICE","HIGH_PRICE","LOW_PRICE","CLOSE_PRICE","TTL_TRD_QNTY","DELIV_QTY","DELIV_PER"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.set_index(df["SYMBOL"].str.strip())
    except Exception: pass
    if df is not None: os.makedirs(f"{CACHE}/cash", exist_ok=True); df.to_parquet(fp)
    _cash_bhav[D] = df; return df

def get_fo_bhav(D):
    if D in _fo_bhav: return _fo_bhav[D]
    fp = f"{CACHE}/fo/{D}.parquet"
    if os.path.exists(fp): _fo_bhav[D] = pd.read_parquet(fp); return _fo_bhav[D]
    dt = pd.to_datetime(D); df = None
    try:
        r = requests.get(f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{dt.strftime('%Y%m%d')}_F_0000.csv.zip", headers=UA, timeout=30)
        if r.status_code==200 and r.content:
            z = zipfile.ZipFile(io.BytesIO(r.content)); df = pd.read_csv(z.open(z.namelist()[0]))
            for c in ["OpnIntrst","ChngInOpnIntrst","StrkPric","ClsPric","HghPric","LwPric","OpnPric"]:
                if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    except Exception: pass
    if df is not None: os.makedirs(f"{CACHE}/fo", exist_ok=True); df.to_parquet(fp)
    _fo_bhav[D] = df; return df

def _biz_before(dt, n):
    d = pd.to_datetime(dt); c = 0
    while c < n:
        d -= pd.Timedelta(days=1)
        if d.weekday() < 5: c += 1
    return d

def fo_metrics_all(D):
    df = get_fo_bhav(D)
    if df is None: return {}
    fut = df[df["FinInstrmTp"]=="STF"]; opt = df[df["FinInstrmTp"]=="STO"]
    if len(fut)==0: return {}
    exps = sorted(fut["XpryDt"].unique()); dd = pd.to_datetime(D); front = None
    for e in exps:
        if dd <= _biz_before(pd.to_datetime(e), ROLL_BUF): front = e; break
    if front is None: front = exps[-1]
    futf = fut[fut["XpryDt"]==front].drop_duplicates("TckrSymb").set_index("TckrSymb")
    of = opt[opt["XpryDt"]==front]
    agg = of.groupby(["TckrSymb","OptnTp"])["OpnIntrst"].sum().unstack(fill_value=0)
    out = {}
    for sym in futf.index:
        cc = float(agg.loc[sym,"CE"]) if (sym in agg.index and "CE" in agg.columns) else 0.0
        pp = float(agg.loc[sym,"PE"]) if (sym in agg.index and "PE" in agg.columns) else 0.0
        out[sym] = {"oi": float(futf.loc[sym,"OpnIntrst"]), "oiChgAbs": float(futf.loc[sym,"ChngInOpnIntrst"]),
                    "callOI": cc, "putOI": pp, "pcr": round(pp/cc, 3) if cc>0 else None}
    return out

def trading_days(seed, end):
    days=[]; d=pd.to_datetime(seed); endd=pd.to_datetime(end)
    while d <= endd:
        if d.weekday() < 5:
            ds = d.strftime("%Y-%m-%d")
            if get_cash_bhav(ds) is not None: days.append(ds)
        d += pd.Timedelta(days=1)
    return days

def _ema(vals, period):
    out=[None]*len(vals); seed=[]; prev=None; k=2.0/(period+1)
    for i,v in enumerate(vals):
        if v is None or (isinstance(v,float) and math.isnan(v)): continue
        if prev is None:
            seed.append(v)
            if len(seed)==period: prev=sum(seed)/period; out[i]=prev
        else:
            prev=v*k+prev*(1-k); out[i]=prev
    return out

def build_universe_md(symbols, start, end, warmup_days=100, ca_gap=CA_GAP, verbose=True):
    seed = (pd.to_datetime(start)-pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    tds = trading_days(seed, end)
    if verbose: print(f"trading days: {len(tds)}  ({tds[0]} .. {tds[-1]})")
    rows=[]
    for D in tds:
        cash = get_cash_bhav(D); fom = fo_metrics_all(D)
        if cash is None: continue
        for sym in symbols:
            if sym not in cash.index: continue
            c = cash.loc[sym]; c = c.iloc[0] if isinstance(c, pd.DataFrame) else c
            m = fom.get(sym, {})
            rows.append({"date":D,"symbol":sym,"open":c["OPEN_PRICE"],"high":c["HIGH_PRICE"],"low":c["LOW_PRICE"],
                "close":c["CLOSE_PRICE"],"volume":c["TTL_TRD_QNTY"],"delivQty":c["DELIV_QTY"],"delivPct":c["DELIV_PER"],
                "oi":m.get("oi"),"oiChgAbs":m.get("oiChgAbs"),"callOI":m.get("callOI"),"putOI":m.get("putOI"),"pcr":m.get("pcr")})
    df = pd.DataFrame(rows)
    if df.empty: return df
    out=[]
    for sym,g in df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        cl = g["close"].values; nA=len(g); A=np.ones(nA); run=1.0
        for i in range(nA-1,0,-1):                                    # back-adjust corporate actions
            r=(cl[i]/cl[i-1]) if cl[i-1] else 1.0
            if r and (r<1-ca_gap or r>1+ca_gap): run*=r
            A[i-1]=run
        for col in ["open","high","low","close"]: g[col]=g[col]*A
        g["ema20"]=_ema(g["close"].tolist(),20); g["ema50"]=_ema(g["close"].tolist(),50)
        g["pxChg"]=(g["close"]/g["close"].shift(1)-1)*100
        prevoi=g["oi"]-g["oiChgAbs"]; g["oiChg"]=(g["oiChgAbs"]/prevoi.replace(0,np.nan))*100
        g["pcrChg"]=g["pcr"].diff()
        out.append(g)
    return pd.concat(out, ignore_index=True).query("date >= @start").reset_index(drop=True)

def build_fno_membership(start, end):
    members={}; ever=set()
    for D in trading_days(start, end):
        fo = get_fo_bhav(D)
        if fo is None: continue
        stf = set(fo.loc[fo["FinInstrmTp"]=="STF","TckrSymb"].astype(str).unique())
        members[D]=stf; ever|=stf
    return members, sorted(ever)

def build_masterdata_pit(start, end, warmup_days=100):
    """Point-in-time masterdata: ever-F&O superset, kept only for the dates each stock was actually in F&O."""
    members, ever = build_fno_membership(start, end)
    print(f"ever-F&O universe: {len(ever)} stocks")
    md = build_universe_md(ever, start, end, warmup_days=warmup_days)
    md["_in"] = [s in members.get(d, ()) for d,s in zip(md["date"], md["symbol"])]
    md = md[md["_in"]].drop(columns="_in").reset_index(drop=True)
    for c in ["open","high","low","close","volume","delivPct","delivQty","oi","oiChg","pxChg","pcr","pcrChg","callOI","putOI"]:
        if c in md.columns: md[c] = pd.to_numeric(md[c], errors="coerce")
    print(f"point-in-time masterdata: {len(md):,} rows | {md['symbol'].nunique()} symbols | {md['date'].min()} .. {md['date'].max()}")
    return md

# ─────────────────────────── FAST incremental masterdata (cached) ───────────────────────────
def _assemble_raw(symbols, days):
    """Raw per-day rows (cash OHLC + F&O metrics) — the slow Drive-I/O part. No CA-adjust/features yet."""
    symset=set(symbols); rows=[]
    for D in days:
        cash=get_cash_bhav(D); fom=fo_metrics_all(D)
        if cash is None: continue
        for sym in symset:
            if sym not in cash.index: continue
            c=cash.loc[sym]; c=c.iloc[0] if isinstance(c,pd.DataFrame) else c; m=fom.get(sym,{})
            rows.append({"date":D,"symbol":sym,"open":c["OPEN_PRICE"],"high":c["HIGH_PRICE"],"low":c["LOW_PRICE"],
                "close":c["CLOSE_PRICE"],"volume":c["TTL_TRD_QNTY"],"delivQty":c["DELIV_QTY"],"delivPct":c["DELIV_PER"],
                "oi":m.get("oi"),"oiChgAbs":m.get("oiChgAbs"),"callOI":m.get("callOI"),"putOI":m.get("putOI"),"pcr":m.get("pcr")})
    return pd.DataFrame(rows)

def _finalize_md(raw, start, ca_gap=CA_GAP):
    """Per-symbol CA back-adjust + EMA + pxChg/oiChg/pcrChg over the raw set (in-memory, fast)."""
    if raw.empty: return raw
    out=[]
    for sym,g in raw.groupby("symbol"):
        g=g.sort_values("date").reset_index(drop=True)
        cl=g["close"].values; nA=len(g); A=np.ones(nA); run=1.0
        for i in range(nA-1,0,-1):
            r=(cl[i]/cl[i-1]) if cl[i-1] else 1.0
            if r and (r<1-ca_gap or r>1+ca_gap): run*=r
            A[i-1]=run
        for col in ["open","high","low","close"]: g[col]=g[col]*A
        g["ema20"]=_ema(g["close"].tolist(),20); g["ema50"]=_ema(g["close"].tolist(),50)
        g["pxChg"]=(g["close"]/g["close"].shift(1)-1)*100
        prevoi=g["oi"]-g["oiChgAbs"]; g["oiChg"]=(g["oiChgAbs"]/prevoi.replace(0,np.nan))*100
        g["pcrChg"]=g["pcr"].diff(); out.append(g)
    return pd.concat(out, ignore_index=True).query("date >= @start").reset_index(drop=True)

def _weekdays_after(last, end):
    days=[]; d=pd.to_datetime(last)+pd.Timedelta(days=1); endd=pd.to_datetime(end)
    while d<=endd:
        if d.weekday()<5:
            ds=d.strftime("%Y-%m-%d")
            if get_cash_bhav(ds) is not None: days.append(ds)
        d+=pd.Timedelta(days=1)
    return days

def build_masterdata_pit_fast(start, end, warmup_days=100, full=False):
    """Same output as build_masterdata_pit but CACHED: raw rows + F&O membership are stored on Drive and only
       NEW sessions are fetched; CA-adjust + features run in-memory. First run ~ full time; later runs ~ seconds.
       Pass full=True occasionally (e.g. monthly) to rebuild cleanly (backfills warmup for newly-added F&O stocks)."""
    import json
    raw_fp=f"{CACHE}/md_raw.parquet"; mem_fp=f"{CACHE}/md_members.json"
    seed=(pd.to_datetime(start)-pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    if (not full) and os.path.exists(raw_fp) and os.path.exists(mem_fp):
        raw=pd.read_parquet(raw_fp); members={k:set(v) for k,v in json.load(open(mem_fp)).items()}
        last=max(raw["date"]); new_days=_weekdays_after(last, end)
        if new_days:
            print(f"incremental: +{len(new_days)} new session(s) [{new_days[0]} .. {new_days[-1]}]")
            for D in new_days:
                fo=get_fo_bhav(D); members[D]=set(fo.loc[fo["FinInstrmTp"]=="STF","TckrSymb"].astype(str).unique()) if fo is not None else set()
            ever=set().union(*members.values())
            raw=pd.concat([raw, _assemble_raw(ever, new_days)], ignore_index=True)
        else:
            print("masterdata cache already current — nothing to fetch")
    else:
        print("full build (no cache / full=True) — this one takes the usual few minutes ...")
        members={}
        for D in trading_days(seed, end):
            fo=get_fo_bhav(D); members[D]=set(fo.loc[fo["FinInstrmTp"]=="STF","TckrSymb"].astype(str).unique()) if fo is not None else set()
        ever=set().union(*members.values()) if members else set()
        raw=_assemble_raw(ever, trading_days(seed, end))
    os.makedirs(CACHE, exist_ok=True)
    raw.to_parquet(raw_fp); json.dump({k:sorted(v) for k,v in members.items()}, open(mem_fp,"w"))
    md=_finalize_md(raw, start)
    md["_in"]=[s in members.get(d, set()) for d,s in zip(md["date"], md["symbol"])]
    md=md[md["_in"]].drop(columns="_in").reset_index(drop=True)
    for c in ["open","high","low","close","volume","delivPct","delivQty","oi","oiChg","pxChg","pcr","pcrChg","callOI","putOI"]:
        if c in md.columns: md[c]=pd.to_numeric(md[c], errors="coerce")
    print(f"masterdata: {len(md):,} rows | {md['symbol'].nunique()} symbols | {md['date'].min()} .. {md['date'].max()}")
    return md

# ─────────────────────────── MICROSTRUCTURE FEATURES ───────────────────────────
def _bu(px, oi):
    if pd.isna(px) or pd.isna(oi): return ""
    if px>0 and oi>0: return "lng_bld"
    if px>0 and oi<0: return "sht_cov"
    if px<0 and oi>0: return "sht_bld"
    if px<0 and oi<0: return "lng_unv"
    return ""
_BULL={"lng_bld":1.0,"sht_cov":0.5}; _BEAR={"sht_bld":1.0,"lng_unv":0.5}
_REC10=np.array([0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.14,0.10,0.06])

def build_features(md, win=10):
    md = md.sort_values(["symbol","date"]).reset_index(drop=True).copy()
    g = md.groupby("symbol")
    md["oi_px_ratio"] = (md["oiChg"]/(md["pxChg"].abs()+0.3)).clip(-20,20)      # coiling vs momentum
    md["volx"]   = (md["volume"]/g["volume"].transform(lambda s: s.rolling(20,min_periods=5).mean())).clip(0,10)
    md["delivx"] = (md["delivPct"]/g["delivPct"].transform(lambda s: s.rolling(20,min_periods=5).mean())).clip(0,5)
    rng = (md["high"]-md["low"]).replace(0,np.nan)
    md["close_pos"] = ((md["close"]-md["low"])/rng).clip(0,1)
    md["impulse"] = (np.minimum(md["oiChg"].abs()/3,2)+np.minimum(md["pxChg"].abs()/2,2)
                     +np.minimum((md["volx"]-1).clip(lower=0),2)+np.minimum((md["delivx"]-1).clip(lower=0),2))/4
    bu=[_bu(p,o) for p,o in zip(md["pxChg"],md["oiChg"])]; md["buildup"]=bu
    bw=np.array([_BULL.get(b,0) for b in bu]); rw=np.array([_BEAR.get(b,0) for b in bu])
    md["day_conv"]=(bw-rw)*(0.5+0.5*np.minimum(md["impulse"].values,2))
    # rolling multi-session VWAP slope on daily candles (3 & 5 session), as %/session
    tp=(md["high"]+md["low"]+md["close"])/3; tpv=tp*md["volume"]
    for W in (3,5):
        num=tpv.groupby(md["symbol"]).transform(lambda s: s.rolling(W,min_periods=W).sum())
        den=md["volume"].groupby(md["symbol"]).transform(lambda s: s.rolling(W,min_periods=W).sum())
        md[f"vwap{W}"]=num/den.replace(0,np.nan)
    md["vwap3_slope"]=((md["vwap3"]/md.groupby("symbol")["vwap3"].shift(2)-1)*100/2).round(3)   # slope over last 3 candles
    md["vwap5_slope"]=((md["vwap5"]/md.groupby("symbol")["vwap5"].shift(4)-1)*100/4).round(3)   # slope over last 5 candles
    md.drop(columns=["vwap3","vwap5"], inplace=True)
    px_a=md["pxChg"].to_numpy(); dc_a=md["day_conv"].to_numpy(); bu_a=np.array(bu)
    dl_a=md["delivPct"].to_numpy(); cl_a=md["close"].to_numpy(); oi_a=md["oi"].to_numpy()
    rows=[]
    for sym, gi in md.groupby("symbol").indices.items():
        gi=np.sort(gi); n=len(gi)
        for k in range(n):
            i=gi[k]; lo=gi[max(0,k-win+1)]; wk=gi[max(0,k-win+1):k+1]; L=len(wk)
            bub=bu_a[wk]; bull=int(np.isin(bub,["lng_bld","sht_cov"]).sum()); bear=int(np.isin(bub,["sht_bld","lng_unv"]).sum())
            rb=0
            for j in range(k,-1,-1):
                if bu_a[gi[j]] in ("lng_bld","sht_cov"): rb+=1
                else: break
            rbr=0
            for j in range(k,-1,-1):
                if bu_a[gi[j]] in ("sht_bld","lng_unv"): rbr+=1
                else: break
            c0=cl_a[lo]; ret_win=(cl_a[i]/c0-1)*100 if c0>0 else 0
            seg=cl_a[wk]; peak=np.maximum.accumulate(seg); pull=float(((seg-peak)/peak).min()*100) if len(seg)>1 else 0.0
            up_frac=float((px_a[wk]>0).mean()); o0=oi_a[lo]; oi_trend=(oi_a[i]/o0-1)*100 if o0>0 else 0
            seg_dl=dl_a[wk]; dtr=float(np.polyfit(np.arange(L),seg_dl,1)[0]) if (L>=3 and not np.isnan(seg_dl).any()) else 0.0
            dc_sum=float(dc_a[wk].sum()); ww=_REC10[-L:]; dc_rec=float(np.dot(dc_a[wk], ww/ww.sum()))
            fh=bu_a[gi[max(0,k-win+1):max(0,k-win+1)+max(1,L//2)]]; last2=bu_a[gi[max(0,k-1):k+1]]
            f_acc=1 if (rb>=3 and oi_trend>0 and ret_win>0) else 0
            f_dist=1 if (np.isin(fh,["lng_bld"]).sum()>=2 and np.isin(last2,["lng_unv","sht_bld"]).any()) else 0
            f_cap=1 if (bear>=3 and (px_a[i]>=0 or bu_a[i]=="sht_cov")) else 0
            f_psh=1 if (rbr>=3 and oi_trend>0 and ret_win<0) else 0
            rows.append((i,bull,bear,rb,rbr,ret_win,pull,up_frac,oi_trend,dtr,dc_sum,dc_rec,f_acc,f_dist,f_cap,f_psh))
    fdf=pd.DataFrame(rows, columns=["_i","bull_days","bear_days","run_bull","run_bear","ret_win","pullback","up_day_frac",
        "oi_trend","deliv_trend","dayconv_sum","dayconv_recent","f_accumulation","f_distribution","f_capitulation",
        "f_persistent_short"]).set_index("_i").sort_index()
    md=md.join(fdf)
    md["mkt_ret"]=md.groupby("date")["ret_win"].transform("median")
    md["ret_rel"]=md["ret_win"]-md["mkt_ret"]
    for c in ["dayconv_recent","ret_win","oi_trend","delivx","impulse","oi_px_ratio"]:
        md[f"xr_{c}"]=md.groupby("date")[c].rank(pct=True)
    return md

FEATURES=["pxChg","oiChg","oi_px_ratio","volx","delivx","delivPct","close_pos","impulse","day_conv","pcr","pcrChg",
    "bull_days","bear_days","run_bull","run_bear","ret_win","pullback","up_day_frac","oi_trend","deliv_trend",
    "dayconv_sum","dayconv_recent","f_accumulation","f_distribution","f_capitulation","f_persistent_short",
    "ret_rel","xr_dayconv_recent","xr_ret_win","xr_oi_trend","xr_delivx","xr_impulse","xr_oi_px_ratio",
    "vwap3_slope","vwap5_slope"]

# ─────────────────────────── LABELS + CALIBRATION ───────────────────────────
def build_label(md, T=4.0, H=7):
    """ONE directional target: dir_label = 1 if the stock resolves UP (+T% before -T%) within H days,
       0 if DOWN first. Same-day both -> the bigger excursion wins. Timeout -> sign of the net H-day move."""
    md = md.sort_values(["symbol","date"]).copy(); lab = pd.Series(index=md.index, dtype="float")
    for sym,g in md.groupby("symbol"):
        idx=g.index.to_numpy(); c=g["close"].to_numpy(); h=g["high"].to_numpy(); l=g["low"].to_numpy(); n=len(g)
        v=np.full(n,np.nan)
        for i in range(n):
            e=c[i]
            if not (e>0): continue
            ut=e*(1+T/100); dt=e*(1-T/100); hi=min(n,i+1+H); r=np.nan
            for j in range(i+1,hi):
                up=h[j]>=ut; dn=l[j]<=dt
                if up and dn: r=1.0 if (h[j]-e)>=(e-l[j]) else 0.0; break
                if up: r=1.0; break
                if dn: r=0.0; break
            if np.isnan(r) and hi>i+1: r=1.0 if c[hi-1]>=e else 0.0
            v[i]=r
        lab.loc[idx]=v
    md["dir_label"]=lab.values
    return md

def calibrate(feat, label, features=None, n_blocks=6, embargo=7, C=0.5, verbose=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    features = features or [c for c in FEATURES if c in feat.columns]
    d = feat.dropna(subset=[label]).copy()
    dates = np.array(sorted(d["date"].unique()))
    blocks = np.array_split(dates[len(dates)//3:], n_blocks)
    oos=[]; coefs=[]
    for bi, blk in enumerate(blocks):
        if len(blk)==0: continue
        t0=blk[0]; prior=dates[dates<t0]; trd=prior[:-embargo] if len(prior)>embargo else prior
        if len(trd)<40: continue
        tr=d[d["date"].isin(trd)]; te=d[(d["date"]>=blk[0])&(d["date"]<=blk[-1])]
        if len(tr)<500 or len(te)<50: continue
        med=tr[features].median()
        Xtr=tr[features].fillna(med).replace([np.inf,-np.inf],0.0); Xte=te[features].fillna(med).replace([np.inf,-np.inf],0.0)
        sc=StandardScaler().fit(Xtr); Xtr_s=sc.transform(Xtr); Xte_s=sc.transform(Xte)
        m=LogisticRegression(max_iter=2000, C=C).fit(Xtr_s, tr[label].astype(int))
        p=m.predict_proba(Xte_s)[:,1]
        r=te[["date","symbol",label]].copy(); r["conv"]=(p*100).round(1); r["_contrib"]=list(Xte_s*m.coef_[0])
        oos.append(r); coefs.append(pd.Series(m.coef_[0],index=features))
        if verbose: print(f"  fold {bi+1}: train {len(tr):,} -> test {len(te):,} [{blk[0]}..{blk[-1]}]")
    O=pd.concat(oos).reset_index(drop=True) if oos else pd.DataFrame()
    coef=pd.concat(coefs,axis=1).mean(axis=1) if coefs else pd.Series(dtype=float)
    coef=coef.reindex(coef.abs().sort_values(ascending=False).index)
    return O, coef, features

def evaluate(oos, label):
    from sklearn.metrics import roc_auc_score
    y=oos[label].astype(int); out={"n":len(oos),"base":round(float(y.mean()),3),"auc":round(float(roc_auc_score(y,oos["conv"])),3)}
    thr=oos["conv"].quantile(0.90); top=oos[oos["conv"]>=thr]
    out["top10%_hit"]=round(float(top[label].mean()),3); out["top10%_n"]=int(len(top))
    return out

def top_drivers(oos, features, i, k=6):
    s=pd.Series(np.array(oos.iloc[i]["_contrib"]), index=features)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(k).round(2)

# ─────────────────────────── LIVE SCORING + DASHBOARD TABLES ───────────────────────────
def fit_final(feat, label, features=None, C=0.5):
    """Final model on ALL labeled data — used to score recent (unlabeled) days for the live dashboard."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    features = features or [c for c in FEATURES if c in feat.columns]
    d = feat.dropna(subset=[label]); med = d[features].median()
    X = d[features].fillna(med).replace([np.inf,-np.inf],0.0)
    sc = StandardScaler().fit(X); m = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X), d[label].astype(int))
    return {"m":m, "sc":sc, "med":med, "features":features, "base": float(d[label].mean())}

def score_all(feat, bundle):
    f=bundle["features"]; X=feat[f].fillna(bundle["med"]).replace([np.inf,-np.inf],0.0); Xs=bundle["sc"].transform(X)
    out=feat[["date","symbol"]].copy(); out["conv"]=(bundle["m"].predict_proba(Xs)[:,1]*100).round(1)
    out["_contrib"]=list(Xs*bundle["m"].coef_[0]); return out

def driver_str(contrib, features, k=3):
    s=pd.Series(np.array(contrib), index=features); s=s.reindex(s.abs().sort_values(ascending=False).index).head(k)
    return ", ".join(f"{f} {'+' if v>=0 else ''}{round(float(v),1)}" for f,v in s.items())

def build_dashboard(feat, months=3, last_n=5):
    """Returns (dash, hist): ONE signed conviction (-100..+100). dash = today's F&O stocks with the last-5
       sessions' score + latest drivers; hist = last-`months` signed-score grid (stock x date)."""
    b=fit_final(feat,"dir_label"); sc=score_all(feat,b)
    sc["score"]=(2*sc["conv"]-100).round(1)                              # signed: +bullish / -bearish
    dates=sorted(feat["date"].unique()); latest=dates[-1]
    today=sorted(feat[feat.date==latest]["symbol"].unique()); l5=dates[-last_n:]
    piv=sc[sc.symbol.isin(today)&sc.date.isin(l5)].pivot_table(index="symbol",columns="date",values="score").reindex(index=today,columns=l5)
    piv.columns=[c[5:] for c in l5]
    lc=sc[(sc.date==latest)&(sc.symbol.isin(today))].set_index("symbol")
    drv={s: driver_str(lc.loc[s,"_contrib"], b["features"]) for s in lc.index}
    vwl=feat[feat.date==latest].drop_duplicates("symbol").set_index("symbol")
    def _vw(s,col):
        return round(float(vwl.loc[s,col]),2) if (s in vwl.index and pd.notna(vwl.loc[s,col])) else ""
    dash=piv.copy()
    dash["vwap3 slp"]=[_vw(s,"vwap3_slope") for s in dash.index]      # rolling VWAP slope next to the score
    dash["vwap5 slp"]=[_vw(s,"vwap5_slope") for s in dash.index]
    dash["Drivers (latest)"]=[drv.get(s,"") for s in dash.index]
    dash=dash.reset_index().rename(columns={"index":"symbol"}).sort_values(piv.columns[-1], ascending=False, na_position="last")
    cutoff=(pd.to_datetime(latest)-pd.DateOffset(months=months)).strftime("%Y-%m-%d"); hd=[d for d in dates if d>=cutoff]
    hist=sc[sc.symbol.isin(today)&sc.date.isin(hd)].pivot_table(index="symbol",columns="date",values="score").reindex(index=today).round(0)
    hist.columns=[c[5:] for c in hist.columns]
    return dash, hist.reset_index()

# ═════════════════════ RULE ENGINE — exact port of "FUT OI Conviction Engine" (.gs) ═════════════════════
# Same 7 components / weights / recency / thresholds / trend-damping as the Apps Script.
# Minor mods applied (user-approved): (#1) null/zero divide guards so data-gap rows don't NaN out;
# (#2) point-in-time universe + (#3) CA-adjusted prices come free from build_masterdata_pit.
CV_WEIGHTS={"BM":20,"TA":20,"OR":15,"BT":15,"DC":10,"PS":10,"TG":10,"VW":12}
CV_BUILDUP_VAL={"lng_bld":1.0,"sht_cov":0.6,"sht_bld":-1.0,"lng_unv":-0.6}
CV_RECENCY5=[0.10,0.15,0.20,0.25,0.30]
CV_MIN_HISTORY=7
CV_OR_WIN=10; CV_BT_WIN=10; CV_NORM_WIN=20   # rule-score lookbacks (bars). DAILY defaults; the WEEKLY pass overrides these shorter.

def _cvnum(v):
    if v is None or v=="": return None
    try: n=float(v)
    except: return None
    return None if n!=n else n
def _cvsign(x): return (x>0)-(x<0)
def _cvclip1(x): return max(-1.0,min(1.0,x))
def _cvmag(oiChg,pxChg):
    m=0.5*min(abs(oiChg or 0)/3.0,1.0)+0.5*min(abs(pxChg or 0)/2.0,1.0)
    return max(m,0.25)
def _cvrollmean(rows,i,field,win,minP):
    lo=max(0,i-win+1); s=0.0; c=0
    for j in range(lo,i+1):
        v=_cvnum(rows[j][field])
        if v is not None: s+=v; c+=1
    return s/c if c>=minP else None
def _cvpriorext(rows,i,field,ismax):
    lo=max(0,i-CV_BT_WIN); vals=[]
    for j in range(lo,i):
        v=_cvnum(rows[j][field])
        if v is not None: vals.append(v)
    if len(vals)<min(5,CV_BT_WIN): return None   # daily (win=10) still needs >=5; short weekly windows need fewer
    return max(vals) if ismax else min(vals)
def _cvsignal(score):
    if score is None: return ""
    if score>=55: return "STRONG BULLISH (Call)"
    if score>=30: return "Bullish watch"
    if score>-30: return "Neutral"
    if score>-55: return "Bearish watch"
    return "STRONG BEARISH (Put)"

def cv_score_symbol(rows):
    """Exact port of cvScoreSymbol. rows: list of dicts (date asc, one symbol) with keys
       dk,high,low,close,volume,delQty,delPct,ema20,ema50,oi,oiChg,pxChg,buildup,pcr,callOI,putOI."""
    n=len(rows); out=[{"score":None,"comps":{},"patterns":"","signal":""} for _ in range(n)]
    for i in range(n):
        if i<CV_MIN_HISTORY-1: continue
        row=rows[i]; notes=[]; comps={}
        volAvg20=_cvrollmean(rows,i,"volume",CV_NORM_WIN,min(5,CV_NORM_WIN)); delAvg20=_cvrollmean(rows,i,"delPct",CV_NORM_WIN,min(5,CV_NORM_WIN))
        hi10=_cvpriorext(rows,i,"high",True); lo10=_cvpriorext(rows,i,"low",False)
        # BM — buildup momentum
        bm=0.0; wsum=0.0
        for k in range(5):
            r=rows[i-4+k]; v=CV_BUILDUP_VAL.get(str(r["buildup"]).strip(),0)
            bm+=CV_RECENCY5[k]*v*_cvmag(_cvnum(r["oiChg"]),_cvnum(r["pxChg"])); wsum+=CV_RECENCY5[k]
        bm/=wsum
        prev=rows[i-1]; oi2=(_cvnum(row["oiChg"]) or 0)+(_cvnum(prev["oiChg"]) or 0); px2=(_cvnum(row["pxChg"]) or 0)+(_cvnum(prev["pxChg"]) or 0)
        if row["buildup"]=="lng_bld" and prev["buildup"]=="lng_bld" and oi2>6 and px2>2.5: bm+=0.3; notes.append("aggressive-long-accel")
        if row["buildup"]=="sht_bld" and prev["buildup"]=="sht_bld" and oi2>6 and px2<-2.5: bm-=0.3; notes.append("aggressive-short-accel")
        comps["BM"]=_cvclip1(bm)
        # TA — trap / absorption
        ta=0.0; decayMap={1:1.0,2:0.75,3:0.5}
        for k in (1,2,3):
            j=i-k
            if j<1: break
            r=rows[j]; rOi=_cvnum(r["oiChg"]) or 0; decay=decayMap[k]
            if r["buildup"]=="sht_bld" and rOi>0.5:
                minClose=float("inf"); sawCov=False
                for m in range(j+1,i+1):
                    cm=_cvnum(rows[m]["close"])
                    if cm is not None: minClose=min(minClose,cm)
                    if rows[m]["buildup"]=="sht_cov": sawCov=True
                if minClose>=(_cvnum(r["low"]) or minClose) and (_cvnum(row["close"]) or 0)>=(_cvnum(r["close"]) or 0):
                    ta+=0.5*decay*min(rOi/2.0,1.5)*(1.3 if sawCov else 1.0); notes.append("trapped-shorts("+str(r["dk"])+")")
            if r["buildup"]=="lng_bld" and rOi>0.5:
                maxClose=float("-inf"); sawUnw=False
                for m in range(j+1,i+1):
                    cm=_cvnum(rows[m]["close"])
                    if cm is not None: maxClose=max(maxClose,cm)
                    if rows[m]["buildup"]=="lng_unv": sawUnw=True
                if maxClose<=(_cvnum(r["high"]) or maxClose) and (_cvnum(row["close"]) or 0)<=(_cvnum(r["close"]) or 0):
                    ta-=0.5*decay*min(rOi/2.0,1.5)*(1.3 if sawUnw else 1.0); notes.append("trapped-longs("+str(r["dk"])+")")
        rng=(_cvnum(row["high"]) or 0)-(_cvnum(row["low"]) or 0); pxChg=_cvnum(row["pxChg"]) or 0; delPct=_cvnum(row["delPct"])
        if rng>0 and pxChg<0 and delAvg20 is not None and delPct is not None:
            if delPct>delAvg20*1.15 and (_cvnum(row["close"])-_cvnum(row["low"]))/rng>0.5: ta+=0.3; notes.append("absorption")
        comps["TA"]=_cvclip1(ta)
        # OR — OI regime (10d)  [#1 guards on oi0/px0/oi(i-2)]
        lo=max(0,i-(CV_OR_WIN-1)); oi0=_cvnum(rows[lo]["oi"]); oi1=_cvnum(row["oi"]); px0=_cvnum(rows[lo]["close"]); px1=_cvnum(row["close"]); oip2=_cvnum(rows[i-2]["oi"])
        orv=0.0
        if oi0 and px0 and oi1 is not None and px1 is not None:
            oiD=(oi1-oi0)/oi0*100; pxD=(px1-px0)/px0*100; oiRecent=(oi1-oip2)/oip2*100 if oip2 else 0.0
            if pxD>1:
                if oiD<-1 and oiRecent>1: orv=0.9; notes.append("base+fresh-longs")
                elif oiD<-1: orv=0.5; notes.append("short-cover-uptrend")
                elif oiD>1: orv=0.7; notes.append("healthy-long-trend")
                else: orv=0.3
            elif pxD<-1:
                if oiD>1: orv=-0.8; notes.append("shorts-in-control")
                elif oiD<-1: orv=-0.4; notes.append("long-liquidation")
                else: orv=-0.3
            else: orv=_cvclip1(0.1*_cvsign(oiRecent)*_cvsign(pxD)) if pxD!=0 else 0.0
        comps["OR"]=orv
        # BT — breakout thrust
        bt=0.0; volX=(_cvnum(row["volume"])/volAvg20) if volAvg20 else 1.0; oiChg=_cvnum(row["oiChg"]) or 0; cl=_cvnum(row["close"])
        if hi10 is not None and cl is not None and cl>hi10:
            bt=0.5
            if volX>1.5: bt+=0.25
            if row["buildup"]=="lng_bld": bt+=0.25
            if oiChg>3: bt=min(bt+0.2,1.0)
            notes.append("breakout-10d")
        elif lo10 is not None and cl is not None and cl<lo10:
            bt=-0.5
            if volX>1.5: bt-=0.25
            if row["buildup"]=="sht_bld": bt-=0.25
            if oiChg>3: bt=max(bt-0.2,-1.0)
            notes.append("breakdown-10d")
        comps["BT"]=bt
        # DC — delivery conviction (last 3 days)
        dc=0.0; dcW=[0.2,0.3,0.5]
        for k in (2,1,0):
            r=rows[i-k]; rDelPct=_cvnum(r["delPct"]); rDelQty=_cvnum(r["delQty"])
            rDelAvg=_cvrollmean(rows,i-k,"delPct",CV_NORM_WIN,min(5,CV_NORM_WIN)); rQtyAvg=_cvrollmean(rows,i-k,"delQty",CV_NORM_WIN,min(5,CV_NORM_WIN))
            if rDelPct is None or rDelQty is None or not rDelAvg or not rQtyAvg: continue
            relPct=(rDelPct/rDelAvg)-1.0; relQty=(rDelQty/rQtyAvg)-1.0
            rel=0.4*_cvclip1(relPct*2.5)+0.6*_cvclip1(relQty*1.5)
            dc+=dcW[2-k]*_cvsign(_cvnum(r["pxChg"]) or 0)*rel
        comps["DC"]=_cvclip1(dc)
        # PS — PCR structure  [#1 guard on close(i-4)]
        ps=0.0; pcrNow=_cvnum(row["pcr"]); pcrThen=_cvnum(rows[i-4]["pcr"]); c4=_cvnum(rows[i-4]["close"])
        pxD5=(_cvnum(row["close"])-c4)/c4*100 if c4 else 0.0
        if pcrNow is not None and pcrThen is not None:
            pcrD5=pcrNow-pcrThen
            if pcrD5>0.03 and pxD5>=-0.5: ps+=0.5; notes.append("put-writing-support")
            if pcrD5<-0.03 and pxD5<-0.5: ps-=0.5; notes.append("call-writing-pressure")
        putNow=_cvnum(row["putOI"]); putThen=_cvnum(rows[i-4]["putOI"]); callNow=_cvnum(row["callOI"]); callThen=_cvnum(rows[i-4]["callOI"])
        if putNow is not None and putThen and callNow is not None and callThen:
            putG=(putNow-putThen)/putThen; callG=(callNow-callThen)/callThen
            if pxD5>1 and callG>0.1 and putG>0.05: ps+=0.3
            elif pxD5<-1 and callG>0.1 and putG<0: ps-=0.3
        comps["PS"]=_cvclip1(ps)
        # TG — trend gate  [#1 guard on close]
        tg=0.0; e20=_cvnum(row["ema20"]); e50=_cvnum(row["ema50"])
        if e20 is not None and e50 is not None and cl:
            spread=(e20-e50)/cl*100; above=(cl-e20)/cl*100; tg=_cvclip1(spread/4.0)*0.6+_cvclip1(above/3.0)*0.4
        comps["TG"]=_cvclip1(tg)
        # VW — rolling multi-session VWAP slope (3 & 5 session): rising VWAP = bullish, steeper = stronger
        vw3=_cvnum(row.get("vwap3_slope")); vw5=_cvnum(row.get("vwap5_slope")); vw=0.0
        if vw3 is not None: vw+=0.6*_cvclip1(vw3/1.5)        # ~1.5%/session saturates the 3-session read
        if vw5 is not None: vw+=0.4*_cvclip1(vw5/1.0)        # ~1.0%/session saturates the 5-session read
        comps["VW"]=_cvclip1(vw)
        # composite + trend damping
        pos=0.0; neg=0.0
        for kk in CV_WEIGHTS:
            v=CV_WEIGHTS[kk]*comps[kk]
            if v>0: pos+=v
            else: neg+=v
        if tg>0.3: neg*=0.6
        elif tg<-0.3: pos*=0.6
        out[i]["score"]=max(-100.0,min(100.0,pos+neg)); out[i]["comps"]=comps
        out[i]["patterns"]=", ".join(notes); out[i]["signal"]=_cvsignal(out[i]["score"])
    return out

def cv_score_all(feat):
    """Run the exact rule engine over the point-in-time, CA-adjusted masterdata (pass `feat` — it carries
       the buildup label + every field the engine reads). Returns date,symbol,score,signal,patterns,comps."""
    d=feat.sort_values(["symbol","date"]); recs=[]
    for sym,g in d.groupby("symbol"):
        rows=[{"dk":dk,"high":h,"low":l,"close":c,"volume":vol,"delQty":dq,"delPct":dp,"ema20":e2,"ema50":e5,
               "oi":oi,"oiChg":oic,"pxChg":pxc,"buildup":str(bu).strip(),"pcr":pcr,"callOI":co,"putOI":po,
               "vwap3_slope":v3,"vwap5_slope":v5}
              for dk,h,l,c,vol,dq,dp,e2,e5,oi,oic,pxc,bu,pcr,co,po,v3,v5 in zip(
                  g["date"],g["high"],g["low"],g["close"],g["volume"],g["delivQty"],g["delivPct"],g["ema20"],g["ema50"],
                  g["oi"],g["oiChg"],g["pxChg"],g["buildup"],g["pcr"],g["callOI"],g["putOI"],
                  g.get("vwap3_slope",pd.Series([None]*len(g))),g.get("vwap5_slope",pd.Series([None]*len(g))))]
        for r,s in zip(rows,cv_score_symbol(rows)):
            recs.append((r["dk"],sym,s["score"],s["signal"],s["patterns"],s["comps"]))
    return pd.DataFrame(recs,columns=["date","symbol","score","signal","patterns","comps"])

_CV_LAB={"BM":"Buildup","TA":"Trap","OR":"Regime","BT":"Brkout","DC":"Deliv","PS":"PCR","TG":"Trend","VW":"VWAPslp"}
def _cv_driver(comps):
    if not comps: return ""
    contrib=sorted(((k,round(CV_WEIGHTS[k]*comps.get(k,0),1)) for k in CV_WEIGHTS), key=lambda kv: abs(kv[1]), reverse=True)
    return ", ".join(f"{_CV_LAB[k]} {'+' if v>=0 else ''}{v}" for k,v in contrib[:3] if v!=0)

def build_rule_dashboard(feat, months=3, last_n=5):
    """Rule-engine dashboard + history, structured exactly like the calibrated CV2 tables:
       dash = today's F&O stocks, last-`last_n` rule scores + latest Signal/Drivers/Patterns;
       hist = last-`months` rule-score grid (stock x date). Signed score -100..+100."""
    sc=cv_score_all(feat)
    dates=sorted(feat["date"].unique()); latest=dates[-1]
    today=sorted(feat[feat.date==latest]["symbol"].unique()); l5=dates[-last_n:]
    piv=sc[sc.symbol.isin(today)&sc.date.isin(l5)].pivot_table(index="symbol",columns="date",values="score").reindex(index=today,columns=l5)
    piv.columns=[c[5:] for c in l5]
    lc=sc[(sc.date==latest)&(sc.symbol.isin(today))].drop_duplicates("symbol").set_index("symbol")
    vwl=feat[feat.date==latest].drop_duplicates("symbol").set_index("symbol")
    def _vw(s,col): return round(float(vwl.loc[s,col]),2) if (s in vwl.index and pd.notna(vwl.loc[s,col])) else ""
    dash=piv.copy()
    dash["vwap3 slp"]=[_vw(s,"vwap3_slope") for s in dash.index]      # rolling VWAP slope next to the score
    dash["vwap5 slp"]=[_vw(s,"vwap5_slope") for s in dash.index]
    dash["Signal (latest)"]=[lc.loc[s,"signal"] if s in lc.index else "" for s in dash.index]
    dash["Drivers (latest)"]=[_cv_driver(lc.loc[s,"comps"]) if s in lc.index else "" for s in dash.index]
    dash["Patterns (latest)"]=[lc.loc[s,"patterns"] if s in lc.index else "" for s in dash.index]
    dash=dash.reset_index().rename(columns={"index":"symbol"}).sort_values(piv.columns[-1],ascending=False,na_position="last")
    cutoff=(pd.to_datetime(latest)-pd.DateOffset(months=months)).strftime("%Y-%m-%d"); hd=[d for d in dates if d>=cutoff]
    hist=sc[sc.symbol.isin(today)&sc.date.isin(hd)].pivot_table(index="symbol",columns="date",values="score").reindex(index=today).round(0)
    hist.columns=[c[5:] for c in hist.columns]
    return dash, hist.reset_index()

# ═════════════════════ FUT OI vs TIME chart (Sensibull-style) ═════════════════════
CV_BU_PLOT={"lng_bld":("#147d6f","Long Buildup"), "sht_cov":("#63c99a","Short Covering"),
            "sht_bld":("#f0576f","Short Buildup"), "lng_unv":("#f6b545","Long Unwinding"),
            "":("#b8bcc2","No Change")}

def plot_fut_oi(feat, symbol, n=25, ax=None, save=None, oi_div=1e7, oi_unit="Cr"):
    """Fut OI vs Time: bars = front-month OI colored by that day's buildup, dark line = close (right axis).
       Reproduces the Sensibull chart from our own masterdata. Returns the figure (or the ax if given one)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch; from matplotlib.lines import Line2D; from matplotlib.ticker import FuncFormatter
    d=feat[feat["symbol"]==symbol].sort_values("date").tail(n)
    if d.empty: print("no data for", symbol); return None
    dates=list(d["date"]); oi=pd.to_numeric(d["oi"],errors="coerce").to_numpy(float)
    close=pd.to_numeric(d["close"],errors="coerce").to_numpy(float)
    bu=[str(b).strip() for b in d["buildup"]]; colors=[CV_BU_PLOT.get(b,CV_BU_PLOT[""])[0] for b in bu]
    x=np.arange(len(d)); own=ax is None
    if own:
        fig,ax=plt.subplots(figsize=(14,7)); fig.patch.set_facecolor("white")
    ax.bar(x, oi, width=0.72, color=colors, zorder=2)
    ax.set_axisbelow(True); ax.grid(axis="y", color="#ececf1", lw=1)
    ax.set_ylabel("Fut OI", color="#6b7280", fontsize=11)
    ax.set_xlim(-0.7, len(d)-0.3); ax.set_ylim(0, (np.nanmax(oi) or 1)*1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: (f"{v/oi_div:.0f}{oi_unit}" if v>0 else "0")))
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0, colors="#6b7280", labelsize=9)
    ax2=ax.twinx(); ax2.plot(x, close, color="#22314f", lw=2.2, zorder=3)
    ax2.set_ylabel(symbol, color="#6b7280", rotation=270, labelpad=20, fontsize=11)
    pad=(np.nanmax(close)-np.nanmin(close))*0.35 or 1; ax2.set_ylim(np.nanmin(close)-pad, np.nanmax(close)+pad)
    for s in ("top","left"): ax2.spines[s].set_visible(False)
    ax2.spines["right"].set_visible(False); ax2.tick_params(length=0, colors="#6b7280", labelsize=9)
    labels=[]; prev_m=None
    for dt in dates:
        ts=pd.to_datetime(dt); lab=ts.strftime("%d")
        if ts.strftime("%b")!=prev_m: lab=ts.strftime("%d\n%b"); prev_m=ts.strftime("%b")
        labels.append(lab)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, color="#6b7280")
    ax.set_title("Fut OI Vs Time", loc="left", fontsize=17, fontweight="bold", color="#2b2f36", pad=18)
    handles=[Patch(color=c, label=l) for c,l in [CV_BU_PLOT[k] for k in ("lng_bld","sht_cov","sht_bld","lng_unv","")]]
    handles.append(Line2D([0],[0], color="#22314f", lw=2.4, label=symbol))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,-0.10), ncol=6, frameon=False, fontsize=10)
    if own:
        plt.tight_layout()
        if save: plt.savefig(save, dpi=130, bbox_inches="tight", facecolor="white"); print("saved", save)
        return fig
    return ax

def plot_fut_oi_interactive(feat, symbol, n=30):
    """Interactive Plotly version — hover a bar to see Px %, OI %, and Vol (x SMA-20). fig.show() in Colab."""
    import plotly.graph_objects as go
    d=feat[feat["symbol"]==symbol].sort_values("date").tail(n)
    if d.empty: print("no data for", symbol); return None
    dx=[str(x)[5:] for x in d["date"]]
    oi=pd.to_numeric(d["oi"],errors="coerce"); close=pd.to_numeric(d["close"],errors="coerce")
    px=pd.to_numeric(d["pxChg"],errors="coerce"); oic=pd.to_numeric(d["oiChg"],errors="coerce")
    vx=pd.to_numeric(d.get("volx"),errors="coerce") if "volx" in d else pd.Series([np.nan]*len(d))
    bu=[str(b).strip() for b in d["buildup"]]
    fig=go.Figure()
    for key in ("lng_bld","sht_cov","sht_bld","lng_unv",""):
        col,lab=CV_BU_PLOT[key]; m=[b==key for b in bu]
        if not any(m): continue
        cd=np.array([[px.iloc[i],oic.iloc[i],vx.iloc[i]] for i in range(len(d)) if m[i]],dtype=float)
        fig.add_bar(x=[dx[i] for i in range(len(d)) if m[i]], y=[oi.iloc[i] for i in range(len(d)) if m[i]],
                    name=lab, marker_color=col, customdata=cd,
                    hovertemplate="<b>%{x}</b><br>Px %{customdata[0]:.2f}%<br>OI %{customdata[1]:.2f}%<br>Vol %{customdata[2]:.2f}x SMA20<extra></extra>")
    fig.add_scatter(x=dx, y=close, name=symbol, yaxis="y2", mode="lines",
                    line=dict(color="#22314f",width=2.4), hovertemplate="Close %{y:.2f}<extra></extra>")
    fig.update_layout(title=dict(text="Fut OI Vs Time",x=0.01,font=dict(size=18,color="#2b2f36")),
        barmode="overlay", plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(categoryorder="array",categoryarray=dx,tickangle=0,showgrid=False),
        yaxis=dict(title="Fut OI",showgrid=True,gridcolor="#ececf1",rangemode="tozero"),
        yaxis2=dict(title=symbol,overlaying="y",side="right",showgrid=False),
        legend=dict(orientation="h",yanchor="top",y=-0.12,x=0.5,xanchor="center"),
        margin=dict(l=60,r=60,t=60,b=70), height=560)
    return fig

def build_chart_data(feat, symbols, n=30):
    """Compact per-stock series for the in-Sheet popup. Returns DataFrame [symbol, data_json] to write as a
       'ChartData' sheet; the Apps Script reads a symbol's JSON and renders the Plotly popup with hover."""
    import json
    def L(s, r=None):
        return [ (None if pd.isna(v) else (round(float(v),r) if r is not None else float(v))) for v in s ]
    recs=[]
    for s in symbols:
        d=feat[feat["symbol"]==s].sort_values("date").tail(n)
        if d.empty: continue
        vx=d["volx"] if "volx" in d else pd.Series([np.nan]*len(d))
        payload={"symbol":s,"date":[str(x)[5:] for x in d["date"]],
                 "close":L(d["close"],2),"oi":L(d["oi"]),"px":L(d["pxChg"],2),"oic":L(d["oiChg"],2),
                 "volx":L(vx,2),"bu":[str(x).strip() for x in d["buildup"]]}
        recs.append((s, json.dumps(payload,separators=(",",":"))))
    return pd.DataFrame(recs, columns=["symbol","data_json"])

_NSE_SESS=None
def _nse_session(force=False):
    """Build (and cache) an NSE session with warmed cookies. NSE blocks bare API hits; it needs a
       homepage visit + a market-data page visit first to set bm_sv/nsit/ak_bmsc cookies."""
    global _NSE_SESS
    if _NSE_SESS is not None and not force: return _NSE_SESS
    hdr={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
         "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language":"en-US,en;q=0.9",
         "Accept-Encoding":"gzip, deflate, br","Connection":"keep-alive","Upgrade-Insecure-Requests":"1"}
    s=requests.Session(); s.headers.update(hdr)
    try:
        s.get("https://www.nseindia.com", timeout=15)
        s.get("https://www.nseindia.com/market-data/live-equity-market", timeout=15)  # warms the api cookies
    except Exception as e:
        print("nse warmup error:", e)
    _NSE_SESS=s; return s

def fetch_live_fno_nse(_retry=2):
    """[FALLBACK] Live snapshot of the whole F&O universe from NSE (one call). Often 404s from cloud/Colab
       IPs (NSE blocks server IPs on /api/). Kept for local runs. Returns {symbol:{'pct','high','low','ltp'}}."""
    def num(x):
        try: return round(float(str(x).replace(",","")),2)
        except: return ""
    from urllib.parse import quote
    # NSE 404s on '+'-encoded spaces (requests' params= default); build the query with %20 / %26 ourselves
    url="https://www.nseindia.com/api/equity-stockIndices?index="+quote("SECURITIES IN F&O", safe="")
    api_hdr={"Accept":"application/json, text/plain, */*","Referer":"https://www.nseindia.com/market-data/live-equity-market",
             "X-Requested-With":"XMLHttpRequest"}
    for attempt in range(_retry+1):
        s=_nse_session(force=(attempt>0))
        try:
            r=s.get(url, headers=api_hdr, timeout=20)
            if r.status_code!=200 or not r.text.strip().startswith(("{","[")):
                print(f"live fetch attempt {attempt+1}: status {r.status_code}, body[:80]={r.text[:80]!r}")
                continue
            data=(r.json() or {}).get("data", [])
            return {str(d.get("symbol")).strip(): {"pct":num(d.get("pChange")), "high":num(d.get("dayHigh")),
                        "low":num(d.get("dayLow")), "ltp":num(d.get("lastPrice"))}
                    for d in data if d.get("symbol")}
        except Exception as e:
            print(f"live fetch attempt {attempt+1} error:", e)
    return {}

# ---- LIVE today's-move via Sensibull (primary; reuses the candle-run auth model) ----
_SENSI_COOKIE=None; _SENSI_BASE="https://oxide.sensibull.com"
def set_sensibull_cookie(c):
    """Paste a fresh Sensibull cookie (same one the candle-run module uses). Not stored to disk."""
    global _SENSI_COOKIE; _SENSI_COOKIE=c; print("sensibull cookie set")

def _sensi_sess():
    s=requests.Session()
    s.headers.update({"cookie":_SENSI_COOKIE or "","content-type":"application/json",
        "origin":"https://web.sensibull.com","referer":"https://web.sensibull.com/","user-agent":"Mozilla/5.0"})
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=24, pool_maxsize=24))
    return s

def fetch_live_fno(symbols, workers=12):
    """Live today's move per F&O stock via Sensibull daily candles (call set_sensibull_cookie() first).
       Uses each symbol's OWN prev-day close, so it needs no bhavcopy join. Threaded for speed.
       Returns {symbol: {'pct','high','low','ltp'}}."""
    if not _SENSI_COOKIE:
        print("live: call set_sensibull_cookie('<cookie>') first"); return {}
    import concurrent.futures as cf
    sess=_sensi_sess()
    frm=(pd.Timestamp.now().normalize()-pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    to =(pd.Timestamp.now().normalize()+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    def num(x):
        try: return round(float(x),2)
        except: return ""
    def pctof(v, base):
        try: return round((float(v)-base)/base*100,2) if base else ""
        except: return ""
    def one(sym):
        try:
            r=sess.post(f"{_SENSI_BASE}/v1/compute/1/candles/{sym}",
                        json={"from_date":frm,"to_date":to,"interval":"1D","skip_last_ts":False}, timeout=20)
            c=((r.json() or {}).get("payload") or {}).get("candles") or []
            if len(c)<2: return sym, None
            prev=float(c[-2]["close"]); cur=c[-1]
            ltp=float(cur["close"]); hi=cur.get("high"); lo=cur.get("low")
            return sym, {"pct":pctof(ltp,prev), "high":num(hi), "low":num(lo), "ltp":num(ltp),
                         "high_pct":pctof(hi,prev), "low_pct":pctof(lo,prev)}
        except Exception:
            return sym, None
    out={}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for sym,val in ex.map(one, list(symbols)):
            if val: out[sym]=val
    return out

print("conviction_engine v2 (self-contained) loaded | features:", len(FEATURES), "| 7-day target")
print("live: set_sensibull_cookie('<cookie>') then fetch_live_fno(symbols)  [NSE fallback: fetch_live_fno_nse()]")
