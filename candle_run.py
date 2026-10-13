# ==================================================================================
# CANDLE RUN — shared engine (local port of the Apps Script "Candle Run" module).
#   Core trigger: a run of N in-direction candles on the underlying (green->CALL, red->PUT),
#   optionally categorised by a MACD histogram cross (AT / Before / After / None). Enters the
#   ATM option, exits on a fixed ₹/% target or stop. Backtest + live share this engine.
#
#   TRIMMED per spec: NO rvol/flags, NO vwap/iv/greeks, NO rsi/momentum, NO move-character.
#   Output columns are joined with our RULE engine: last-5 daily conviction, 3 weekly
#   conviction scores (Wk-Curr/Last/Prev), and vwap3/vwap5 slopes.
#
#   >>> ALL TUNABLE KNOBS ARE IN THE CONFIG BLOCK BELOW — edit there, nothing else. <<<
# ==================================================================================
import os, math, datetime as dt
import requests
import pandas as pd, numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  the only place you toggle knobs
# ══════════════════════════════════════════════════════════════════════════════
# ── RUN TRIGGER (this engine's core) ──
INTERVAL       = "5M"     # underlying candle interval the run is measured on: 1M|3M|5M|15M
RUN_LEN        = 5        # required in-direction candles in the run window
RUN_MAX_OPP    = 1        # tolerate up to N opposite candles inside the window (0 = strict)
RUN_OPP_MAXPCT = 60       # a tolerated opposite candle's body must be <= this % of avg in-dir body (0 = off)
RUN_NET        = True     # also require the window's net close-to-close move to be in-direction
RUN_POS        = "ALL"    # keep which runs: ALL | AT | BEFORE | AFTER | NONE (tagged in 'Trigger Case')
XOVER_WINDOW   = 3        # bars before/after the run to look for the MACD cross (defines Before/After)
HIST_SIDE_BARS = 0        # require MACD histogram on the in-direction side for N bars ending at trigger (0=off)

# ── SIGNAL SOURCE ──
SIGNAL_SOURCE  = "UNDERLYING"   # UNDERLYING (stock: green run->call, red run->put) | OPTION (premium run)
OPT_INTERVAL   = "5M"           # OPTION source only: interval for the option-premium run/entry

# ── INSTRUMENT ──
STRIKE_OFFSET  = 0        # 0 = ATM, +1 = 1 strike ITM, -1 = 1 strike OTM
EXPIRY_BUFFER  = 2        # avoid last N sessions of a contract (roll to next); 0 = always front

# ── EXIT ──
TARGET_MODE    = "PERCENTAGE"   # PERCENTAGE | INR
TARGET_PCT     = 50
TARGET_INR     = 6000
STOP_MODE      = "PERCENTAGE"   # PERCENTAGE | INR
STOP_PCT       = 35
STOP_INR       = 5000

# ── MACD (used to categorise the run: AT/Before/After) + active noise filters ──
FAST           = 12
SLOW           = 26
SIGNAL         = 9
MA_TYPE        = "EMA"    # EMA | SMA | WMA  (oscillator lines)
SIGNAL_MA_TYPE = "EMA"    # EMA | SMA | WMA  (signal line)
MACD_ABOVE_ZERO= False    # only take crosses with MACD on the in-direction side of zero
COOLDOWN_BARS  = 3        # min bars between accepted signals, same option, same day (collapses clusters)
HEIKIN_ASHI    = False    # compute MACD on Heikin-Ashi candles (entry still fills at the REAL price)
REQUIRE_OPP_CROSS = False # require the opposite ATM strike's MACD histogram < 0 at the trigger

# ── BACKTEST WINDOW ──
BT_START       = "2026-07-30"
BT_END         = "2026-08-21"

# ── data / auth ──
COOKIE_PATH    = os.path.expanduser("~/live-refresh/cookie.txt")
CANDLES_URL    = "https://oxide.sensibull.com/v1/compute/1/candles/{sym}"
OICHART_URL    = "https://oxide.sensibull.com/v1/compute/compute_intraday_oi_chart"
META_URL       = "https://oxide.sensibull.com/v1/compute/1/instrument_metadata/{sym}"

# ══════════════════════════════════════════════════════════════════════════════
#  Sensibull data
# ══════════════════════════════════════════════════════════════════════════════
def _hdr():
    ck=open(COOKIE_PATH).read().strip()
    return {"cookie":ck,"content-type":"application/json","origin":"https://web.sensibull.com",
            "referer":"https://web.sensibull.com/","user-agent":"Mozilla/5.0"}

def _norm_interval(iv):
    iv=str(iv).strip().upper()
    if iv.isdigit(): iv=iv+"M"
    if iv=="60M": iv="1H"
    return iv

def fetch_candles(sym, from_date, to_date, interval):
    d2=(dt.date.fromisoformat(to_date)+dt.timedelta(days=1)).isoformat()
    body={"from_date":from_date,"to_date":d2,"interval":_norm_interval(interval),"skip_last_ts":False}
    try:
        r=requests.post(CANDLES_URL.format(sym=sym), json=body, headers=_hdr(), timeout=30)
        c=((r.json() or {}).get("payload") or {}).get("candles") or []
    except Exception as e:
        print(sym,"candle err",e); return []
    c=[x for x in c if from_date <= x["ts"].split("T")[0] <= to_date]
    return sorted(c, key=lambda x:x["ts"])

def fetch_atm_ref(sym):
    try:
        r=requests.post(OICHART_URL, json={"chart_selection":"oi","expiries":{},"underlying":sym}, headers=_hdr(), timeout=30)
        j=r.json()
        if not (j.get("success") and j.get("payload")): return None
        p=j["payload"]; atm=p.get("atm_strike"); exp=p.get("expiries") or {}
        enabled=sorted(e for e in exp if exp[e].get("is_enabled"))
        monthly=sorted(e for e in exp if exp[e].get("is_enabled") and not exp[e].get("is_weekly"))
        expiry=enabled[0] if enabled else None; nextE=enabled[1] if len(enabled)>1 else None
        mon=monthly[0] if monthly else expiry
        if not expiry or atm is None: return None
        qty=fetch_lot_qty(fut_sym(sym,mon)) or fetch_lot_qty(build_sym(sym,mon,atm,"CE")) or 1
        return {"atm":atm,"expiry":expiry,"monthly":mon,"nextExpiry":nextE,"qty":qty}
    except Exception as e:
        print(sym,"atm err",e); return None

def fetch_lot_qty(tradingsymbol):
    if not tradingsymbol: return None
    try:
        r=requests.get(META_URL.format(sym=tradingsymbol), headers=_hdr(), timeout=20); j=r.json()
        p=(j or {}).get("payload") or {}
        if p.get("lot_size") is not None:
            lot=int(p.get("lot_size") or 0); mult=float(p.get("multiplier") or 1)
            if lot>0: return max(1, round(lot*mult))
    except Exception: pass
    return None

# ══════════════════════════════════════════════════════════════════════════════
#  symbol / strike / expiry helpers
# ══════════════════════════════════════════════════════════════════════════════
def _mmm(expiry): return dt.date.fromisoformat(expiry).strftime("%y%b").upper()      # e.g. 26AUG
def fut_sym(sym, expiry):   return (sym+"FUT") if not expiry else sym+_mmm(expiry)+"FUT"
def build_sym(sym, expiry, strike, typ):
    s=int(strike) if float(strike)==int(strike) else strike
    return f"{sym}{_mmm(expiry)}{s}{typ}"

def strike_step(price):
    for hi,st in [(25,0.5),(50,1),(100,2.5),(250,5),(500,10),(1000,20),(2500,25),(5000,50),(10000,100),(25000,200),(50000,500)]:
        if price<hi: return st
    return 1000

def strike_for(atm, direction, offset):
    if not offset: return atm
    step=strike_step(atm)
    return atm-offset*step if direction=="LONG" else atm+offset*step

def _biz_days_before(date_str, n):
    d=dt.date.fromisoformat(date_str); c=0
    while c<n:
        d-=dt.timedelta(days=1)
        if d.weekday()<5: c+=1
    return d.isoformat()

def _last_wkday_of_month(y, m, wd):          # wd: Mon=0..Sun=6
    d=dt.date(y+(m==12), (m%12)+1, 1)-dt.timedelta(days=1)   # last day of month m
    while d.weekday()!=wd: d-=dt.timedelta(days=1)
    return d.isoformat()

def _monthly_expiry_for(date_str, wd):
    d=dt.date.fromisoformat(date_str); return _last_wkday_of_month(d.year, d.month, wd)

def expiry_for_date(D, wd, buffer=None):
    buffer=EXPIRY_BUFFER if buffer is None else buffer
    this_exp=_monthly_expiry_for(D, wd)
    if buffer<=0 or D<=_biz_days_before(this_exp, buffer): return this_exp
    d=dt.date.fromisoformat(D); nm=dt.date(d.year+(d.month==12), (d.month%12)+1, 15)
    return _last_wkday_of_month(nm.year, nm.month, wd)

def force_close_date(expiry): return _biz_days_before(expiry, EXPIRY_BUFFER) if (EXPIRY_BUFFER>0 and expiry) else ""

def _wd_of(date_str): return dt.date.fromisoformat(date_str).weekday()

# ══════════════════════════════════════════════════════════════════════════════
#  MACD + candle-run detection (port of cr_findSignals_)
# ══════════════════════════════════════════════════════════════════════════════
def _heikin(cnds):
    ha=[None]*len(cnds)
    for i,c in enumerate(cnds):
        o=float(c.get("open") or c["close"]); h=float(c["high"]); l=float(c["low"]); cl=float(c["close"])
        haC=(o+h+l+cl)/4; haO=(o+cl)/2 if (i==0 or ha[i-1] is None) else (ha[i-1][0]+ha[i-1][1])/2
        ha[i]=(haO, haC)
    return [x[1] for x in ha]

def _src_closes(cnds): return _heikin(cnds) if HEIKIN_ASHI else [float(c["close"]) for c in cnds]

def _ma(vals, period, typ):
    n=len(vals); r=[None]*n
    start=next((i for i,v in enumerate(vals) if v is not None and not (isinstance(v,float) and math.isnan(v))), -1)
    if start<0 or start+period-1>=n: return r
    t=(typ or "EMA").upper()
    if t=="SMA":
        for i in range(start+period-1,n): r[i]=sum(vals[i-j] for j in range(period))/period
    elif t=="WMA":
        wsum=period*(period+1)/2
        for i in range(start+period-1,n): r[i]=sum(vals[i-j]*(period-j) for j in range(period))/wsum
    else:
        k=2/(period+1); r[start+period-1]=sum(vals[start:start+period])/period
        for i in range(start+period,n): r[i]=vals[i]*k+r[i-1]*(1-k)
    return r

def macd_calc(closes):
    maF=_ma(closes,FAST,MA_TYPE); maS=_ma(closes,SLOW,MA_TYPE)
    macd=[(maF[i]-maS[i]) if (maF[i] is not None and maS[i] is not None) else None for i in range(len(closes))]
    signal=_ma(macd,SIGNAL,SIGNAL_MA_TYPE)
    hist=[(macd[i]-signal[i]) if (macd[i] is not None and signal[i] is not None) else None for i in range(len(closes))]
    return macd, signal, hist

def find_signals(cnds, direction, sign, start_date, end_date):
    sign=sign or 1
    RUN_W=RUN_LEN+max(0,RUN_MAX_OPP)
    need=SLOW+SIGNAL+max(RUN_W,1)+XOVER_WINDOW+2
    if not cnds or len(cnds)<need: return []
    closes=[float(c["close"]) for c in cnds]; opens=[float(c.get("open") or "nan") for c in cnds]
    macd,signal,hist=macd_calc(_src_closes(cnds))
    def dircandle(j):
        if j<1: return False
        if math.isfinite(opens[j]) and opens[j]>0:
            return closes[j]>opens[j] if sign>0 else closes[j]<opens[j]
        return closes[j]>closes[j-1] if sign>0 else closes[j]<closes[j-1]
    def body(j):
        base=opens[j] if (math.isfinite(opens[j]) and opens[j]>0) else (closes[j-1] if j>=1 else closes[j])
        return abs(closes[j]-base)
    def run_ends_at(j):
        if j<RUN_W: return False
        start=j-RUN_W+1
        if not dircandle(j): return False
        dcnt=0; ocnt=0; dsum=0.0
        for i in range(start,j+1):
            if dircandle(i): dcnt+=1; dsum+=body(i)
        avgdir=dsum/dcnt if dcnt else 0
        smallmax=(avgdir*RUN_OPP_MAXPCT/100) if (RUN_OPP_MAXPCT>0 and avgdir>0) else float("inf")
        for i in range(start,j+1):
            if not dircandle(i):
                if body(i)>smallmax: return False
                ocnt+=1
        if dcnt<RUN_LEN or ocnt>RUN_MAX_OPP: return False
        if RUN_NET and not (sign*(closes[j]-closes[start-1])>0): return False
        return True
    def cross_at(j):
        if j<1 or hist[j] is None or hist[j-1] is None: return False
        return (hist[j-1]<=0 and hist[j]>0) if sign>0 else (hist[j-1]>=0 and hist[j]<0)
    def hist_side_ok(j):
        if HIST_SIDE_BARS<=0: return True
        for k in range(HIST_SIDE_BARS):
            h=hist[j-k]
            if h is None or not (sign*h>0): return False
        return True
    def run_start(i): return i-RUN_W+1
    def classify(i):
        for j in range(i, max(run_start(i),1)-1, -1):
            if cross_at(j): return "AT-Exact" if j==i else "AT-Within"
        for w in range(1,XOVER_WINDOW+1):
            if i+w<len(cnds) and cross_at(i+w): return "Before"
        s0=run_start(i)
        for w in range(1,XOVER_WINDOW+1):
            if s0-w>=1 and cross_at(s0-w): return "After"
        return "None"
    def keep(cat):
        if RUN_POS=="ALL": return True
        if RUN_POS=="AT": return cat.startswith("AT")
        if RUN_POS=="BEFORE": return cat=="Before"
        if RUN_POS=="AFTER": return cat=="After"
        if RUN_POS=="NONE": return cat=="None"
        return True
    out=[]; last_by_cat={}
    for i in range(RUN_LEN, len(cnds)):
        if not run_ends_at(i): continue
        cat=classify(i)
        if not keep(cat): continue
        T=i
        if not hist_side_ok(T): continue
        if MACD_ABOVE_ZERO and not (macd[T] is not None and sign*macd[T]>0): continue
        dd=cnds[T]["ts"].split("T")[0]
        if dd<start_date or dd>end_date: continue
        li=last_by_cat.get(cat)
        if li is not None and cnds[li]["ts"].split("T")[0]==dd and (T-li)<COOLDOWN_BARS: continue
        ts=cnds[T]["ts"]
        out.append({"direction":direction,"date":dd,"time":ts.split("T")[1][:5],"ts":ts,"idx":T,"case":cat,
                    "price":round(closes[T],2),
                    "macd":round(macd[T],3) if macd[T] is not None else "",
                    "signal":round(signal[T],3) if signal[T] is not None else "",
                    "hist":round(hist[T],3) if hist[T] is not None else ""})
        last_by_cat[cat]=T
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  run-move %, target/stop, exit sim
# ══════════════════════════════════════════════════════════════════════════════
def _idx_at(cnds, ts):
    idx=-1
    for i,c in enumerate(cnds):
        if c["ts"]==ts: return i
        if c["ts"]<=ts: idx=i
        else: break
    return idx

def run_move_pct(cnds, ts, n):
    if not cnds: return ""
    j=_idx_at(cnds, ts)
    if j<0 or j-n<0: return ""
    base=float(cnds[j-n]["close"]); end=float(cnds[j]["close"])
    if not (base>0 and end>0): return ""
    return round((end/base-1)*100, 2)

def target_sl(ep, qty):
    q=qty if (qty and qty>0) else 1
    t=ep+TARGET_INR/q if TARGET_MODE=="INR" else ep*(1+TARGET_PCT/100)
    s=ep-STOP_INR/q  if STOP_MODE=="INR"   else ep*(1-STOP_PCT/100)
    if s<0: s=0
    return round(t,2), round(s,2)

def sim_exit(optC, eIdx, ep, sig_date, qty, max_date):
    tgt,sl=target_sl(ep,qty); held=0; last_in=-1; hit_buffer=False
    for i in range(eIdx+1, len(optC)):
        c=optC[i]; d=c["ts"].split("T")[0]
        if d<sig_date: continue
        if max_date and d>max_date: hit_buffer=True; break
        held+=1; last_in=i
        if float(c["high"])>=tgt:
            return dict(outcome="Target Hit",exit_date=d,exit_time=c["ts"].split("T")[1][:5],exit_price=tgt,
                        pnl_pct=round((tgt-ep)/ep*100,2),pnl_inr=round((tgt-ep)*qty,2),held=held)
        if float(c["low"])<=sl:
            return dict(outcome="Stop Loss Hit",exit_date=d,exit_time=c["ts"].split("T")[1][:5],exit_price=sl,
                        pnl_pct=round((sl-ep)/ep*100,2),pnl_inr=round((sl-ep)*qty,2),held=held)
    if hit_buffer and last_in>=0:
        c=optC[last_in]; px=float(c["close"])
        return dict(outcome="Expiry Close",exit_date=c["ts"].split("T")[0],exit_time=c["ts"].split("T")[1][:5],
                    exit_price=round(px,2),pnl_pct=round((px-ep)/ep*100,2),pnl_inr=round((px-ep)*qty,2),held=held)
    last=optC[-1] if optC else None; ltp=float(last["close"]) if last else ep
    return dict(outcome="Open",exit_date=last["ts"].split("T")[0] if last else "",exit_time=last["ts"].split("T")[1][:5] if last else "",
                exit_price=round(ltp,2),pnl_pct=round((ltp-ep)/ep*100,2),pnl_inr=round((ltp-ep)*qty,2),held=held)

# ── PURE candle-run trigger (no MACD, no cross, no cooldown): does a run end at bar j? ──
def _run_ends_at(closes, opens, j, sign):
    RUN_W=RUN_LEN+max(0,RUN_MAX_OPP)
    if j<RUN_W: return False
    def dircandle(i):
        if i<1: return False
        if math.isfinite(opens[i]) and opens[i]>0:
            return closes[i]>opens[i] if sign>0 else closes[i]<opens[i]
        return closes[i]>closes[i-1] if sign>0 else closes[i]<closes[i-1]
    def body(i):
        base=opens[i] if (math.isfinite(opens[i]) and opens[i]>0) else (closes[i-1] if i>=1 else closes[i])
        return abs(closes[i]-base)
    start=j-RUN_W+1
    if not dircandle(j): return False
    dcnt=0; ocnt=0; dsum=0.0
    for i in range(start,j+1):
        if dircandle(i): dcnt+=1; dsum+=body(i)
    avg=dsum/dcnt if dcnt else 0
    smax=(avg*RUN_OPP_MAXPCT/100) if (RUN_OPP_MAXPCT>0 and avg>0) else float("inf")
    for i in range(start,j+1):
        if not dircandle(i):
            if body(i)>smax: return False
            ocnt+=1
    if dcnt<RUN_LEN or ocnt>RUN_MAX_OPP: return False
    if RUN_NET and not (sign*(closes[j]-closes[start-1])>0): return False
    return True

def run_signal_now(cnds, sign):
    """True if a run (RUN_LEN in-dir, <=RUN_MAX_OPP small opposites, net in-dir) ends at the LAST candle."""
    if not cnds or len(cnds) < RUN_LEN+max(0,RUN_MAX_OPP)+1: return False
    closes=[float(c["close"]) for c in cnds]; opens=[float(c.get("open") or "nan") for c in cnds]
    return _run_ends_at(closes, opens, len(cnds)-1, sign)

print("candle_run engine loaded | CONFIG at top | find_signals / run_signal_now / sim_exit / fetch_* ready")
