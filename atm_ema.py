# ==================================================================================
# ATM EMA 12/26 — live updater for two dashboard columns:
#     "ATM CE (12/26)"  and  "ATM PE (12/26)"
#   For each stock it finds the ATM strike from the CURRENT LTP (Sensibull OI chart),
#   fetches that ATM CALL and ATM PUT's 15-min candles, computes EMA12 & EMA26 on the
#   option premium, and writes:  "<strike> True"  if EMA12 >= EMA26 (fast above slow),
#   else "<strike> False".  True = that option's premium is in an up-cross state.
#
#   The two column HEADERS are created by the nightly conviction_local.py; this runner
#   just fills the cells, every 15 min during market hours.
#   RUN once:  cd ~/conviction && python atm_ema.py once
#   RUN loop:  cd ~/conviction && python atm_ema.py
# ==================================================================================
import os, sys, time, datetime as dt
import concurrent.futures as cf
import pandas as pd
import gspread
import candle_run as CR
import live_refresh_local as LR      # open_sheet / header_map / sheet_symbols / LIVE_SHEETS / IST

# ── CONFIG ──
EMA_FAST   = 12
EMA_SLOW   = 26
INTERVAL   = "1H"
SEED_DAYS  = 15           # calendar days of history to fetch (needs >= EMA_SLOW bars; 15 covers up to 1H)
EVERY_MIN  = 15
WORKERS    = 6
SHEETS     = ["Rule Dashboard", "Calibrated Dashboard"]   # where the two columns live
COLS       = [("ATM","atm"),("CE Δ%","ceg"),("CE bars","ceb"),("PE Δ%","peg"),("PE bars","peb")]

def _ema(vals, period):
    return pd.Series(vals).ewm(span=period, adjust=False).mean().tolist()

def _cross_state(osym):
    """Returns (gap_pct, streak): gap_pct=(EMA12-EMA26)/EMA26*100 (sign = which way the cross is);
       streak = how many consecutive candles the current above/below state has held (1 = just flipped)."""
    seed=(dt.date.today()-dt.timedelta(days=SEED_DAYS)).isoformat(); today=dt.date.today().isoformat()
    c=CR.fetch_candles(osym, seed, today, INTERVAL)
    if len(c) < EMA_SLOW+2: return None
    closes=[float(x["close"]) for x in c]
    ef=_ema(closes,EMA_FAST); es=_ema(closes,EMA_SLOW)
    state=ef[-1]>=es[-1]; gap_pct=round((ef[-1]-es[-1])/es[-1]*100,1) if es[-1] else 0.0
    streak=0
    for i in range(len(ef)-1,-1,-1):
        if (ef[i]>=es[i])==state: streak+=1
        else: break
    return gap_pct, streak

def atm_state(sym):
    try:
        ref=CR.fetch_atm_ref(sym)
        if not ref: return None
        atm=ref["atm"]; exp=ref["expiry"]; out={"atm":atm}
        for typ,gk,bk in (("CE","ceg","ceb"),("PE","peg","peb")):
            r=_cross_state(CR.build_sym(sym, exp, atm, typ)); time.sleep(0.08)
            if r is None: out[gk]=""; out[bk]=""
            else: gap,streak=r; out[gk]=gap; out[bk]=streak
        return out
    except Exception:
        return None

def refresh_once(sh):
    tabs=[(nm, sh.worksheet(nm)) for nm in SHEETS if nm in [w.title for w in sh.worksheets()]]
    allsyms=set()
    for _,ws in tabs: allsyms |= set(LR.sheet_symbols(ws, LR.header_map(ws)))
    syms=sorted(allsyms)
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results=list(ex.map(atm_state, syms))
    data={s:r for s,r in zip(syms, results) if r}
    for _,ws in tabs:
        m=LR.header_map(ws); s=LR.sheet_symbols(ws, m); n=len(s)
        for name,key in COLS:
            col=m.get(name.strip().lower())
            if not col: continue
            vals=[[ data.get(sym,{}).get(key,"") ] for sym in s]
            a1=f"{gspread.utils.rowcol_to_a1(2,col)}:{gspread.utils.rowcol_to_a1(1+n,col)}"
            ws.update(range_name=a1, values=vals, value_input_option="USER_ENTERED")
    return len(data)

def in_hours():
    now=dt.datetime.now(LR.IST); t=now.hour*60+now.minute
    return now.weekday()<5 and 9*60+15 <= t <= 15*60+30

def main():
    sh=LR.open_sheet()
    arg=sys.argv[1] if len(sys.argv)>1 else ""
    if arg=="once":                                  # manual: run now regardless of hours
        n=refresh_once(sh); print(f"ATM EMA once: updated {n} symbols"); return
    if arg=="tick":                                  # scheduled: one guarded pass (skip off-hours)
        if not in_hours(): print(f"{dt.datetime.now(LR.IST):%H:%M} outside market hours — skip"); return
        n=refresh_once(sh); print(f"ATM EMA tick: updated {n} symbols"); return
    print("ATM EMA 12/26 monitor — every", EVERY_MIN, "min")
    while True:
        now=dt.datetime.now(LR.IST)
        if now.weekday()>=5: print("weekend — exit"); break
        if now.hour*60+now.minute > 15*60+30: print("market closed — exit"); break
        if not in_hours(): print(f"{now:%H:%M} pre-market…"); time.sleep(60); continue
        try:
            n=refresh_once(sh); print(f"{now:%H:%M} ATM EMA updated {n} symbols")
        except Exception as e:
            print("cycle error:", e)
        time.sleep(EVERY_MIN*60)

if __name__=="__main__":
    main()
