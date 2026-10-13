# ==================================================================================
# LIVE FVG — one runner that refreshes BOTH the price %s AND the live intraday
# nearest-gap on the dashboards, every 10 min during market hours.
#
#   Supersedes live_refresh_local.py: a single 1-min fetch per symbol (days=4) gives
#   the previous close (for %s), today's session high/low, AND today's intraday FVGs.
#   No extra API calls vs the old price-only refresher.
#
#   Writes 4 live columns:  LTP, Today %, High %, Low %.
#   The imb_sup/imb_res/in_imb columns are left ALONE — they hold the nightly
#   DAILY-candle FVG levels from conviction_local.py (wider, meaningful S/R).
#   (Intraday 1-min gaps were too tightly stacked to separate support from resistance.)
#
#   Reuses your Sheet URL + service-account config by importing live_refresh_local
#   (nothing to re-enter). Reads the Sensibull cookie from ~/live-refresh/cookie.txt.
#
#   RUN:  cd ~/live-refresh && python live_fvg.py     (loops till 15:30, then exits)
#   ONE PASS NOW:  python -c "import live_fvg as V; V.refresh_once(V.LR.open_sheet())"
# ==================================================================================
import os, json, time, re, html, datetime as dt
import concurrent.futures as cf
import gspread
import intraday_fvg as F               # fetch_1m + _detect (+ cookie.txt)
import intraday_regime as RG           # day-type classifier (Trend/Coiling/Fired/Chop)
import live_refresh_local as LR        # reuse open_sheet / header_map / sheet_symbols / LIVE_SHEETS / IST

SEG      = getattr(F, "CASH_SEG", 1)
WORKERS  = 8
# (dashboard header name, key in the computed dict)
# NOTE: imb_sup/imb_res/in_imb are intentionally NOT here — those stay the nightly
# DAILY-candle values from conviction_local.py (wider, meaningful S/R). Live refresh
# only touches the price + %s.
LIVE_MAP = [("LTP","ltp"), ("Today %","pct"), ("High %","high_pct"), ("Low %","low_pct"),
            ("Regime","regime"), ("Eff%","eff"), ("Exp×","exp"), ("S/R Flag","srflag")]

# ── LIVE S/R reaction flag ──────────────────────────────────────────────────
# The nightly build writes the daily-candle imbalance levels (imb_sup1..N / imb_res1..N)
# into the dashboard. Here we read those levels back each cycle and, using TODAY's 1-min
# bars, flag whether price has just reacted at one:
#     Bounced ↑  — a recent bar's LOW tagged a support level, price now back above it
#     Rejected ↓ — a recent bar's HIGH tagged a resistance level, price now back below it
#     Broke S ↓  — price is now trading below the nearest support
#     Broke R ↑  — price is now trading above the nearest resistance
# We check bar HIGHS/LOWS over a window a little longer than the refresh interval, NOT the
# 5-min LTP snapshot — so a touch that happens BETWEEN two 5-min writes is still caught
# (down to 1-min resolution). REACT_BARS>refresh gap => consecutive checks overlap, no blind spot.
IMB_LEVELS  = {}      # sym -> (supports desc, resistances asc); refreshed from the sheet each cycle
CONV_SCORES = {}      # sym -> list of up to 5 recent daily conviction scores, oldest->newest
VOL_MULT3   = {}      # sym -> MAX 3-day volume multiple vs 20d SMA (from the 'Vol×' column)
VOL_FILTER_ON = False # True once the sheet actually has a 'Vol×' column (else we don't gate)
VOL_MULT    = 2.0     # alert only if ANY of the last 3 days' volume >= VOL_MULT x its 20-day SMA (matches the scanner: OR, 2x)
REACT_BARS  = 7       # 1-min bars to look back for a level touch (> the 5-min refresh cadence)

def _isnum(x):
    try: float(x); return True
    except Exception: return False

def _read_sheet_refs(ws):
    """Read per-symbol S/R levels (imb_sup*/imb_res*), the last 5 daily conviction scores,
       and the 3-day volume multiple ('Vol×') from a dashboard tab.
       -> (levels, scores, vol_mult, has_vol_col)."""
    try: vals = ws.get_all_values()
    except Exception: return {}, {}, {}, False
    if not vals: return {}, {}, {}, False
    hdr=[h.strip() for h in vals[0]]
    symi=next((i for i,h in enumerate(hdr) if h.lower()=="symbol"), 0)
    supc=[i for i,h in enumerate(hdr) if re.fullmatch(r"imb_sup\d+", h)]
    resc=[i for i,h in enumerate(hdr) if re.fullmatch(r"imb_res\d+", h)]
    datec=[i for i,h in enumerate(hdr) if re.fullmatch(r"\d{2}-\d{2}", h)]  # MM-DD daily score cols, sheet order = ascending
    scorecols=datec[-5:]                                 # up to the 5 most recent date columns, oldest->newest
    voli=next((i for i,h in enumerate(hdr) if h=="Vol×"), None)   # 3-day volume-filter column
    lv={}; sc={}; vm={}
    for row in vals[1:]:
        if symi>=len(row): continue
        sym=row[symi].strip().upper()
        if not sym: continue
        sups=sorted({float(row[i]) for i in supc if i<len(row) and _isnum(row[i])}, reverse=True)  # [0]=nearest support
        ress=sorted({float(row[i]) for i in resc if i<len(row) and _isnum(row[i])})                # [0]=nearest resistance
        lv[sym]=(sups, ress)
        sc[sym]=[float(row[i]) for i in scorecols if i<len(row) and _isnum(row[i])]   # 5 recent scores, oldest->newest
        if voli is not None and voli<len(row) and _isnum(row[voli]):
            vm[sym]=float(row[voli])
    return lv, sc, vm, (voli is not None)

def _deepest_rank(levels, cond):
    """levels are in rank order (index 0 = rank 1). Return the 1-based rank of the FURTHEST
       level satisfying cond (i.e. how far price has reached), or 0 if none."""
    r=0
    for i,x in enumerate(levels):
        if cond(x): r=i+1
    return r

def _sr_flag(sym, h, l, ltp):
    """Live reaction of today's price against this symbol's nightly S/R levels, tagged with the
       designated level: e.g. 'Broke S2 ↓', 'Bounced S1 ↑', 'Rejected R1 ↓', 'Broke R1 ↑'.
       sups are ranked nearest-first (S1 = nearest support below); ress likewise (R1 = nearest above)."""
    lv=IMB_LEVELS.get(sym)
    if not lv: return ""
    sups,ress=lv
    if len(l)==0 or len(h)==0: return ""
    rl=float(l[-REACT_BARS:].min()); rh=float(h[-REACT_BARS:].max())
    S1=sups[0] if sups else None; R1=ress[0] if ress else None
    if S1 is not None and ltp < S1:                      # trading below the nearest support -> break down
        return f"Broke S{_deepest_rank(sups, lambda s: ltp < s)} ↓"
    if R1 is not None and ltp > R1:                      # trading above the nearest resistance -> break up
        return f"Broke R{_deepest_rank(ress, lambda r: ltp > r)} ↑"
    if S1 is not None and rl <= S1:                      # dipped to support in-window, held (ltp >= S1)
        return f"Bounced S{_deepest_rank(sups, lambda s: rl <= s)} ↑"
    if R1 is not None and rh >= R1:                      # spiked to resistance in-window, held (ltp <= R1)
        return f"Rejected R{_deepest_rank(ress, lambda r: rh >= r)} ↓"
    return ""

# ── alert dispatch: ping only when a symbol's flag CHANGES to an alert state ──────
# live_fvg runs as a fresh process every ~5 min (launchd), and a flag usually persists
# across cycles — so we persist the last-sent flag per symbol to a small JSON and only
# notify on a *transition* (blank->flag or flag->different flag). Reset daily.
ALERT_STATE = os.path.join(os.environ.get("STATE_DIR", os.path.expanduser("~/live-refresh")), "sr_alert_state.json")
CONFIRM_CYCLES = 2    # a flag must HOLD this many consecutive cycles before it alerts (debounce)

# Telegram can't colour text/cells, so we use emoji as a direction proxy (flags now carry a level, e.g. 'Broke S2 ↓').
def _flag_emoji(fl):
    if fl.startswith("Bounced"):  return "🟢"       # bounced off support (bullish reaction)
    if fl.startswith("Broke R"):  return "🟩"       # broke resistance up (bullish break)
    if fl.startswith("Rejected"): return "🔴"       # rejected at resistance (bearish reaction)
    if fl.startswith("Broke S"):  return "🟥"       # broke support down (bearish break)
    return "⚪"

def _heat_emoji(s):
    """Conviction score -> diverging red/orange/white/green square (thresholds tweakable)."""
    if s is None:  return "⬜"
    if s >=  20:   return "🟩"
    if s <= -50:   return "🟥"
    if s <= -20:   return "🟧"
    return "⬜"

def _load_alert_state():
    try:
        with open(ALERT_STATE) as f: s = json.load(f)
        if s.get("date") == dt.date.today().isoformat(): return dict(s.get("syms", {}))
    except Exception:
        pass
    return {}

def _save_alert_state(syms):
    try:
        with open(ALERT_STATE, "w") as f:
            json.dump({"date": dt.date.today().isoformat(), "syms": syms}, f)
    except Exception as e:
        print("alert state save failed:", e)

def dispatch_alerts(live):
    """Telegram alert only when a flag has HELD the same alert-state for >= CONFIRM_CYCLES
       consecutive cycles (debounce), and only ONCE per continuous run — so a stock that
       flips around within a few minutes doesn't reappear in every trigger."""
    try:
        import notify
    except Exception:
        return 0
    if not notify.enabled(): return 0
    st = _load_alert_state(); lines = []
    for sym, v in sorted(live.items()):
        fl  = v.get("srflag", "") or ""
        rec = st.get(sym) or {"fl": "", "n": 0, "alerted": False}
        if fl == rec.get("fl"):
            rec["n"] = rec.get("n", 0) + 1                 # same state persists -> extend streak
        else:
            rec = {"fl": fl, "n": 1, "alerted": False}     # state changed -> restart streak
        vol_ok = (not VOL_FILTER_ON) or (VOL_MULT3.get(sym, 0.0) >= VOL_MULT)   # 3-day volume filter (fail-open if no column yet)
        if fl and rec["n"] >= CONFIRM_CYCLES and not rec["alerted"] and vol_ok:
            rec["alerted"] = True                          # confirmed & not yet pinged -> alert once
            seq = CONV_SCORES.get(sym) or []               # up to 5 recent scores, oldest->newest
            cv  = " | ".join(f"{x:+.0f}" for x in seq) if seq else "—"
            esym = html.escape(sym)                        # <code> => tap-to-copy just the ticker
            lines.append(f"{_flag_emoji(fl)} <code>{esym}</code>  {html.escape(fl)} · "
                         f"LTP {v.get('ltp','')} · CV {cv}")
        st[sym] = rec
    if lines:
        now  = dt.datetime.now(LR.IST)
        head = f"🔔 <b>S/R Alerts</b>\n📅 {now:%d %b %Y} · {now:%H:%M}\n"
        notify.send(head + "\n".join(lines), parse_mode="HTML")
    _save_alert_state(st)
    return len(lines)

def compute_one(sym):
    """One 1-min fetch -> live price %s + nearest unfilled intraday gap for the current session."""
    df=None
    for _ in range(3):                        # retry transient timeouts / connection resets
        try:
            df = F.fetch_1m(sym, days=4, seg=SEG)
            if df is not None and not df.empty: break
        except Exception:
            df=None
        time.sleep(0.4)
    if df is None or df.empty: return sym, None
    sessions = sorted(df["session"].astype(str).unique())
    today = sessions[-1]; prev = sessions[-2] if len(sessions) >= 2 else None
    g = df[df["session"].astype(str) == today]
    if g.empty or len(g) < 2: return sym, None
    h = g["high"].to_numpy(float); l = g["low"].to_numpy(float); c = g["close"].to_numpy(float)
    ltp = float(c[-1]); hi = float(h.max()); lo = float(l.min())
    if prev is not None:
        pc = float(df[df["session"].astype(str) == prev]["close"].iloc[-1])
    else:
        pc = float(g["close"].iloc[0])
    out = {"ltp":      round(ltp, 2),
           "pct":      round((ltp-pc)/pc*100, 2),
           "high_pct": round((hi-pc)/pc*100, 2),
           "low_pct":  round((lo-pc)/pc*100, 2)}
    tag,eff,exp = RG.compute_regime(h, l, c)          # intraday day-type from today's 1-min bars
    out["regime"]=tag; out["eff"]=eff; out["exp"]=exp
    out["srflag"]=_sr_flag(sym, h, l, ltp)            # live reaction vs nightly S/R levels
    return sym, out

def fetch_live(symbols):
    live = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for sym, v in ex.map(compute_one, list(symbols)):
            if v: live[sym] = v
    return live

def write_live(ws, m, symbols, live):
    n = len(symbols)
    for name, key in LIVE_MAP:
        col = m.get(name.strip().lower())
        if not col: continue
        vals = [[live.get(s, {}).get(key, "")] for s in symbols]
        a1 = f"{gspread.utils.rowcol_to_a1(2,col)}:{gspread.utils.rowcol_to_a1(1+n,col)}"
        ws.update(range_name=a1, values=vals, value_input_option="USER_ENTERED")

def refresh_once(sh):
    tabs = [(nm, sh.worksheet(nm)) for nm in LR.LIVE_SHEETS if nm in [w.title for w in sh.worksheets()]]
    global IMB_LEVELS, CONV_SCORES, VOL_MULT3, VOL_FILTER_ON
    allsyms = set(); lvls = {}; scores = {}; vms = {}; volcol = False
    for nm, ws in tabs:
        allsyms |= set(LR.sheet_symbols(ws, LR.header_map(ws)))
        l, s, vm, hv = _read_sheet_refs(ws)    # S/R levels + conviction scores + 3-day volume multiple
        lvls.update(l); vms.update(vm); volcol = volcol or hv
        if "rule" in nm.lower(): scores = s    # Rule Dashboard = authoritative conviction score
        elif not scores: scores.update(s)
    IMB_LEVELS = lvls; CONV_SCORES = scores; VOL_MULT3 = vms; VOL_FILTER_ON = volcol
    live = fetch_live(sorted(allsyms))
    for _, ws in tabs:
        m = LR.header_map(ws); syms = LR.sheet_symbols(ws, m); write_live(ws, m, syms, live)
    try: dispatch_alerts(live)                    # Telegram ping on flag changes (no-op if unconfigured)
    except Exception as e: print("alert dispatch error:", e)
    return len(live)

def run_once_guarded():
    """ONE refresh if we're in market hours, else no-op. For a launchd job that fires
    every 10 min — resilient to the laptop being asleep at the open (the next tick catches up)."""
    now = dt.datetime.now(LR.IST); t = now.hour*60 + now.minute
    if now.weekday() >= 5 or not (9*60+15 <= t <= 15*60+30):
        print(f"{now:%H:%M} outside market hours — skip"); return 0
    n = refresh_once(LR.open_sheet()); print(f"refreshed {n} symbols @ {now:%H:%M:%S}"); return n

def main(every_min=10):
    sh = LR.open_sheet(); print("connected:", LR.SHEET_URL)
    while True:
        now = dt.datetime.now(LR.IST); t = now.hour*60 + now.minute
        if now.weekday() >= 5: print("weekend — exiting"); break
        if t > 15*60+30:
            try: n = refresh_once(sh); print(f"final refresh {n} @ {now:%H:%M} — market closed, exiting")
            except Exception as e: print("final refresh error:", e)
            break
        if t < 9*60+15: print(f"{now:%H:%M} pre-market, waiting…"); time.sleep(60); continue
        try:
            n = refresh_once(sh); print(f"refreshed {n} symbols (price + intraday FVG) @ {now:%H:%M:%S}")
        except Exception as e:
            print("refresh error:", e)
        time.sleep(every_min*60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        run_once_guarded()          # launchd fires this every 10 min (see the plist)
    else:
        main()                      # foreground: loops itself till 15:30
