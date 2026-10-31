# ==================================================================================
# EMA CROSS ALERTS — the STOCK's 1-hour EMA 12 / EMA 26 cross, confirmed by the rolling 3- and 5-session VWAP.
#   (This replaces the earlier alert that fired on the OPTION's 1-hour EMA 9 / 26. That alert is gone.)
#
#   CALL alert:  the stock's hourly EMA12 crosses ABOVE EMA26 on a just-completed hourly candle
#                AND the stock price (that candle's close) is ABOVE both the 3-session VWAP and the 5-session VWAP.
#   PUT alert:   exactly the opposite — EMA12 crosses BELOW EMA26 AND price is BELOW both VWAPs.
#   VWAP is calculated LIVE every cycle from the stock's own hourly candles: the last 3 (or 5) sessions INCLUDING today's trading so far
#   (today's partial candle = high/low so far, latest price, volume so far; typical price = (H+L+C)/3, volume-weighted). It rolls every cycle — nothing restarts.
#   The same live numbers are written back to the "Rule Dashboard" columns vwap3 / vwap5 during market hours (the nightly job resets them to the closing values).
#   Price tested against the VWAPs = the stock's latest price (last candle close, including the hour in progress).
#   The alert names the ATM CALL (or ATM PUT) with its premium, plus the stock's last 5 conviction scores (information only — not a gate).
#   Telegram: separate channel (EMA_TELEGRAM_CHAT_ID; token = EMA_TELEGRAM_TOKEN, else TELEGRAM_TOKEN).
#   LOG: tab "EMA12-26 Triggers" (one row per alert, performance filled in later) + "EMA12-26 Scorecard". Paper-trade only; NO ORDERS are placed.
#        (The old tabs "EMA Cross Triggers" / "EMA Cross Scorecard" are left untouched as history.)
#   Entry premium (paper) = the ATM option's hourly close on the candle where the stock's cross completed.
#   Performance (option premium % vs entry): +1h, +2h, end of day, next-day close, peak and low that day, and the premium when the stock's EMA12 crosses back.
#
#   Usage:  python ema_cross_alert.py once | loop | test | --selftest
#   Env (optional): EMA_LOOP_MIN=5  EMA_MAX_PER_DAY=2  FORCE=true (rehearsal: prints, writes nothing)
# ==================================================================================
import os, re, sys, time, html, datetime as dt, urllib.parse, urllib.request
from zoneinfo import ZoneInfo
import concurrent.futures as cf
import numpy as np, pandas as pd
import trigger_engine as TE

IST = ZoneInfo("Asia/Kolkata")
FAST, SLOW = 12, 26
LOOP_MIN = int(os.environ.get("EMA_LOOP_MIN", 5))
MAX_PER_DAY = int(os.environ.get("EMA_MAX_PER_DAY", 2)); FRESH_MIN = 95; SEED_DAYS = int(os.environ.get("EMA_SEED_DAYS", 30)); WORKERS = 6
FORCE = os.environ.get("FORCE", "").lower() == "true"
TAB, SC_TAB = "EMA12-26 Triggers", "EMA12-26 Scorecard"
HDR = ["date", "time", "alert time", "symbol", "side", "option", "strike", "expiry", "spot", "entry prem", "ema12", "ema26", "gap%", "vwap3", "vwap5",
       "s1", "s2", "s3", "s4", "s5", "opt+1h%", "opt+2h%", "optEOD%", "opt+1D%", "peak%", "low%", "recross time", "recross%"]
OUTS = ["opt+1h%", "opt+2h%", "optEOD%", "opt+1D%", "peak%", "low%", "recross time", "recross%"]

# ───────────────────────────── data access (overridden in the self-test) ─────────────────────────────
def get_hourly(sym, now):
    """1-hour candles for a STOCK symbol or an option symbol (same endpoint)."""
    import candle_run as CR
    seed = (now.date() - dt.timedelta(days=SEED_DAYS)).isoformat()
    return TE._candles_df(CR.fetch_candles(sym, seed, now.date().isoformat(), "1H"))

def get_minute(osym, d0, now):
    import candle_run as CR
    return TE._candles_df(CR.fetch_candles(osym, d0, now.date().isoformat(), "1M"))

def get_atm(sym):
    import candle_run as CR
    ref = CR.fetch_atm_ref(sym)
    if not ref: return None
    return {"atm": ref["atm"], "expiry": ref["expiry"], "CE": lambda: CR.build_sym(sym, ref["expiry"], ref["atm"], "CE"),
            "PE": lambda: CR.build_sym(sym, ref["expiry"], ref["atm"], "PE")}

# ───────────────────────────── dashboard: scores + VWAP prices ─────────────────────────────
def read_scores(ws):
    vals = ws.get_all_values()
    if not vals: return {}
    hdr = [h.strip() for h in vals[0]]; si = hdr.index("symbol") if "symbol" in hdr else 0
    dcols = [i for i, h in enumerate(hdr) if re.fullmatch(r"\d{2}-\d{2}", h)]
    li = hdr.index("LTP") if "LTP" in hdr else None; out = {}
    cell = lambda row, i: TE._num(row[i]) if (i is not None and i < len(row)) else None
    for row in vals[1:]:
        if si >= len(row) or not row[si].strip(): continue
        sc = [TE._num(row[i]) if i < len(row) else None for i in dcols]; sc = [x for x in sc if x is not None][-5:]
        out[row[si].strip().upper()] = {"sc": sc, "ltp": cell(row, li)}
    return out

def write_live_vwaps(ws, live):
    """Writes today's LIVE vwap3 / vwap5 into the dashboard columns (only for stocks we have a value for; other cells are left as they were)."""
    vals = ws.get_all_values()
    if not vals: return 0
    hdr = [h.strip() for h in vals[0]]
    if "symbol" not in hdr or "vwap3" not in hdr or "vwap5" not in hdr: print("dashboard has no vwap3 / vwap5 columns yet (run the nightly job once)"); return 0
    si, c3, c5 = hdr.index("symbol"), hdr.index("vwap3"), hdr.index("vwap5"); n = 0; col3, col5 = [], []
    import gspread
    for row in vals[1:]:
        row = row + [""] * (len(hdr) - len(row)); v = live.get(row[si].strip().upper())
        if v and v.get(3) is not None and v.get(5) is not None: col3.append([round(v[3], 2)]); col5.append([round(v[5], 2)]); n += 1
        else: col3.append([row[c3]]); col5.append([row[c5]])
    for c, col in ((c3, col3), (c5, col5)):
        a = gspread.utils.rowcol_to_a1(2, c + 1); b = gspread.utils.rowcol_to_a1(1 + len(col), c + 1)
        ws.update(values=col, range_name=f"{a}:{b}", value_input_option="USER_ENTERED")
    return n

def vwap_ok(side, px, v3, v5):
    """CALL: price above BOTH VWAPs.  PUT: price below BOTH VWAPs."""
    if px is None or v3 is None or v5 is None: return False
    return (px > v3 and px > v5) if side == "CALL" else (px < v3 and px < v5)

def live_vwaps(h, now, sym=None):
    """LIVE rolling VWAP over the last 3 and 5 sessions, today's partial session included.
       Daily bars are rebuilt from the hourly candles (high = max, low = min, close = last, volume = sum), so today's bar is simply 'the day so far'.
       Returns {3: vwap3, 5: vwap5, 'px': latest price, 'session': date of the newest bar}  (a window with too few sessions -> None)."""
    d = h[h["ts"] <= pd.Timestamp(now)].copy()
    if d.empty: return {3: None, 5: None, "px": None, "session": None}
    hi = d["high"] if "high" in d else d["close"]; lo = d["low"] if "low" in d else d["close"]
    d = d.assign(_h=hi, _l=lo)
    g = d.groupby("session").agg(h=("_h", "max"), l=("_l", "min"), c=("close", "last"), v=("volume", "sum")).sort_index()
    tpv = (g["h"] + g["l"] + g["c"]) / 3 * g["v"]; out = {"px": float(d["close"].iloc[-1]), "session": g.index[-1]}
    for W in (3, 5):
        den = float(g["v"].iloc[-W:].sum()) if len(g) >= W else 0.0
        out[W] = float(tpv.iloc[-W:].sum()) / den if den > 0 else None
    return out

# ───────────────────────────── the cross ─────────────────────────────
def completed_hourly(df, now):
    d = df.copy(); end = d["ts"] + pd.Timedelta(minutes=60); close_t = d["ts"].dt.normalize() + pd.Timedelta(hours=15, minutes=30)
    d["end"] = end.where(end <= close_t, close_t)
    return d[d["end"] <= now].reset_index(drop=True)

def emas(close):
    c = pd.Series(close, dtype=float); return c.ewm(span=FAST, adjust=False).mean().to_numpy(), c.ewm(span=SLOW, adjust=False).mean().to_numpy()

def find_cross(df, now, side):
    """Latest completed hourly candle (or the one before) on which the STOCK's EMA12 crossed ABOVE EMA26 (CALL) or BELOW it (PUT)."""
    d = completed_hourly(df, now)
    if len(d) < SLOW + 3: return None
    e12, e26 = emas(d["close"])
    for i in (len(d) - 1, len(d) - 2):
        hit = (e12[i] > e26[i] and e12[i - 1] <= e26[i - 1]) if side == "CALL" else (e12[i] < e26[i] and e12[i - 1] >= e26[i - 1])
        if hit:
            age = (now - d["end"].iloc[i].to_pydatetime()).total_seconds() / 60
            if age > FRESH_MIN: return None
            return {"end": d["end"].iloc[i].to_pydatetime(), "close": float(d["close"].iloc[i]), "e12": float(e12[i]), "e26": float(e26[i])}
    return None

# ───────────────────────────── alert ─────────────────────────────
def tg_send(text):
    tok = (os.environ.get("EMA_TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_TOKEN") or "").strip(); chat = os.environ.get("EMA_TELEGRAM_CHAT_ID", "").strip()
    if not tok or not chat: print("EMA Telegram channel not configured (EMA_TELEGRAM_CHAT_ID missing)"); return False
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"}).encode()
        with urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=data), timeout=20) as r:
            return r.status == 200
    except Exception as e: print("telegram failed:", e); return False

def alert_text(ev):
    call = ev["side"] == "CALL"; e = "🟢" if call else "🔴"; sc = " | ".join(f"{x:+.0f}" for x in ev["sc"]) or "—"; c = ev["cross"]
    prem = f"{ev['prem']:.2f}" if ev.get("prem") is not None else "—"
    return (f"{e} <b>{ev['side']}</b> · <code>{html.escape(ev['sym'])}</code>\n"
            f"Stock 1H EMA 12 crossed {'above' if call else 'below'} EMA 26 (EMA12 {c['e12']:.2f} / EMA26 {c['e26']:.2f})\n"
            f"Stock price {ev['spot']:.2f} is {'above' if call else 'below'} live VWAP-3d {ev['vwap3']:.2f} and VWAP-5d {ev['vwap5']:.2f}\n"
            f"ATM {ev['side']}: {html.escape(ev['osym'])} (strike {ev['strike']}, exp {ev['expiry']}) · premium {prem}\n"
            f"Conviction (last {len(ev['sc'])}, old→new): {sc}\n"
            f"⏱ candle closed {c['end']:%H:%M} IST · paper-trade log, not advice")

# ───────────────────────────── outcomes ─────────────────────────────
def _first_at_or_after(d, t):
    g = d[d["ts"] >= t]; return g.iloc[0] if len(g) else None

def outcomes_for_row(row, ix, now, mcache, hcache):
    out = {}
    try: t0 = dt.datetime.strptime(row[ix["date"]] + " " + row[ix["time"]], "%Y-%m-%d %H:%M")
    except Exception: return out
    if (now - t0).days > 14: return out
    osym = row[ix["option"]]; entry = TE._num(row[ix["entry prem"]]); sym = row[ix["symbol"]]; side = row[ix["side"]]
    if not osym or not entry: return out
    cell = lambda c: row[ix[c]] if ix[c] < len(row) else ""
    if all(cell(c) != "" for c in OUTS): return out
    sess_over = (now.date() > t0.date()) or (now.hour * 60 + now.minute >= 15 * 60 + 31)
    if osym not in mcache:
        try: mcache[osym] = get_minute(osym, t0.date().isoformat(), now)
        except Exception: mcache[osym] = None
    m = mcache[osym]
    pct = lambda px: round((float(px) / entry - 1) * 100, 2)
    if m is not None and len(m):
        day = m[m["session"] == t0.date()]; day_after = day[day["ts"] >= t0]
        for col, mins in (("opt+1h%", 60), ("opt+2h%", 120)):
            if cell(col) != "": continue
            T = t0 + dt.timedelta(minutes=mins)
            if T.hour * 60 + T.minute > 15 * 60 + 29:
                if sess_over: out[col] = "n/a"
                continue
            if now < T: continue
            b = _first_at_or_after(day, T)
            if b is not None: out[col] = pct(b["close"])
            elif sess_over: out[col] = "n/a"
        if sess_over and len(day):
            if cell("optEOD%") == "": out["optEOD%"] = pct(day["close"].iloc[-1])
            hi = (day_after["high"] if "high" in day_after else day_after["close"]).max() if len(day_after) else np.nan
            lo = (day_after["low"] if "low" in day_after else day_after["close"]).min() if len(day_after) else np.nan
            if cell("peak%") == "" and hi == hi: out["peak%"] = pct(hi)
            if cell("low%") == "" and lo == lo: out["low%"] = pct(lo)
        if cell("opt+1D%") == "":
            later = sorted(s for s in m["session"].unique() if s > t0.date())
            if later and (later[0] < now.date() or now.hour * 60 + now.minute >= 15 * 60 + 31):
                out["opt+1D%"] = pct(m[m["session"] == later[0]]["close"].iloc[-1])
            elif (now.date() - t0.date()).days > 5: out["opt+1D%"] = "n/a"
    if cell("recross%") == "" and m is not None and len(m):
        # the cross-back is judged on the STOCK: for a CALL, EMA12 back below EMA26; for a PUT, back above. We report what the OPTION premium was then.
        if sym not in hcache:
            try: hcache[sym] = get_hourly(sym, now)
            except Exception: hcache[sym] = None
        h = hcache[sym]
        if h is not None and len(h):
            d = completed_hourly(h, now)
            if len(d) > SLOW + 2:
                e12, e26 = emas(d["close"])
                after = [i for i in range(len(d)) if d["end"].iloc[i].to_pydatetime() > t0 and ((e12[i] < e26[i]) if side == "CALL" else (e12[i] > e26[i]))]
                if after:
                    tend = d["end"].iloc[after[0]].to_pydatetime(); b = _first_at_or_after(m, tend - dt.timedelta(minutes=1))
                    if b is not None: out["recross time"] = tend.strftime("%Y-%m-%d %H:%M"); out["recross%"] = pct(b["close"])
    return out

def fill_outcomes(table, now):
    ix = {h: i for i, h in enumerate(table[0])}; ups = {}; mc, hc = {}, {}
    for r, row in enumerate(table[1:], start=1):
        row = row + [""] * (len(table[0]) - len(row))
        for col, v in outcomes_for_row(row, ix, now, mc, hc).items(): ups[(r, ix[col])] = v
    return ups

def scorecard(table):
    if len(table) < 2: return pd.DataFrame()
    df = pd.DataFrame([r + [""] * (len(table[0]) - len(r)) for r in table[1:]], columns=table[0])
    cols = ["opt+1h%", "opt+2h%", "optEOD%", "opt+1D%", "peak%", "low%", "recross%"]
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    rows = []
    for name, g in [("ALL", df)] + [(s, x) for s, x in df.groupby("side")]:
        r = {"group": name, "alerts": len(g)}
        for c in cols[:4] + ["recross%"]:
            v = g[c].dropna(); r[f"{c} avg"] = v.mean() if len(v) else np.nan; r[f"{c} win%"] = (v > 0).mean() * 100 if len(v) else np.nan; r[f"{c} n"] = len(v)
        r["peak% avg"] = g["peak%"].mean(); r["low% avg"] = g["low%"].mean(); rows.append(r)
    return pd.DataFrame(rows).round(2)

# ───────────────────────────── cycle ─────────────────────────────
def option_premium(osym, cross_end, now):
    """ATM option's hourly close on the candle that completed with the stock's cross (fallback: its latest completed hourly close)."""
    h = get_hourly(osym, now)
    if h is None or h.empty: return None
    d = completed_hourly(h, now)
    if d.empty: return None
    m = d[d["end"] == pd.Timestamp(cross_end)]
    return float((m if len(m) else d).iloc[-1]["close"])

def scan(scores, now, existing_counts, existing_keys):
    """Returns new alert events for this cycle. One stock-hourly fetch per stock; the ATM option is fetched only when the stock passes every test."""
    day = now.strftime("%Y-%m-%d"); stat = {"stocks": 0, "no candles": 0, "too short": 0, "no live vwap": 0, "cross (call)": 0, "cross (put)": 0, "failed vwap": 0, "passed": 0, "no ATM": 0, "errors": 0}
    bars_seen = []; live = {}
    def one(item):
        sym, d = item; stat["stocks"] += 1
        try:
            h = get_hourly(sym, now)
            if h is None or h.empty: stat["no candles"] += 1; return None
            vw = live_vwaps(h, now, sym); live[sym] = vw
            if vw[3] is None or vw[5] is None: stat["no live vwap"] += 1; return None
            dd = completed_hourly(h, now); bars_seen.append(len(dd))
            if len(dd) < SLOW + 3: stat["too short"] += 1; return None
            for side, typ in (("CALL", "CE"), ("PUT", "PE")):
                cr = find_cross(h, now, side)
                if not cr: continue
                stat["cross (call)" if side == "CALL" else "cross (put)"] += 1
                if not vwap_ok(side, vw["px"], vw[3], vw[5]): stat["failed vwap"] += 1; continue
                key = (cr["end"].strftime("%Y-%m-%d"), cr["end"].strftime("%H:%M"), sym, side)
                if key in existing_keys or existing_counts.get((day, sym, side), 0) >= MAX_PER_DAY: continue
                stat["passed"] += 1
                a = get_atm(sym)
                if not a: stat["no ATM"] += 1; return None
                osym = a[typ](); prem = option_premium(osym, cr["end"], now)
                return {"sym": sym, "side": side, "osym": osym, "strike": a["atm"], "expiry": str(a["expiry"]), "cross": cr, "prem": prem,
                        "vwap3": vw[3], "vwap5": vw[5], "sc": d["sc"], "spot": vw["px"], "key": key}
        except Exception as e: stat["errors"] += 1; print(f"  {sym}: {type(e).__name__}: {e}")
        return None
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex: res = [r for r in ex.map(one, scores.items()) if r]
    scan.live = live
    print("summary:", ", ".join(f"{k} {v}" for k, v in stat.items()), f"| hourly candles per stock: median {int(np.median(bars_seen)) if bars_seen else 0}, need {SLOW + 3}")
    return res

def event_row(ev, now):
    sc = (["", "", "", "", ""] + [round(x, 1) for x in ev["sc"]])[-5:]; c = ev["cross"]
    rec = {"date": c["end"].strftime("%Y-%m-%d"), "time": c["end"].strftime("%H:%M"), "alert time": now.strftime("%H:%M"), "symbol": ev["sym"], "side": ev["side"],
           "option": ev["osym"], "strike": ev["strike"], "expiry": ev["expiry"], "spot": round(ev["spot"], 2), "entry prem": round(ev["prem"], 2) if ev.get("prem") is not None else "",
           "ema12": round(c["e12"], 2), "ema26": round(c["e26"], 2), "gap%": round((c["e12"] / c["e26"] - 1) * 100, 2), "vwap3": round(ev["vwap3"], 2), "vwap5": round(ev["vwap5"], 2)}
    rec.update({f"s{i+1}": sc[i] for i in range(5)}); return [rec.get(h, "") for h in HDR]

def put_scorecard(sh, table):
    import gspread
    sc = scorecard(table)
    if sc.empty: return
    try: w = sh.worksheet(SC_TAB); w.clear()
    except gspread.WorksheetNotFound: w = sh.add_worksheet(title=SC_TAB, rows=30, cols=30)
    o = sc.astype(object).where(sc.notna(), ""); w.update(values=[list(o.columns)] + o.values.tolist(), range_name="A1", value_input_option="USER_ENTERED")

def run_cycle(sh):
    import gspread
    now = dt.datetime.now(IST).replace(tzinfo=None)
    dash_ws = sh.worksheet("Rule Dashboard"); scores = read_scores(dash_ws)
    try: ws = sh.worksheet(TAB)
    except gspread.WorksheetNotFound:
        if FORCE: ws = None
        else:
            ws = sh.add_worksheet(title=TAB, rows=2000, cols=len(HDR) + 2); ws.update(values=[HDR], range_name="A1")
            try: ws.freeze(rows=1)
            except Exception: pass
    table = ws.get_all_values() if ws else [HDR]
    if not table: table = [HDR]
    ix = {h: i for i, h in enumerate(table[0])}
    keys = {(r[ix["date"]], r[ix["time"]], r[ix["symbol"]], r[ix["side"]]) for r in table[1:] if len(r) > ix["side"]}
    counts = {}
    for r in table[1:]:
        if len(r) > ix["side"]: k = (r[ix["date"]], r[ix["symbol"]], r[ix["side"]]); counts[k] = counts.get(k, 0) + 1
    t = now.hour * 60 + now.minute
    events = scan(scores, now, counts, keys) if (FORCE or 9 * 60 + 15 <= t <= 15 * 60 + 45) else []
    if t >= 9 * 60 + 15 and t <= 15 * 60 + 45 and not FORCE and getattr(scan, "live", None):
        try: print(f"live vwap3 / vwap5 written for {write_live_vwaps(dash_ws, scan.live)} stocks")
        except Exception as e: print("could not write live vwap:", type(e).__name__, e)
    for ev in events:
        txt = alert_text(ev); print(("[REHEARSAL] " if FORCE else "") + txt.replace("\n", " | "))
        if not FORCE: tg_send(txt)
    if events and not FORCE: ws.append_rows([event_row(ev, now) for ev in events], value_input_option="USER_ENTERED")
    print(f"{now:%H:%M} new EMA 12/26 alerts: {len(events)}")
    if ws is not None and not FORCE and len(table) + len(events) > 1:
        table = ws.get_all_values(); ups = fill_outcomes(table, now)
        if ups:
            print(f"outcomes: filling {len(ups)} cells")
            ws.batch_update([{"range": gspread.utils.rowcol_to_a1(r + 1, c + 1), "values": [[v]]} for (r, c), v in ups.items()], value_input_option="USER_ENTERED")
            for (r, c), v in ups.items(): table[r] = table[r] + [""] * (len(table[0]) - len(table[r])); table[r][c] = v
        if events or ups: put_scorecard(sh, table)
    return len(events)

def setup_test(sh):
    """Creates the log tab (header only) and sends one test message to the EMA channel. Works at any hour."""
    import gspread
    try: sh.worksheet(TAB); print(f"tab '{TAB}' already exists")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TAB, rows=2000, cols=len(HDR) + 2); ws.update(values=[HDR], range_name="A1")
        try: ws.freeze(rows=1)
        except Exception: pass
        print(f"created tab '{TAB}'")
    ok = tg_send("✅ EMA 12/26 + VWAP alert channel is connected. Real alerts will appear here during market hours.")
    print("telegram test message sent" if ok else "telegram test FAILED: check EMA_TELEGRAM_CHAT_ID (starts with -100) and that the bot is an admin of the channel")

def main(mode):
    import live_refresh_local as LR
    sh = LR.open_sheet()
    if mode == "test": setup_test(sh); return
    print("REHEARSAL (FORCE)" if FORCE else "live", "| mode", mode)
    while True:
        try: run_cycle(sh)
        except Exception as e: print("cycle error:", type(e).__name__, e)
        now = dt.datetime.now(IST)
        if mode != "loop" or now.hour * 60 + now.minute >= 15 * 60 + 45: break
        time.sleep(LOOP_MIN * 60)

# ───────────────────────────── self-test ─────────────────────────────
def _hourly(start="2026-10-12", sessions=6, closes=None):
    rows = []; k = 0
    for d in pd.bdate_range(start, periods=sessions):
        for h in range(7):
            ts = d + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(hours=h); p = closes[k]; k += 1
            rows.append({"ts": ts, "open": p, "high": p * 1.01, "low": p * .99, "close": p, "volume": 100})
    return TE._norm(pd.DataFrame(rows))

def _find_cross_series(sign):
    """closes that fall (rise) then reverse so EMA12 crosses EMA26 inside the session window (10:15-14:15)."""
    for L in range(26, 38):
        base = np.concatenate([100 - sign * 0.5 * np.arange(L), (100 - sign * 0.5 * (L - 1)) + sign * 4 * np.arange(1, 43 - L + 1)])[:42]
        cl = base                                                                      # sign>0: down then up (CALL cross) ; sign<0: up then down (PUT cross)
        h = _hourly(closes=cl); d = completed_hourly(h, pd.Timestamp("2026-10-30 23:00")); e12, e26 = emas(d["close"])
        ks = [i for i in range(1, len(d)) if ((e12[i] > e26[i] and e12[i - 1] <= e26[i - 1]) if sign > 0 else (e12[i] < e26[i] and e12[i - 1] >= e26[i - 1]))]
        if ks and 9 * 60 + 45 <= d["end"].iloc[ks[0]].hour * 60 + d["end"].iloc[ks[0]].minute <= 12 * 60 + 30: return h, d["end"].iloc[ks[0]].to_pydatetime()
    raise AssertionError("no test series")

def selftest():
    hc, endc = _find_cross_series(+1); hp, endp = _find_cross_series(-1)
    # CALL: right after the crossing candle completes -> found; before it completes -> not; long after -> stale; and NOT found as a PUT
    assert find_cross(hc, endc + dt.timedelta(minutes=3), "CALL") is not None and find_cross(hc, endc + dt.timedelta(minutes=3), "PUT") is None
    c1 = find_cross(hc, endc - dt.timedelta(minutes=1), "CALL"); assert c1 is None or c1["end"] < endc
    assert find_cross(hc, endc + dt.timedelta(minutes=300), "CALL") is None
    # PUT is the exact mirror
    assert find_cross(hp, endp + dt.timedelta(minutes=3), "PUT") is not None and find_cross(hp, endp + dt.timedelta(minutes=3), "CALL") is None
    assert find_cross(hp, endp + dt.timedelta(minutes=300), "PUT") is None
    # VWAP rule: CALL needs price above BOTH, PUT below BOTH, anything missing -> no alert
    assert vwap_ok("CALL", 101, 100, 99) and not vwap_ok("CALL", 99.5, 100, 99) and not vwap_ok("CALL", 98, 100, 99) and not vwap_ok("CALL", 101, None, 99)
    assert vwap_ok("PUT", 98, 100, 99) and not vwap_ok("PUT", 99.5, 100, 99) and not vwap_ok("PUT", 101, 100, 99) and not vwap_ok("PUT", None, 100, 99)
    print("cross CALL at", endc, "| PUT at", endp, "| vwap rule OK")
    # scan(): stock hourly drives the cross; dashboard supplies vwap3/vwap5 + scores; option fetched only after every test passes
    fetched = []
    class A(dict): pass
    def fake_atm(sym): return {"atm": 100, "expiry": "2026-11-24", "CE": lambda: sym + "26NOV100CE", "PE": lambda: sym + "26NOV100PE"}
    ohr = _hourly(closes=[5.0 + 0.01 * i for i in range(42)])
    def fake_hourly(sym, now): fetched.append(sym); return hc if sym.startswith("UP") else (hp if sym.startswith("DN") else ohr)
    globals().update(get_atm=fake_atm, get_hourly=fake_hourly)
    px = float(hc[hc["ts"] <= pd.Timestamp(endc + dt.timedelta(minutes=3))]["close"].iloc[-1]); pxp = float(hp[hp["ts"] <= pd.Timestamp(endp + dt.timedelta(minutes=3))]["close"].iloc[-1])   # latest price incl. the hour in progress
    # live vwap is computed from the candles; here we pin it per stock (the maths itself is tested separately below)
    PIN = {"UPOK": (px - 1, px - 2), "UPBAD": (px + 1, px - 2), "UPNOV": (None, None), "DNOK": (pxp + 1, pxp + 2), "DNBAD": (pxp - 1, pxp + 2)}
    real_live = live_vwaps
    globals()["live_vwaps"] = lambda h, now, sym=None: {3: PIN[sym][0], 5: PIN[sym][1], "px": float(h[h["ts"] <= pd.Timestamp(now)]["close"].iloc[-1]), "session": None}
    sc = {"UPOK": {"sc": [10, 20, 30, 41, 35], "ltp": px}, "UPBAD": {"sc": [1, 2], "ltp": px}, "UPNOV": {"sc": [1], "ltp": px}}
    now = endc + dt.timedelta(minutes=3); ev = scan(sc, now, {}, set())
    assert [e["sym"] for e in ev] == ["UPOK"] and ev[0]["side"] == "CALL" and ev[0]["osym"] == "UPOK26NOV100CE" and ev[0]["prem"] is not None, ev
    assert "UPOK26NOV100CE" in fetched and "UPBAD26NOV100CE" not in fetched, fetched          # ATM option only fetched for the stock that passes every test
    assert set(scan.live) == {"UPOK", "UPBAD", "UPNOV"}
    sc2 = {"DNOK": {"sc": [-10, -20, -30], "ltp": pxp}, "DNBAD": {"sc": [-5], "ltp": pxp}}
    ev2 = scan(sc2, endp + dt.timedelta(minutes=3), {}, set()); assert [e["sym"] for e in ev2] == ["DNOK"] and ev2[0]["side"] == "PUT" and ev2[0]["osym"].endswith("100PE"), ev2
    assert scan(sc, now, {}, {ev[0]["key"]}) == [] and scan(sc, now, {(now.strftime("%Y-%m-%d"), "UPOK", "CALL"): MAX_PER_DAY}, set()) == []
    globals()["live_vwaps"] = real_live
    # live VWAP maths: rebuild daily bars from the hourly candles, today's partial session included, and roll (today replaces the oldest day)
    nowv = pd.Timestamp("2026-10-16 12:20"); lv = live_vwaps(hc, nowv); dv = hc[hc["ts"] <= nowv]
    gg = dv.groupby("session").agg(h=("high", "max"), l=("low", "min"), c=("close", "last"), v=("volume", "sum")).sort_index()
    ex = lambda W: float(((gg["h"] + gg["l"] + gg["c"]) / 3 * gg["v"]).iloc[-W:].sum() / gg["v"].iloc[-W:].sum())
    assert str(gg.index[-1]) == "2026-10-16" and abs(lv[3] - ex(3)) < 1e-9 and abs(lv[5] - ex(5)) < 1e-9, (lv, ex(3), ex(5))
    nowv2 = pd.Timestamp("2026-10-16 14:20"); lv2 = live_vwaps(hc, nowv2); assert lv2[3] != lv[3], "must update as the day progresses (live, not as of previous close)"
    lvp = live_vwaps(hc, pd.Timestamp("2026-10-15 15:30")); assert str(lvp["session"]) == "2026-10-15"       # no future bars leak in
    short = hc[hc["session"].isin(sorted(hc["session"].unique())[:2])]; assert live_vwaps(short, pd.Timestamp("2026-10-30 23:00"))[3] is None      # fewer than 3 sessions -> no value
    # dashboard write: only stocks with live values change, other cells untouched
    class WS:
        def __init__(s): s.vals = [["symbol", "LTP", "vwap3", "vwap5"], ["AAA", "1", "10", "11"], ["BBB", "2", "20", "21"], ["CCC", "3", "30", "31"]]; s.w = {}
        def get_all_values(s): return s.vals
        def update(s, values, range_name, value_input_option=None): s.w[range_name] = values
    w = WS(); n = write_live_vwaps(w, {"AAA": {3: 12.345, 5: 13.0}, "CCC": {3: None, 5: None}}); assert n == 1 and w.w["C2:C4"] == [[12.35], ["20"], ["30"]] and w.w["D2:D4"] == [[13.0], ["21"], ["31"]], w.w
    print(alert_text(ev[0])); print(alert_text(ev2[0]))
    # outcomes with synthetic minute data: premium 100 at entry, +5 after 1h, +10 at 2h, EOD 12, next day 20; stock EMA12 later crosses back -> recross uses the OPTION price then
    cr = find_cross(hc, endc + dt.timedelta(minutes=3), "CALL"); t0 = cr["end"]; mins = []
    for dd in sorted(set([t0.date(), (pd.Timestamp(t0.date()) + pd.offsets.BDay(1)).date()])):
        for i in range(375):
            ts = pd.Timestamp(dd) + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(minutes=i)
            if dd == t0.date(): m = (ts - t0).total_seconds() / 60; px_ = 100 + (5 * m / 60 if m <= 60 else 5 + 5 * (m - 60) / 60 if m <= 120 else 10 + 2 * (m - 120) / 100) if m >= 0 else 100
            else: px_ = 120
            mins.append({"ts": ts, "open": px_, "high": px_ + .5, "low": px_ - .5, "close": px_, "volume": 10})
    mdf = TE._norm(pd.DataFrame(mins)); globals()["get_minute"] = lambda o, d0, nw: mdf
    e0 = {"sym": "TEST", "side": "CALL", "osym": "TEST26OCT100CE", "strike": 100, "expiry": "2026-10-27", "cross": {**cr, "close": 250.0}, "prem": 100.0, "vwap3": 240.0, "vwap5": 235.0, "sc": [10, 22, 30, 41, 35], "spot": 250.0}
    row = event_row(e0, t0 + dt.timedelta(minutes=3)); assert len(row) == len(HDR) and row[HDR.index("entry prem")] == 100.0 and row[HDR.index("vwap5")] == 235.0
    after = (pd.Timestamp(t0.date()) + pd.offsets.BDay(2)).to_pydatetime().replace(hour=10)
    ups = fill_outcomes([HDR, row], after); got = {HDR[c]: v for (r, c), v in ups.items()}; print(got)
    assert abs(got["opt+1h%"] - 5) < 1.2 and abs(got["opt+2h%"] - 10) < 1.2 and got["opt+1D%"] == 20.0 and got["peak%"] >= got["optEOD%"] > 10, got
    tbl = [HDR] + [[(got.get(hh, row[i]) if row[i] == "" else row[i]) for i, hh in enumerate(HDR)]]
    print(scorecard(tbl).T.to_string()); print("SELFTEST OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: main(sys.argv[1] if len(sys.argv) > 1 else "once")
