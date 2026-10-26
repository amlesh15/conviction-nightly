# ==================================================================================
# EMA CROSS ALERTS — hourly EMA 9 / EMA 26 cross on the ATM option, gated by daily conviction.
#
#   RULE (CALL and PUT use the SAME cross logic, applied to that option's own premium):
#       the option's hourly EMA9 crosses ABOVE its EMA26 on a just-completed hourly candle
#   GATE (Rule Dashboard daily conviction, last two scores):
#       CALL option: both of the last 2 scores  >  +20
#       PUT  option: both of the last 2 scores  <  -20
#   ALERT: separate Telegram channel (EMA_TELEGRAM_CHAT_ID; token = EMA_TELEGRAM_TOKEN, else TELEGRAM_TOKEN)
#       stock, CALL/PUT, ATM option name + premium, spot, last 5 conviction scores.
#   LOG:   tab "EMA Cross Triggers" (one row per alert) + performance filled in later, and a
#          "EMA Cross Scorecard" tab. Paper-trade only; NO ORDERS are placed.
#   Entry premium (paper) = close of the hourly candle on which the cross completed.
#   Performance (premium % vs entry): +1h, +2h, end of day, next-day close, best (peak) and worst (low)
#          premium reached that day, and the result of exiting when EMA9 crosses back below EMA26.
#
#   Usage:  python ema_cross_alert.py once | loop | --selftest
#   Env (optional): EMA_LOOP_MIN=5  EMA_SCORE_MIN=20  EMA_MAX_PER_DAY=2  FORCE=true (rehearsal: prints, writes nothing)
# ==================================================================================
import os, re, sys, time, html, datetime as dt, urllib.parse, urllib.request
from zoneinfo import ZoneInfo
import concurrent.futures as cf
import numpy as np, pandas as pd
import trigger_engine as TE

IST = ZoneInfo("Asia/Kolkata")
FAST, SLOW = 9, 26
SCORE_MIN = float(os.environ.get("EMA_SCORE_MIN", 20)); LOOP_MIN = int(os.environ.get("EMA_LOOP_MIN", 5))
MAX_PER_DAY = int(os.environ.get("EMA_MAX_PER_DAY", 2)); FRESH_MIN = 95; SEED_DAYS = 15; WORKERS = 6
FORCE = os.environ.get("FORCE", "").lower() == "true"
TAB, SC_TAB = "EMA Cross Triggers", "EMA Cross Scorecard"
HDR = ["date", "time", "alert time", "symbol", "side", "option", "strike", "expiry", "spot", "entry prem", "ema9", "ema26", "gap%",
       "s1", "s2", "s3", "s4", "s5", "opt+1h%", "opt+2h%", "optEOD%", "opt+1D%", "peak%", "low%", "recross time", "recross%"]
OUTS = ["opt+1h%", "opt+2h%", "optEOD%", "opt+1D%", "peak%", "low%", "recross time", "recross%"]

# ───────────────────────────── data access (overridden in the self-test) ─────────────────────────────
def get_hourly(osym, now):
    import candle_run as CR
    seed = (now.date() - dt.timedelta(days=SEED_DAYS)).isoformat()
    return TE._candles_df(CR.fetch_candles(osym, seed, now.date().isoformat(), "1H"))

def get_minute(osym, d0, now):
    import candle_run as CR
    return TE._candles_df(CR.fetch_candles(osym, d0, now.date().isoformat(), "1M"))

def get_atm(sym):
    import candle_run as CR
    ref = CR.fetch_atm_ref(sym)
    if not ref: return None
    return {"atm": ref["atm"], "expiry": ref["expiry"], "CE": lambda: CR.build_sym(sym, ref["expiry"], ref["atm"], "CE"),
            "PE": lambda: CR.build_sym(sym, ref["expiry"], ref["atm"], "PE")}

# ───────────────────────────── conviction ─────────────────────────────
def read_scores(ws):
    vals = ws.get_all_values()
    if not vals: return {}
    hdr = [h.strip() for h in vals[0]]; si = hdr.index("symbol") if "symbol" in hdr else 0
    dcols = [i for i, h in enumerate(hdr) if re.fullmatch(r"\d{2}-\d{2}", h)]
    li = hdr.index("LTP") if "LTP" in hdr else None; out = {}
    for row in vals[1:]:
        if si >= len(row) or not row[si].strip(): continue
        sc = [TE._num(row[i]) if i < len(row) else None for i in dcols]; sc = [x for x in sc if x is not None][-5:]
        out[row[si].strip().upper()] = {"sc": sc, "ltp": TE._num(row[li]) if li is not None and li < len(row) else None}
    return out

def gate(side, sc):
    if len(sc) < 2: return False
    return min(sc[-2:]) > SCORE_MIN if side == "CALL" else max(sc[-2:]) < -SCORE_MIN

# ───────────────────────────── the cross ─────────────────────────────
def completed_hourly(df, now):
    d = df.copy(); end = d["ts"] + pd.Timedelta(minutes=60); close_t = d["ts"].dt.normalize() + pd.Timedelta(hours=15, minutes=30)
    d["end"] = end.where(end <= close_t, close_t)
    return d[d["end"] <= now].reset_index(drop=True)

def emas(close):
    c = pd.Series(close, dtype=float); return c.ewm(span=FAST, adjust=False).mean().to_numpy(), c.ewm(span=SLOW, adjust=False).mean().to_numpy()

def find_cross(df, now):
    """Latest completed hourly candle (or the one before) on which EMA9 moved from <= EMA26 to > EMA26."""
    d = completed_hourly(df, now)
    if len(d) < SLOW + 3: return None
    e9, e26 = emas(d["close"])
    for i in (len(d) - 1, len(d) - 2):
        if e9[i] > e26[i] and e9[i - 1] <= e26[i - 1]:
            age = (now - d["end"].iloc[i].to_pydatetime()).total_seconds() / 60
            if age > FRESH_MIN: return None
            return {"end": d["end"].iloc[i].to_pydatetime(), "close": float(d["close"].iloc[i]), "e9": float(e9[i]), "e26": float(e26[i])}
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
    e = "🟢" if ev["side"] == "CALL" else "🔴"; sc = " | ".join(f"{x:+.0f}" for x in ev["sc"]) or "—"
    return (f"{e} <b>{ev['side']}</b> · <code>{html.escape(ev['sym'])}</code>\n"
            f"1H EMA 9 crossed above EMA 26 on the {ev['side']} option\n"
            f"Option: {html.escape(ev['osym'])} (ATM {ev['strike']}, exp {ev['expiry']})\n"
            f"Premium: {ev['cross']['close']:.2f} · EMA9 {ev['cross']['e9']:.2f} / EMA26 {ev['cross']['e26']:.2f}\n"
            f"Spot: {ev['spot'] if ev['spot'] is not None else '—'}\n"
            f"Conviction (last {len(ev['sc'])}, old→new): {sc}\n"
            f"⏱ candle closed {ev['cross']['end']:%H:%M} IST · paper-trade log, not advice")

# ───────────────────────────── outcomes ─────────────────────────────
def _first_at_or_after(d, t):
    g = d[d["ts"] >= t]; return g.iloc[0] if len(g) else None

def outcomes_for_row(row, ix, now, mcache, hcache):
    out = {}
    try: t0 = dt.datetime.strptime(row[ix["date"]] + " " + row[ix["time"]], "%Y-%m-%d %H:%M")
    except Exception: return out
    if (now - t0).days > 14: return out
    osym = row[ix["option"]]; entry = TE._num(row[ix["entry prem"]])
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
    if cell("recross%") == "":
        if osym not in hcache:
            try: hcache[osym] = get_hourly(osym, now)
            except Exception: hcache[osym] = None
        h = hcache[osym]
        if h is not None and len(h):
            d = completed_hourly(h, now)
            if len(d) > SLOW + 2:
                e9, e26 = emas(d["close"]); after = [i for i in range(len(d)) if d["end"].iloc[i].to_pydatetime() > t0 and e9[i] < e26[i]]
                if after: out["recross time"] = d["end"].iloc[after[0]].strftime("%Y-%m-%d %H:%M"); out["recross%"] = pct(d["close"].iloc[after[0]])
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
def scan(scores, now, existing_counts, existing_keys):
    """Returns new alert events for this cycle."""
    work = []
    for sym, d in scores.items():
        for side, typ in (("CALL", "CE"), ("PUT", "PE")):
            if gate(side, d["sc"]) and existing_counts.get((now.strftime("%Y-%m-%d"), sym, side), 0) < MAX_PER_DAY: work.append((sym, side, typ))
    print(f"gate passed: {sum(1 for w in work if w[1]=='CALL')} CALL candidates, {sum(1 for w in work if w[1]=='PUT')} PUT candidates")
    stat = {"checked": 0, "no ATM": 0, "no candles": 0, "too short": 0, "9 above 26 now": 0, "cross in last 2 candles": 0, "errors": 0}
    def one(w):
        sym, side, typ = w
        stat["checked"] += 1
        try:
            a = get_atm(sym)
            if not a: stat["no ATM"] += 1; return None
            osym = a[typ](); h = get_hourly(osym, now)
            if h is None or h.empty: stat["no candles"] += 1; return None
            dd = completed_hourly(h, now)
            if len(dd) < SLOW + 3: stat["too short"] += 1; return None
            e9, e26 = emas(dd["close"]); stat["9 above 26 now"] += int(e9[-1] > e26[-1])
            stat["cross in last 2 candles"] += int(any(e9[i] > e26[i] and e9[i - 1] <= e26[i - 1] for i in (len(dd) - 1, len(dd) - 2)))
            cr = find_cross(h, now)
            if not cr: return None
            key = (cr["end"].strftime("%Y-%m-%d"), cr["end"].strftime("%H:%M"), sym, side)
            if key in existing_keys: return None
            return {"sym": sym, "side": side, "osym": osym, "strike": a["atm"], "expiry": str(a["expiry"]), "cross": cr,
                    "sc": scores[sym]["sc"], "spot": scores[sym]["ltp"], "key": key}
        except Exception as e: stat["errors"] += 1; print(f"  {sym} {side}: {type(e).__name__}: {e}"); return None
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex: res = [r for r in ex.map(one, work) if r]
    print("summary:", ", ".join(f"{k} {v}" for k, v in stat.items()))
    return res

def event_row(ev, now):
    sc = (["", "", "", "", ""] + [round(x, 1) for x in ev["sc"]])[-5:]; c = ev["cross"]
    rec = {"date": c["end"].strftime("%Y-%m-%d"), "time": c["end"].strftime("%H:%M"), "alert time": now.strftime("%H:%M"), "symbol": ev["sym"], "side": ev["side"],
           "option": ev["osym"], "strike": ev["strike"], "expiry": ev["expiry"], "spot": ev["spot"] if ev["spot"] is not None else "", "entry prem": round(c["close"], 2),
           "ema9": round(c["e9"], 2), "ema26": round(c["e26"], 2), "gap%": round((c["e9"] / c["e26"] - 1) * 100, 2)}
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
    scores = read_scores(sh.worksheet("Rule Dashboard"))
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
    for ev in events:
        txt = alert_text(ev); print(("[REHEARSAL] " if FORCE else "") + txt.replace("\n", " | "))
        if not FORCE: tg_send(txt)
    if events and not FORCE: ws.append_rows([event_row(ev, now) for ev in events], value_input_option="USER_ENTERED")
    print(f"{now:%H:%M} new EMA-cross alerts: {len(events)}")
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
    ok = tg_send("✅ EMA cross alert channel is connected. Real alerts will appear here during market hours.")
    print("telegram test message sent" if ok else "telegram test FAILED: check EMA_TELEGRAM_CHAT_ID (starts with -100) and that the bot is an admin of the channel")

def main(mode):
    import live_refresh_local as LR
    sh = LR.open_sheet()
    if mode == "test": setup_test(sh); return; print("REHEARSAL (FORCE)" if FORCE else "live", "| mode", mode)
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

def selftest():
    for L in range(26, 38):
        cl = np.concatenate([100 - 0.5 * np.arange(L), (100 - 0.5 * (L - 1)) + 4 * np.arange(1, 43 - L + 1)])[:42]; h = _hourly(closes=cl)
        d = completed_hourly(h, pd.Timestamp("2026-10-30 23:00")); e9, e26 = emas(d["close"])
        ks = [i for i in range(1, len(d)) if e9[i] > e26[i] and e9[i - 1] <= e26[i - 1]]
        if ks and 9 * 60 + 45 <= d["end"].iloc[ks[0]].hour * 60 + d["end"].iloc[ks[0]].minute <= 12 * 60 + 30: k = ks[0]; break
    endk = d["end"].iloc[k].to_pydatetime()
    # right after the crossing candle completes -> alert; before it completes -> nothing; long after -> stale
    assert find_cross(h, endk + dt.timedelta(minutes=3)) is not None
    assert find_cross(h, endk - dt.timedelta(minutes=1)) is None or find_cross(h, endk - dt.timedelta(minutes=1))["end"] < endk
    assert find_cross(h, endk + dt.timedelta(minutes=300)) is None
    # gate
    assert gate("CALL", [5, 25, 30]) and not gate("CALL", [5, 25, 15]) and gate("PUT", [-30, -25]) and not gate("PUT", [-30, -10]) and not gate("CALL", [30])
    print("cross at", endk, "| gate OK")
    # outcomes with synthetic minute data: premium 100 at entry, +5 after 1h, +10 at 2h, EOD 12, next day 20
    cr = find_cross(h, endk + dt.timedelta(minutes=3)); t0 = cr["end"]
    mins = []
    for dd in sorted(set([t0.date(), (pd.Timestamp(t0.date()) + pd.offsets.BDay(1)).date()])):
        for i in range(375):
            ts = pd.Timestamp(dd) + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(minutes=i)
            if dd == t0.date(): m = (ts - t0).total_seconds() / 60; px = 100 + (5 * m / 60 if m <= 60 else 5 + 5 * (m - 60) / 60 if m <= 120 else 10 + 2 * (m - 120) / 100) if m >= 0 else 100
            else: px = 120
            mins.append({"ts": ts, "open": px, "high": px + .5, "low": px - .5, "close": px, "volume": 10})
    mdf = TE._norm(pd.DataFrame(mins)); globals()["get_minute"] = lambda o, d0, now: mdf; globals()["get_hourly"] = lambda o, now: h
    ev = {"sym": "TEST", "side": "CALL", "osym": "TEST26OCT100CE", "strike": 100, "expiry": "2026-10-27", "cross": {**cr, "close": 100.0}, "sc": [10, 22, 30, 41, 35], "spot": 250.0}
    row = event_row(ev, t0 + dt.timedelta(minutes=3)); print(alert_text(ev))
    after = (pd.Timestamp(t0.date()) + pd.offsets.BDay(2)).to_pydatetime().replace(hour=10)
    ups = fill_outcomes([HDR, row], after); ixh = {hh: i for i, hh in enumerate(HDR)}; got = {HDR[c]: v for (r, c), v in ups.items()}; print(got)
    assert abs(got["opt+1h%"] - 5) < 1.2 and abs(got["opt+2h%"] - 10) < 1.2 and got["opt+1D%"] == 20.0 and got["peak%"] >= got["optEOD%"] > 10, got
    tbl = [HDR] + [[(got.get(hh, row[i]) if row[i] == "" else row[i]) for i, hh in enumerate(HDR)]]
    print(scorecard(tbl).T.to_string()); print("SELFTEST OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: main(sys.argv[1] if len(sys.argv) > 1 else "once")
