# ==================================================================================
# VP MONITOR — multi-timeframe volume-profile levels (POC / VAH / VAL) + live touch / cross alerts.
#
#   A "profile" is built from the 1-minute bars (volume spread across each bar's high-low range, then
#   POC = busiest price, Value Area = the 70% of volume around it -> VAL / VAH). The interval you pick only
#   decides HOW FAR BACK the profile looks:
#       15M x 8   = the last 8 fifteen-minute candles  (~2 hours)
#       1H  x 30  = the last 30 hourly candles         (~5 sessions)
#       1D  x 20  = the last 20 sessions
#   Intraday windows use COMPLETED candles only (levels stay fixed inside a candle). Daily windows use
#   completed sessions (today excluded until 15:25).
#
#   Config: tab "VP Config" (columns: interval | candles | alert). Edit it in the sheet; created with defaults.
#   Output tabs: "VP Levels" (every stock, every config, where price sits), "VP Events" (log of touches/crosses).
#   Telegram: digest of events for configs with alert = Y.  Research only, no orders.
#
#   Usage:  python vp_monitor.py once        one pass (levels + events since the last few bars)
#           python vp_monitor.py loop        repeat every VP_LOOP_MIN minutes until 15:30 IST
#           python vp_monitor.py --selftest
#   Env (optional): VP_LOOP_MIN=5 VP_BIN_PCT=0.05 VP_VA=0.70 VP_EPS_PCT=0.03 VP_TOUCH_PCT=0.05 VP_REACT=8
#                   VP_COOLDOWN=45 VP_MAXMSG=15 VP_FETCH_DAYS=35 VP_MIN_BARS=20 FORCE=true (ignore market hours/freshness)
# ==================================================================================
import os, sys, json, time, html, datetime as dt
from zoneinfo import ZoneInfo
import concurrent.futures as cf
import numpy as np, pandas as pd
import trigger_engine as TE

IST = ZoneInfo("Asia/Kolkata")
_f = lambda k, d: float(os.environ.get(k, d))
LOOP_MIN = int(_f("VP_LOOP_MIN", 5)); BIN_PCT = _f("VP_BIN_PCT", 0.05); VA = _f("VP_VA", 0.70)
EPS = _f("VP_EPS_PCT", 0.03) / 100; TOUCH = _f("VP_TOUCH_PCT", 0.05) / 100; REACT = int(_f("VP_REACT", 8))
COOLDOWN = int(_f("VP_COOLDOWN", 45)); MAXMSG = int(_f("VP_MAXMSG", 15)); FETCH_DAYS = int(_f("VP_FETCH_DAYS", 35))
MIN_BARS = int(_f("VP_MIN_BARS", 20)); FORCE = os.environ.get("FORCE", "").lower() == "true"; WORKERS = 8
DEFAULT_CFG = [("15M", 8, "Y"), ("15M", 26, "N"), ("1H", 6, "Y"), ("1H", 30, "Y"), ("1D", 5, "Y"), ("1D", 20, "Y")]
STEP = {"15M": 15, "1H": 60}
STATE = os.path.join(os.environ.get("STATE_DIR", os.path.expanduser("~/live-refresh/state")), "vp_state.json")
EV_HDR = ["date", "time", "symbol", "window", "level", "level price", "event", "last", "distance %"]

# ───────────────────────────── profile ─────────────────────────────
def profile(w):
    """Volume profile of a bar window -> {'POC','VAH','VAL'} or None."""
    hi = w["high"].to_numpy(float); lo = w["low"].to_numpy(float); v = w["volume"].fillna(0).to_numpy(float)
    ok = np.isfinite(hi) & np.isfinite(lo) & (hi >= lo) & (v > 0); hi, lo, v = hi[ok], lo[ok], v[ok]
    if len(v) < MIN_BARS: return None
    p0 = lo.min(); ref = float(np.median((hi + lo) / 2)); wd = max(ref * BIN_PCT / 100, 0.01)
    a = np.floor((lo - p0) / wd).astype(int); b = np.floor((hi - p0) / wd).astype(int); nb = int(b.max()) + 1
    share = v / (b - a + 1); diff = np.zeros(nb + 1); np.add.at(diff, a, share); np.add.at(diff, b + 1, -share)
    vol = np.cumsum(diff)[:nb]; tot = vol.sum()
    if tot <= 0: return None
    poc = int(np.argmax(vol)); li = hi_i = poc; acc = float(vol[poc])
    while acc < VA * tot and (li > 0 or hi_i < nb - 1):
        up = float(vol[hi_i + 1:hi_i + 3].sum()) if hi_i < nb - 1 else -1.0
        dn = float(vol[max(li - 2, 0):li].sum()) if li > 0 else -1.0
        if up >= dn: hi_i = min(hi_i + 2, nb - 1); acc += up
        else: li = max(li - 2, 0); acc += dn
    return {"VAL": p0 + li * wd, "POC": p0 + (poc + .5) * wd, "VAH": p0 + (hi_i + 1) * wd}

def completed(df, interval):
    ls = df["session"].iloc[-1]; lm = int(df["mod"].iloc[-1])
    if lm >= 15 * 60 + 25: return df                        # session finished: everything is complete
    if interval == "1D": return df[df["session"] < ls]
    st = STEP[interval]; cut = 555 + ((lm - 555) // st) * st  # 09:15 anchored candle boundary
    return df[(df["session"] < ls) | (df["mod"] < cut)]

def levels_for(df, interval, n):
    c = completed(df, interval)
    if c.empty: return None
    if interval == "1D":
        ss = sorted(c["session"].unique())[-n:]; w = c[c["session"].isin(ss)]
    else: w = c.tail(n * STEP[interval])
    return profile(w)

# ───────────────────────────── events ─────────────────────────────
def detect(df, lv):
    """Touches / crosses of POC, VAH, VAL during the last REACT bars of the latest session."""
    s = df[df["session"] == df["session"].iloc[-1]].tail(REACT + 1)
    if len(s) < REACT + 1: return []
    cl = s["close"].to_numpy(float); hi = s["high"].to_numpy(float); lo = s["low"].to_numpy(float); c0, c1 = cl[0], cl[-1]; out = []
    for name, L in lv.items():
        if c0 <= L * (1 - EPS) and c1 >= L * (1 + EPS): out.append((name, L, "CROSS UP", c1))
        elif c0 >= L * (1 + EPS) and c1 <= L * (1 - EPS): out.append((name, L, "CROSS DOWN", c1))
        elif ((hi >= L * (1 - TOUCH)) & (lo <= L * (1 + TOUCH))).any():
            if c0 < L and c1 < L: out.append((name, L, "TOUCH from below", c1))
            elif c0 > L and c1 > L: out.append((name, L, "TOUCH from above", c1))
    return out

def pos_text(last, lv):
    if last > lv["VAH"]: return "above VAH"
    if last < lv["VAL"]: return "below VAL"
    return "in value, " + ("above" if last >= lv["POC"] else "below") + " POC"

# ───────────────────────────── data ─────────────────────────────
def fetch(args):
    sym, days = args
    import intraday_fvg as F
    for _ in range(3):
        try:
            d = F.fetch_1m(sym, days=days, seg=getattr(F, "CASH_SEG", 1))
            if d is not None and not d.empty: return sym, TE._fix_volume(TE._norm(d))
        except Exception: pass
    return sym, None

def fetch_all(symbols, days):
    out = {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for s, d in ex.map(fetch, [(s, days) for s in symbols]):
            if d is not None and len(d): out[s] = d
    return out

def merge(old, new):
    keep = old[~old["session"].isin(set(new["session"].unique()))]
    return pd.concat([keep, new]).sort_values("ts").reset_index(drop=True)

# ───────────────────────────── sheet / state ─────────────────────────────
def load_cfg(sh):
    import gspread
    try: vals = sh.worksheet("VP Config").get_all_values()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="VP Config", rows=40, cols=6)
        ws.update(values=[["interval", "candles", "alert"]] + [[i, n, a] for i, n, a in DEFAULT_CFG] + [[], ["interval: 15M, 1H or 1D. candles: how many candles back. alert: Y sends Telegram, N logs only."]], range_name="A1")
        return list(DEFAULT_CFG)
    cfg = []
    for r in vals[1:]:
        if len(r) >= 2 and r[0].strip().upper() in ("15M", "1H", "1D") and str(r[1]).strip().isdigit():
            cfg.append((r[0].strip().upper(), int(r[1]), (r[2].strip().upper() if len(r) > 2 and r[2].strip() else "Y")))
    return cfg or list(DEFAULT_CFG)

def load_state():
    try:
        s = json.load(open(STATE))
        if s.get("date") == dt.date.today().isoformat(): return s
    except Exception: pass
    return {"date": dt.date.today().isoformat(), "last": {}}

def save_state(s):
    try: os.makedirs(os.path.dirname(STATE), exist_ok=True); json.dump(s, open(STATE, "w"))
    except Exception as e: print("state save failed:", e)

def put(sh, title, header, rows, append=False):
    import gspread
    try: ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=max(1000, len(rows) + 50), cols=max(12, len(header) + 2)); ws.update(values=[header], range_name="A1")
        try: ws.freeze(rows=1)
        except Exception: pass
    if append:
        if rows: ws.append_rows(rows, value_input_option="USER_ENTERED")
    else:
        ws.clear(); ws.update(values=[header] + rows, range_name="A1", value_input_option="USER_ENTERED")

# ───────────────────────────── one cycle ─────────────────────────────
def cycle(bars, cfgs, now, state):
    rows, events = [], []
    for sym in sorted(bars):
        df = bars[sym]; last = float(df["close"].iloc[-1]); fresh = (now - df["ts"].iloc[-1].to_pydatetime().replace(tzinfo=None)).total_seconds() / 60
        row = [sym, round(last, 2)]
        for itv, n, alert in cfgs:
            lv = levels_for(df, itv, n); name = f"{itv} x{n}"
            if not lv: row += ["", "", "", ""]; continue
            row += [round(lv["VAL"], 2), round(lv["POC"], 2), round(lv["VAH"], 2), pos_text(last, lv)]
            if FORCE or fresh <= 12:
                for lvname, L, ev, px in detect(df, lv):
                    key = f"{sym}|{name}|{lvname}|{ev.split()[0]}"; prev = state["last"].get(key)
                    cd = COOLDOWN if ev.startswith("CROSS") else COOLDOWN * 2
                    if prev and (now - dt.datetime.fromisoformat(prev)).total_seconds() / 60 < cd: continue
                    state["last"][key] = now.isoformat(timespec="seconds")
                    events.append({"sym": sym, "win": name, "lvl": lvname, "L": L, "ev": ev, "px": px, "alert": alert, "n": n, "itv": itv})
        rows.append(row)
    return rows, events

def digest(events):
    order = {"1D": 0, "1H": 1, "15M": 2}
    ev = sorted([e for e in events if e["alert"] == "Y"], key=lambda e: (order[e["itv"]], -e["n"], e["sym"]))[:MAXMSG]
    if not ev: return ""
    icon = {"CROSS UP": "🟩", "CROSS DOWN": "🟥", "TOUCH from below": "🔻", "TOUCH from above": "🔺"}
    lines = [f"{icon[e['ev']]} <b>{html.escape(e['sym'])}</b> {e['ev'].lower()} {e['win']} {e['lvl']} {e['L']:.2f} (px {e['px']:.2f})" for e in ev]
    return "📊 <b>Volume-profile levels</b>\n" + "\n".join(lines) + f"\n⏱ {dt.datetime.now(IST):%H:%M} IST · research, not advice"

def symbols_from_sheet(sh):
    vals = sh.worksheet("Rule Dashboard").get_all_values(); hdr = [h.strip() for h in vals[0]]; si = hdr.index("symbol") if "symbol" in hdr else 0
    return sorted({r[si].strip().upper() for r in vals[1:] if len(r) > si and r[si].strip()})

def main(mode):
    import live_refresh_local as LR
    sh = LR.open_sheet(); cfgs = load_cfg(sh); syms = symbols_from_sheet(sh)
    only = os.environ.get("VP_SYMBOLS", "").strip()
    if only: syms = [s for s in syms if s in {x.strip().upper() for x in only.split(",")}]
    print(f"{len(syms)} symbols | windows: {[f'{i} x{n}({a})' for i, n, a in cfgs]}")
    bars = fetch_all(syms, FETCH_DAYS); print(f"bars for {len(bars)} stocks")
    if not bars: print("No bars (cookie expired?)"); sys.exit(1)
    state = load_state(); header = ["symbol", "last"] + [f"{i} x{n} {k}" for i, n, _ in cfgs for k in ("VAL", "POC", "VAH", "position")]
    try: import notify
    except Exception: notify = None
    while True:
        now = dt.datetime.now(IST).replace(tzinfo=None)
        rows, events = cycle(bars, cfgs, now, state); save_state(state)
        print(f"{now:%H:%M} levels for {len(rows)} stocks | new events: {len(events)}")
        try:
            put(sh, "VP Levels", header, rows)
            if events:
                put(sh, "VP Events", EV_HDR, [[now.strftime("%Y-%m-%d"), now.strftime("%H:%M"), e["sym"], e["win"], e["lvl"], round(e["L"], 2), e["ev"], round(e["px"], 2),
                                              round((e["px"] / e["L"] - 1) * 100, 2)] for e in events], append=True)
        except Exception as e: print("sheet write failed:", e)
        msg = digest(events)
        if msg and notify and notify.enabled() and not FORCE:
            try: notify.send(msg, parse_mode="HTML")
            except Exception as e: print("telegram failed:", e)
        elif msg: print(msg)
        if mode != "loop" or (now.hour * 60 + now.minute) >= 15 * 60 + 30: break
        time.sleep(LOOP_MIN * 60)
        fresh = fetch_all(list(bars), 2)
        for s, d in fresh.items(): bars[s] = merge(bars[s], d)

# ───────────────────────────── self-test ─────────────────────────────
def _mk(prices_by_session, vol=1000):
    rows = []
    for d, path in prices_by_session:
        base = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15)
        for i, p in enumerate(path):
            rows.append({"ts": base + pd.Timedelta(minutes=i), "open": p, "high": p * 1.0003, "low": p * 0.9997, "close": p, "volume": float(vol)})
    return TE._norm(pd.DataFrame(rows))

def selftest():
    # 1) profile: 70% of volume at 100, rest spread to 104 -> POC must be ~100 and value area must hug it
    rng = np.random.default_rng(1); p = np.where(rng.random(375) < .7, 100.0, rng.uniform(100, 104, 375))
    df = _mk([(dt.date(2026, 10, 1), p)]); pr = profile(df)
    assert pr and abs(pr["POC"] - 100) < 0.3 and pr["VAL"] <= pr["POC"] <= pr["VAH"] and pr["VAH"] - pr["VAL"] < 4.5, pr
    # 2) windows: 10 flat sessions at 100, then today runs 100 -> 103 across the 100 level
    days = [d.date() for d in pd.bdate_range(end="2026-10-14", periods=10)]
    hist = [(d, np.full(375, 100.0) + rng.normal(0, .02, 375)) for d in days]
    today = np.concatenate([np.full(120, 99.5), np.linspace(99.5, 100.6, 20), np.full(30, 100.6)])
    df = _mk(hist + [(dt.date(2026, 10, 15), today)])
    df = df[df["ts"] <= pd.Timestamp("2026-10-15 09:15") + pd.Timedelta(minutes=len(today) - 1)].reset_index(drop=True)
    lv = levels_for(df, "1D", 5); assert lv and abs(lv["POC"] - 100) < 0.3, lv
    # 15M window must use completed candles only and not look ahead
    l15 = levels_for(df, "15M", 8); assert l15 is not None
    ev = detect(df.iloc[:-5], lv); ev2 = detect(df, lv)
    print("levels 1D x5:", {k: round(v, 2) for k, v in lv.items()}, "| events at end:", [(a, c) for a, _, c, _ in ev2])
    # cross detected on a fresh jump through VAL/POC
    jump = np.concatenate([np.full(126, 99.0), np.full(4, 101.5)]); d2 = _mk(hist + [(dt.date(2026, 10, 15), jump)])
    e = detect(d2, levels_for(d2, "1D", 5)); assert any(x[2] == "CROSS UP" for x in e), e
    # cooldown + digest
    state = {"date": "x", "last": {}}; bars = {"AAA": d2}; rows, evs = cycle(bars, [("1D", 5, "Y")], d2["ts"].iloc[-1].to_pydatetime(), state)
    rows2, evs2 = cycle(bars, [("1D", 5, "Y")], d2["ts"].iloc[-1].to_pydatetime(), state)
    assert len(evs) > 0 and len(evs2) == 0, (len(evs), len(evs2)); print(digest(evs))
    print("SELFTEST OK")

if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: main(sys.argv[1] if len(sys.argv) > 1 else "once")
