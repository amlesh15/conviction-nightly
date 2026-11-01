# ==================================================================================
# IMBALANCE REACTION STUDY — how does price behave when it reaches an imbalance level?  (understanding first, no trigger yet)
#   For every F&O stock and every session with 1-minute data (Sensibull keeps ~60 days):
#     1. levels are the ones that existed BEFORE the session opened: fvg_lab.nearest_imbalances (unchanged) on daily bars up to the previous day only
#        -> no hindsight (a chart drawn today with today's levels flatters past touches; this does not).
#     2. the session's 1-min bars are rolled up to 15-min bars, and each level is followed from its first touch:
#          S_BOUNCE  price touched a support level and closed back above it            -> CALL idea
#          S_BREAK   price closed below the support level                              -> PUT idea
#          R_BREAK   price closed above a resistance level                              -> CALL idea (+ does it retest and hold?)
#          R_REJECT  price touched resistance and closed back below it                  -> PUT idea
#          CHOP      touched, did nothing decisive before the close
#     3. from the confirming bar's close we measure what happened next: return after 1h / 2h / close, best and worst excursion over 2h,
#        and a symmetric race: does price go +0.5 x daily range first, or -0.5 x daily range first?  (50% = no edge)
#     4. CONTROLS: the same test on fake levels a full daily range further away, and on every 15-min bar of every stock (the market's own drift).
#        A level only "works" if it beats those.
#   Output tabs: "IMB React Summary", "IMB React Events".   Usage: python imb_react_study.py [--selftest]
#   env: GCP_SA_JSON, SHEET_ID, SENSIBULL cookie (setup_env.py); optional IR_DAYS (60), IR_SYMBOLS ("A,B"), IR_TOL/IR_BRK (fractions of daily range, 0.10 / 0.12)
# ==================================================================================
import os, sys, time, datetime as dt
import concurrent.futures as cf
import numpy as np, pandas as pd

SELFTEST = "--selftest" in sys.argv
COOKIE_PATH = os.path.expanduser("~/live-refresh/cookie.txt")
CANDLE = "https://oxide.sensibull.com/v1/compute/1/candles/{sym}"
DAYS = int(os.environ.get("IR_DAYS", "56")); TOLF = float(os.environ.get("IR_TOL", "0.10")); BRKF = float(os.environ.get("IR_BRK", "0.12"))
RACEF = 0.5          # race target / stop = 0.5 x daily range
LAST_ENTRY = 22      # confirming bar must be 15:00 or earlier (bar 22) so there is room to measure

# ───────────────────────────── data ─────────────────────────────
def _hdr():
    ck = open(COOKIE_PATH).read().strip() if os.path.exists(COOKIE_PATH) else ""
    return {"cookie": ck, "content-type": "application/json", "origin": "https://web.sensibull.com", "referer": "https://web.sensibull.com/", "user-agent": "Mozilla/5.0"}

STATUS = {}   # last HTTP status / reason per symbol (written to the summary tab so a failed run explains itself)

def fetch_1m(sym, days=DAYS, tries=3):
    import requests
    end = dt.date.today()
    for k in range(tries + 2):
        body = {"from_date": (end - dt.timedelta(days=days)).isoformat(), "to_date": (end + dt.timedelta(days=1)).isoformat(), "interval": "1M", "skip_last_ts": False}
        try:
            r = requests.post(CANDLE.format(sym=sym), json=body, headers=_hdr(), timeout=(8, 40))
            if r.status_code in (429, 502, 503, 504): time.sleep(1.5 * (k + 1)); continue
            STATUS[sym] = f"HTTP {r.status_code}"
            if r.status_code == 400 and "exceeds limit" in r.text and days > 20: days -= 4; continue     # server counts the range inclusively; shrink and retry
            if r.status_code != 200: STATUS[sym] += " " + r.text[:80].replace("\n", " "); return None
            c = ((r.json() or {}).get("payload") or {}).get("candles") or []
            if not c: STATUS[sym] += " no candles"; return None
            df = pd.DataFrame(c); ts = pd.to_datetime(df["ts"])
            if getattr(ts.dt, "tz", None) is not None: ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            df["ts"] = ts
            for col in ("open", "high", "low", "close", "volume"): df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.sort_values("ts").reset_index(drop=True)
        except Exception as ex_: STATUS[sym] = "error " + str(ex_)[:80]; time.sleep(1.0 * (k + 1))
    return None

def to_15m(df):
    """1-min -> 15-min bars per session. Returns {session(str): DataFrame(open,high,low,close) indexed 0..n-1 in time order}."""
    d = df.copy(); d["session"] = d["ts"].dt.strftime("%Y-%m-%d"); d["b"] = (d["ts"].dt.hour * 60 + d["ts"].dt.minute - 555) // 15
    d = d[(d["b"] >= 0) & (d["b"] <= 24)]
    g = d.groupby(["session", "b"]).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).reset_index()
    return {s: x.sort_values("b").reset_index(drop=True) for s, x in g.groupby("session") if len(x) >= 20}

# ───────────────────────────── the measurement ─────────────────────────────
def _forward(c, h, l, j, dirn, atr):
    """What happened after confirming bar j, seen from a trade in direction dirn (+1 call idea / -1 put idea). All in % of entry price, 'good' = in our favour."""
    e = c[j]; n = len(c); k4 = min(j + 4, n - 1); k8 = min(j + 8, n - 1)
    sg = lambda x: dirn * (x / e - 1) * 100
    hi = h[j + 1:j + 9]; lo = l[j + 1:j + 9]
    if len(hi) == 0: return None
    best = sg(hi.max()) if dirn > 0 else sg(lo.min()); worst = sg(lo.min()) if dirn > 0 else sg(hi.max())
    tg = RACEF * atr; race = "none"
    for k in range(j + 1, n):
        up = h[k] >= e * (1 + tg / 100); dn = l[k] <= e * (1 - tg / 100)
        if dirn < 0: up, dn = dn, up
        # for a short idea 'up' means price fell by tg (win)
        if up and dn: race = "L"; break           # both in one bar: count against us
        if up: race = "W"; break
        if dn: race = "L"; break
    return {"fwd 1h %": round(sg(c[k4]), 3), "fwd 2h %": round(sg(c[k8]), 3), "to close %": round(sg(c[-1]), 3), "best 2h %": round(best, 3), "worst 2h %": round(worst, 3), "race": race}

def scan_level(o, h, l, c, L, kind, atr):
    """kind 'S' (level below the previous close) or 'R' (above). atr = daily range in %. Returns an event dict or None (never touched / gapped through)."""
    tol = TOLF * atr / 100; brk = BRKF * atr / 100; n = len(c)
    if kind == "S":
        if o[0] < L * (1 - brk): return None                      # opened already through the level
        t = next((i for i in range(n) if l[i] <= L * (1 + tol)), None)
        if t is None: return None
        for j in range(t, n):
            if c[j] < L * (1 - brk): out, dirn = "S_BREAK", -1; break
            if c[j] >= L * (1 + brk): out, dirn = "S_BOUNCE", 1; break
        else: return {"outcome": "CHOP", "touch bar": t, "confirm bar": None}
        wick = bool(l[t:j + 1].min() < L * (1 - brk)) if out == "S_BOUNCE" else False
    else:
        if o[0] > L * (1 + brk): return None
        t = next((i for i in range(n) if h[i] >= L * (1 - tol)), None)
        if t is None: return None
        for j in range(t, n):
            if c[j] > L * (1 + brk): out, dirn = "R_BREAK", 1; break
            if c[j] <= L * (1 - brk): out, dirn = "R_REJECT", -1; break
        else: return {"outcome": "CHOP", "touch bar": t, "confirm bar": None}
        wick = bool(h[t:j + 1].max() > L * (1 + brk)) if out == "R_REJECT" else False
    ev = {"outcome": out, "touch bar": t, "confirm bar": j, "wick through": "Y" if wick else ""}
    if j > LAST_ENTRY: ev["outcome"] = out + "_LATE"; return ev
    f = _forward(c, h, l, j, dirn, atr)
    if f is None: return None
    ev.update(f); ev["entry"] = round(float(c[j]), 2)
    if out == "R_BREAK":                                          # does the broken level turn into support?
        k = next((i for i in range(j + 1, min(j + 9, n)) if l[i] <= L * (1 + tol)), None)
        ev["retest"] = "Y" if k is not None else "N"
        ev["retest held"] = ("Y" if not (c[k:min(k + 3, n)] < L * (1 - brk)).any() else "N") if k is not None else ""
    return ev

def base_rate(c, h, l, atr, step=1):
    rows = []
    for j in range(0, len(c) - 9, step):
        f = _forward(c, h, l, j, 1, atr)
        if f: rows.append(f)
    return rows

# ───────────────────────────── levels, as they stood before each session ─────────────────────────────
def levels_before(md_sym, sess):
    import fvg_lab as FL
    g = md_sym[md_sym["date"] < sess]
    if len(g) < 20: return None
    r = FL.nearest_imbalances(g[["symbol", "date", "high", "low", "close"]]) if len(g) else None
    if r is None or r.empty: return None
    r = r.iloc[0]; P = float(g["close"].iloc[-1]); gg = g.tail(14)
    atr = float(((gg["high"] - gg["low"]) / gg["close"]).mean() * 100)
    out = []
    for k in (1, 2, 3):
        for kind, col in (("S", f"imb_sup{k}"), ("R", f"imb_res{k}")):
            v = r.get(col)
            if v not in ("", None) and v == v: out.append((kind, k, float(v)))
    return P, atr, out

def study_symbol(sym, md_sym, bars_by_sess, latest):
    ev, base = [], []
    for sess, b in bars_by_sess.items():
        if sess > latest: continue
        lv = levels_before(md_sym, sess)
        if lv is None: continue
        P, atr, levels = lv
        if not (0.3 < atr < 15): continue
        o, h, l, c = (b[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        base += base_rate(c, h, l, atr, step=2)
        for kind, k, L in levels:
            for ctrl in (0, 1):
                LL = L if not ctrl else (L * (1 - atr / 100) if kind == "S" else L * (1 + atr / 100))
                e = scan_level(o, h, l, c, LL, kind, atr)
                if e is None: continue
                e.update({"symbol": sym, "date": sess, "level": f"{kind}{k}", "price": round(LL, 2), "away %": round((LL / P - 1) * 100, 2), "daily range %": round(atr, 2), "control": "fake" if ctrl else "real"})
                if e.get("confirm bar") is not None: e["time"] = f"{9 + (15 + 15 * e['confirm bar']) // 60}:{(15 + 15 * e['confirm bar']) % 60:02d}"
                ev.append(e)
    return ev, base

# ───────────────────────────── summary ─────────────────────────────
def summarise(ev, base):
    E = pd.DataFrame(ev); B = pd.DataFrame(base); rows = []
    def stat(name, d, note=""):
        d = d[d["race"].isin(["W", "L", "none"])] if "race" in d else d
        n = len(d)
        if n == 0: return
        dec = d[d["race"].isin(["W", "L"])]
        win = round((dec["race"] == "W").mean() * 100, 1) if len(dec) else ""
        se = round(100 * (0.25 / max(len(dec), 1)) ** 0.5, 1) if len(dec) else ""
        rows.append([name, n, win, se, round(d["fwd 1h %"].mean(), 3), round(d["fwd 2h %"].mean(), 3), round(d["fwd 2h %"].median(), 3), round(d["to close %"].mean(), 3), round(d["best 2h %"].mean(), 3), round(d["worst 2h %"].mean(), 3), note])
    stat("ALL 15-min bars, any stock (market drift, call side)", B, "baseline for CALL ideas")
    if not E.empty:
        done = E[E["outcome"].isin(["S_BOUNCE", "S_BREAK", "R_BREAK", "R_REJECT"])]
        for ctrl in ("real", "fake"):
            for oc in ("S_BOUNCE", "R_BREAK", "S_BREAK", "R_REJECT"):
                for lvl in ("ALL", "1", "2", "3"):
                    d = done[(done["control"] == ctrl) & (done["outcome"] == oc)]
                    if lvl != "ALL": d = d[d["level"].str.endswith(lvl)]
                    stat(f"{ctrl.upper()} levels | {oc} | {'any rank' if lvl == 'ALL' else 'rank ' + lvl}", d, "CALL idea" if oc in ("S_BOUNCE", "R_BREAK") else "PUT idea (gain shown = price fell)")
        d = done[(done["control"] == "real") & (done["outcome"] == "S_BOUNCE") & (done["wick through"] == "Y")]; stat("REAL | S_BOUNCE after a wick below the level (reclaim)", d, "CALL idea")
        d = done[(done["control"] == "real") & (done["outcome"] == "S_BOUNCE") & (done["wick through"] != "Y")]; stat("REAL | S_BOUNCE clean (no wick below)", d, "CALL idea")
        d = done[(done["control"] == "real") & (done["outcome"] == "R_BREAK") & (done["retest"] == "Y")]; stat("REAL | R_BREAK that later retested the level", d, "CALL idea")
        d = done[(done["control"] == "real") & (done["outcome"] == "R_BREAK") & (done["retest held"] == "Y")]; stat("REAL | R_BREAK, retest HELD (level flipped to support)", d, "CALL idea")
        d = done[(done["control"] == "real") & (done["outcome"] == "R_BREAK") & (done["retest held"] == "N")]; stat("REAL | R_BREAK, retest FAILED", d, "CALL idea")
        rows.append([]); rows.append(["How often each thing happens (REAL levels, all touches)"]); r = E[E["control"] == "real"]["outcome"].value_counts()
        for k_, v_ in r.items(): rows.append([k_, int(v_), round(100 * v_ / r.sum(), 1)])
    head = ["bucket", "events", "race win % (0.5 daily range first)", "± 1 sd", "fwd 1h % avg", "fwd 2h % avg", "fwd 2h % median", "to close % avg", "best 2h % avg", "worst 2h % avg", "note"]
    return head, rows

def main():
    import run_nightly as RN, gspread, live_refresh_local as LR
    today = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")
    G = RN.load_engine(); RN.seed_cache(); md = G["build_masterdata_pit_fast"](RN.START, today, full=False)
    md["date"] = md["date"].astype(str); latest = str(md["date"].max())
    syms = [s.strip().upper() for s in (os.environ.get("IR_SYMBOLS") or "").split(",") if s.strip()] or sorted(md["symbol"].unique())
    print(f"{len(syms)} symbols; daily data through {latest}; pulling {DAYS} days of 1-min candles")
    byS = {s: g.sort_values("date").reset_index(drop=True) for s, g in md[md["symbol"].isin(syms)].groupby("symbol")}
    def job(s):
        df = fetch_1m(s)
        return s, (to_15m(df) if df is not None and len(df) else None)
    EV, BASE, nodata = [], [], []
    with cf.ThreadPoolExecutor(6) as ex:
        for i, (s, bars) in enumerate(ex.map(job, syms)):
            if not bars: nodata.append(s); continue
            e, b = study_symbol(s, byS[s], bars, latest); EV += e; BASE += b
            if (i + 1) % 25 == 0: print(f"  {i + 1}/{len(syms)} done, {len(EV)} events so far")
    print(f"events {len(EV)}, no intraday data for {len(nodata)} symbols: {nodata[:20]}")
    head, rows = summarise(EV, BASE)
    from collections import Counter
    notes = [[], ["RUN NOTES"], ["symbols asked", len(syms)], ["symbols with intraday data", len(syms) - len(nodata)], ["symbols with NO intraday data", len(nodata)], ["events found", len(EV)], ["15-min baseline bars", len(BASE)], ["daily data through", latest]]
    for k_, v_ in Counter(STATUS.get(x, "?") for x in nodata).most_common(5): notes.append(["no-data reason", k_, v_])
    rows = rows + notes
    for r_ in rows[:60]: print(" | ".join(str(x) for x in r_))
    sh = LR.open_sheet()
    def put(title, header, body):
        vals = [header] + body
        try: ws = sh.worksheet(title); ws.clear()
        except gspread.WorksheetNotFound: ws = sh.add_worksheet(title=title, rows=len(vals) + 20, cols=max(len(header), 12) + 2)
        if ws.row_count < len(vals) + 5: ws.resize(rows=len(vals) + 20)
        ws.update(values=vals, range_name="A1", value_input_option="USER_ENTERED")
    put("IMB React Summary", head, rows)
    E = pd.DataFrame(EV)
    if not E.empty:
        cols = ["date", "symbol", "control", "level", "price", "away %", "daily range %", "outcome", "time", "entry", "wick through", "retest", "retest held", "fwd 1h %", "fwd 2h %", "to close %", "best 2h %", "worst 2h %", "race"]
        for cc in cols:
            if cc not in E: E[cc] = ""
        E = E[E["control"] == "real"].sort_values(["date", "symbol"])[cols].fillna("").head(9000)
        put("IMB React Events", cols, E.values.tolist())
    print("tabs written")

def selftest():
    n = 25; z = np.zeros(n)
    def mk(path):   # path = list of closes; highs/lows +-0.1%
        c = np.array(path, float); o = np.r_[c[0], c[:-1]]; return o, c * 1.001, c * 0.999, c
    atr = 2.0; L = 100.0
    # support bounce: drifts down, touches 100, closes back above, then rallies
    p = list(np.linspace(103, 100.3, 8)) + [100.0, 100.6, 101.0] + list(np.linspace(101.2, 104, 14)); o, h, l, c = mk(p)
    e = scan_level(o, h, l, c, L, "S", atr); assert e["outcome"] == "S_BOUNCE" and e["race"] == "W" and e["to close %"] > 1, e
    # support break: closes under 100 and keeps falling (a PUT idea -> gain positive)
    p = list(np.linspace(103, 100.2, 8)) + [99.5, 99.0, 98.5] + list(np.linspace(98.3, 96, 14)); o, h, l, c = mk(p)
    e = scan_level(o, h, l, c, L, "S", atr); assert e["outcome"] == "S_BREAK" and e["race"] == "W" and e["to close %"] > 1, e
    # gap through support at the open -> skipped
    p = [97.0] * n; o, h, l, c = mk(p); assert scan_level(o, h, l, c, L, "S", atr) is None
    # never touched
    p = [105.0] * n; o, h, l, c = mk(p); assert scan_level(o, h, l, c, L, "S", atr) is None
    # resistance break + retest + hold + rally
    p = list(np.linspace(97, 99.8, 8)) + [100.5, 101.0, 100.4, 100.2, 100.8] + list(np.linspace(101, 103.5, 12)); o, h, l, c = mk(p)
    e = scan_level(o, h, l, c, L, "R", atr); assert e["outcome"] == "R_BREAK" and e["retest"] == "Y" and e["retest held"] == "Y" and e["race"] == "W", e
    # resistance rejection (PUT idea)
    p = list(np.linspace(97, 99.8, 8)) + [100.0, 99.5, 99.0] + list(np.linspace(98.8, 96, 14)); o, h, l, c = mk(p)
    e = scan_level(o, h, l, c, L, "R", atr); assert e["outcome"] == "R_REJECT" and e["to close %"] > 1, e
    # chop
    p = [100.0 + (0.05 if i % 2 else -0.05) for i in range(n)]; o, h, l, c = mk([103] + p[1:]); o[0] = 103.0
    e = scan_level(o, h, l, c, L, "S", atr); assert e is not None and e["outcome"] == "CHOP", e
    # 1-min -> 15-min roll-up
    ts = pd.date_range("2026-10-29 09:15", periods=375, freq="min"); df = pd.DataFrame({"ts": ts, "open": 1.0, "high": np.arange(375) + 2.0, "low": 0.5, "close": np.arange(375) + 1.0, "volume": 1})
    b = to_15m(df)["2026-10-29"]; assert len(b) == 25 and b["high"].iloc[0] == 16.0 and b["close"].iloc[0] == 15.0 and b["close"].iloc[24] == 375.0
    # levels use only data BEFORE the session (no hindsight)
    d = pd.bdate_range("2026-09-01", periods=40).strftime("%Y-%m-%d"); px = np.r_[np.linspace(100, 120, 20), np.linspace(120, 100, 20)]
    md = pd.DataFrame({"symbol": "AAA", "date": d, "high": px + 1, "low": px - 1, "close": px}); full = levels_before(md, "2099-01-01"); early = levels_before(md, d[25])
    assert full is not None and early is not None and early[0] == px[24]
    # summary runs on mixed events
    ev = [{"symbol": "A", "date": "2026-10-29", "level": "S1", "control": "real", "outcome": "S_BOUNCE", "wick through": "Y", "fwd 1h %": .3, "fwd 2h %": .5, "to close %": .8, "best 2h %": 1.0, "worst 2h %": -.2, "race": "W"},
          {"symbol": "A", "date": "2026-10-29", "level": "R1", "control": "real", "outcome": "R_BREAK", "retest": "Y", "retest held": "Y", "wick through": "", "fwd 1h %": .1, "fwd 2h %": -.1, "to close %": -.3, "best 2h %": .4, "worst 2h %": -.5, "race": "L"}]
    head, rows = summarise(ev, [{"fwd 1h %": 0, "fwd 2h %": 0, "to close %": 0, "best 2h %": .3, "worst 2h %": -.3, "race": "none"}]); assert len(rows) > 3
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
