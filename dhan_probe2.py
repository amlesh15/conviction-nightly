# ==================================================================================
# DHAN PROBE 2 — the other four data sources behind Dhan's charts
#   fo_list      GET  https://static-scanx.dhan.co/staticscanx/fo_list        all F&O stocks + security ids
#   getDataH     POST https://ticks.dhan.co/getDataH      INTERVAL "D"  daily candles          (START/END in SECONDS)
#   getData      POST https://ticks.dhan.co/getData       INTERVAL "1/5/15/60" intraday candles (START/END in SECONDS)
#   getOrderFlow POST https://ticks.dhan.co/orderflow/getOrderFlow   footprint data (request shape unknown -> several tried)
#  For each: does it answer, response shape, how far back history goes, and a sample saved to the sheet.
#  Needs the same secret as before (DHAN_HEADERS_JSON = {"Auth":"eyJ..."}).  Research only; nothing here places orders.
#  Usage: python dhan_probe2.py [--selftest]
# ==================================================================================
import os, sys, json, time, datetime as dt
import numpy as np, pandas as pd
import dhan_probe as D

SELFTEST = "--selftest" in sys.argv
IST = D.IST; OUT = D.OUT; SAMPLES = {}
U_FO = "https://static-scanx.dhan.co/staticscanx/fo_list"; U_H = "https://ticks.dhan.co/getDataH"; U_G = "https://ticks.dhan.co/getData"; U_OF = "https://ticks.dhan.co/orderflow/getOrderFlow"
log = D.log
TRANSPORT = None

def call(method, url, body=None, auth=True):
    if TRANSPORT: return TRANSPORT(method, url, body)
    import requests
    h = D.headers(auth)
    return requests.get(url, headers=h, timeout=60) if method == "GET" else requests.post(url, json=body, headers=h, timeout=120)

def shape(o, depth=0, maxd=3):
    if isinstance(o, dict): return "{...}" if depth >= maxd else "{" + ", ".join(f"{k}: {shape(v, depth + 1, maxd)}" for k, v in list(o.items())[:16]) + "}"
    if isinstance(o, list): return f"list[{len(o)}]" + (" of " + shape(o[0], depth + 1, maxd) if o and depth < maxd else "")
    return type(o).__name__ + (f"={str(o)[:24]}" if o is not None else "")

def times(js):
    d = js.get("data", js) if isinstance(js, dict) else js
    if isinstance(d, dict):
        for k in ("t", "time", "Time", "timestamp", "ts"):
            a = d.get(k)
            if isinstance(a, list) and a and isinstance(a[0], (int, float)):
                a = np.array(a, float); return pd.to_datetime(a, unit="ms" if a.max() > 1e12 else "s", utc=True).tz_convert(IST).tz_localize(None)
    if isinstance(d, list) and d and isinstance(d[0], dict):
        for k in ("t", "time", "Time", "timestamp", "ts", "date"):
            if k in d[0] and isinstance(d[0][k], (int, float)):
                a = np.array([x[k] for x in d], float); return pd.to_datetime(a, unit="ms" if a.max() > 1e12 else "s", utc=True).tz_convert(IST).tz_localize(None)
    return None

def describe(name, r, keep=None):
    txt = r.text or ""
    try: js = r.json()
    except Exception: log(name, "response", f"HTTP {r.status_code}, not JSON", txt[:300]); return None
    ts = times(js)
    log(name, "response", f"HTTP {r.status_code}, {len(txt):,} chars", shape(js))
    if ts is not None and len(ts): log(name, "time range", f"{len(ts)} rows", f"{ts.min()} -> {ts.max()}")
    return js

def sec(date_or_days_back=None, days_back=None):
    return int(time.time()) - int(days_back * 86400)

def body(sid, interval, start, end, **kw):
    b = {"EXCH": "NSE", "SEG": "E", "INST": "EQUITY", "SEC_ID": sid, "INTERVAL": interval, "START": start, "END": end}; b.update(kw); return b

def rows_to_sheet(name, js, limit=400):
    d = js.get("data", js) if isinstance(js, dict) else js
    try:
        if isinstance(d, dict):
            ks = [k for k, v in d.items() if isinstance(v, list)]; n = min(len(d[k]) for k in ks); G = pd.DataFrame({k: d[k][-limit:] if len(d[k]) >= n else d[k] for k in ks if len(d[k]) == len(d[ks[0]])})
        elif isinstance(d, list) and d and isinstance(d[0], dict): G = pd.DataFrame(d[-limit:])
        else: return
        SAMPLES[name] = G
    except Exception: pass

def raw_to_sheet(name, r, n=45, w=800):
    t = r.text or ""; SAMPLES[name] = pd.DataFrame({"raw response (first %d chars)" % (n * w): [t[i: i + w] for i in range(0, min(len(t), n * w), w)]})

def main():
    sid = int(os.environ.get("DHAN_SEC_ID", "13538")); D.jwt_info()
    # ── fo_list
    r = call("GET", U_FO, None, auth=False)
    if r.status_code != 200: log("fo_list", "without token", f"HTTP {r.status_code}", (r.text or "")[:150]); r = call("GET", U_FO, None, auth=True)
    js = describe("fo_list", r)
    if js is not None:
        def biggest(o):
            best = []
            def rec(x):
                nonlocal best
                if isinstance(x, list):
                    if x and isinstance(x[0], dict) and len(x) > len(best): best = x
                    for y in x[:3]: rec(y)
                elif isinstance(x, dict):
                    for y in x.values(): rec(y)
            rec(o); return best
        L = biggest(js); log("fo_list", "stock entries found", len(L), "first: " + json.dumps(L[0])[:300] if L else "none")
        hit = [x for x in L if "TECHM" in json.dumps(x).upper()]; log("fo_list", "TECHM entry (should have SEC_ID 13538)", "found" if hit else "not found", json.dumps(hit[0])[:300] if hit else "")
        if L: SAMPLES["Dhan FO List"] = pd.json_normalize(L[:400])
    # ── getDataH (daily)
    now = int(time.time())
    for days in (400, 1500, 3000, 6000):
        r = call("POST", U_H, body(sid, "D", now - days * 86400, now, EXPCODE=0, SYM="TECHM"))
        if r.status_code != 200: log("getDataH", f"daily, {days} days back", f"HTTP {r.status_code}", (r.text or "")[:200]); continue
        try: js = r.json(); ts = times(js); n = len(ts) if ts is not None else -1
        except Exception: js, ts, n = None, None, -1
        log("getDataH", f"daily, asked {days} days back", f"{n} candles", f"{ts.min()} -> {ts.max()}" if ts is not None and len(ts) else "no times found; shape " + (shape(js) if js else "?"))
        if days == 400 and js is not None: describe("getDataH", r); rows_to_sheet("Dhan Daily Sample", js)
        time.sleep(0.4)
    # ── getData (intraday candles): how far back for each interval
    for iv, spans in (("15", (10, 30, 90, 180, 365)), ("5", (10, 30, 90, 180)), ("1", (5, 10, 30, 60, 120)), ("60", (30, 180, 365))):
        for days in spans:
            r = call("POST", U_G, body(sid, iv, now - days * 86400, now + 86400))
            if r.status_code != 200: log("getData", f"{iv}-min, {days} days back", f"HTTP {r.status_code}", (r.text or "")[:160]); continue
            try: js = r.json(); ts = times(js); n = len(ts) if ts is not None else -1
            except Exception: js, ts, n = None, None, -1
            log("getData", f"{iv}-min, asked {days} days back", f"{n} candles", f"{ts.min()} -> {ts.max()}" if ts is not None and len(ts) else "no times found")
            if iv == "15" and days == 10 and js is not None: describe("getData", r); rows_to_sheet("Dhan 15m Sample", js, 300)
            time.sleep(0.3)
    # ── getOrderFlow: exact request shape copied from the browser (no INST, no INTERVAL; times in seconds + ISO strings)
    end = int(pd.Timestamp.now(tz=IST).normalize().timestamp())
    def ofbody(days):
        st = (end - days * 86400) // 86400 * 86400; iso = lambda t: pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return {"EXCH": "NSE", "SEG": "E", "SEC_ID": sid, "START": st, "END": end, "START_TIME": iso(st), "END_TIME": iso(end)}
    good = None
    for days in (73, 5, 30, 180, 365, 730, 1500):
        try: r = call("POST", U_OF, ofbody(days))
        except Exception as e: log("getOrderFlow", f"{days} days", "error", f"{type(e).__name__}: {str(e)[:150]}"); continue
        txt = r.text or ""; js = None
        try: js = r.json()
        except Exception: pass
        ts = times(js) if js is not None else None
        log("getOrderFlow", f"asked {days} days back", f"HTTP {r.status_code}, {len(txt):,} chars" + (f", {len(ts)} time rows" if ts is not None else ""), (f"{ts.min()} -> {ts.max()}" if ts is not None and len(ts) else txt[:260].replace("\n", " ")))
        if r.status_code == 200 and len(txt) > 200 and good is None:
            good = r; describe("getOrderFlow", r); raw_to_sheet("Dhan OrderFlow Raw", r)
            if js is not None: rows_to_sheet("Dhan OrderFlow Sample", js, 300)
        time.sleep(0.5)
    if good is None:
        for lab, b_ in (("with INST+INTERVAL", {**ofbody(5), "INST": "EQUITY", "INTERVAL": "1"}), ("seconds only", {k: v for k, v in ofbody(5).items() if "TIME" not in k})):
            try: r = call("POST", U_OF, b_); log("getOrderFlow", lab, f"HTTP {r.status_code}, {len(r.text or ''):,} chars", (r.text or "")[:260].replace("\n", " "))
            except Exception as e: log("getOrderFlow", lab, "error", str(e)[:120])
        log("getOrderFlow", "no request worked", "tell me the status/message above", "")
    finish()

def finish():
    pd.DataFrame(D.REPORT, columns=["stage", "check", "result", "detail"]).to_csv(f"{OUT}/probe2_report.csv", index=False)
    if SELFTEST: return
    try:
        import oi_probe as P; sh = P.open_sheet(); P.put_tab(sh, "Dhan Probe 2", ["source", "check", "result", "detail  (run " + dt.datetime.now().strftime("%Y-%m-%d %H:%M") + ")"], D.REPORT)
        for name, G in SAMPLES.items():
            G = G.copy().astype(object).where(G.notna(), ""); P.put_tab(sh, name, [str(c) for c in G.columns], G.astype(str).values.tolist())
        print("sheet tabs written")
    except Exception as e: print("sheet write skipped:", type(e).__name__, str(e)[:120])

# ───────────────────────────── self-test: fake server ─────────────────────────────
class _R:
    def __init__(s, js, code=200): s._j = js; s.status_code = code; s.text = json.dumps(js)
    def json(s): return s._j

def _fake(method, url, b):
    now = int(time.time())
    if url == U_FO: return _R({"data": [{"SecId": 13538, "Sym": "TECHM"}, {"SecId": 2885, "Sym": "RELIANCE"}]})
    if url == U_H or url == U_G:
        step = 86400 if b["INTERVAL"] == "D" else int(b["INTERVAL"]) * 60; start = max(b["START"], now - (400 if b["INTERVAL"] == "D" else 20) * 86400)
        t = list(range(start, min(b["END"], now), step))[:3000]; return _R({"data": {"t": t, "o": [1] * len(t), "h": [1] * len(t), "l": [1] * len(t), "c": [1] * len(t), "v": [1] * len(t)}})
    if url == U_OF:
        return _R({"success": True, "data": {"t": [b["START"] + 86400 * i for i in range(60)], "buy": [5] * 60, "sell": [4] * 60}}) if "INTERVAL" not in b and "START_TIME" in b else _R({"error": "bad request"}, 400)
    return _R({}, 404)

if __name__ == "__main__":
    if SELFTEST:
        TRANSPORT = _fake; os.environ.setdefault("DHAN_HEADERS_JSON", "{}"); main()
        rep = {(r[0], r[1]): r for r in D.REPORT}
        assert rep[("fo_list", "TECHM entry (should have SEC_ID 13538)")][2] == "found"
        assert any(r[0] == "getDataH" and "candles" in r[2] and int(r[2].split()[0]) > 100 for r in D.REPORT); assert any(r[0] == "getOrderFlow" and r[2].startswith("HTTP 200") for r in D.REPORT) and "Dhan OrderFlow Raw" in SAMPLES
        print("SELFTEST OK")
    else: main()
