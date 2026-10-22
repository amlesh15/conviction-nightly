# ==================================================================================
# DHAN ORDER-FLOW PROBE — can we read the tick-by-tick feed behind the footprint chart, and how far back does it go?
#   POST https://tickbytick.dhan.co/api/ticks   body {EXCH,SEG,INST,SEC_ID,INTERVAL,START,END}   (shape copied from your browser)
#   response {success, data:{Time[],t[],c[],bid[],ask[],v[],oi[]}, nextTime}
#  Checks (each is one row in the sheet tab "Dhan Probe"):
#   1 does it answer WITHOUT a login token?  (if not: needs DHAN_HEADERS_JSON secret = JSON of header name -> value)
#   2 one full day: ticks, first/last time, pagination via nextTime, gaps between ticks
#   3 is each tick a TRADE or a quote snapshot?  (price vs bid/ask, how often the price sits at ask / at bid / between)
#   4 how many days back can we go?  (1,2,5,10,20,40,90,180,365 days ago)
#   5 other instruments: a few F&O stocks, NIFTY futures (ids from Dhan's public scrip master, if reachable)
#   6 delta quality: classify every tick buy/sell, compare with candle direction
#  Saves a one-day sample to out/dhan/sample_ticks.csv.gz.   Paper research only; nothing here places orders.
#  Usage: python dhan_probe.py [--selftest]      env: DHAN_HEADERS_JSON (optional), DHAN_SEC_ID (default 13538), GCP_SA_JSON, SHEET_ID
# ==================================================================================
import os, sys, io, json, time, gzip, datetime as dt
import numpy as np, pandas as pd

URL = "https://tickbytick.dhan.co/api/ticks"
OUT = "out/dhan"; os.makedirs(OUT, exist_ok=True)
SELFTEST = "--selftest" in sys.argv
IST = "Asia/Kolkata"
REPORT = []

def log(stage, check, result, detail=""):
    REPORT.append([stage, check, str(result), str(detail)[:800]]); print(f"[{stage}] {check}: {result}  {str(detail)[:300]}", flush=True)

def headers(with_auth=True):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/plain, */*", "Origin": "https://dext.dhan.co", "Referer": "https://dext.dhan.co/",
         "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    if with_auth and os.environ.get("DHAN_HEADERS_JSON"):
        try: h.update(json.loads(os.environ["DHAN_HEADERS_JSON"]))
        except Exception as e: log("setup", "DHAN_HEADERS_JSON", "unreadable", type(e).__name__)
    return h

def ms(date, hh=0, mm=0, ss=0):
    return int(pd.Timestamp(f"{date} {hh:02d}:{mm:02d}:{ss:02d}", tz=IST).value // 10**6)

TRANSPORT = None          # selftest replaces this
def post(body, auth=True, timeout=60):
    if TRANSPORT: return TRANSPORT(body)
    import requests
    return requests.post(URL, json=body, headers=headers(auth), timeout=timeout)

def body_for(sec_id, start, end, exch="NSE", seg="E", inst="EQUITY", interval="1"):
    return {"EXCH": exch, "SEG": seg, "INST": inst, "SEC_ID": sec_id, "INTERVAL": interval, "START": start, "END": end}

def frame(js):
    d = (js or {}).get("data") or {}
    if not d or "t" not in d: return pd.DataFrame()
    n = len(d["t"]); out = {"t": d["t"]}
    for k in ("c", "bid", "ask", "v", "oi"): out[k] = d.get(k, [np.nan] * n)[:n]
    F = pd.DataFrame(out); F["ts"] = pd.to_datetime(F["t"], unit="ms", utc=True).dt.tz_convert(IST).dt.tz_localize(None); return F

def fetch_day(sec_id, date, auth=True, max_pages=40, **kw):
    """all ticks for one IST calendar day, following nextTime. returns (frame, pages, status_note)."""
    start, end = ms(date), ms(date, 23, 59, 59); parts = []; pages = 0; note = ""
    cur = start
    while pages < max_pages:
        try: r = post(body_for(sec_id, cur, end, **kw), auth)
        except Exception as e: return pd.concat(parts) if parts else pd.DataFrame(), pages, f"{type(e).__name__}: {str(e)[:120]}"
        pages += 1
        if r.status_code != 200: return pd.concat(parts) if parts else pd.DataFrame(), pages, f"HTTP {r.status_code}: {r.text[:160]}"
        try: js = r.json()
        except Exception: return pd.concat(parts) if parts else pd.DataFrame(), pages, "not JSON: " + r.text[:120]
        F = frame(js)
        if F.empty: note = "empty page"; break
        parts.append(F); nt = js.get("nextTime")
        if not nt or nt <= cur or nt >= end: break
        if F["t"].max() >= end - 60_000: break
        cur = int(nt)
    if not parts: return pd.DataFrame(), pages, note or "no data"
    A = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return A, pages, note

def last_weekdays(n_back):
    d = pd.Timestamp.now(tz=IST).normalize().tz_localize(None); out = []
    while len(out) < n_back:
        if d.weekday() < 5: out.append(d.date())
        d -= pd.Timedelta(days=1)
    return out

def classify(F):
    """+1 = trade at/above ask (buyer aggressive), -1 = at/below bid, else tick rule vs previous price."""
    c, b, a = F["c"].to_numpy(float), F["bid"].to_numpy(float), F["ask"].to_numpy(float)
    side = np.zeros(len(F)); side[(c >= a) & (a > 0)] = 1; side[(c <= b) & (b > 0)] = -1
    mid = side == 0; prev = np.r_[np.nan, c[:-1]]; tr = np.sign(c - prev); tr[np.isnan(tr)] = 0
    side[mid] = tr[mid]; return side, (c >= a) & (a > 0), (c <= b) & (b > 0)

def analyse_day(F, label):
    if F.empty: return
    g = F["ts"].diff().dt.total_seconds().dropna()
    log("day", f"{label}: ticks", len(F), f"{F['ts'].iloc[0]} -> {F['ts'].iloc[-1]}; gap median {g.median():.2f}s, p95 {g.quantile(.95):.1f}s, max {g.max():.0f}s")
    cover = "covers the whole session" if F["ts"].iloc[0].hour * 60 + F["ts"].iloc[0].minute <= 9 * 60 + 17 and F["ts"].iloc[-1].hour * 60 + F["ts"].iloc[-1].minute >= 15 * 60 + 25 else "DOES NOT cover the whole session"
    log("day", f"{label}: coverage", cover, f"volume sum {F['v'].sum():,.0f}; oi non-zero share {(F['oi'] != 0).mean():.2f}")
    side, atask, atbid = classify(F)
    log("trade-or-quote", f"{label}: price at ask / at bid / between", f"{atask.mean()*100:.1f}% / {atbid.mean()*100:.1f}% / {(~atask & ~atbid).mean()*100:.1f}%",
        "if most ticks sit exactly at bid or ask they are real trades -> buy/sell delta is exact; if mostly between they are snapshots and delta is an estimate")
    spread = (F["ask"] - F["bid"]).replace(0, np.nan); log("quote", f"{label}: median spread", f"{spread.median():.3f}", f"crossed/zero quotes {(F['ask'] <= F['bid']).mean()*100:.1f}%")
    F = F.copy(); F["side"] = side; F["signed"] = side * F["v"]; F["min"] = F["ts"].dt.floor("1min")
    m = F.groupby("min").agg(c=("c", "last"), o=("c", "first"), v=("v", "sum"), delta=("signed", "sum")).reset_index(); m = m[m["v"] > 0]
    ret = (m["c"] - m["o"]).to_numpy(); ok = ret != 0
    if ok.sum() > 30: log("delta quality", f"{label}: 1-min delta sign agrees with candle direction", f"{(np.sign(m['delta'])[ok] == np.sign(ret)[ok]).mean()*100:.1f}%", f"day total delta {m['delta'].sum():,.0f} of volume {m['v'].sum():,.0f}")
    return m

def jwt_info():
    """report token lifetime and claim NAMES only (never values) so we know how long it lasts and what it is."""
    try:
        import base64
        tok = json.loads(os.environ.get("DHAN_HEADERS_JSON", "{}")); tok = next((v for k, v in tok.items() if str(v).startswith("eyJ")), None)
        if not tok: return
        p = tok.split(".")[1]; p += "=" * (-len(p) % 4); c = json.loads(base64.urlsafe_b64decode(p))
        exp = c.get("exp"); iat = c.get("iat")
        fmt = lambda t: pd.Timestamp(t, unit="s", tz="UTC").tz_convert(IST).strftime("%Y-%m-%d %H:%M IST")
        life = f"{(exp - iat) / 3600:.1f} hours" if exp and iat else "unknown"
        log("token", "expiry", fmt(exp) if exp else "no exp claim", f"lifetime {life}; issued {fmt(iat) if iat else '?'}; claim names: {', '.join(sorted(c.keys()))}")
    except Exception as e: log("token", "decode", "could not read", type(e).__name__)

def main():
    jwt_info()
    sec = int(os.environ.get("DHAN_SEC_ID", "13538")); today = pd.Timestamp.now(tz=IST).date(); days = last_weekdays(3)
    # 1 auth
    d0 = days[0]
    try: r = post(body_for(sec, ms(d0), ms(d0, 23, 59, 59)), auth=False)
    except Exception as e: log("access", "request without login", "FAILED to connect", f"{type(e).__name__}: {str(e)[:200]}"); return finish(None)
    log("access", "request without login token", f"HTTP {r.status_code}", r.text[:200] if r.status_code != 200 else f"{len(r.content):,} bytes")
    auth = r.status_code != 200
    if auth:
        if not os.environ.get("DHAN_HEADERS_JSON"): log("access", "login needed", "STOP", "add repository secret DHAN_HEADERS_JSON (JSON of the request-header names and values your browser sends) and run again"); return finish(None)
        r2 = post(body_for(sec, ms(d0), ms(d0, 23, 59, 59)), True); log("access", "request WITH your headers", f"HTTP {r2.status_code}", r2.text[:200] if r2.status_code != 200 else "ok")
        if r2.status_code != 200: return finish(None)
    # 2 full recent days
    got = None
    for d in days:
        F, pages, note = fetch_day(sec, d, auth)
        log("day", f"{d} fetch", f"{len(F)} ticks in {pages} page(s)", note)
        if len(F) and got is None: got = (d, F)
        if len(F): analyse_day(F, str(d))
    if got:
        got[1].to_csv(f"{OUT}/sample_ticks.csv.gz", index=False)
    # 4 history depth
    ago = [1, 2, 5, 10, 20, 40, 90, 180, 365]; found = {}
    for k in ago:
        d = pd.Timestamp.now(tz=IST).normalize().tz_localize(None) - pd.Timedelta(days=k)
        while d.weekday() >= 5: d -= pd.Timedelta(days=1)
        try: r = post(body_for(sec, ms(d.date()), ms(d.date(), 23, 59, 59)), auth, timeout=60); js = r.json() if r.status_code == 200 else {}; n = len(((js or {}).get("data") or {}).get("t", []))
        except Exception as e: n = -1
        found[k] = n; log("history", f"~{k} days back ({d.date()})", f"{n} ticks" if n >= 0 else "error", "first page only"); time.sleep(0.4)
    deepest = max([k for k, n in found.items() if n and n > 0], default=0); log("history", "deepest day with data in this test", f"{deepest} days back", "tells us whether we can backfill or must record forward")
    # 5 other instruments
    others = []
    try:
        import requests
        sm = requests.get("https://images.dhan.co/api-data/api-scrip-master.csv", timeout=90); sm.raise_for_status()
        S = pd.read_csv(io.StringIO(sm.text), low_memory=False); log("instruments", "Dhan scrip master", f"{len(S):,} rows", "columns: " + ",".join(S.columns[:12]))
        cols = {c.upper(): c for c in S.columns}; sym = cols.get("SEM_TRADING_SYMBOL") or cols.get("SM_SYMBOL_NAME"); sid = cols.get("SEM_SMST_SECURITY_ID"); exch = cols.get("SEM_EXM_EXCH_ID"); seg = cols.get("SEM_SEGMENT"); ins = cols.get("SEM_INSTRUMENT_NAME")
        for s_ in ("RELIANCE", "HDFCBANK", "TCS", "INFY"):
            q = S[(S[sym] == s_) & (S[exch] == "NSE") & (S[ins] == "EQUITY")]
            if len(q): others.append((s_, "NSE", "E", "EQUITY", int(q[sid].iloc[0])))
        q = S[(S[ins] == "FUTIDX") & (S[sym].astype(str).str.startswith("NIFTY-")) & ~S[sym].astype(str).str.contains("BANK|FIN|MID|NXT", regex=True)]
        if len(q): r_ = q.iloc[0]; others.append((str(r_[sym]), "NSE", "D", "FUTIDX", int(r_[sid])))
        q = S[(S[ins] == "INDEX") & (S[sym] == "NIFTY")]
        if len(q): others.append(("NIFTY (index)", "NSE", "I", "INDEX", int(q[sid].iloc[0])))
    except Exception as e: log("instruments", "scrip master", "not reachable", f"{type(e).__name__}: {str(e)[:150]}")
    for name, ex, sg, inst, sid_ in others:
        F, pages, note = fetch_day(sid_, days[0], auth, exch=ex, seg=sg, inst=inst); log("instruments", f"{name} ({ex}/{sg}/{inst}/{sid_})", f"{len(F)} ticks", note)
        if len(F): analyse_day(F, name)
    finish(None)

def finish(_):
    print("\n".join(" | ".join(r) for r in REPORT[-3:]))
    pd.DataFrame(REPORT, columns=["stage", "check", "result", "detail"]).to_csv(f"{OUT}/probe_report.csv", index=False)
    if SELFTEST: return
    try:
        import oi_probe as P; sh = P.open_sheet(); P.put_tab(sh, "Dhan Probe", ["stage", "check", "result", "detail  (run " + dt.datetime.now().strftime("%Y-%m-%d %H:%M") + ")"], REPORT); print("sheet tab written")
    except Exception as e: print("sheet write skipped:", type(e).__name__, str(e)[:120])

# ───────────────────────────── self-test with a fake server ─────────────────────────────
def _fake_server(body):
    rng = np.random.default_rng(1); day0 = pd.Timestamp(body["START"], unit="ms", tz="UTC").tz_convert(IST).normalize()
    t0 = int(day0.value // 1e6) + int((9 * 60 + 15) * 60_000); t1 = int(day0.value // 1e6) + int((15 * 60 + 30) * 60_000)
    if day0.weekday() >= 5 or (pd.Timestamp.now(tz=IST).normalize() - day0).days > 30: data = {}
    else:
        ts = np.arange(max(body["START"], t0), t1, 900)[:30000]; n = len(ts); px = 1800 + np.cumsum(rng.normal(0, .15, n)); bid = np.round(px - .05, 1); ask = np.round(px + .05, 1)
        c = np.where(rng.random(n) < .5, ask, bid); data = {"t": ts.tolist(), "c": c.tolist(), "bid": bid.tolist(), "ask": ask.tolist(), "v": rng.integers(1, 5000, n).tolist(), "oi": [0] * n}
        data["Time"] = ["x"] * n
    cut = 12000; nxt = None
    if data and len(data["t"]) > cut: nxt = data["t"][cut]; data = {k: v[:cut] for k, v in data.items()}
    class R:
        status_code = 200; text = "{}"; content = b"x"
        def json(s): return {"success": True, "data": data, "nextTime": nxt}
    return R()

if __name__ == "__main__":
    if SELFTEST:
        TRANSPORT = _fake_server; main()
        rows = {r[1]: r for r in REPORT}; assert any("ticks in" in r[2] and int(r[2].split()[0]) > 20000 for r in REPORT if r[1].endswith("fetch")), "pagination did not stitch pages"
        assert any(r[0] == "history" and r[1].startswith("~5 days") for r in REPORT); assert any(r[1].endswith("coverage") and r[2].startswith("covers") for r in REPORT)
        print("SELFTEST OK")
    else: main()
