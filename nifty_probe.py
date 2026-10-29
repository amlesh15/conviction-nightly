# ==================================================================================
# NIFTY PROBE — can we pull footprint (price-level) data for Nifty index / Nifty futures from Dhan?
#   Uses the same chart back-end and login header as the stock order-flow archive:
#     POST https://ticks.dhan.co/orderflow/getOrderFlow   footprint candles (buy / sell volume per price level)
#     POST https://ticks.dhan.co/getData                  1-minute price + volume candles
#   Candidates tried:  Nifty 50 index (SEG I, id 13) ... Nifty futures (SEG D, ids from Dhan's scrip master: current + next expiry)
#   For each: does it answer, how many candles, do candles carry price levels, how far back, and does it have volume.
#   Result goes to the sheet tab "Nifty Probe" (+ "Nifty Probe Sample") and prints in the log.  Research only.
#   Needs the secret DHAN_HEADERS_JSON (same one the stock archive uses).
#   Usage: python nifty_probe.py [--selftest]
# ==================================================================================
import os, sys, json, time, datetime as dt
import numpy as np, pandas as pd

SELFTEST = "--selftest" in sys.argv
IST = "Asia/Kolkata"
U_OF = "https://ticks.dhan.co/orderflow/getOrderFlow"; U_G = "https://ticks.dhan.co/getData"
MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"
REPORT = []; SAMPLE = {}; TRANSPORT = None

def log(stage, check, result, detail=""):
    REPORT.append([stage, check, str(result), str(detail)[:800]]); print(f"[{stage}] {check}: {result}  {str(detail)[:300]}", flush=True)

def headers():
    h = {"Content-Type": "application/json", "Accept": "application/json, text/plain, */*", "Origin": "https://dext.dhan.co", "Referer": "https://dext.dhan.co/",
         "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    try: h.update(json.loads(os.environ.get("DHAN_HEADERS_JSON") or "{}"))
    except Exception: log("setup", "DHAN_HEADERS_JSON", "unreadable", "check the secret is valid JSON")
    return h

def post(url, body):
    if TRANSPORT: return TRANSPORT(url, body)
    import requests
    return requests.post(url, json=body, headers=headers(), timeout=120)

def iso(t): return pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")

def candidates():
    """[(label, SEG, INST, SEC_ID)] — index first, then futures from the scrip master."""
    c = [("Nifty 50 index", "I", "INDEX", 13)]
    try:
        if TRANSPORT: m = pd.DataFrame()
        else:
            import requests
            r = requests.get(MASTER, timeout=90); r.raise_for_status(); open("/tmp/dhan_master.csv", "wb").write(r.content); m = pd.read_csv("/tmp/dhan_master.csv", low_memory=False)
        if len(m):
            idc = next(x for x in m.columns if "SECURITY_ID" in x and "UNDERLYING" not in x); ts = next(x for x in m.columns if "TRADING_SYMBOL" in x)
            f = m[(m["SEM_EXM_EXCH_ID"] == "NSE") & (m["SEM_INSTRUMENT_NAME"] == "FUTIDX") & (m[ts].astype(str).str.upper().str.startswith("NIFTY-"))].copy()
            f["exp"] = pd.to_datetime(f["SEM_EXPIRY_DATE"], errors="coerce").dt.date; f = f[f["exp"] >= dt.date.today()].sort_values("exp").head(2)
            for _, r in f.iterrows(): c.append((f"Nifty futures {r['exp']} ({r[ts]})", "D", "FUTIDX", int(r[idc])))
            log("master", "Nifty futures found", len(f), "; ".join(f"{r[ts]} id {int(r[idc])}" for _, r in f.iterrows()))
    except Exception as e: log("master", "scrip master", "failed", f"{type(e).__name__}: {str(e)[:150]}")
    return c

def probe_of(label, seg, inst, sid):
    now_day = int(pd.Timestamp.now(tz=IST).normalize().timestamp()) + 86400
    for days in (5, 30, 60):
        end = now_day; start = (end - days * 86400) // 86400 * 86400
        for variant, extra in (("", {}), ("+INST", {"INST": inst})):
            b = {"EXCH": "NSE", "SEG": seg, "SEC_ID": sid, "START": start, "END": end, "START_TIME": iso(start), "END_TIME": iso(end), **extra}
            try: r = post(U_OF, b)
            except Exception as e: log("footprint", f"{label} {days}d{variant}", "error", f"{type(e).__name__}: {str(e)[:120]}"); continue
            txt = r.text or ""
            try: cs = (r.json().get("data") or {}).get("candles") or []
            except Exception: cs = []
            if r.status_code == 401: log("footprint", label, "401", "login token expired: refresh DHAN_HEADERS_JSON"); return None
            if not cs:
                log("footprint", f"{label} {days}d{variant}", f"HTTP {r.status_code}, no candles", txt[:200].replace("\n", " ")); continue
            nl = [len(c.get("levels") or []) for c in cs]; t = pd.to_datetime([c["t"] for c in cs], unit="s", utc=True).tz_convert(IST).tz_localize(None)
            has_vol = sum(c.get("total_volume", 0) or 0 for c in cs)
            log("footprint", f"{label} {days}d{variant}", f"OK {len(cs)} candles", f"{t.min()} -> {t.max()} | price levels per candle: median {int(np.median(nl))}, max {max(nl)} | total volume {has_vol:,.0f} | keys {sorted(cs[0].keys())[:14]}")
            SAMPLE[label] = pd.DataFrame([{"t": str(x), "high": c.get("high"), "low": c.get("low"), "buy": c.get("buy_volume"), "sell": c.get("sell_volume"), "neutral": c.get("neutral_volume"),
                                           "total": c.get("total_volume"), "delta": c.get("delta"), "levels": len(c.get("levels") or [])} for x, c in zip(t[-60:], cs[-60:])])
            return cs
        time.sleep(0.3)
    return None

def probe_price(label, seg, inst, sid):
    end = int(time.time()) + 86400; start = end - 10 * 86400
    try: r = post(U_G, {"EXCH": "NSE", "SEG": seg, "INST": inst, "SEC_ID": sid, "INTERVAL": "1", "START": start, "END": end})
    except Exception as e: log("price", label, "error", str(e)[:120]); return
    try: d = (r.json().get("data") or {})
    except Exception: d = {}
    n = len(d.get("t") or []); v = sum(d.get("v") or [])
    log("price", f"{label} 1-min, 10d", f"HTTP {r.status_code}, {n} candles", f"volume total {v:,.0f}" + (" (zero volume: index has no traded volume)" if n and not v else ""))

def main():
    for label, seg, inst, sid in candidates():
        cs = probe_of(label, seg, inst, sid); probe_price(label, seg, inst, sid)
        log("verdict", label, "FOOTPRINT AVAILABLE" if cs else "no footprint", "")
        time.sleep(0.5)
    finish()

def finish():
    os.makedirs("out/dhan", exist_ok=True); pd.DataFrame(REPORT, columns=["stage", "check", "result", "detail"]).to_csv("out/dhan/nifty_probe.csv", index=False)
    if SELFTEST: return
    try:
        import live_refresh_local as LR, gspread
        sh = LR.open_sheet()
        def put(title, header, rows):
            try: ws = sh.worksheet(title); ws.clear()
            except gspread.WorksheetNotFound: ws = sh.add_worksheet(title=title, rows=500, cols=max(8, len(header)))
            ws.update(values=[header] + rows, range_name="A1", value_input_option="USER_ENTERED")
        put("Nifty Probe", ["stage", "check", "result", "detail (run " + dt.datetime.now().strftime("%Y-%m-%d %H:%M") + ")"], REPORT)
        rows = []
        for lab, G in SAMPLE.items(): rows += [[lab] + [("" if pd.isna(x) else x) for x in r] for r in G.values.tolist()]
        if rows: put("Nifty Probe Sample", ["instrument"] + list(next(iter(SAMPLE.values())).columns), rows)
        print("sheet tabs written")
    except Exception as e: print("sheet write skipped:", type(e).__name__, str(e)[:150])

# ───────────────────────────── self-test (fake server) ─────────────────────────────
class _R:
    def __init__(s, js, code=200): s._j = js; s.status_code = code; s.text = json.dumps(js)[:400]
    def json(s): return s._j

def _fake(url, b):
    if url == U_OF:
        if b["SEG"] == "I": return _R({"data": {"candles": []}})                       # index: nothing
        t0 = b["END"] - 86400 * 2 + 33300
        cs = [{"t": t0 + 60 * i, "high": 25000.0 + i, "low": 24990.0 + i, "buy_volume": 900, "sell_volume": 800, "neutral_volume": 5, "total_volume": 1705, "delta": 100,
               "levels": [{"price": 24990.0 + k, "buy_volume": 100, "sell_volume": 90, "total_volume": 190} for k in range(10)]} for i in range(50)]
        return _R({"data": {"candles": cs}})
    return _R({"data": {"t": [1, 2, 3], "v": [10, 20, 30]}})

if __name__ == "__main__":
    if SELFTEST:
        TRANSPORT = _fake; os.environ["DHAN_HEADERS_JSON"] = "{}"
        for lab, seg, inst, sid in [("Nifty 50 index", "I", "INDEX", 13), ("Nifty futures X", "D", "FUTIDX", 1)]:
            cs = probe_of(lab, seg, inst, sid); probe_price(lab, seg, inst, sid)
            assert (cs is None) == (seg == "I")
        assert any(r[0] == "footprint" and r[2].startswith("OK 50") for r in REPORT); finish(); print("SELFTEST OK")
    else: main()
