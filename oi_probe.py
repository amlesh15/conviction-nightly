# ==================================================================================
# OPEN INTEREST ENGINE — step 0: DATA ACCESS PROBE
#
# Answers, with evidence, before we build anything:
#   1. Do the 3 Sensibull endpoints answer with our cookie?  What do they look like?
#   2. Can we list every option contract (underlying / expiry / strike / CE-PE) from the metacache?
#   3. Do option candles carry OHLCV + OI, to the minute?  How often does OI actually change?
#   4. Does it work for the WHOLE FnO universe (ATM CE + PE for each stock)?
#   5. Full option chain for FEDERALBNK (every strike, every minute) -> our first case study.
#   6. How far back does 1-min history go, and are EXPIRED contracts still served?
#
# Output: Google Sheet tabs  "OI Probe" (verdict table), "OI Probe Schema" (shape of the metadata JSON),
#         "OI Probe Universe" (one row per stock), "OI Probe FED Chain" (FEDERALBNK, all strikes)
#         + raw files in out/oi/ (uploaded as a workflow artifact).
# Usage:  python oi_probe.py             (cookie via setup_env.py; sheet via GCP_SA_JSON / SHEET_ID)
#         python oi_probe.py --selftest  (simulated Sensibull, checks the logic offline)
# ==================================================================================
import os, re, sys, json, time, random, datetime as dt
from collections import Counter, defaultdict
import concurrent.futures as cf
import numpy as np, pandas as pd

COOKIE_PATH = os.path.expanduser("~/live-refresh/cookie.txt")
CANDLE = "https://oxide.sensibull.com/v1/compute/1/candles/{sym}"
META_CACHE = "https://oxide.sensibull.com/v1/compute/cache/instrument_metacache_addon/1"
META = "https://oxide.sensibull.com/v1/compute/1/instrument_metadata/"
CASE = os.environ.get("OI_CASE", "FEDERALBNK")       # the case-study stock
WORKERS = int(os.environ.get("OI_WORKERS", "6"))
OUT = "out/oi"; os.makedirs(OUT, exist_ok=True)
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
OPT_RE = re.compile(r"^([A-Z0-9&\-]+?)(\d{2})(" + "|".join(MONTHS) + r")(\d+(?:\.\d+)?)(CE|PE)$")

BUDGET_MIN = float(os.environ.get("OI_BUDGET_MIN", "25"))      # hard stop: after this the probe stops calling and writes what it has
DEADLINE = time.time() + BUDGET_MIN * 60
def out_of_time(): return time.time() > DEADLINE

STATS = {"calls": 0, "lat": [], "codes": Counter(), "errors": Counter()}
REPORT = []          # [stage, check, result, detail]

def add(stage, check, ok, detail=""):
    res = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    REPORT.append([stage, check, res, str(detail)[:900]]); print(f"[{res}] {stage} | {check} | {str(detail)[:300]}")

# ───────────────────────────── transport ─────────────────────────────
def _hdr():
    ck = open(COOKIE_PATH).read().strip() if os.path.exists(COOKIE_PATH) else ""
    return {"cookie": ck, "content-type": "application/json", "origin": "https://web.sensibull.com",
            "referer": "https://web.sensibull.com/", "user-agent": "Mozilla/5.0"}

def _real_transport(method, url, body=None):
    import requests
    return requests.request(method, url, json=body, headers=_hdr(), timeout=(8, 20))

TRANSPORT = _real_transport

def http(method, url, body=None, tries=3):
    """Returns (response|None, error|None). Records latency/status. Retries 429/5xx/network."""
    last = None
    for k in range(tries):
        if out_of_time(): return None, "probe time budget used up"
        t0 = time.time()
        try:
            r = TRANSPORT(method, url, body); dtm = time.time() - t0
            STATS["calls"] += 1; STATS["lat"].append(dtm); STATS["codes"][r.status_code] += 1
            if r.status_code in (429, 502, 503, 504): last = f"HTTP {r.status_code}"; time.sleep(1.5 * (k + 1)); continue
            return r, None
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"; STATS["errors"][type(e).__name__] += 1; time.sleep(1 + k)
    return None, last

def candles(sym, days=3, interval="1M"):
    """-> (DataFrame|None, info). Mirrors the request our other jobs use."""
    end = dt.date.today()
    body = {"from_date": (end - dt.timedelta(days=days)).isoformat(), "to_date": (end + dt.timedelta(days=1)).isoformat(),
            "interval": interval, "skip_last_ts": False}
    r, err = http("POST", CANDLE.format(sym=sym), body)
    if r is None: return None, {"status": None, "err": err}
    info = {"status": r.status_code, "err": None}
    if r.status_code != 200: info["err"] = r.text[:160]; return None, info
    try: js = r.json()
    except Exception: info["err"] = "not json: " + r.text[:100]; return None, info
    c = ((js or {}).get("payload") or {}).get("candles") or []
    if not c: info["err"] = "no candles (keys=%s)" % (list(js.keys())[:6] if isinstance(js, dict) else type(js).__name__); return None, info
    df = pd.DataFrame(c)
    if "ts" not in df.columns: info["err"] = "candle columns: %s" % list(df.columns)[:10]; return None, info
    ts = pd.to_datetime(df["ts"])
    if getattr(ts.dt, "tz", None) is not None: ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["ts"] = ts; df["session"] = ts.dt.date
    for col in ("open", "high", "low", "close", "volume", "oi"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True), info

def profile(df):
    """Facts about one contract's candles for the latest session."""
    if df is None or df.empty: return {}
    s = df["session"].max(); d = df[df["session"] == s]
    out = {"sessions": int(df["session"].nunique()), "latest": str(s), "rows": len(d), "first": d["ts"].iloc[0].strftime("%H:%M"),
           "last": d["ts"].iloc[-1].strftime("%H:%M"), "close": float(d["close"].iloc[-1]) if "close" in d else None,
           "volume": float(d["volume"].fillna(0).sum()) if "volume" in d else None, "has_oi": "oi" in d.columns}
    if "oi" in d.columns:
        o = d["oi"]; out["oi_nonnull%"] = round(o.notna().mean() * 100, 1)
        oo = o.dropna()
        if len(oo):
            out["oi_first"], out["oi_last"] = float(oo.iloc[0]), float(oo.iloc[-1]); out["oi_chg"] = out["oi_last"] - out["oi_first"]
            ch = (oo.diff().fillna(0) != 0); out["oi_updates"] = int(ch.sum()); out["oi_update%"] = round(ch.mean() * 100, 1)
    return out

# ───────────────────────────── schema + symbol harvest ─────────────────────────────
def outline(obj, max_lines=160):
    """Compact description of an unknown JSON so we can write a parser: path, type, size, example."""
    lines = []
    def ex(v):
        s = json.dumps(v, default=str)
        return s if len(s) <= 90 else s[:87] + "..."
    def walk(o, path, depth):
        if len(lines) >= max_lines: return
        if isinstance(o, dict):
            lines.append(f"{path or '$'}  dict[{len(o)}]  keys: {', '.join(list(map(str, o.keys()))[:12])}{' ...' if len(o) > 12 else ''}")
            if depth >= 5: return
            keys = list(o.keys())
            for k in (keys if len(keys) <= 14 else keys[:4]): walk(o[k], f"{path}.{k}" if path else str(k), depth + 1)
        elif isinstance(o, list):
            lines.append(f"{path}  list[{len(o)}]  first: {ex(o[0]) if o else '-'}")
            if o and depth < 5 and isinstance(o[0], (dict, list)): walk(o[0], path + "[0]", depth + 1)
        else: lines.append(f"{path}  {type(o).__name__}  e.g. {ex(o)}")
    walk(obj, "", 0); return lines

def harvest(obj):
    """Find every string that looks like an option symbol; keep the dict that contained it (carries ids etc.)."""
    recs = {}; stack = [(obj, None)]
    while stack:
        o, parent = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and OPT_RE.match(k): recs.setdefault(k, v if isinstance(v, dict) else {"value": v})
                stack.append((v, o))
        elif isinstance(o, list):
            for v in o: stack.append((v, parent))
        elif isinstance(o, str) and OPT_RE.match(o):
            recs.setdefault(o, parent if isinstance(parent, dict) else {})
    return recs

def parse_sym(s):
    m = OPT_RE.match(s)
    if not m: return None
    u, yy, mon, k, t = m.groups(); return u, int(yy), MONTHS.index(mon), float(k), t

def build_chains(recs, universe=None):
    ch = defaultdict(lambda: defaultdict(lambda: {"CE": {}, "PE": {}}))
    for s in recs:
        p = parse_sym(s)
        if not p: continue
        u, yy, mi, k, t = p
        if universe and u not in universe: continue
        ch[u][(yy, mi)][t][k] = s
    return ch

def nearest_expiry(exps, today=None):
    today = today or dt.date.today(); key = (today.year % 100, today.month - 1)
    ok = sorted(e for e in exps if e >= key)
    return ok[0] if ok else (sorted(exps)[-1] if exps else None)

def exp_label(e): return f"{e[0]:02d}{MONTHS[e[1]]}"

# ───────────────────────────── stages ─────────────────────────────
def stage_auth():
    df, info = candles("RELIANCE", days=3)
    add("1 auth", "cash 1-min candles (RELIANCE)", df is not None, f"status={info['status']} {info['err'] or ''} rows={0 if df is None else len(df)}")
    return df is not None

def stage_metacache():
    r, err = http("GET", META_CACHE)
    if r is None: add("2 metacache", "GET instrument_metacache_addon", False, err); return None, {}
    ct = r.headers.get("content-type", ""); size = len(r.content)
    if r.status_code != 200: add("2 metacache", "GET instrument_metacache_addon", False, f"HTTP {r.status_code} {r.text[:200]}"); return None, {}
    try: js = r.json()
    except Exception: add("2 metacache", "response is JSON", False, f"{ct} {r.text[:200]}"); return None, {}
    with open(f"{OUT}/metacache.json", "w") as f: json.dump(js, f)
    add("2 metacache", "GET instrument_metacache_addon", True, f"HTTP 200, {size/1e6:.1f} MB, {ct}")
    recs = harvest(js)
    und = {parse_sym(s)[0] for s in recs}
    exps = sorted({(parse_sym(s)[1], parse_sym(s)[2]) for s in recs})
    add("2 metacache", "option contracts found", len(recs) > 0, f"{len(recs):,} option symbols | {len(und)} underlyings | expiries: {', '.join(exp_label(e) for e in exps[:8])}")
    case = [s for s in recs if parse_sym(s)[0] == CASE]
    add("2 metacache", f"{CASE} contracts", len(case) > 0, f"{len(case)} contracts; sample record: {json.dumps(recs.get(case[0]) if case else None, default=str)[:300]}")
    return js, recs

def stage_metadata(recs):
    r, err = http("GET", META)
    if r is None: add("3 metadata", "GET instrument_metadata/", False, err)
    else:
        txt = r.text[:260].replace("\n", " ")
        add("3 metadata", "GET instrument_metadata/", r.status_code == 200, f"HTTP {r.status_code} {len(r.content)} bytes: {txt}")
        if r.status_code == 200:
            try: js = r.json(); open(f"{OUT}/instrument_metadata.json", "w").write(json.dumps(js)[:5_000_000]); add("3 metadata", "shape", None, "; ".join(outline(js, 8)))
            except Exception: pass
    # try with an id / symbol suffix, taken from a real record
    probe_sym = f"{CASE}26OCT340CE"
    rec = recs.get(probe_sym) or next((v for k, v in recs.items() if parse_sym(k)[0] == CASE), {}) if recs else {}
    ids = [str(v) for k, v in (rec or {}).items() if re.search(r"id|token|key", str(k), re.I) and isinstance(v, (str, int))][:2]
    for suf in ids + [probe_sym]:
        r2, e2 = http("GET", META + suf)
        add("3 metadata", f"GET instrument_metadata/{suf[:30]}", None if r2 is None else r2.status_code == 200,
            e2 or f"HTTP {r2.status_code}: {r2.text[:200]}".replace("\n", " "))

def stage_case_contract():
    sym = f"{CASE}26OCT340CE"
    df, info = candles(sym, days=5)
    if df is None: add("4 contract", f"candles {sym}", False, f"status={info['status']} {info['err']}"); return None
    p = profile(df)
    add("4 contract", f"candles {sym}", True, f"columns={list(df.columns)}")
    add("4 contract", "latest session profile", None, json.dumps(p))
    add("4 contract", "has OI column", bool(p.get("has_oi")) and p.get("oi_nonnull%", 0) > 0, f"oi non-null {p.get('oi_nonnull%')}%")
    if p.get("rows"): add("4 contract", "OI changes minute-to-minute?", None,
        f"OI changed on {p.get('oi_updates')} of {p['rows']} minutes ({p.get('oi_update%')}%) — "
        f"{'near-continuous' if (p.get('oi_update%') or 0) > 60 else 'updates in steps, so minute-level OI deltas are coarse'}")
    df.tail(400).to_csv(f"{OUT}/{sym}_sample.csv", index=False)
    return df

def fetch_contract(sym, days=2):
    df, info = candles(sym, days=days); return sym, df, info

def stage_case_chain(chains):
    exps = chains.get(CASE, {})
    e = nearest_expiry(list(exps.keys()))
    if e is None: add("5 chain", f"{CASE} option chain", False, "no contracts for case stock in metacache"); return
    syms = [s for t in ("CE", "PE") for s in exps[e][t].values()]
    t0 = time.time(); rows = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for sym, df, info in ex.map(fetch_contract, syms):
            p = profile(df); u, yy, mi, k, t = parse_sym(sym)
            rows.append({"strike": k, "type": t, "symbol": sym, "ok": df is not None, "err": info.get("err") or "", **{x: p.get(x) for x in
                         ("rows", "first", "last", "close", "volume", "oi_first", "oi_last", "oi_chg", "oi_updates", "oi_update%")}})
    T = pd.DataFrame(rows).sort_values(["strike", "type"]); T.to_csv(f"{OUT}/{CASE}_chain.csv", index=False)
    okc = int(T["ok"].sum())
    add("5 chain", f"{CASE} {exp_label(e)} full chain", okc > 0.6 * len(T), f"{okc}/{len(T)} contracts returned candles in {time.time()-t0:.0f}s ({len(exps[e]['CE'])} CE + {len(exps[e]['PE'])} PE strikes)")
    live = T[T["ok"] & (T["rows"].fillna(0) > 0)]
    if len(live): add("5 chain", "contracts with trades today", None, f"{int((live['volume'].fillna(0) > 0).sum())} traded | total ΔOI CE {live[live.type=='CE']['oi_chg'].sum():+,.0f} PE {live[live.type=='PE']['oi_chg'].sum():+,.0f}")
    return T

def stage_universe(chains, universe):
    t0 = time.time(); missing_meta = [u for u in universe if u not in chains]
    add("6 universe", "universe stocks present in metacache", len(missing_meta) == 0, f"{len(universe)-len(missing_meta)}/{len(universe)} found; missing: {', '.join(missing_meta[:25])}")
    todo = [u for u in universe if u in chains]
    def one(u):
        if out_of_time(): return {"symbol": u, "status": "skipped (time budget)"}
        res = {"symbol": u, "expiry": "", "strikes": "", "spot": "", "atm": "", "ce_rows": "", "pe_rows": "", "ce_oi%": "", "pe_oi%": "",
               "ce_last": "", "pe_last": "", "ce_oi_upd%": "", "pe_oi_upd%": "", "status": "", "error": ""}
        e = nearest_expiry(list(chains[u].keys())); ch = chains[u][e]; res["expiry"] = exp_label(e); res["strikes"] = len(ch["CE"])
        sp, info = candles(u, days=3)
        if sp is None: res["status"] = "no spot"; res["error"] = f"{info['status']} {info['err']}"; return res
        spot = float(sp["close"].iloc[-1]); res["spot"] = round(spot, 2)
        ks = sorted(set(ch["CE"]) & set(ch["PE"]) or set(ch["CE"]))
        if not ks: res["status"] = "no strikes"; return res
        atm = min(ks, key=lambda k: abs(k - spot)); res["atm"] = atm
        for t, pre in (("CE", "ce"), ("PE", "pe")):
            s = ch[t].get(atm)
            if not s: res[f"{pre}_rows"] = "no contract"; continue
            df, inf = candles(s, days=2); p = profile(df)
            if df is None: res[f"{pre}_rows"] = f"ERR {inf['status']}"; res["error"] += f" {t}:{inf['err']}"; continue
            res[f"{pre}_rows"] = p.get("rows"); res[f"{pre}_oi%"] = p.get("oi_nonnull%"); res[f"{pre}_last"] = p.get("last"); res[f"{pre}_oi_upd%"] = p.get("oi_update%")
        ok = all(isinstance(res[x], (int, float)) and res[x] > 0 for x in ("ce_rows", "pe_rows"))
        res["status"] = "OK" if ok else "partial"; return res
    rows = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            rows.append(r)
            if i % 10 == 0: print(f"  universe {i}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
    U = pd.DataFrame(rows); U.to_csv(f"{OUT}/universe.csv", index=False)
    n_ok = int((U["status"] == "OK").sum()) if len(U) else 0
    add("6 universe", "ATM CE+PE candles for every stock", n_ok >= 0.9 * max(1, len(todo)), f"{n_ok}/{len(todo)} OK in {time.time()-t0:.0f}s")
    if len(U):
        oi_ok = pd.to_numeric(U["ce_oi%"], errors="coerce").dropna()
        add("6 universe", "OI present on ATM contracts", (oi_ok > 50).mean() > 0.9 if len(oi_ok) else False, f"median OI coverage {oi_ok.median() if len(oi_ok) else 'n/a'}%")
        upd = pd.to_numeric(U["ce_oi_upd%"], errors="coerce").dropna()
        if len(upd): add("6 universe", "how often OI changes (ATM CE, % of minutes)", None, f"median {upd.median():.0f}% | p25 {upd.quantile(.25):.0f}% | p75 {upd.quantile(.75):.0f}%")
        bad = U[U["status"] != "OK"]["symbol"].tolist()
        if bad: add("6 universe", "stocks not OK", None, ", ".join(bad[:40]))
    return U

def stage_history(chains):
    exps = chains.get(CASE, {}); e = nearest_expiry(list(exps.keys()))
    if e is None: return
    ks = sorted(exps[e]["CE"]); sp, _ = candles(CASE, days=3)
    spot = float(sp["close"].iloc[-1]) if sp is not None else ks[len(ks) // 2]
    atm = min(ks, key=lambda k: abs(k - spot)); sym = exps[e]["CE"][atm]
    for d in (5, 15, 30, 60, 90):
        df, info = candles(sym, days=d)
        add("7 history", f"{sym} back {d} days", None if df is None else True,
            f"{'ERR ' + str(info['err']) if df is None else str(df['session'].nunique()) + ' sessions, ' + str(len(df)) + ' bars, from ' + str(df['session'].min())}")
    # expired contracts: same strikes, previous month(s)
    for back in (1, 2):
        mi = e[1] - back; yy = e[0]
        while mi < 0: mi += 12; yy -= 1
        pe = (yy, mi); near = sorted(ks, key=lambda k: abs(k - spot))[:8]
        hit = 0; best = ""
        for k in near:
            s = f"{CASE}{yy:02d}{MONTHS[mi]}{k:g}CE"; df, info = candles(s, days=90)
            if df is not None: hit += 1; best = best or f"{s}: {df['session'].nunique()} sessions from {df['session'].min()}"
        add("7 history", f"EXPIRED {exp_label(pe)} contracts served?", hit > 0, f"{hit}/{len(near)} strikes returned candles. {best}")

def stage_latency():
    lat = STATS["lat"]
    if lat: add("8 speed", "requests", None, f"{STATS['calls']} calls | avg {np.mean(lat):.2f}s | p95 {np.percentile(lat,95):.2f}s | status codes {dict(STATS['codes'])} | errors {dict(STATS['errors'])}")

# ───────────────────────────── sheet output ─────────────────────────────
def open_sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    info = json.loads(os.environ["GCP_SA_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return gspread.authorize(creds).open_by_key(os.environ["SHEET_ID"])

def put_tab(sh, title, header, rows):
    import gspread
    vals = [header] + [[("" if (isinstance(v, float) and v != v) or v is None else (v if isinstance(v, (int, float)) else str(v)[:900])) for v in r] for r in rows]
    try: ws = sh.worksheet(title); ws.clear(); ws.resize(rows=max(50, len(vals) + 5), cols=max(8, len(header) + 1))
    except gspread.WorksheetNotFound: ws = sh.add_worksheet(title=title, rows=max(50, len(vals) + 5), cols=max(8, len(header) + 1))
    ws.update(values=vals, range_name="A1", value_input_option="RAW")

def publish(schema_lines, T, U):
    sh = open_sheet(); now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    put_tab(sh, "OI Probe", ["stage", "check", "result", "detail  (run " + now + " server time)"], REPORT)
    if schema_lines: put_tab(sh, "OI Probe Schema", ["shape of metadata JSON (path  type  example)"], [[l] for l in schema_lines])
    if U is not None and len(U): put_tab(sh, "OI Probe Universe", list(U.columns), U.values.tolist())
    if T is not None and len(T): put_tab(sh, "OI Probe FED Chain", list(T.columns), T.values.tolist())
    print("sheet tabs written")

def universe_from_sheet():
    sh = open_sheet(); v = sh.worksheet("Rule Dashboard").get_all_values(); h = [x.strip() for x in v[0]]; i = h.index("symbol") if "symbol" in h else 0
    return sorted({r[i].strip().upper() for r in v[1:] if len(r) > i and r[i].strip()})

def run(universe):
    if not stage_auth():
        add("1 auth", "STOP", False, "cookie rejected or Sensibull unreachable — refresh SENSIBULL_COOKIE secret"); return None, None, None, []
    js, recs = stage_metacache()
    schema = outline(js) if js is not None else []
    stage_metadata(recs)
    stage_case_contract()
    chains = build_chains(recs, set(universe) | {CASE}) if recs else {}
    T = stage_case_chain(chains) if chains else None
    U = stage_universe(chains, universe) if chains else None
    if chains: stage_history(chains)
    stage_latency()
    return T, U, schema, REPORT

def main():
    universe = universe_from_sheet()
    print(f"{len(universe)} stocks in the conviction universe; case study: {CASE}; time budget {BUDGET_MIN:g} min", flush=True)
    T = U = None; schema = []
    try: T, U, schema, _ = run(universe)
    except Exception as e: add("run", "unexpected error", False, f"{type(e).__name__}: {e}")
    if out_of_time(): add("run", "time budget", False, f"stopped after {BUDGET_MIN:g} min — results above are partial")
    try: publish(schema, T, U)
    except Exception as e: print("sheet write failed:", e)
    json.dump(REPORT, open(f"{OUT}/report.json", "w"), indent=1)
    print("\n=== SUMMARY ==="); [print(f"{r[2]:5s} {r[0]} | {r[1]}") for r in REPORT]

# ───────────────────────────── offline self-test (simulated Sensibull) ─────────────────────────────
class _R:
    def __init__(self, code, js=None, text=""): self.status_code = code; self._js = js; self.text = text or json.dumps(js)[:300]; self.content = self.text.encode(); self.headers = {"content-type": "application/json"}
    def json(self): return self._js

def _fake_universe():
    return {"FEDERALBNK": 200.0, "RELIANCE": 1300.0, "ABB": 5500.0, "BAJAJ-AUTO": 9000.0}

def _fake_transport(method, url, body=None):
    U = _fake_universe()
    if url.endswith("instrument_metacache_addon/1"):
        data = {}
        for u, sp in U.items():
            step = 5 if sp < 1000 else 50
            for e in ("26OCT", "26NOV"):
                for k in range(int(sp * .8 // step * step), int(sp * 1.2), step):
                    for t in ("CE", "PE"): data[f"{u}{e}{k}{t}"] = {"instrument_token": random.randint(1, 10**7), "strike": k, "type": t}
        return _R(200, {"data": {"instruments": data}, "version": 3})
    if url.endswith("instrument_metadata/"): return _R(405, text="Method not allowed")
    if "instrument_metadata/" in url: return _R(200, {"payload": {"ok": 1}})
    sym = url.rsplit("/", 1)[1]
    if "26SEP" in sym: return _R(200, {"payload": {"candles": []}})
    rnd = random.Random(hash(sym) % 1000); m = OPT_RE.match(sym); rows = []; px = rnd.uniform(5, 40) if m else U.get(sym, 100.0); oi = rnd.randint(1000, 90000)
    days = 3 if body and body["from_date"] else 1
    base = dt.datetime.combine(dt.date.today(), dt.time(9, 15))
    for d in range(2):
        for i in range(375):
            px *= 1 + rnd.gauss(0, .002); oi += rnd.choice([0, 0, 0, 50, -50, 100]) if m else 0
            c = {"ts": (base - dt.timedelta(days=d) + dt.timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S+05:30"), "open": px, "high": px * 1.001, "low": px * .999, "close": px, "volume": rnd.randint(0, 500)}
            if m: c["oi"] = oi
            rows.append(c)
    return _R(200, {"payload": {"candles": rows}})

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        TRANSPORT = _fake_transport; WORKERS = 3
        T, U, schema, rep = run(sorted(_fake_universe()))
        assert T is not None and len(T) > 10 and U is not None and len(U) == 4, "stages did not produce output"
        assert (U["status"] == "OK").all(), U.to_string()
        print("\n".join(schema[:12])); print("SELFTEST OK:", len(rep), "report lines")
    else:
        main()
