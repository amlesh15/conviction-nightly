# ==================================================================================
# STOCK FUTURES REPOSITORY — every stock-futures contract NSE lists on every trading day (near, next AND far month), from NSE's own daily F&O file.
#   Source: NSE archives "F&O bhavcopy" (the same end-of-day file behind NSE's security-wise F&O report), both formats:
#     new  (UDiFF)  BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip     legacy  fo<DD><MON><YYYY>bhav.csv.zip   (older dates; spot price taken from the cash file)
#   Universe = whatever NSE lists as a stock future that day, so new F&O entrants (and exits) are captured automatically; index futures are excluded.
#   Stored RAW (not corporate-action adjusted) as ONE SMALL PARQUET FILE PER DAY:  data/futures/by_day/YYYY/YYYY-MM-DD.parquet   + data/futures/manifest.json
#   Columns: date, symbol, expiry, exp_rank (1=nearest listed expiry that day, 2=next, 3=far), dte (calendar days to expiry), open, high, low, close, settle,
#            prev_close, last, underlying (spot), contracts, turnover_cr, oi (units), oi_chg (units), lot, trades, src
#   Resumable: days already in the manifest are skipped; weekdays NSE has no file for (holidays) are remembered as "none".  Missing files are NEVER guessed.
#   Usage:  python futures_repo.py --selftest
#           python futures_repo.py            env: FUT_START (default 2019-01-01), FUT_END (default today IST), FUT_MAX_MIN (stop after N minutes, default 330), FUT_SHEET=1 (write "Futures Roll" tab)
# ==================================================================================
import os, io, sys, json, time, zipfile, datetime as dt
import numpy as np, pandas as pd
import functools; print = functools.partial(print, flush=True)     # show every line in the Actions log as it happens

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data", "futures"); DAYS = os.path.join(ROOT, "by_day"); MANIFEST = os.path.join(ROOT, "manifest.json")
UDIFF_FROM = "2024-07-08"          # NSE's new-format file starts here (the other format is tried as a fallback in both directions)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36", "Accept": "*/*"}
COLS = ["date", "symbol", "expiry", "exp_rank", "dte", "open", "high", "low", "close", "settle", "prev_close", "last", "underlying", "contracts", "turnover_cr", "oi", "oi_chg", "lot", "trades", "src"]

# ───────────────────────────── parsing ─────────────────────────────
def _num(s): return pd.to_numeric(s, errors="coerce")

def parse_udiff(raw, D):
    f = raw[raw["FinInstrmTp"].astype(str).str.strip() == "STF"].copy()
    if f.empty: return pd.DataFrame(columns=COLS)
    lot = _num(f["NewBrdLotQty"]) if "NewBrdLotQty" in f else pd.Series(np.nan, index=f.index)
    vol = _num(f["TtlTradgVol"]) if "TtlTradgVol" in f else pd.Series(np.nan, index=f.index)
    g = lambda c: _num(f[c]) if c in f else pd.Series(np.nan, index=f.index)
    out = pd.DataFrame({"date": D, "symbol": f["TckrSymb"].astype(str).str.strip(), "expiry": pd.to_datetime(f["XpryDt"]).dt.strftime("%Y-%m-%d"),
        "open": g("OpnPric"), "high": g("HghPric"), "low": g("LwPric"), "close": g("ClsPric"), "settle": g("SttlmPric"), "prev_close": g("PrvsClsgPric"), "last": g("LastPric"),
        "underlying": g("UndrlygPric"), "contracts": (vol / lot).round(), "turnover_cr": g("TtlTrfVal") / 1e7, "oi": g("OpnIntrst"), "oi_chg": g("ChngInOpnIntrst"),
        "lot": lot, "trades": g("TtlNbOfTxsExctd"), "src": "udiff"})
    return out

def parse_legacy(raw, D, spot=None):
    raw = raw.copy(); raw.columns = [c.strip() for c in raw.columns]
    f = raw[raw["INSTRUMENT"].astype(str).str.strip() == "FUTSTK"].copy()
    if f.empty: return pd.DataFrame(columns=COLS)
    sym = f["SYMBOL"].astype(str).str.strip()
    out = pd.DataFrame({"date": D, "symbol": sym, "expiry": pd.to_datetime(f["EXPIRY_DT"].astype(str).str.strip(), format="%d-%b-%Y").dt.strftime("%Y-%m-%d"),
        "open": _num(f["OPEN"]), "high": _num(f["HIGH"]), "low": _num(f["LOW"]), "close": _num(f["CLOSE"]), "settle": _num(f["SETTLE_PR"]), "prev_close": np.nan, "last": np.nan,
        "underlying": sym.map(spot) if spot is not None else np.nan, "contracts": _num(f["CONTRACTS"]), "turnover_cr": _num(f["VAL_INLAKH"]) / 100.0,
        "oi": _num(f["OPEN_INT"]), "oi_chg": _num(f["CHG_IN_OI"]), "lot": np.nan, "trades": np.nan, "src": "legacy"})
    return out

def finish(df, D):
    """dedupe, expiry rank among the expiries listed THAT day for that stock, days to expiry; stable column order."""
    if df is None or df.empty: return pd.DataFrame(columns=COLS)
    df = df.drop_duplicates(["symbol", "expiry"], keep="last").copy()
    df["exp_rank"] = df.groupby("symbol")["expiry"].rank(method="dense").astype(int)
    df["dte"] = (pd.to_datetime(df["expiry"]) - pd.Timestamp(D)).dt.days
    return df[COLS].sort_values(["symbol", "expiry"]).reset_index(drop=True)

# ───────────────────────────── download ─────────────────────────────
class Blocked(Exception): pass

def _get(sess, url, tries=3):
    """-> bytes | None (404 = no such file) ; raises Blocked on persistent non-404 failures."""
    last = None
    for a in range(tries):
        try:
            r = sess.get(url, headers=UA, timeout=45)
            if r.status_code == 200 and len(r.content) > 300: return r.content
            if r.status_code == 404: return None
            last = f"HTTP {r.status_code}"
        except Exception as e: last = repr(e)[:80]
        time.sleep(1.5 * (a + 1))
    raise Blocked(f"{url} -> {last}")

def _zip_csv(b):
    z = zipfile.ZipFile(io.BytesIO(b)); return pd.read_csv(z.open(z.namelist()[0]), low_memory=False)

def url_udiff(D): return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{D.replace('-', '')}_F_0000.csv.zip"
def url_legacy(D):
    t = pd.Timestamp(D); return f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{t.year}/{t.strftime('%b').upper()}/fo{t.strftime('%d%b%Y').upper()}bhav.csv.zip"
def url_cash(D):
    t = pd.Timestamp(D); return f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{t.strftime('%d%m%Y')}.csv"

def url_cash_old(D):
    t = pd.Timestamp(D); return f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{t.year}/{t.strftime('%b').upper()}/cm{t.strftime('%d%b%Y').upper()}bhav.csv.zip"

def parse_cash(c, close_col):
    c = c.copy(); c.columns = [x.strip() for x in c.columns]; c = c[c["SERIES"].astype(str).str.strip() == "EQ"]
    return pd.Series(_num(c[close_col]).values, index=c["SYMBOL"].astype(str).str.strip().values).groupby(level=0).last()

def spot_for(sess, D):
    """spot (cash close) per symbol: NSE's full cash file, else (before it existed, 2019) the older cm bhavcopy zip."""
    try:
        b = _get(sess, url_cash(D), tries=2)
        if b is not None: return parse_cash(pd.read_csv(io.StringIO(b.decode("utf-8", "ignore"))), "CLOSE_PRICE")
    except Exception: pass
    try:
        b = _get(sess, url_cash_old(D), tries=2)
        if b is not None: return parse_cash(_zip_csv(b), "CLOSE")
    except Exception: pass
    return None

def fetch_day(sess, D):
    """-> (DataFrame | None). None = NSE has no file that day. Raises Blocked if NSE refuses."""
    order = ("udiff", "legacy") if D >= UDIFF_FROM else ("legacy", "udiff")
    for kind in order:
        b = _get(sess, url_udiff(D) if kind == "udiff" else url_legacy(D))
        if b is None: continue
        raw = _zip_csv(b)
        df = parse_udiff(raw, D) if kind == "udiff" else parse_legacy(raw, D, spot_for(sess, D))
        return finish(df, D)
    return None

# ───────────────────────────── storage ─────────────────────────────
def day_path(D): return os.path.join(DAYS, D[:4], f"{D}.parquet")
def load_manifest():
    try: return json.load(open(MANIFEST))
    except Exception: return {}
def save_manifest(m):
    os.makedirs(ROOT, exist_ok=True); json.dump(dict(sorted(m.items())), open(MANIFEST, "w"), indent=0)
def save_day(D, df):
    os.makedirs(os.path.dirname(day_path(D)), exist_ok=True); df.to_parquet(day_path(D), index=False)

def load(start=None, end=None, symbols=None):
    """Read the repository back into one DataFrame (all expiries)."""
    fs = []
    for y in sorted(os.listdir(DAYS)) if os.path.isdir(DAYS) else []:
        for fn in sorted(os.listdir(os.path.join(DAYS, y))):
            D = fn[:10]
            if (start and D < start) or (end and D > end): continue
            fs.append(pd.read_parquet(os.path.join(DAYS, y, fn)))
    df = pd.concat(fs, ignore_index=True) if fs else pd.DataFrame(columns=COLS)
    return df[df.symbol.isin(symbols)] if symbols else df

# ───────────────────────────── views / checks ─────────────────────────────
def roll_view(df, D=None):
    """One row per stock for a day: near / next / far prices, spot, basis, annualised carry, OI and rollover % (next / (near + next))."""
    D = D or df.date.max(); d = df[df.date == D]; rows = []
    for s, g in d.groupby("symbol"):
        g = g.sort_values("expiry"); n = g[g.exp_rank == 1]; x = g[g.exp_rank == 2]; f = g[g.exp_rank == 3]
        if n.empty: continue
        n = n.iloc[0]; sp = n.underlying if pd.notna(n.underlying) else np.nan
        basis = (n.close - sp) / sp * 100 if sp and sp == sp else np.nan; carry = basis * 365 / n.dte if (n.dte and n.dte > 0 and basis == basis) else np.nan
        nxt = x.iloc[0] if len(x) else None; far = f.iloc[0] if len(f) else None
        no, xo = float(n.oi), (float(nxt.oi) if nxt is not None else np.nan)
        rows.append({"symbol": s, "near expiry": n.expiry, "dte": int(n.dte), "spot": sp, "near close": n.close, "basis %": round(basis, 3) if basis == basis else "", "annualised carry %": round(carry, 2) if carry == carry else "",
                     "next expiry": nxt.expiry if nxt is not None else "", "next close": nxt.close if nxt is not None else "", "next-near spread %": round((nxt.close / n.close - 1) * 100, 3) if nxt is not None and n.close else "",
                     "far expiry": far.expiry if far is not None else "", "far close": far.close if far is not None else "",
                     "near OI": no, "next OI": xo, "far OI": float(far.oi) if far is not None else "", "rollover %": round(xo / (no + xo) * 100, 1) if (no + xo) > 0 and xo == xo else "",
                     "near contracts": n.contracts, "next contracts": nxt.contracts if nxt is not None else ""})
    return pd.DataFrame(rows).sort_values("rollover %", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce"))

def active_contract(df, roll_days=3):
    """The contract a trader would be in: nearest expiry, but the NEXT one once the nearest has `roll_days` or fewer calendar days left."""
    d = df.sort_values(["symbol", "date", "expiry"]); first = d[d.exp_rank == 1].set_index(["symbol", "date"]); second = d[d.exp_rank == 2].set_index(["symbol", "date"])
    use2 = first.dte <= roll_days; out = first.copy(); idx = out.index[use2 & out.index.isin(second.index)]
    out.loc[idx] = second.loc[idx]; return out.reset_index()

def checks(df):
    """Integrity report: lines to print. Looks for what would silently poison later analysis."""
    L = []; per = df.groupby("date").agg(n=("symbol", "nunique"), r=("symbol", "size"))
    L.append(f"{df.date.nunique()} sessions {df.date.min()} .. {df.date.max()} | {df.symbol.nunique()} distinct stocks | {len(df):,} contract-days")
    L.append(f"stocks per day: min {per.n.min()} (on {per.n.idxmin()}), median {int(per.n.median())}, max {per.n.max()}")
    nt = df[df.contracts > 0]; bad = nt[(nt.close > nt.high * 1.0001) | (nt.close < nt.low * 0.9999)]
    L.append(f"contracts that did not trade (open/high/low are 0, value it with 'settle'): {int((df.contracts == 0).sum()):,} rows = {(df.contracts == 0).mean() * 100:.0f}% (mostly far-month)")
    L.append(f"traded contracts with close outside the day's high/low: {len(bad)} of {len(nt):,} rows" + (f" e.g. {bad.iloc[0].symbol} {bad.iloc[0].date}" if len(bad) else ""))
    z = df[(df.close <= 0) | df.close.isna()]; L.append(f"zero / missing close: {len(z)} rows (a contract with no trade that day keeps its settlement price in 'settle')")
    zz = df[(df.exp_rank == 1) & ((df.oi <= 0) | df.oi.isna())]; L.append(f"nearest-expiry rows with no open interest: {len(zz)}")
    # unit check at the legacy/new boundary: the two formats must report OI in the same unit (median ratio of consecutive days' OI should be ~1, not ~lot size)
    s = df.sort_values("date"); dates = sorted(s.date.unique()); b = [i for i in range(1, len(dates)) if s[s.date == dates[i]].src.iloc[0] != s[s.date == dates[i - 1]].src.iloc[0]]
    for i in b[:2]:
        a = s[(s.date == dates[i - 1]) & (s.exp_rank == 1)].set_index("symbol").oi; c = s[(s.date == dates[i]) & (s.exp_rank == 1)].set_index("symbol").oi
        r = (c / a.reindex(c.index)).replace([np.inf, -np.inf], np.nan).dropna(); L.append(f"format boundary {dates[i-1]} -> {dates[i]}: median day-over-day OI ratio {r.median():.2f} (about 1 means both formats use the same unit)")
    return L

# ───────────────────────────── run ─────────────────────────────
def weekdays(a, b):
    """every calendar day: NSE sometimes trades on a Saturday/Sunday (budget day, live-trading drills); weekends with no file are simply recorded as 'none'."""
    d = pd.Timestamp(a); e = pd.Timestamp(b)
    while d <= e:
        yield d.strftime("%Y-%m-%d"); d += pd.Timedelta(days=1)

def ensure_readme():
    p = os.path.join(ROOT, "README.md")
    if not os.path.exists(p):
        os.makedirs(ROOT, exist_ok=True)
        open(p, "w").write("# Stock futures repository\n\nEvery NSE stock-futures contract (all listed expiries) per trading day, raw (not corporate-action adjusted). One parquet file per day under `by_day/`.\n"
            "Columns: " + ", ".join(COLS) + ".\n\n`exp_rank` 1 = nearest listed expiry that day, 2 = next, 3 = far. `oi`, `oi_chg` are in units (shares); `contracts` = number of contracts; `turnover_cr` in rupees crore.\n"
            "A contract that did not trade has open/high/low = 0; use `settle` to value it. Lot sizes change over time, so convert OI to lots with each row's own `lot`. The 2024-11-01 Diwali muhurat session is a one-hour session.\n"
            "`manifest.json` lists every date tried: `ok` (file stored) or `none` (NSE has no file that day, e.g. holiday). Read back with `futures_repo.load()`.\n")

def main():
    import requests
    start = os.environ.get("FUT_START", "2019-01-01"); end = os.environ.get("FUT_END") or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).strftime("%Y-%m-%d")
    budget = float(os.environ.get("FUT_MAX_MIN", "330")) * 60; t0 = time.time(); m = load_manifest(); sess = requests.Session(); ensure_readme()
    todo = [D for D in weekdays(start, end) if D not in m or (m[D].get("status") == "none" and D >= (pd.Timestamp(end) - pd.Timedelta(days=3)).strftime("%Y-%m-%d"))]
    if os.environ.get("FUT_REPAIR", "") in ("1", "true"):
        fix = []
        for D, v in sorted(m.items()):
            if v.get("status") != "ok" or D < start or D > end: continue
            try: n = pd.read_parquet(day_path(D), columns=["underlying", "exp_rank"])
            except Exception: continue
            if n[n.exp_rank == 1].underlying.isna().mean() > 0.3: fix.append(D)
        todo = sorted(set(todo) | set(fix)); print(f"repair mode: {len(fix)} stored days have no spot price and will be re-fetched")
    print(f"{len(todo)} days to fetch between {start} and {end} ({len(m)} already in the manifest)")
    ok = none = 0; consec = 0
    for i, D in enumerate(todo):
        if time.time() - t0 > budget: print(f"time budget reached after {i} days - run again to continue"); break
        try: df = fetch_day(sess, D)
        except Blocked as e:
            consec += 1; print("  BLOCKED/ERROR:", e)
            if consec >= 8: print("NSE keeps refusing - stopping so nothing is recorded wrongly; rerun later"); break
            continue
        consec = 0
        if df is None or df.empty: m[D] = {"status": "none"}; none += 1
        else: save_day(D, df); m[D] = {"status": "ok", "rows": int(len(df)), "stocks": int(df.symbol.nunique()), "src": str(df.src.iloc[0])}; ok += 1
        if (i + 1) % 50 == 0: save_manifest(m); print(f"  {D}: {ok} stored, {none} no-file so far")
        time.sleep(0.25)
    save_manifest(m); print(f"stored {ok} new days, {none} no-file days; repository now has {sum(1 for v in m.values() if v.get('status') == 'ok')} sessions")
    df = load(start=start)
    if df.empty: raise SystemExit("nothing in the repository yet")
    for line in checks(df): print(" ", line)
    if os.environ.get("FUT_SHEET", "") in ("1", "true") and os.environ.get("SHEET_ID"):
        import gspread, live_refresh_local as LR
        rv = roll_view(df); sh = LR.open_sheet(); H = list(rv.columns)
        def clean(v):
            if hasattr(v, "item"): v = v.item()
            return "" if (v is None or (isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))))) else v
        vals = [H] + [[clean(x) for x in r] for r in rv.values.tolist()]
        try: ws = sh.worksheet("Futures Roll"); ws.clear()
        except gspread.WorksheetNotFound: ws = sh.add_worksheet(title="Futures Roll", rows=len(vals) + 20, cols=len(H) + 2)
        ws.update(values=vals, range_name="A1", value_input_option="USER_ENTERED"); print(f"'Futures Roll' tab written for {df.date.max()} ({len(rv)} stocks)")

# ───────────────────────────── self-test ─────────────────────────────
def selftest():
    import tempfile, shutil
    # new-format sample (2 stocks, 3 expiries for A, plus an index future and an option that must be ignored)
    u = pd.DataFrame({"FinInstrmTp": ["STF"] * 4 + ["IDF", "STO"], "TckrSymb": ["A", "A", "A", "B", "NIFTY", "A"], "XpryDt": ["2026-11-24", "2026-12-29", "2027-01-26", "2026-11-24", "2026-11-24", "2026-11-24"],
        "OpnPric": [100, 101, 102, 50, 1, 1], "HghPric": [103, 104, 105, 52, 1, 1], "LwPric": [99, 100, 101, 49, 1, 1], "ClsPric": [102, 103, 104, 51, 1, 1], "SttlmPric": [102.5, 103.5, 104.5, 51.2, 1, 1],
        "PrvsClsgPric": [100, 101, 102, 50, 1, 1], "LastPric": [102, 103, 104, 51, 1, 1], "UndrlygPric": [101.5] * 3 + [50.5, 1, 1], "TtlTradgVol": [2500, 500, 250, 3000, 1, 1], "TtlTrfVal": [2.55e7, 5e6, 2.5e6, 1.5e7, 1, 1],
        "OpnIntrst": [50000, 10000, 5000, 90000, 1, 1], "ChngInOpnIntrst": [100, 50, 25, -300, 1, 1], "NewBrdLotQty": [250, 250, 250, 1000, 1, 1], "TtlNbOfTxsExctd": [40, 8, 4, 60, 1, 1]})
    d = finish(parse_udiff(u, "2026-11-06"), "2026-11-06")
    assert len(d) == 4 and set(d.symbol) == {"A", "B"}, d
    a = d[d.symbol == "A"].sort_values("expiry"); assert list(a.exp_rank) == [1, 2, 3] and list(a.dte) == [18, 53, 81], a
    assert a.iloc[0].contracts == 10 and abs(a.iloc[0].turnover_cr - 2.55) < 1e-9 and a.iloc[0].oi == 50000 and a.iloc[0].lot == 250
    # legacy sample
    l = pd.DataFrame({"INSTRUMENT": ["FUTSTK", "FUTSTK", "FUTIDX", "OPTSTK"], "SYMBOL": ["A", "A", "NIFTY", "A"], "EXPIRY_DT": ["30-Mar-2023", "27-Apr-2023", "30-Mar-2023", "30-Mar-2023"], "STRIKE_PR": [0, 0, 0, 100], "OPTION_TYP": ["XX", "XX", "XX", "CE"],
        "OPEN": [100, 101, 1, 1], "HIGH": [103, 104, 1, 1], "LOW": [99, 100, 1, 1], "CLOSE": [102, 103, 1, 1], "SETTLE_PR": [102.4, 103.4, 1, 1], "CONTRACTS": [10, 2, 1, 1], "VAL_INLAKH": [255.0, 51.0, 1, 1],
        "OPEN_INT": [50000, 10000, 1, 1], "CHG_IN_OI": [100, 50, 1, 1], "TIMESTAMP": ["15-MAR-2023"] * 4})
    d2 = finish(parse_legacy(l, "2023-03-15", pd.Series({"A": 101.5})), "2023-03-15")
    assert len(d2) == 2 and list(d2.exp_rank) == [1, 2] and list(d2.dte) == [15, 43] and abs(d2.iloc[0].turnover_cr - 2.55) < 1e-9 and d2.iloc[0].underlying == 101.5, d2
    cm = pd.DataFrame({"SYMBOL": ["A", "B", "A"], "SERIES": ["EQ", "EQ", "BE"], "CLOSE": [101.5, 50.5, 9.0]})
    assert parse_cash(cm, "CLOSE").to_dict() == {"A": 101.5, "B": 50.5} and url_cash_old("2019-03-15").endswith("/EQUITIES/2019/MAR/cm15MAR2019bhav.csv.zip")
    assert url_legacy("2023-03-15").endswith("/2023/MAR/fo15MAR2023bhav.csv.zip") and url_udiff("2026-11-06").endswith("20261106_F_0000.csv.zip")
    # storage round trip, views
    global ROOT, DAYS, MANIFEST
    tmp = tempfile.mkdtemp(); ROOT = tmp; DAYS = os.path.join(tmp, "by_day"); MANIFEST = os.path.join(tmp, "manifest.json")
    try:
        save_day("2026-11-06", d); save_day("2023-03-15", d2); save_manifest({"2026-11-06": {"status": "ok"}, "2026-11-07": {"status": "none"}})
        assert os.path.exists(day_path("2026-11-06")) and load_manifest()["2026-11-07"]["status"] == "none"
        al = load(); assert len(al) == 6 and set(load(start="2026-01-01").date) == {"2026-11-06"}
        rv = roll_view(al, "2026-11-06"); r = rv[rv.symbol == "A"].iloc[0]
        assert abs(r["rollover %"] - 16.7) < 0.05 and r["next expiry"] == "2026-12-29" and abs(r["basis %"] - ((102 - 101.5) / 101.5 * 100)) < 0.001, r
        ac = active_contract(d.assign(dte=np.where(d.exp_rank == 1, 2, d.dte)), roll_days=3); assert ac[ac.symbol == "A"].iloc[0].expiry == "2026-12-29"
        assert len(list(weekdays("2026-11-02", "2026-11-08"))) == 7
        print("\n".join(checks(al)[:3]))
    finally: shutil.rmtree(tmp)
    print("SELFTEST OK - both NSE formats parse (stock futures only, all expiries ranked), URLs, storage, roll view, active-contract rule")

if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
