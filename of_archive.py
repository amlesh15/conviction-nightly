# ==================================================================================
# ORDER-FLOW ARCHIVE + DATA-QUALITY GATE
#   Source (Dhan chart back-end, your login token):
#     POST https://ticks.dhan.co/orderflow/getOrderFlow   1-minute footprint candles, buy/sell/neutral volume per price level
#     POST https://ticks.dhan.co/getData                  1-minute price candles (o,h,l,c,v) - used to check the order-flow numbers
#   For every F&O stock it saves ONE row per minute with: buy / sell / neutral volume, delta, cumulative delta, point-of-control,
#   volume at the high and low, imbalance stacks (3:1 diagonal), delta in the top / bottom third of the minute's range, vwap,
#   plus o,h,l,c,v.  The full price-level grid is reduced to those features (keeps the archive ~100-200 MB instead of ~5 GB).
#   History the server holds (about 3 Aug 2026 onwards) is rolling - every run MERGES with the previous archive so days are never lost.
#   Quality gate (tab "OF Quality"): does the order-flow volume match the price feed, how much is 'neutral', and does the
#   minute's delta point the same way as the minute's price move?  If not, nothing built on it can be trusted.
#  Tabs: "OF Quality", "OF Coverage", "OF Today".   Research only - nothing here places orders.
#  Usage: python of_archive.py [--selftest]      env: DHAN_HEADERS_JSON (Auth token), OF_LIMIT (test: first N stocks), OF_BUDGET_MIN (default 100)
# ==================================================================================
import os, sys, io, json, time, glob, datetime as dt, threading
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import dhan_probe as D

SELFTEST = "--selftest" in sys.argv
U_OF = "https://ticks.dhan.co/orderflow/getOrderFlow"; U_G = "https://ticks.dhan.co/getData"; U_H = "https://ticks.dhan.co/getDataH"
OUT = "out/of"; ARCH = f"{OUT}/archive"; PREV = f"{OUT}/prev"
for p in (ARCH, PREV): os.makedirs(p, exist_ok=True)
BUDGET = float(os.environ.get("OF_BUDGET_MIN", "100")); T0 = time.time(); IST = "Asia/Kolkata"
TRANSPORT = None; LOCK = threading.Lock()

class AuthExpired(Exception): pass

# ───────────────────────────── network ─────────────────────────────
def post(url, body, tries=4):
    if TRANSPORT: return TRANSPORT(url, body)
    import requests
    for k in range(tries):
        try: r = requests.post(url, json=body, headers=D.headers(True), timeout=180)
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(2 + 3 * k); continue
        if r.status_code == 401: raise AuthExpired("401 from Dhan - copy a fresh Auth token into the DHAN_HEADERS_JSON secret")
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 + 10 * k); continue
        return r
    return r

def iso(t): return pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")

def of_body(sid, start, end):
    return {"EXCH": "NSE", "SEG": "E", "SEC_ID": sid, "START": start, "END": end, "START_TIME": iso(start), "END_TIME": iso(end)}

def fetch_of(sid, max_chunks=8, span=45):
    """getOrderFlow candles, newest 45-day chunks first, until the server has nothing older."""
    today = int(pd.Timestamp.now(tz=IST).normalize().timestamp()) + 86400; out = []; empty = 0
    for c in range(max_chunks):
        end = today - c * span * 86400; start = (end - span * 86400) // 86400 * 86400
        r = post(U_OF, of_body(sid, start, end))
        if r.status_code != 200: break
        try: cs = (r.json().get("data") or {}).get("candles") or []
        except Exception: cs = []
        if not cs: empty += 1
        else: empty = 0; out += cs
        if empty >= 1 and c > 0: break
        time.sleep(0.25)
    return out

def fetch_1m(sid, max_chunks=6, span=45):
    today = int(time.time()) + 86400; parts = []
    for c in range(max_chunks):
        end = today - c * span * 86400; start = end - span * 86400
        r = post(U_G, {"EXCH": "NSE", "SEG": "E", "INST": "EQUITY", "SEC_ID": sid, "INTERVAL": "1", "START": start, "END": end})
        if r.status_code != 200: break
        d = (r.json().get("data") or {}) if r.text else {}
        if not d or not d.get("t"): break
        parts.append(pd.DataFrame({k: d[k] for k in ("t", "o", "h", "l", "c", "v") if k in d}))
        time.sleep(0.25)
    if not parts: return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    return pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)

def fetch_daily(sid, sym):
    now = int(time.time()); r = post(U_H, {"EXCH": "NSE", "SEG": "E", "INST": "EQUITY", "SEC_ID": sid, "INTERVAL": "D", "START": now - 200 * 86400, "END": now, "EXPCODE": 0, "SYM": sym})
    if r.status_code != 200: return pd.DataFrame(columns=["date", "d_o", "d_h", "d_l", "d_c", "d_v"])
    d = (r.json().get("data") or {})
    if not d.get("t"): return pd.DataFrame(columns=["date", "d_o", "d_h", "d_l", "d_c", "d_v"])
    ts = pd.to_datetime(np.array(d["t"], float), unit="s") + pd.Timedelta(hours=5, minutes=30)
    return pd.DataFrame({"date": ts.strftime("%Y-%m-%d"), "d_o": d["o"], "d_h": d["h"], "d_l": d["l"], "d_c": d["c"], "d_v": d["v"]}).drop_duplicates("date")

# ───────────────────────────── features from one footprint candle ─────────────────────────────
def longest_run(m):
    if not m.any(): return 0
    x = np.r_[0, m.astype(np.int8), 0]; d = np.diff(x); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]; return int((en - st).max())

def candle_features(c):
    lv = c.get("levels") or []; tot = c.get("total_volume", 0) or 0
    row = {"t": c["t"], "hi": c.get("high"), "lo": c.get("low"), "buy": c.get("buy_volume", 0), "sell": c.get("sell_volume", 0), "neu": c.get("neutral_volume", 0), "tot": tot,
           "delta": c.get("delta", 0), "cd": c.get("cumulative_delta", 0), "nlev": len(lv)}
    if not lv: return row
    p = np.array([x["price"] for x in lv], float); b = np.array([x.get("buy_volume", 0) for x in lv], float); s = np.array([x.get("sell_volume", 0) for x in lv], float)
    t_ = np.array([x.get("total_volume", 0) for x in lv], float); o = np.argsort(p); p, b, s, t_ = p[o], b[o], s[o], t_[o]
    k = int(np.argmax(t_)); row["poc"] = p[k]; row["poc_share"] = t_[k] / tot if tot else np.nan
    row["hi_b"], row["hi_s"], row["lo_b"], row["lo_s"] = b[-1], s[-1], b[0], s[0]
    mv = max(1.0, 0.02 * tot); n = len(p)
    if n >= 2:
        bi = np.zeros(n, bool); si = np.zeros(n, bool)
        bi[1:] = (b[1:] >= 3 * np.maximum(s[:-1], 1)) & (b[1:] >= mv); si[:-1] = (s[:-1] >= 3 * np.maximum(b[1:], 1)) & (s[:-1] >= mv)
        row["bstack"], row["sstack"], row["nb"], row["ns"] = longest_run(bi), longest_run(si), int(bi.sum()), int(si.sum())
    th = max(1, n // 3); dl = b - s; row["d_top"] = dl[-th:].sum(); row["d_bot"] = dl[:th].sum()
    row["vwap"] = (p * t_).sum() / t_.sum() if t_.sum() > 0 else np.nan
    return row

def profile_frame(of_candles):
    """volume-at-price by 30-minute period and price bin (bin width ~5 bps of price, constant per stock): buy and sell volume. 09:15-15:14 only."""
    D, Pm, PX, Bv, Sv = [], [], [], [], []
    for c in of_candles:
        t = c["t"]; m = ((t + 19800) % 86400) // 60 - 555; lv = c.get("levels") or []
        if m < 0 or m >= 360 or not lv: continue
        n = len(lv); D += [(t + 19800) // 86400] * n; Pm += [m // 30] * n; PX += [x["price"] for x in lv]; Bv += [x.get("buy_volume", 0) for x in lv]; Sv += [x.get("sell_volume", 0) for x in lv]
    if not D: return pd.DataFrame(columns=["date", "per", "bin", "w", "buy", "sell"])
    P = pd.DataFrame({"day": D, "per": Pm, "px": PX, "buy": Bv, "sell": Sv}); w = max(0.05, round(0.0005 * float(P["px"].median()) / 0.05) * 0.05)
    P["bin"] = np.floor(P["px"] / w + 1e-9).astype(int); P["date"] = pd.to_datetime(P["day"] * 86400, unit="s").dt.strftime("%Y-%m-%d")
    G = P.groupby(["date", "per", "bin"], as_index=False)[["buy", "sell"]].sum(); G["w"] = np.float32(w); G["buy"] = G["buy"].astype("int64"); G["sell"] = G["sell"].astype("int64"); return G[["date", "per", "bin", "w", "buy", "sell"]]

def merge_prev_profile(sym, G):
    for ext, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        pth = f"{PREV}/prof_{sym}{ext}"
        if os.path.exists(pth):
            try: P = fn(pth); return pd.concat([G, P[~P["date"].isin(G["date"].unique())]]).sort_values(["date", "per", "bin"]).reset_index(drop=True)
            except Exception: pass
    return G

def save_profile(G, sym):
    try: G.to_parquet(f"{ARCH}/prof_{sym}.parquet", index=False, compression="zstd")
    except Exception: G.to_csv(f"{ARCH}/prof_{sym}.csv.gz", index=False)

def build_frame(sym, of_candles, gd, daily=None):
    if not of_candles: return pd.DataFrame()
    F = pd.DataFrame([candle_features(c) for c in of_candles]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    F = F.merge(gd, on="t", how="left") if len(gd) else F.assign(o=np.nan, h=np.nan, l=np.nan, c=np.nan, v=np.nan)
    ts = pd.to_datetime(F["t"], unit="s") + pd.Timedelta(hours=5, minutes=30); F["date"] = ts.dt.strftime("%Y-%m-%d"); F["hhmm"] = ts.dt.hour * 100 + ts.dt.minute
    F.insert(0, "symbol", sym)
    if daily is not None and len(daily): F = F.merge(daily, on="date", how="left")
    else:
        for c in ("d_o", "d_h", "d_l", "d_c", "d_v"): F[c] = np.nan
    for c in F.columns:
        if F[c].dtype == "float64" and c not in ("t",): F[c] = F[c].astype("float32")
    return F

def merge_prev(sym, F):
    for ext, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        pth = f"{PREV}/{sym}{ext}"
        if os.path.exists(pth):
            try:
                P = fn(pth); allr = pd.concat([F, P[~P["t"].isin(F["t"])]]); return allr.sort_values("t").reset_index(drop=True)
            except Exception: pass
    return F

def save(F, sym):
    try: F.to_parquet(f"{ARCH}/{sym}.parquet", index=False, compression="zstd"); return "parquet"
    except Exception: F.to_csv(f"{ARCH}/{sym}.csv.gz", index=False); return "csv.gz"

# ───────────────────────────── quality ─────────────────────────────
def quality_row(F):
    M = F[(F["hhmm"] >= 915) & (F["hhmm"] < 1530)]; both = M.dropna(subset=["v", "o", "c"])
    r = {"symbol": F["symbol"].iloc[0], "first day": F["date"].min(), "last day": F["date"].max(), "sessions": F["date"].nunique(), "minutes/session (median)": F[(F.hhmm >= 915) & (F.hhmm < 1530)].groupby("date").size().median(),
         "neutral share of volume %": M["neu"].sum() / max(M["tot"].sum(), 1) * 100}
    if len(both) > 200:
        r["volume vs price-feed (ratio)"] = both["tot"].sum() / max(both["v"].sum(), 1); ret = (both["c"] - both["o"]).to_numpy(); dl = both["delta"].to_numpy(float)
        ok = (ret != 0) & (dl != 0); r["delta sign agrees with minute move %"] = (np.sign(dl[ok]) == np.sign(ret[ok])).mean() * 100 if ok.sum() else np.nan
        x = (both["delta"] / both["tot"].replace(0, np.nan)).to_numpy(float); y = (ret / both["o"].to_numpy(float)); g = np.isfinite(x) & np.isfinite(y)
        r["corr(delta/volume, return)"] = np.corrcoef(x[g], y[g])[0, 1] if g.sum() > 100 else np.nan
        r["share of minutes with zero-volume sell at high %"] = (M["hi_s"] == 0).mean() * 100
    r.update(daily_check(F)); return r

def daily_check(F):
    """per day: minute feed (09:15..15:14) vs Dhan's official daily candle."""
    M = F[(F.hhmm >= 915) & (F.hhmm < 1530)].dropna(subset=["c", "o"])
    if M.empty or F["d_c"].notna().sum() == 0: return {}
    g = M.groupby("date").agg(o=("o", "first"), c=("c", "last"), h=("h", "max"), l=("l", "min"), v=("tot", "sum"), d_o=("d_o", "first"), d_c=("d_c", "first"), d_h=("d_h", "first"), d_l=("d_l", "first"), d_v=("d_v", "first")).dropna(subset=["d_c"])
    if g.empty: return {}
    pct = lambda a, b: (a / b - 1) * 100
    return {"days compared to official daily candle": len(g), "open: minute feed vs official, median abs diff %": pct(g.o, g.d_o).abs().median(), "last-minute (15:14) price vs official close, median abs diff %": pct(g.c, g.d_c).abs().median(),
            "days where 15:14 vs official close differs > 0.2%": (pct(g.c, g.d_c).abs() > 0.2).mean() * 100, "high vs official, median abs diff %": pct(g.h, g.d_h).abs().median(), "low vs official, median abs diff %": pct(g.l, g.d_l).abs().median(),
            "volume (minute feed) / official daily volume": (g.v.sum() / g.d_v.sum()) if g.d_v.sum() else np.nan}

def today_row(F):
    ds = sorted(F["date"].unique()); d = ds[-1]; S = F[(F.date == d) & (F.hhmm >= 915) & (F.hhmm < 1530)]
    if len(S) < 50: return None
    vol = S["tot"].sum(); last = S[S.hhmm >= 1500]; o, c = S["o"].dropna(), S["c"].dropna()
    pc = np.nan
    if len(ds) > 1:
        off = F[F.date == ds[-2]]["d_c"].dropna(); P = F[(F.date == ds[-2]) & (F.hhmm >= 915) & (F.hhmm < 1530)]["c"].dropna()
        pc = off.iloc[0] if len(off) else (P.iloc[-1] if len(P) else np.nan)
    o0 = o.iloc[0] if len(o) else np.nan; offc = F[F.date == d]["d_c"].dropna()
    c1 = offc.iloc[0] if len(offc) else (c.iloc[-1] if len(c) else np.nan); basis = "official daily close" if len(offc) else "15:14 price (PROVISIONAL - last 15 min missing)"
    return {"symbol": F["symbol"].iloc[0], "date": d, "volume (09:15-15:14 only)": vol, "delta": S["delta"].sum(), "delta % of volume": S["delta"].sum() / max(vol, 1) * 100,
            "last-30m delta % of its volume": last["delta"].sum() / max(last["tot"].sum(), 1) * 100,
            "gap % (open vs yesterday's close)": (o0 / pc - 1) * 100 if pc == pc else np.nan, "intraday % (open to close)": (c1 / o0 - 1) * 100 if o0 == o0 else np.nan,
            "change % vs yesterday's close": (c1 / pc - 1) * 100 if pc == pc else np.nan, "close used": basis}

# ───────────────────────────── per stock ─────────────────────────────
def process(sym, sid):
    if time.time() - T0 > BUDGET * 60: return sym, None, "time budget reached"
    of = fetch_of(sid)
    if not of: return sym, None, "no order-flow candles"
    F = merge_prev(sym, build_frame(sym, of, fetch_1m(sid), fetch_daily(sid, sym))); G = merge_prev_profile(sym, profile_frame(of))
    with LOCK: fmt = save(F, sym); save_profile(G, sym)
    return sym, F, fmt

def universe_ids(symbols):
    import requests
    if TRANSPORT: S = TRANSPORT("scrip", None)
    else:
        r = requests.get("https://images.dhan.co/api-data/api-scrip-master.csv", timeout=120); r.raise_for_status(); S = pd.read_csv(io.StringIO(r.text), low_memory=False)
    c = {x.upper(): x for x in S.columns}; q = S[(S[c["SEM_EXM_EXCH_ID"]] == "NSE") & (S[c["SEM_INSTRUMENT_NAME"]] == "EQUITY")]
    m = dict(zip(q[c["SEM_TRADING_SYMBOL"]].astype(str).str.upper(), q[c["SEM_SMST_SECURITY_ID"]].astype(int)))
    return {s: m[s] for s in symbols if s in m}, [s for s in symbols if s not in m]

def main():
    D.jwt_info()
    if SELFTEST: symbols = [f"S{i:02d}" for i in range(8)]
    else:
        import oi_probe as P; symbols = P.universe_from_sheet()
    lim = int(os.environ.get("OF_LIMIT") or 0); symbols = symbols[:lim] if lim else symbols
    ids, missing = universe_ids(symbols); D.log("setup", "stocks to fetch", len(ids), f"not found in Dhan scrip master: {missing[:25]}")
    rows, tod, cov, errs = [], [], [], []; done = 0
    try:
        with ThreadPoolExecutor(3) as ex:
            futs = {ex.submit(process, s, i): s for s, i in ids.items()}
            for f in as_completed(futs):
                s = futs[f]
                try: sym, F, note = f.result()
                except AuthExpired as e: errs.append(str(e)); [x.cancel() for x in futs]; break
                except Exception as e: errs.append(f"{s}: {type(e).__name__} {str(e)[:100]}"); continue
                if F is None: errs.append(f"{s}: {note}"); continue
                done += 1; rows.append(quality_row(F)); t = today_row(F); tod += [t] if t else []
                cov.append({"symbol": s, "first day": F["date"].min(), "last day": F["date"].max(), "sessions": F["date"].nunique(), "minutes": len(F), "has price feed %": F["c"].notna().mean() * 100})
                if done % 10 == 0: print(f"  {done}/{len(ids)} stocks ({time.time()-T0:.0f}s)", flush=True)
    except AuthExpired as e: errs.append(str(e))
    Q = pd.DataFrame(rows); C = pd.DataFrame(cov); T = pd.DataFrame(tod)
    if len(Q):
        num = Q.select_dtypes("number"); agg = {"symbol": "ALL (median)", "first day": Q["first day"].min(), "last day": Q["last day"].max()}; agg.update(num.median().to_dict()); Q = pd.concat([pd.DataFrame([agg]), Q.sort_values("symbol")])
    D.log("result", "stocks saved", done, "; ".join(errs[:6]) if errs else "no errors")
    if len(Q): print(Q.head(1).T.round(3).to_string())
    Q.to_csv(f"{OUT}/of_quality.csv", index=False); C.to_csv(f"{OUT}/of_coverage.csv", index=False); T.to_csv(f"{OUT}/of_today.csv", index=False)
    if SELFTEST: return Q, C, T
    import oi_probe as P; sh = P.open_sheet(); rnd = lambda d: d.round(4).replace([np.inf, -np.inf], np.nan).astype(object).where(d.notna(), "")
    if len(Q): P.put_tab(sh, "OF Quality", list(Q.columns), rnd(Q).values.tolist())
    if len(C): P.put_tab(sh, "OF Coverage", list(C.columns), rnd(C).values.tolist())
    if len(T): P.put_tab(sh, "OF Today", list(T.columns), rnd(T.sort_values("delta % of volume")).values.tolist())
    P.put_tab(sh, "OF Run Log", ["stage", "check", "result", "detail"], D.REPORT[-6:] + [["errors", e, "", ""] for e in errs[:30]]); print("sheet tabs written")

# ───────────────────────────── self-test ─────────────────────────────
def _synth_of(sid, start, end, rng):
    out = []; days = [d for d in pd.bdate_range("2026-09-14", "2026-10-22")]
    for d in days:
        base = float(1000 + sid % 500); cd = 0; px = base
        for m in range(375 + 8):
            hh = 9 * 60 + 7 + m if m < 1 else 9 * 60 + 15 + (m - 1)
            if m >= 376: break
            ts = int((pd.Timestamp(d) + pd.Timedelta(minutes=hh)).timestamp()) - 19800; strength = rng.normal(0, 1); ret = strength * 0.0004 + rng.normal(0, 0.0003)
            o = px; px = px * (1 + ret); tot = int(rng.integers(2000, 20000)); buyshare = min(.95, max(.05, .5 + strength * .12 + rng.normal(0, .04))); b = int(tot * buyshare); s = tot - b
            lv = []; ps = np.round(np.linspace(min(o, px), max(o, px) + 0.5, 9), 1)
            for p_ in ps: bb = int(b / 9 * rng.uniform(.3, 1.7)); ss = int(s / 9 * rng.uniform(.3, 1.7)); lv.append({"price": float(p_), "buy_volume": bb, "sell_volume": ss, "neutral_volume": 0, "total_volume": bb + ss, "delta": bb - ss})
            cd += b - s; out.append({"t": ts, "time": "x", "high": float(max(o, px) + 0.5), "low": float(min(o, px)), "buy_volume": b, "sell_volume": s, "neutral_volume": 0, "total_volume": tot, "delta": b - s, "cumulative_delta": cd, "levels": lv, "_o": o, "_c": px})
    return out

class _R:
    def __init__(s, j, code=200): s._j = j; s.status_code = code; s.text = json.dumps(j)[:50] if code == 200 else "err"
    def json(s): return s._j

def _fake(url, body):
    if url == "scrip": return pd.DataFrame({"SEM_EXM_EXCH_ID": ["NSE"] * 8, "SEM_INSTRUMENT_NAME": ["EQUITY"] * 8, "SEM_TRADING_SYMBOL": [f"S{i:02d}" for i in range(8)], "SEM_SMST_SECURITY_ID": list(range(100, 108))})
    sid = body["SEC_ID"]; rng = np.random.default_rng(sid)
    if url == U_H:
        cs = _synth_of(sid, 0, 4e9, rng); G = pd.DataFrame([{"d": pd.Timestamp(c["t"] + 19800, unit="s").strftime("%Y-%m-%d"), "o": c["_o"], "c": c["_c"] * 1.002, "h": c["high"], "l": c["low"], "v": c["total_volume"], "t": c["t"]} for c in cs])
        g = G.groupby("d").agg(o=("o", "first"), c=("c", "last"), h=("h", "max"), l=("l", "min"), v=("v", "sum"), t=("t", "first")).reset_index()
        return _R({"success": True, "data": {"o": g.o.tolist(), "h": g.h.tolist(), "l": g.l.tolist(), "c": g.c.tolist(), "v": g.v.tolist(), "t": [int(pd.Timestamp(x).timestamp()) - 19800 for x in g.d], "Time": g.d.tolist()}})
    cs = _synth_of(sid, body["START"], body["END"], rng)
    lo = body["START"]; hi = body["END"]; cs = [c for c in cs if lo <= c["t"] <= hi]
    if url == U_OF: return _R({"success": True, "data": {"instrument": "x", "interval": "1m", "candles": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cs]}})
    cs = [c for c in cs if c["time"] == "x"]; return _R({"success": True, "data": {"t": [c["t"] for c in cs], "o": [c["_o"] for c in cs], "h": [c["high"] for c in cs], "l": [c["low"] for c in cs], "c": [c["_c"] for c in cs], "v": [c["total_volume"] for c in cs]}})

if __name__ == "__main__":
    if SELFTEST:
        TRANSPORT = _fake; Q, C, T = main(); a = Q.iloc[0]
        assert a["sessions"] >= 20 and 0.9 < a["volume vs price-feed (ratio)"] < 1.1, a.to_dict()
        assert a["delta sign agrees with minute move %"] > 60 and a["corr(delta/volume, return)"] > 0.3, "synthetic planted relation not detected"
        pf = [f for f in os.listdir(ARCH) if f.startswith("prof_")]; assert len(pf) == 8, pf
        G = pd.read_csv(f"{ARCH}/prof_S00.csv.gz"); Fm = pd.read_csv(f"{ARCH}/S00.csv.gz"); Fm = Fm[(Fm.hhmm >= 915) & (Fm.hhmm < 1515)]
        assert abs((G.buy.sum() + G.sell.sum()) / Fm["tot"].sum() - 1) < 0.01, "profile volume != minute volume"; print("profile rows", len(G), "dates", G.date.nunique())
        print(sorted(os.listdir(ARCH))[:3], "SELFTEST OK")
    else: main()
