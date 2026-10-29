# ==================================================================================
# NIFTY LEVELS — support / resistance from Nifty FUTURES footprint data (price-level buy/sell volume), plus an honest test.
#   Source: Dhan chart back-end getOrderFlow (1-minute footprint candles) for the nearest-expiry NIFTY future (contract looked up in the scrip master).
#   Level sources (all from traded volume, not from price gaps):
#     value area  POC / VAH / VAL  of: yesterday, last 5 sessions, last month (22 sessions) (70% value area)
#     HVN         high-volume nodes of the 1-month profile (local volume peaks)
#     stacked     3+ adjacent price bins where one side dominates 3:1 (diagonal) inside a 30-minute block; buy stacks mark support, sell stacks resistance
#     absorption  minutes with big volume and a lopsided delta where price barely moved
#     yesterday's high and low
#   Nearby sources are merged into one level (within MERGE points); strength = how many sources agree (weighted).
#   OUTPUT  tab "Nifty Levels" (ranked supports below / resistances above the last price, with the sources behind each)
#           tab "Nifty Level Test" (does price actually react at these levels more than at random prices next to them?)
#   Research only. No orders.
#   Usage: python nifty_levels.py [--selftest]      env: DHAN_HEADERS_JSON, GCP_SA_JSON, SHEET_ID, NL_DAYS (history to fetch, default 60)
# ==================================================================================
import os, sys, json, time, datetime as dt
import numpy as np, pandas as pd

SELFTEST = "--selftest" in sys.argv
IST = "Asia/Kolkata"; U_OF = "https://ticks.dhan.co/orderflow/getOrderFlow"; MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"
BIN = 5.0            # price bin, index points
MERGE = 12.0         # sources closer than this become one level
VA = 0.70
W = {"prevPOC": 2, "prevVAH": 2, "prevVAL": 2, "5dPOC": 2, "5dVAH": 2, "5dVAL": 2, "1MPOC": 3, "1MVAH": 3, "1MVAL": 3, "HVN": 2, "buyStack": 1, "sellStack": 1, "absorb": 1, "prevHigh": 1, "prevLow": 1}
MONTH = 22           # ~1 calendar month of sessions
TRANSPORT = None; M_REACT = 10.0; LOOK = 30; CONTROL = (-80, -40, 40, 80)

# ───────────────────────────── data ─────────────────────────────
def headers():
    h = {"Content-Type": "application/json", "Accept": "application/json, text/plain, */*", "Origin": "https://dext.dhan.co", "Referer": "https://dext.dhan.co/",
         "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    h.update(json.loads(os.environ.get("DHAN_HEADERS_JSON") or "{}")); return h

def post(body):
    if TRANSPORT: return TRANSPORT(body)
    import requests
    for k in range(4):
        r = requests.post(U_OF, json=body, headers=headers(), timeout=180)
        if r.status_code == 401: raise SystemExit("401 from Dhan: refresh the DHAN_HEADERS_JSON secret")
        if r.status_code in (429, 500, 502, 503, 504): time.sleep(5 + 10 * k); continue
        return r
    return r

def iso(t): return pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")

def nearest_future():
    import requests
    r = requests.get(MASTER, timeout=90); r.raise_for_status(); open("/tmp/dhan_master.csv", "wb").write(r.content); m = pd.read_csv("/tmp/dhan_master.csv", low_memory=False)
    idc = next(x for x in m.columns if "SECURITY_ID" in x and "UNDERLYING" not in x); ts = next(x for x in m.columns if "TRADING_SYMBOL" in x)
    f = m[(m["SEM_EXM_EXCH_ID"] == "NSE") & (m["SEM_INSTRUMENT_NAME"] == "FUTIDX") & (m[ts].astype(str).str.upper().str.startswith("NIFTY-"))].copy()
    f["exp"] = pd.to_datetime(f["SEM_EXPIRY_DATE"], errors="coerce").dt.date; f = f[f["exp"] >= dt.date.today()].sort_values("exp")
    return int(f.iloc[0][idc]), str(f.iloc[0][ts]), f.iloc[0]["exp"]

def fetch(sid, days, span=45):
    end0 = int(pd.Timestamp.now(tz=IST).normalize().timestamp()) + 86400; out = []
    for c in range((days + span - 1) // span):
        end = end0 - c * span * 86400; start = (end - span * 86400) // 86400 * 86400
        r = post({"EXCH": "NSE", "SEG": "D", "SEC_ID": sid, "START": start, "END": end, "START_TIME": iso(start), "END_TIME": iso(end)})
        try: cs = (r.json().get("data") or {}).get("candles") or []
        except Exception: cs = []
        if not cs: break
        out += cs; time.sleep(0.3)
    return out

def to_frames(cs):
    """-> bars (one row per minute: t, date, mod, hi, lo, buy, sell, tot) and levels (t, date, blk, bin, buy, sell)."""
    bars, lv = [], []
    for c in cs:
        t = c["t"]; ist = t + 19800; mod = (ist % 86400) // 60; date = pd.Timestamp(ist // 86400 * 86400, unit="s").strftime("%Y-%m-%d")
        if mod < 555 or mod > 930: continue
        bars.append((t, date, mod, c.get("high"), c.get("low"), c.get("buy_volume", 0) or 0, c.get("sell_volume", 0) or 0, c.get("total_volume", 0) or 0))
        for x in c.get("levels") or []: lv.append((date, (mod - 555) // 30, int(np.floor(x["price"] / BIN + 1e-9)), x.get("buy_volume", 0) or 0, x.get("sell_volume", 0) or 0))
    B = pd.DataFrame(bars, columns=["t", "date", "mod", "hi", "lo", "buy", "sell", "tot"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    L = pd.DataFrame(lv, columns=["date", "blk", "bin", "buy", "sell"]); return B, L

# ───────────────────────────── building blocks ─────────────────────────────
def profile(L):
    g = L.groupby("bin")[["buy", "sell"]].sum(); g["v"] = g["buy"] + g["sell"]; return g.sort_index()

def value_area(g, frac=VA):
    """POC, VAH, VAL (bin lower edges + half bin = centre) by expanding from the POC towards the heavier neighbouring pair."""
    if g.empty: return None
    idx = list(g.index); v = g["v"].to_numpy(float); k = int(np.argmax(v)); lo = hi = k; got = v[k]; tot = v.sum()
    while got < frac * tot and (lo > 0 or hi < len(v) - 1):
        up = v[hi + 1:hi + 3].sum() if hi < len(v) - 1 else -1; dn = v[max(lo - 2, 0):lo].sum() if lo > 0 else -1
        if up >= dn: step = v[hi + 1:hi + 3]; hi += len(step); got += step.sum()
        else: step = v[max(lo - 2, 0):lo]; lo -= len(step); got += step.sum()
    c = lambda i: (idx[i] + 0.5) * BIN
    return c(k), c(hi), c(lo)

def hvns(g, win=3, mult=1.3):
    v = g["v"].reindex(range(int(g.index.min()), int(g.index.max()) + 1), fill_value=0).to_numpy(float); base = v[v > 0].mean(); out = []
    for i in range(len(v)):
        w = v[max(0, i - win):i + win + 1]
        if v[i] == w.max() and v[i] >= mult * base and (not out or i - out[-1] > win): out.append(i)
    return [(int(g.index.min()) + i + 0.5) * BIN for i in out]

def stacks(L, min_run=3, ratio=3.0):
    """Per (date, 30-min block): runs of >= min_run adjacent bins with 3:1 diagonal dominance. -> [(date, side, price_mid)]"""
    out = []
    for (d, b), g in L.groupby(["date", "blk"]):
        g = g.groupby("bin")[["buy", "sell"]].sum().sort_index(); full = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1), fill_value=0)
        buy = full["buy"].to_numpy(float); sell = full["sell"].to_numpy(float); mv = max(1.0, 0.03 * (buy.sum() + sell.sum())); n = len(full); base = int(full.index.min())
        bi = np.zeros(n, bool); si = np.zeros(n, bool)
        if n > 1: bi[1:] = (buy[1:] >= ratio * np.maximum(sell[:-1], 1)) & (buy[1:] >= mv); si[:-1] = (sell[:-1] >= ratio * np.maximum(buy[1:], 1)) & (sell[:-1] >= mv)
        for side, m in (("buy", bi), ("sell", si)):
            i = 0
            while i < n:
                if m[i]:
                    j = i
                    while j + 1 < n and m[j + 1]: j += 1
                    if j - i + 1 >= min_run: out.append((d, side, (base + (i + j) / 2 + 0.5) * BIN))
                    i = j + 1
                else: i += 1
    return out

def absorption(B):
    """Minutes with volume >= 3x median, |delta| >= 40% of volume, and a high-low range no bigger than the median range. -> [(date, price_mid)]"""
    B = B.copy(); B["rng"] = B["hi"] - B["lo"]; mv = B["tot"].median(); mr = B["rng"].median()
    a = B[(B["tot"] >= 3 * mv) & ((B["buy"] - B["sell"]).abs() >= 0.4 * B["tot"]) & (B["rng"] <= mr)]
    return [(r.date, (r.hi + r.lo) / 2) for r in a.itertuples()]

def cluster(items, merge=MERGE):
    """items = [(price, source, weight)] -> [{price, strength, sources}] merged by proximity."""
    items = sorted(items); cl = []
    for p, s, w in items:
        if cl and p - cl[-1]["last"] <= merge: c = cl[-1]; c["m"].append((p, s, w)); c["last"] = p
        else: cl.append({"m": [(p, s, w)], "last": p})
    out = []
    for c in cl:
        ws = sum(w for _, _, w in c["m"]); px = sum(p * w for p, _, w in c["m"]) / ws; src = sorted({s for _, s, _ in c["m"]})
        out.append({"price": round(px, 1), "strength": round(ws, 1), "sources": ", ".join(src), "n": len({s for s in src})})
    return out

def build_levels(B, L):
    days = sorted(B["date"].unique()); items = []
    if len(days) < 2: return [], None, days
    base = days[:-1]                                       # completed sessions; days[-1] is today (possibly partial) and is not used for levels
    ydate = base[-1]
    for tag, ds in (("prev", [ydate]), ("5d", base[-5:]), ("1M", base[-MONTH:])):
        va = value_area(profile(L[L["date"].isin(ds)]))
        if va:
            for nm, p in zip(("POC", "VAH", "VAL"), va): items.append((p, tag + nm, W[tag + nm]))
    for p in hvns(profile(L[L["date"].isin(base[-MONTH:])])): items.append((p, "HVN", W["HVN"]))
    yb = B[B["date"] == ydate]; items += [(float(yb["hi"].max()), "prevHigh", 1), (float(yb["lo"].min()), "prevLow", 1)]
    recent = set(base[-5:])
    for d, side, p in stacks(L[L["date"].isin(recent)]): items.append((p, side + "Stack", W[side + "Stack"]))
    for d, p in absorption(B[B["date"].isin(recent)]): items.append((p, "absorb", 1))
    last = float((B["hi"].iloc[-1] + B["lo"].iloc[-1]) / 2)
    return cluster(items), last, days

# ───────────────────────────── honest test: do prior-session levels beat nearby random prices? ─────────────────────────────
def first_touch_outcome(hi, lo, L, m=M_REACT, look=LOOK):
    """hi/lo arrays of one session. First bar whose range contains L; came from below or above (previous bar mid). Returns 'react', 'break' or None."""
    for i in range(1, len(hi)):
        if lo[i] <= L <= hi[i]:
            from_below = (hi[i - 1] + lo[i - 1]) / 2 < L
            for j in range(i, min(len(hi), i + look)):
                if from_below:
                    if lo[j] <= L - m: return "react"
                    if hi[j] >= L + m: return "break"
                else:
                    if hi[j] >= L + m: return "react"
                    if lo[j] <= L - m: return "break"
            return None
    return None

def level_test(B, L, top=6):
    days = sorted(B["date"].unique()); res = {"level": [0, 0], "control": [0, 0]}; per = []
    for k in range(5, len(days)):
        d = days[k]; Lh = L[L["date"].isin(days[max(0, k - MONTH):k])]; Bh = B[B["date"].isin(days[max(0, k - MONTH):k])]
        if Lh.empty: continue
        lv, last, _ = build_levels(pd.concat([Bh, B[B["date"] == d].head(1)]), Lh)       # one bar of day d marks "today"; levels use only earlier sessions
        lv = sorted(lv, key=lambda c: -c["strength"])[:top]; s = B[B["date"] == d]; hi = s["hi"].to_numpy(float); lo = s["lo"].to_numpy(float)
        for c in lv:
            for name, off in [("level", 0)] + [("control", o) for o in CONTROL]:
                o = first_touch_outcome(hi, lo, c["price"] + off)
                if o: res[name][0 if o == "react" else 1] += 1
    out = []
    for name in ("level", "control"):
        r, b = res[name]; out.append([name, r, b, round(r / (r + b) * 100, 1) if r + b else ""])
    return out

# ───────────────────────────── output ─────────────────────────────
def sheet_put(sh, title, header, rows):
    import gspread
    try: ws = sh.worksheet(title); ws.clear()
    except gspread.WorksheetNotFound: ws = sh.add_worksheet(title=title, rows=300, cols=max(8, len(header)))
    ws.update(values=[header] + rows, range_name="A1", value_input_option="USER_ENTERED")

def imbalance_refs(sh):
    """Nifty daily-imbalance levels as they stand in the NIFTY tab (shown for comparison only; their calculation is untouched)."""
    try:
        v = sh.worksheet("NIFTY").get_all_values(); h = [x.strip() for x in v[0]]; r = v[1]; g = lambda k: float(str(r[h.index(k)]).replace(",", "")) if k in h and r[h.index(k)] not in ("", None) else None
        return [x for x in (g("imb_sup"), g("imb_res")) if x], (r[h.index("updated")] if "updated" in h else "")
    except Exception: return [], ""

def report(levels, last, label, imb, imb_asof):
    sup = sorted([c for c in levels if c["price"] < last], key=lambda c: -c["price"]); res = sorted([c for c in levels if c["price"] >= last], key=lambda c: c["price"])
    rows = []
    def add(side, c, rank):
        near = [i for i in imb if abs(i - c["price"]) <= 20]
        rows.append([side + str(rank), c["price"], round((c["price"] / last - 1) * 100, 2), c["strength"], c["n"], c["sources"], f"{near[0]:g}" if near else ""])
    for i, c in enumerate(res[:6], 1): add("R", c, i)
    for i, c in enumerate(sup[:6], 1): add("S", c, i)
    rows.sort(key=lambda r: -r[1])
    hdr = ["level", "price", "% from last", "strength", "sources agreeing", "what it is made of", f"daily imbalance within 20 pts (sheet NIFTY tab, as of {imb_asof or 'n/a'})"]
    return hdr, rows

def main():
    import live_refresh_local as LR
    days = int(os.environ.get("NL_DAYS", "45")); sid, name, exp = nearest_future(); print("contract:", name, sid, exp)
    cs = fetch(sid, days); B, L = to_frames(cs); print(len(cs), "candles;", B["date"].nunique(), "sessions;", B["date"].min(), "->", B["date"].max())
    last = float((B["hi"].iloc[-1] + B["lo"].iloc[-1]) / 2)
    dummy = pd.DataFrame([{"t": 0, "date": "9999-12-31", "mod": 555, "hi": last, "lo": last, "buy": 0, "sell": 0, "tot": 0}])      # marks "next session", so every completed session feeds the levels
    levels, _, ds = build_levels(pd.concat([B, dummy], ignore_index=True), L); ds = [d for d in ds if d != "9999-12-31"]; sh = LR.open_sheet(); imb, asof = imbalance_refs(sh)
    hdr, rows = report(levels, last, name, imb, asof)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    sheet_put(sh, "Nifty Levels", hdr, rows + [[], ["contract", name, "last price (mid of last minute)", round(last, 1), "sessions used", len(ds), f"run {stamp}"]])
    drows = []
    for d in ds[-MONTH:]:
        va = value_area(profile(L[L["date"] == d])); b = B[B["date"] == d]
        if va: drows.append([d, round(float(b["hi"].max()), 1), round(float(b["lo"].min()), 1), round(va[0], 1), round(va[1], 1), round(va[2], 1), int(b["tot"].sum()), int((b["buy"] - b["sell"]).sum())])
    sheet_put(sh, "Nifty Daily Profile", ["date", "high", "low", "POC", "VAH", "VAL", "volume", "net delta"], drows[::-1])
    t = level_test(B, L)
    note = [[], ["reading", "react % = share of first touches where price moved 10 pts back away before 10 pts through, within 30 minutes. If 'level' is not clearly above 'control' (same prices shifted +-40/80 pts), the levels add nothing beyond noise."],
            ["data", f"{len(ds)} sessions {ds[0]} -> {ds[-1]}; levels for each day built only from earlier sessions"]]
    sheet_put(sh, "Nifty Level Test", ["set", "reacted", "broke through", "react %"], t + note)
    print("tabs written"); print(hdr); [print(r) for r in rows]; print(t)

# ───────────────────────────── self-test ─────────────────────────────
def _synth():
    rng = np.random.default_rng(3); cs = []; base = 23000.0
    for d in range(8):
        day0 = int(pd.Timestamp("2026-10-12").timestamp()) + d * 86400 - 19800; px = base + rng.normal(0, 20)
        for m in range(375):
            px += rng.normal(0, 2) + (23010 - px) * 0.03          # drifts toward 23010, so 23010 is the heavy POC
            hi, lo = px + 3, px - 3; lv = []
            for k in range(int(lo // 1), int(hi // 1) + 1, 1):
                w = np.exp(-((k - 23010) / 15) ** 2); lv.append({"price": float(k), "buy_volume": int(100 * w + 5 + rng.integers(0, 20)), "sell_volume": int(100 * w + 5 + rng.integers(0, 20))})
            cs.append({"t": day0 + (555 + m) * 60, "high": hi, "low": lo, "buy_volume": sum(x["buy_volume"] for x in lv), "sell_volume": sum(x["sell_volume"] for x in lv), "total_volume": sum(x["buy_volume"] + x["sell_volume"] for x in lv), "levels": lv})
    return cs

def selftest():
    cs = _synth(); B, L = to_frames(cs); assert B["date"].nunique() == 8, B["date"].nunique()
    g = profile(L); poc, vah, val = value_area(g); print("POC/VAH/VAL", poc, vah, val); assert abs(poc - 23010) <= 8 and val < poc < vah
    lv, last, ds = build_levels(B, L); assert lv and any(abs(c["price"] - 23010) <= 12 for c in lv), lv[:3]
    top = max(lv, key=lambda c: c["strength"]); print("top level", top); assert abs(top["price"] - 23010) <= 15
    # stacked imbalance detector: 4 bins where buys hit the bin above's sells 3:1
    rows = [("d", 0, 100 + i, 400, 100) for i in range(4)]; S = stacks(pd.DataFrame(rows, columns=["date", "blk", "bin", "buy", "sell"])); print("stacks", S); assert S and S[0][1] == "buy"
    # reaction test classification
    hi = np.array([100, 101, 105, 120, 130.0]); lo = np.array([99, 100, 99, 112, 125.0])
    assert first_touch_outcome(np.array([95., 97, 99, 101, 112, 120]), np.array([93., 95, 97, 99, 100, 111]), 100) == "break"
    assert first_touch_outcome(np.array([95., 97, 100.5, 98, 95, 90]), np.array([93., 95, 96, 92, 85, 80]), 100) == "react"
    t = level_test(B, L); print("level test", t); assert t[0][0] == "level"
    cl = cluster([(100, "a", 2), (105, "b", 2), (200, "c", 1)]); assert len(cl) == 2 and cl[0]["n"] == 2
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
