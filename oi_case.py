# ==================================================================================
# OPEN INTEREST ENGINE — step 1: CASE STUDY (default FEDERALBNK, nearest expiry, 10 strikes either side of the open)
#
#  Pulls spot + every option contract 1-min (price + OI), then answers:
#   "OI FED Windows"  one session, many time windows (open->10:00, open->10:30, 10:00-10:30, 30-min blocks ...):
#                     spot move, OI change on calls/puts, who is doing what (writing / covering / long build / unwinding),
#                     a net 'OI pressure' number, and where the call wall / put wall is.
#   "OI FED Strikes"  the same windows strike by strike (the heat-map view)
#   "OI FED 5min"     5-minute tape for the session: spot, cumulative OI change, pressure
#   "OI FED Days"     one row per session (all ~20 sessions we can see)
#   "OI FED Test"     does OI pressure line up with — or LEAD — the stock's next move, across ALL sessions?
#   "OI FED Notes"    definitions
#
#  How "who is doing what" is inferred (we cannot see traders, only OI and the option's own price):
#     OI up + option price down  = WRITING (new shorts)          OI up + price up   = LONG BUILD (new buyers)
#     OI down + option price up  = SHORT COVERING (writers leave) OI down + price down = LONG UNWIND (buyers leave)
#   Bullish: CE short-cover, CE long-build, PE writing, PE long-unwind.   Bearish: the other four.
#   pressure = bullish OI moved - bearish OI moved   (lakh shares; 1 lakh = 100,000 shares)
#
# Usage: python oi_case.py [--selftest]      env: OI_CASE, OI_DATE (YYYY-MM-DD, default latest session), OI_WINDOWS ("10:00-10:30,11:50-12:30")
# ==================================================================================
import os, sys, json, time, datetime as dt
import concurrent.futures as cf
import numpy as np, pandas as pd
import oi_probe as P

P.DEADLINE = time.time() + float(os.environ.get("OI_BUDGET_MIN", "20")) * 60
CASE = os.environ.get("OI_CASE", P.CASE)
BAND = int(os.environ.get("OI_BAND", "10"))                 # strikes each side of the opening price
DAYS = int(os.environ.get("OI_DAYS", "40"))                 # calendar days of history (API max 60)
OUT = P.OUT
TRANSPORT_DAYS = None
MIN_OI_SHARE = float(os.environ.get("OI_MIN_OI_SHARE", "0.3"))   # session counts only if band OI >= 30% of the biggest session

BULL = {"CE short cover", "CE long build", "PE writing", "PE long unwind"}
BEAR = {"CE writing", "CE long unwind", "PE short cover", "PE long build"}
CLASSES = ["CE writing", "CE short cover", "CE long build", "CE long unwind", "PE writing", "PE short cover", "PE long build", "PE long unwind"]
L = 1e5   # lakh

def hm(s): h, m = s.split(":"); return int(h) * 60 + int(m)
def fmt_hm(x): return f"{x // 60:02d}:{x % 60:02d}"

# ───────────────────────────── data ─────────────────────────────
def pull(case):
    body, r = P._discover_body([case])
    if body is None: raise SystemExit("contract list unavailable")
    js = P.unwrap(r.json()); recs = P.harvest(js)
    chains = P.build_chains(recs, {case}); exps = chains.get(case, {})
    e = P.nearest_expiry(list(exps.keys()))
    if e is None: raise SystemExit("no contracts for " + case)
    lot = 2500
    try: lot = int(js["payload"]["derivatives"][case]["default_lot_size"])
    except Exception:
        r2, _ = P.http("GET", P.META + f"{case}{P.exp_label(e)}{int(sorted(exps[e]['CE'])[len(exps[e]['CE'])//2]):d}CE")
        try: lot = int(r2.json()["payload"]["lot_size"])
        except Exception: pass
    syms = [s for t in ("CE", "PE") for s in exps[e][t].values()]
    print(f"{case}: expiry {P.exp_label(e)}, {len(syms)} contracts, lot {lot}", flush=True)
    spot, info = P.candles(case, days=DAYS)
    if spot is None: raise SystemExit(f"no spot candles: {info}")
    frames = []
    with cf.ThreadPoolExecutor(max_workers=P.WORKERS) as ex:
        for sym, df, inf in ex.map(lambda s: P.fetch_contract(s, DAYS), syms):
            if df is None: print("  no data:", sym, inf.get("err")); continue
            u, yy, mi, k, t = P.parse_sym(sym)
            d = df[["ts", "session", "close", "volume", "oi"]].copy(); d["strike"] = k; d["typ"] = t; frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(f"{OUT}/{case}_{P.exp_label(e)}_panel.csv.gz", index=False)
    spot[["ts", "session", "open", "high", "low", "close", "volume"]].to_csv(f"{OUT}/{case}_spot.csv.gz", index=False)
    return panel, spot, lot, P.exp_label(e)

class Session:
    """One trading day: aligned minute grids of OI / option price / volume for the strike band."""
    def __init__(self, panel, spot, s, band=BAND):
        d = panel[panel["session"] == s]
        mk = lambda col: d.pivot_table(index="ts", columns=["typ", "strike"], values=col, aggfunc="last").sort_index()
        self.oi = mk("oi"); self.px = mk("close").ffill(); self.vol = mk("volume").fillna(0)
        self.oi = self.oi.ffill().fillna(0)
        sp = spot[spot["session"] == s].set_index("ts").sort_index()
        self.idx = self.oi.index
        self.spot = sp["close"].reindex(self.idx, method="ffill").ffill().bfill()
        self.s = s
        both = sorted(set(self.oi["CE"].columns) & set(self.oi["PE"].columns)) if "CE" in self.oi and "PE" in self.oi else []
        open_px = float(sp["close"].iloc[0]) if len(sp) else float(self.spot.iloc[0])
        i = int(np.argmin([abs(k - open_px) for k in both])) if both else 0
        self.strikes = both[max(0, i - band): i + band + 1]; self.open_px = open_px
        self.hi = float(sp["high"].max()) if len(sp) else np.nan; self.lo = float(sp["low"].min()) if len(sp) else np.nan
    def pos(self, hhmm):
        t = pd.Timestamp(f"{self.s} {hhmm}")
        return max(0, int(self.idx.searchsorted(t, side="right")) - 1)

def klass(side, doi, dpx, vol, min_oi):
    if vol <= 0 or abs(doi) < min_oi or dpx == 0 or dpx != dpx: return ""
    return f"{side} " + {(1, 0): "writing", (1, 1): "long build", (0, 1): "short cover", (0, 0): "long unwind"}[(int(doi > 0), int(dpx > 0))]

def window(S, a, b, lot, min_lots=2):
    """a,b = 'HH:MM'. Returns (per-strike DataFrame, summary dict)."""
    i0, i1 = S.pos(a), S.pos(b); rows = []; tot = {c: 0.0 for c in CLASSES}
    for k in S.strikes:
        r = {"strike": k}
        for t in ("CE", "PE"):
            if (t, k) not in S.oi.columns: continue
            oi0, oi1 = float(S.oi[(t, k)].iloc[i0]), float(S.oi[(t, k)].iloc[i1])
            p0, p1 = float(S.px[(t, k)].iloc[i0]) if (t, k) in S.px else np.nan, float(S.px[(t, k)].iloc[i1]) if (t, k) in S.px else np.nan
            v = float(S.vol[(t, k)].iloc[i0 + 1: i1 + 1].sum()) if (t, k) in S.vol else 0.0
            doi = oi1 - oi0; c = klass(t, doi, p1 - p0, v, min_lots * lot)
            r.update({f"{t} OI": oi1 / L, f"{t} dOI": doi / L, f"{t} px": p1, f"{t} dpx%": (p1 / p0 - 1) * 100 if p0 and p0 == p0 and p0 >= 0.5 else np.nan, f"{t} vol": v / L, f"{t} class": c})
            if c: tot[c] += abs(doi) / L
        rows.append(r)
    T = pd.DataFrame(rows)
    bull = sum(tot[c] for c in BULL); bear = sum(tot[c] for c in BEAR)
    ce_cols = [("CE", k) for k in S.strikes if ("CE", k) in S.oi.columns]; pe_cols = [("PE", k) for k in S.strikes if ("PE", k) in S.oi.columns]
    band_oi0 = (S.oi[ce_cols].iloc[i0].sum() + S.oi[pe_cols].iloc[i0].sum()) / L if ce_cols and pe_cols else np.nan
    def wall(cols, i): return float(S.oi[cols].iloc[i].idxmax()[1]) if cols else np.nan
    top = sorted(((v, c) for c, v in tot.items() if v > 0), reverse=True)[:2]
    sp0, sp1 = float(S.spot.iloc[i0]), float(S.spot.iloc[i1])
    dce = float(S.oi[ce_cols].iloc[i1].sum() - S.oi[ce_cols].iloc[i0].sum()) / L if ce_cols else np.nan
    dpe = float(S.oi[pe_cols].iloc[i1].sum() - S.oi[pe_cols].iloc[i0].sum()) / L if pe_cols else np.nan
    summ = {"window": f"{a}-{b}", "spot start": sp0, "spot end": sp1, "spot chg%": (sp1 / sp0 - 1) * 100,
            "CE dOI (L sh)": dce, "PE dOI (L sh)": dpe, **{f"{c} (L)": tot[c] for c in CLASSES},
            "bullish OI (L)": bull, "bearish OI (L)": bear, "PRESSURE (L)": bull - bear,
            "pressure % of band OI": (bull - bear) / band_oi0 * 100 if band_oi0 else np.nan, "band OI at start (L)": band_oi0,
            "call wall start→end": f"{wall(ce_cols, i0):g}→{wall(ce_cols, i1):g}", "put wall start→end": f"{wall(pe_cols, i0):g}→{wall(pe_cols, i1):g}",
            "story": " + ".join(f"{c} {v:.1f}L" for v, c in top) if top else "quiet"}
    return T, summ

# ───────────────────────────── outputs ─────────────────────────────
def default_windows(extra):
    cum = ["09:30", "10:00", "10:30", "11:00", "11:50", "12:30", "13:30", "15:30"]
    W = [("cum", "09:15", c) for c in cum]
    W += [("block", fmt_hm(hm("09:15") + 30 * k), fmt_hm(hm("09:15") + 30 * (k + 1))) for k in range(12)]
    for w in (extra or "").split(","):
        w = w.strip()
        if "-" in w: a, b = w.split("-"); W.append(("custom", a, b))
    return W

def tape(S, lot, step=5):
    rows = []; i0 = S.pos("09:15"); prev = None
    for m in range(hm("09:15") + step, hm("15:30") + 1, step):
        a, b = fmt_hm(m - step), fmt_hm(m)
        _, s_blk = window(S, a, b, lot); _, s_cum = window(S, "09:15", b, lot)
        rows.append({"time": b, "spot": s_blk["spot end"], "5m chg%": s_blk["spot chg%"], "since-open chg%": s_cum["spot chg%"],
                     "CE dOI 5m (L)": s_blk["CE dOI (L sh)"], "PE dOI 5m (L)": s_blk["PE dOI (L sh)"], "PRESSURE 5m (L)": s_blk["PRESSURE (L)"],
                     "CE dOI since open (L)": s_cum["CE dOI (L sh)"], "PE dOI since open (L)": s_cum["PE dOI (L sh)"], "PRESSURE since open (L)": s_cum["PRESSURE (L)"],
                     "story 5m": s_blk["story"]})
    return pd.DataFrame(rows)

def days_table(sessions, lot):
    rows = []; blocks = []
    for S in sessions:
        if len(S.strikes) < 5 or len(S.idx) < 100: continue
        _, d = window(S, "09:15", "15:30", lot)
        rows.append({"session": str(S.s), "band OI at open (L)": d["band OI at start (L)"], "open": S.open_px, "close": d["spot end"], "day chg%": (d["spot end"] / S.open_px - 1) * 100,
                     "high vs open%": (S.hi / S.open_px - 1) * 100, "low vs open%": (S.lo / S.open_px - 1) * 100,
                     "CE dOI (L)": d["CE dOI (L sh)"], "PE dOI (L)": d["PE dOI (L sh)"], "PRESSURE (L)": d["PRESSURE (L)"],
                     "call wall": d["call wall start→end"], "put wall": d["put wall start→end"], "story": d["story"]})
        for k in range(12):
            a, b = fmt_hm(hm("09:15") + 30 * k), fmt_hm(hm("09:15") + 30 * (k + 1))
            _, w = window(S, a, b, lot)
            b0 = w["band OI at start (L)"] or np.nan
            blocks.append({"session": str(S.s), "band": d["band OI at start (L)"], "k": k, "ret": w["spot chg%"], "pressure": w["pressure % of band OI"],
                           "ce": w["CE dOI (L sh)"] / b0 * 100, "pe": w["PE dOI (L sh)"] / b0 * 100})
    D = pd.DataFrame(rows); B = pd.DataFrame(blocks)
    # Sessions before the previous expiry only have this contract as a thin FAR-month series (tiny OI). Keep sessions whose
    # band OI is at least MIN_OI_SHARE of the biggest session; those are the ones where this contract was the live one.
    if len(D):
        thr = MIN_OI_SHARE * D["band OI at open (L)"].max()
        D["used in test"] = np.where(D["band OI at open (L)"] >= thr, "yes", "no (thin far-month OI)")
        B = B[B["session"].isin(D.loc[D["used in test"] == "yes", "session"])].drop(columns=["band"], errors="ignore")
    return D, B

def _resid(y, z):
    A = np.column_stack([np.ones(len(z)), z]); beta = np.linalg.lstsq(A, y, rcond=None)[0]; return y - A @ beta

def _corrs(d, f):
    """d has session, ret, next_ret and feature f. Everything is de-meaned WITHIN each session first (removes whole-day drift
    and whole-day OI level effects), then: coincident corr, lead corr, and lead corr after removing what the stock's own
    same-block move already explains ('does OI add anything price hasn't already told us?')."""
    d = d.dropna(subset=[f, "ret", "next_ret"]).copy()
    if len(d) < 20: return (np.nan, np.nan, np.nan)
    for c in (f, "ret", "next_ret"): d[c] = d[c] - d.groupby("session")[c].transform("mean")
    x, r, y = d[f].to_numpy(float), d["ret"].to_numpy(float), d["next_ret"].to_numpy(float)
    if x.std() == 0 or y.std() == 0: return (np.nan, np.nan, np.nan)
    same = np.corrcoef(x, r)[0, 1]; lead = np.corrcoef(x, y)[0, 1]
    rx, ry = _resid(x, r), _resid(y, r)
    part = np.corrcoef(rx, ry)[0, 1] if rx.std() > 0 and ry.std() > 0 else np.nan
    return (same, lead, part)

def lead_test(B, boot=400):
    """Across every session x 30-min block. Blocks inside one day are NOT independent, so uncertainty comes from
    resampling whole SESSIONS (cluster bootstrap), not from n = number of blocks."""
    if B.empty: return pd.DataFrame()
    B = B.sort_values(["session", "k"]).copy(); B["next_ret"] = B.groupby("session")["ret"].shift(-1)
    B["ce_drop"] = -B["ce"]; B["put_minus_call"] = B["pe"] - B["ce"]
    feats = [("pressure (uses option price => circular with same-block move)", "pressure"),
             ("call OI change % of band", "ce"), ("put OI change % of band", "pe"),
             ("call writers LEAVING (-call OI change)", "ce_drop"), ("put build minus call build", "put_minus_call")]
    sess = B["session"].unique(); rng = np.random.default_rng(11); groups = {s: B[B["session"] == s] for s in sess}; out = []
    for name, f in feats:
        same, lead, part = _corrs(B, f); bs = []
        for _ in range(boot):
            pick = rng.choice(sess, size=len(sess), replace=True)
            d = pd.concat([groups[s].assign(session=f"{s}#{i}") for i, s in enumerate(pick)])
            bs.append(_corrs(d, f)[2])
        bs = np.array([v for v in bs if v == v])
        lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if len(bs) > 20 else (np.nan, np.nan)
        verdict = "no evidence beyond price" if (lo != lo or lo <= 0 <= hi) else ("OI LEADS (adds to price)" if lo > 0 else "OI leads the OTHER way")
        out.append({"feature": name, "sessions": len(sess), "blocks": int(B[[f, "ret", "next_ret"]].dropna().shape[0]),
                    "corr w/ same-block move": round(same, 3), "corr w/ NEXT block": round(lead, 3),
                    "NEXT block after removing price momentum": round(part, 3), "95% CI low": round(lo, 3), "95% CI high": round(hi, 3), "verdict": verdict})
    return pd.DataFrame(out)

NOTES = [
    ["term", "meaning"],
    ["lakh (L)", "100,000 shares. FEDERALBNK lot = 2,500 shares, so 1 L = 40 lots"],
    ["band", f"{BAND} strikes either side of the session's opening price (fixed for that day)"],
    ["CE writing", "call OI up while the call's price falls => new call sellers (caps the stock: bearish/resistance)"],
    ["CE short cover", "call OI down while price rises => call writers closing (bullish)"],
    ["CE long build", "call OI up and price up => new call buyers (bullish)"],
    ["CE long unwind", "call OI down and price down => call buyers leaving (bearish-ish)"],
    ["PE writing", "put OI up while price falls => new put sellers (support: bullish)"],
    ["PE short cover", "put OI down while price rises => put writers closing (bearish)"],
    ["PE long build", "put OI up and price up => new put buyers (bearish)"],
    ["PE long unwind", "put OI down and price down => put buyers leaving (bullish-ish)"],
    ["PRESSURE", "bullish OI moved minus bearish OI moved, in lakh shares. + means OI flows favour upside"],
    ["pressure % of band OI", "PRESSURE divided by total band OI at window start — comparable across days"],
    ["call wall / put wall", "strike with the largest call OI / put OI (acts as resistance / support)"],
    ["OI FED Test", "uses ALL sessions. Everything is de-meaned inside each day, then we ask: does this OI measure predict the NEXT 30-min move even after removing what the stock's own move already explains? 95% CI comes from resampling whole sessions. 'pressure' is partly circular with the same-block move (it uses the option's price, which follows the stock) — read only its NEXT-block row"],
    ["sessions used", "OI FED Days column 'used in test': early sessions saw this contract as a thin far-month series (tiny OI), so they are left out of the cross-day test"],
    ["caveat", "we infer intent from OI + option price; option prices also move with the stock, IV and time decay. Min move to count = 2 lots"],
]

def main(selftest=False):
    panel, spot, lot, elabel = pull(CASE)
    sess_all = sorted(panel["session"].unique())
    sessions = []
    for s in sess_all:
        try: sessions.append(Session(panel, spot, s))
        except Exception as e: print("session skipped", s, e)
    target = dt.date.fromisoformat(os.environ["OI_DATE"]) if os.environ.get("OI_DATE") else sessions[-1].s
    S = next((x for x in sessions if x.s == target), sessions[-1]); print(f"case session: {S.s} | band strikes {S.strikes[0]:g}..{S.strikes[-1]:g} | open {S.open_px}")
    W = default_windows(os.environ.get("OI_WINDOWS", "")); rows = []; strike_rows = []
    for kind, a, b in W:
        T, summ = window(S, a, b, lot); rows.append({"type": kind, **summ})
        T = T.copy(); T.insert(0, "window", f"{a}-{b}"); T.insert(1, "spot chg%", summ["spot chg%"]); strike_rows.append(T)
    WIN = pd.DataFrame(rows); STR = pd.concat(strike_rows, ignore_index=True); TAPE = tape(S, lot)
    DAYS_T, BLK = days_table(sessions, lot); TEST = lead_test(BLK)
    for n, df in (("windows", WIN), ("strikes", STR), ("5min", TAPE), ("days", DAYS_T), ("test", TEST)): df.to_csv(f"{OUT}/{CASE}_{n}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(WIN[["type", "window", "spot chg%", "CE dOI (L sh)", "PE dOI (L sh)", "PRESSURE (L)", "story"]].round(2).to_string(index=False))
    print(TEST.to_string(index=False))
    if selftest: return WIN, STR, TAPE, DAYS_T, TEST
    sh = P.open_sheet(); tag = {"windows": WIN, "strikes": STR, "5min": TAPE, "days": DAYS_T, "test": TEST}
    short = {"FEDERALBNK": "FED"}.get(CASE, CASE)
    for n, df in tag.items():
        if len(df): P.put_tab(sh, f"OI {short} {n.capitalize() if n != '5min' else '5min'}", list(df.columns), df.round(3).values.tolist())
    P.put_tab(sh, f"OI {short} Notes", NOTES[0], NOTES[1:] + [["session shown", str(S.s)], ["expiry", elabel], ["lot size", lot]])
    print("sheet tabs written for", short)

# ───────────────────────────── offline self-test ─────────────────────────────
def _fake_transport_factory():
    base = P._fake_transport
    def tr(method, url, body=None):
        if "/candles/" in url:
            sym = url.rsplit("/", 1)[1]; m = P.OPT_RE.match(sym); rnd = np.random.default_rng(abs(hash(sym)) % 10**6)
            days = [d.date() for d in pd.bdate_range(end="2026-10-19", periods=12)]; rows = []
            spx = 200.0
            for di, d in enumerate(days):
                srng = np.random.default_rng(di + 7); drift = srng.normal(0, .0004)
                path = spx * np.cumprod(1 + drift + srng.normal(0, .0006, 385)); spx = path[-1]
                px0 = 8.0; oi = 100000 + di * 1000 if m else 0
                for i in range(385):
                    t = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15 + i)
                    if m:
                        k = float(m.group(4)); typ = m.group(5)
                        intr = max(path[i] - k, 0) if typ == "CE" else max(k - path[i], 0); c = intr + 3.0 + abs(rnd.normal(0, .05))
                        oi += rnd.choice([0, 0, 2500, -2500, 5000]); rows.append({"ts": t.isoformat() + "+05:30", "open": c, "high": c, "low": c, "close": c, "volume": float(rnd.integers(0, 3000)), "oi": float(oi)})
                    else: rows.append({"ts": t.isoformat() + "+05:30", "open": path[i], "high": path[i] * 1.0004, "low": path[i] * .9996, "close": path[i], "volume": float(rnd.integers(500, 4000))})
            return P._R(200, {"payload": {"candles": rows}})
        return base(method, url, body)
    return tr

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        P.TRANSPORT = _fake_transport_factory(); P.WORKERS = 3
        WIN, STR, TAPE, D, TEST = main(selftest=True)
        assert len(WIN) >= 20 and len(TAPE) > 60 and len(D) >= 10 and len(TEST) >= 4, "outputs too small"
        print("SELFTEST OK")
    else:
        main()
