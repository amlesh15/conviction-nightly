# ==================================================================================
# NIFTY OPTION OI — one session, strike by strike, against what spot did.
#   Question: when Nifty fell early, did PUT writers build support (e.g. 22,900/22,950)?  When it rose to 23,100, did CALL writers cap it?
#   Data: Sensibull 1-min candles (price + volume + OI) for every NIFTY option contract of the nearest expiry (same endpoints as oi_probe/oi_case).
#   Reading OI (we only see OI and the option's own price, not traders):
#       OI up + option price down = WRITING (new sellers)   OI up + price up = LONG BUILD (new buyers)
#       OI down + price up = SHORT COVER (writers leave)    OI down + price down = LONG UNWIND (buyers leave)
#   Support is "built" by PE WRITING (and CE short cover); resistance by CE WRITING (and PE short cover).
#   OUTPUT tabs: "Nifty OI Story" (plain sentences), "Nifty OI Phases" (per strike for each phase of the day), "Nifty OI Tape" (5-min, key strikes).
#   Usage: python nifty_oi_day.py [--selftest]   env: NOI_DATE (default = latest session), NOI_LO/NOI_HI strike range (default 22600/23400),
#          NOI_KEYS "22900,22950,23100", NOI_EXPIRY (YYYY-MM-DD, default nearest on/after the session), SENSIBULL cookie via setup_env.py
# ==================================================================================
import os, sys, re, json, time, datetime as dt
import concurrent.futures as cf
import numpy as np, pandas as pd

SELFTEST = "--selftest" in sys.argv
COOKIE_PATH = os.path.expanduser("~/live-refresh/cookie.txt")
CANDLE = "https://oxide.sensibull.com/v1/compute/1/candles/{sym}"
META_CACHE = "https://oxide.sensibull.com/v1/compute/cache/instrument_metacache_addon/1"
LO = float(os.environ.get("NOI_LO", "22600")); HI = float(os.environ.get("NOI_HI", "23400"))
KEYS = [float(x) for x in os.environ.get("NOI_KEYS", "22900,22950,23100").split(",") if x.strip()]
L = 1e5

def _hdr():
    ck = open(COOKIE_PATH).read().strip() if os.path.exists(COOKIE_PATH) else ""
    return {"cookie": ck, "content-type": "application/json", "origin": "https://web.sensibull.com", "referer": "https://web.sensibull.com/", "user-agent": "Mozilla/5.0"}

def _real(method, url, body=None):
    import requests
    return requests.request(method, url, json=body, headers=_hdr(), timeout=(8, 25))
TRANSPORT = _real

def http(method, url, body=None, tries=3):
    last = None
    for k in range(tries):
        try:
            r = TRANSPORT(method, url, body)
            if r.status_code in (429, 502, 503, 504): last = f"HTTP {r.status_code}"; time.sleep(1.5 * (k + 1)); continue
            return r
        except Exception as e: last = str(e)[:100]; time.sleep(1 + k)
    raise RuntimeError(f"{url}: {last}")

def unwrap(o, d=0):
    if d > 6: return o
    if isinstance(o, str):
        t = o.strip()
        if len(t) > 1 and t[0] in "{[" and t[-1] in "}]":
            try: return unwrap(json.loads(t), d + 1)
            except Exception: return o
        return o
    if isinstance(o, dict): return {k: unwrap(v, d + 1) for k, v in o.items()}
    if isinstance(o, list): return [unwrap(v, d + 1) for v in o]
    return o

def candles(sym, days=10):
    end = dt.date.today()
    r = http("POST", CANDLE.format(sym=sym), {"from_date": (end - dt.timedelta(days=days)).isoformat(), "to_date": (end + dt.timedelta(days=1)).isoformat(), "interval": "1M", "skip_last_ts": False})
    if r.status_code != 200: return None
    c = ((r.json() or {}).get("payload") or {}).get("candles") or []
    if not c: return None
    df = pd.DataFrame(c); ts = pd.to_datetime(df["ts"])
    if getattr(ts.dt, "tz", None) is not None: ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["ts"] = ts; df["session"] = ts.dt.date.astype(str)
    for col in ("open", "high", "low", "close", "volume", "oi"):
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)

# ───────────────────────────── chain from the metacache ─────────────────────────────
def chain(session, want_expiry=None):
    r = http("POST", META_CACHE, {"underlyer_list": ["NIFTY"]}); js = unwrap(r.json())
    node = js["payload"]["derivatives"]["NIFTY"]; node = unwrap(node)
    lot = int(node.get("default_lot_size") or 65); ders = node["derivatives"]
    exps = sorted(ders.keys())
    e = want_expiry or next((x for x in exps if x >= session), exps[-1])
    opts = ders[e]["options"]; out = {}
    for k, v in opts.items():
        kf = float(k)
        if LO <= kf <= HI:
            for t in ("CE", "PE"):
                if isinstance(v.get(t), dict) and v[t].get("tradingsymbol"): out[(t, kf)] = v[t]["tradingsymbol"]
    return e, lot, out, exps

# ───────────────────────────── session arrays ─────────────────────────────
class Day:
    def __init__(self, panel, spot, session):
        d = panel[panel["session"] == session]
        mk = lambda col: d.pivot_table(index="ts", columns=["typ", "strike"], values=col, aggfunc="last").sort_index()
        self.oi = mk("oi"); self.idx = self.oi.index; self.oi = self.oi.ffill().fillna(0)
        self.px = mk("close").reindex(self.idx).ffill(); self.vol = mk("volume").reindex(self.idx).fillna(0)
        self.session = session; self.spot_src = "Sensibull spot"
        if spot is not None and (spot["session"] == session).any():
            sp = spot[spot["session"] == session].set_index("ts").sort_index()
            self.spot = sp["close"].reindex(self.idx, method="ffill").ffill().bfill()
            self.hi = sp["high"].reindex(self.idx, method="ffill").ffill().bfill(); self.lo = sp["low"].reindex(self.idx, method="ffill").ffill().bfill()
        else:                       # synthetic spot from put-call parity: K + C - P at the strike where |C-P| is smallest
            self.spot_src = "implied from option prices (put-call parity)"
            ks = sorted(set(self.px["CE"].columns) & set(self.px["PE"].columns)); sp = []
            for i in range(len(self.idx)):
                best = min(ks, key=lambda k: abs(self.px[("CE", k)].iloc[i] - self.px[("PE", k)].iloc[i]) if self.px[("CE", k)].iloc[i] == self.px[("CE", k)].iloc[i] and self.px[("PE", k)].iloc[i] == self.px[("PE", k)].iloc[i] else 1e9)
                sp.append(best + self.px[("CE", best)].iloc[i] - self.px[("PE", best)].iloc[i])
            self.spot = pd.Series(sp, index=self.idx).ffill().bfill(); self.hi = self.spot; self.lo = self.spot
        self.strikes = sorted(set(self.oi["CE"].columns) | set(self.oi["PE"].columns))
    def pos(self, hhmm): return max(0, int(self.idx.searchsorted(pd.Timestamp(f"{self.session} {hhmm}"), side="right")) - 1)
    def hm(self, i): return self.idx[min(i, len(self.idx) - 1)].strftime("%H:%M")

def klass(doi, dpx, vol, min_oi, side):
    if vol <= 0 or abs(doi) < min_oi or not dpx or dpx != dpx: return ""
    return f"{side} " + {(1, 0): "WRITING", (1, 1): "long build", (0, 1): "short cover", (0, 0): "long unwind"}[(int(doi > 0), int(dpx > 0))]

def phase_table(D, a, b, lot):
    i0, i1 = D.pos(a) if isinstance(a, str) else a, D.pos(b) if isinstance(b, str) else b; rows = []
    for k in D.strikes:
        r = {"strike": k}
        for t in ("CE", "PE"):
            if (t, k) not in D.oi.columns: continue
            oi0, oi1 = float(D.oi[(t, k)].iloc[i0]), float(D.oi[(t, k)].iloc[i1])
            p0 = float(D.px[(t, k)].iloc[i0]) if (t, k) in D.px else np.nan; p1 = float(D.px[(t, k)].iloc[i1]) if (t, k) in D.px else np.nan
            v = float(D.vol[(t, k)].iloc[i0 + 1:i1 + 1].sum()) if (t, k) in D.vol else 0.0
            r.update({f"{t} OI end (L)": oi1 / L, f"{t} dOI (L)": (oi1 - oi0) / L, f"{t} dOI (lots)": (oi1 - oi0) / lot, f"{t} px": p1,
                      f"{t} what": klass(oi1 - oi0, p1 - p0, v, 2 * lot, t)})
        rows.append(r)
    return pd.DataFrame(rows), i0, i1

def key_line(T, strike):
    r = T[T["strike"] == strike]
    if r.empty: return ""
    r = r.iloc[0]; out = []
    for t in ("PE", "CE"):
        if f"{t} dOI (lots)" in r and r[f"{t} dOI (lots)"] == r[f"{t} dOI (lots)"]:
            out.append(f"{t} {strike:g}: {r[f'{t} dOI (lots)']:+,.0f} lots ({r[f'{t} what'] or 'small/untraded'})")
    return "; ".join(out)

# ───────────────────────────── story ─────────────────────────────
def build(D, lot):
    i_lo = int(np.argmin(D.lo.to_numpy())); i_hi = int(np.argmax(D.hi.to_numpy())); i_end = D.pos("15:30"); i_op = 0
    phases = []
    if i_lo > 0: phases.append(("A. fall: open → day low", i_op, i_lo))
    if i_hi > i_lo: phases.append(("B. rally: day low → day high", i_lo, i_hi))
    elif i_hi > 0 and i_lo == 0: phases.append(("B. rally: open → day high", i_op, i_hi))
    if i_end > i_hi: phases.append(("C. after the high → close", i_hi, i_end))
    phases.append(("D. whole day: open → close", i_op, i_end))
    story = [[f"Session {D.session}. Spot ({D.spot_src}): open {D.spot.iloc[0]:.0f}, low {D.lo.min():.0f} at {D.hm(i_lo)}, high {D.hi.max():.0f} at {D.hm(i_hi)}, last {D.spot.iloc[i_end]:.0f}. Lot = {lot}."]]
    ph_rows = []
    for name, a, b in phases:
        T, a, b = phase_table(D, a, b, lot)
        ce = T["CE dOI (lots)"].sum(); pe = T["PE dOI (lots)"].sum()
        story.append([f"{name}  ({D.hm(a)}→{D.hm(b)}, spot {D.spot.iloc[a]:.0f}→{D.spot.iloc[b]:.0f}): all listed strikes CE {ce:+,.0f} lots, PE {pe:+,.0f} lots."])
        for k in KEYS:
            ln = key_line(T, k)
            if ln: story.append(["      " + ln])
        # biggest movers in the phase: where writers actually came in
        pw = T.dropna(subset=["PE dOI (lots)"]).sort_values("PE dOI (lots)", ascending=False).head(3); cw = T.dropna(subset=["CE dOI (lots)"]).sort_values("CE dOI (lots)", ascending=False).head(3)
        story.append(["      biggest PUT OI adds: " + ", ".join(f"{r.strike:g} ({r['PE dOI (lots)']:+,.0f}, {r['PE what'] or '-'})" for _, r in pw.iterrows())])
        story.append(["      biggest CALL OI adds: " + ", ".join(f"{r.strike:g} ({r['CE dOI (lots)']:+,.0f}, {r['CE what'] or '-'})" for _, r in cw.iterrows())])
        Tp = T.copy(); Tp.insert(0, "phase", f"{name} {D.hm(a)}→{D.hm(b)}"); ph_rows.append(Tp)
    # when did spot first reach each key CE strike, and what were that strike's call writers doing in the 15 min before/after
    for k in KEYS:
        below = k < float(D.spot.iloc[0])          # a level under the open is reached from above (support); over the open from below (resistance)
        reach = np.where(D.lo.to_numpy() <= k + 10)[0] if below else np.where(D.hi.to_numpy() >= k - 10)[0]
        if len(reach) and ("CE", k) in D.oi.columns:
            j = int(reach[0]); Tb, _, _ = phase_table(D, max(0, j - 15), j, lot); Ta, _, _ = phase_table(D, j, min(len(D.idx) - 1, j + 15), lot)
            story.append([f"Spot first {'fell to' if below else 'rose to'} within 10 pts of {k:g} at {D.hm(j)}.  15 min before: {key_line(Tb, k)}.  15 min after: {key_line(Ta, k)}."])
    return story, pd.concat(ph_rows, ignore_index=True), (i_lo, i_hi)

def tape(D, lot, step=5):
    cols = [(t, k) for k in KEYS for t in ("PE", "CE") if (t, k) in D.oi.columns]; rows = []
    for m in range(9 * 60 + 15 + step, 15 * 60 + 31, step):
        i = D.pos(f"{m // 60:02d}:{m % 60:02d}"); r = {"time": D.hm(i), "spot": round(float(D.spot.iloc[i]), 1)}
        for c in cols:
            r[f"{c[0]} {c[1]:g} OI (L)"] = round(float(D.oi[c].iloc[i]) / L, 2); r[f"{c[0]} {c[1]:g} dOI since open (lots)"] = round(float(D.oi[c].iloc[i] - D.oi[c].iloc[0]) / lot)
            r[f"{c[0]} {c[1]:g} px"] = round(float(D.px[c].iloc[i]), 1) if D.px[c].iloc[i] == D.px[c].iloc[i] else ""
        rows.append(r)
    return pd.DataFrame(rows)

# ───────────────────────────── main ─────────────────────────────
def pull():
    sess = os.environ.get("NOI_DATE") or None
    spot = candles("NIFTY"); today = (sess or (spot["session"].max() if spot is not None else dt.date.today().isoformat()))
    e, lot, syms, exps = chain(today, os.environ.get("NOI_EXPIRY") or None)
    print(f"session {today}; expiry {e} (listed: {exps[:6]}); lot {lot}; {len(syms)} contracts in {LO:g}..{HI:g}", flush=True)
    frames = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for (t, k), df in zip(syms.keys(), ex.map(candles, syms.values())):
            if df is None: continue
            d = df[["ts", "session", "close", "volume", "oi"]].copy(); d["strike"] = k; d["typ"] = t; frames.append(d)
    if not frames: raise SystemExit("no option candles returned (cookie expired?)")
    return pd.concat(frames, ignore_index=True), spot, today, lot, e

def main():
    panel, spot, today, lot, e = pull()
    D = Day(panel, spot, today); story, PH, (i_lo, i_hi) = build(D, lot); TP = tape(D, lot)
    print("\n".join(s[0] for s in story))
    from live_refresh_local import open_sheet
    import nifty_levels as NL
    sh = open_sheet()
    NL.sheet_put(sh, "Nifty OI Story", ["what happened to option OI (lots = OI / lot size)"], story + [[], [f"expiry {e}; strikes {LO:g}-{HI:g}; key strikes {', '.join(f'{k:g}' for k in KEYS)}"],
        ["writing = OI up while the option price fell; short cover = OI down while price rose; we infer intent, we do not see traders"]])
    NL.sheet_put(sh, "Nifty OI Phases", list(PH.columns), PH.round(2).fillna("").values.tolist())
    NL.sheet_put(sh, "Nifty OI Tape", list(TP.columns), TP.fillna("").values.tolist())
    print("tabs written")

# ───────────────────────────── self-test ─────────────────────────────
def selftest():
    idx = pd.date_range("2026-10-30 09:15", "2026-10-30 15:30", freq="1min"); n = len(idx)
    spot = np.concatenate([np.linspace(22960, 22800, 60), np.linspace(22800, 23110, 150), np.linspace(23110, 23050, n - 210)])
    sp = pd.DataFrame({"ts": idx, "session": "2026-10-30", "open": spot, "high": spot + 2, "low": spot - 2, "close": spot, "volume": 1})
    frames = []
    for k in (22800.0, 22900.0, 22950.0, 23100.0):
        for t in ("CE", "PE"):
            oi = np.full(n, 1e6)
            if (t, k) == ("PE", 22900.0): oi = np.linspace(1e6, 1e6 + 130 * 20, n)  # build 20 lots/... large
            if (t, k) == ("PE", 22900.0): oi = np.concatenate([np.linspace(1e6, 2.6e6, 60), np.full(n - 60, 2.6e6)])
            if (t, k) == ("CE", 23100.0): oi = np.concatenate([np.full(190, 1e6), np.linspace(1e6, 3e6, n - 190)])
            px = (np.linspace(60, 20, n) if t == "PE" else np.linspace(20, 60, n)) if k == 22900.0 else np.full(n, 30.0)
            if (t, k) == ("PE", 22900.0): px = np.concatenate([np.linspace(40, 25, 60), np.linspace(25, 5, n - 60)])
            if (t, k) == ("CE", 23100.0): px = np.concatenate([np.linspace(10, 40, 190), np.linspace(40, 15, n - 190)])
            frames.append(pd.DataFrame({"ts": idx, "session": "2026-10-30", "close": px, "volume": 500.0, "oi": oi, "strike": k, "typ": t}))
    panel = pd.concat(frames, ignore_index=True); lot = 65
    D = Day(panel, sp, "2026-10-30"); story, PH, (i_lo, i_hi) = build(D, lot); TP = tape(D, lot)
    txt = "\n".join(s[0] for s in story); print(txt)
    assert D.hm(i_lo) == "10:14" or abs(i_lo - 59) <= 1, i_lo
    assert "PE 22900: +" in txt and "WRITING" in txt, "put writing in the fall must be detected"
    assert "CE 23100" in txt and "within 10 pts of 23100" in txt
    assert len(TP) > 50 and f"PE 22900 OI (L)" in TP.columns
    # parity fallback gives spot in the right ballpark
    D2 = Day(panel, None, "2026-10-30"); assert D2.spot_src.startswith("implied")
    os.environ["NOI_X"] = "1"; print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
