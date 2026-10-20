# ==================================================================================
# OPEN INTEREST ENGINE — step 2: UNIVERSE EVENT STUDY + HISTORY ARCHIVE
#
#  For every F&O stock (nearest expiry, 10 strikes either side of each day's open) it
#   A) SAVES the 1-minute price/volume/OI history to out/oi/archive/*.parquet   (October contracts vanish after 27 Oct)
#   B) turns each stock-session into a 5-minute tape: spot, band call-OI and put-OI changes, forward returns
#   C) finds OI EVENTS and measures what the stock did next, against honest controls:
#        E1 up    call writers EXIT during an up-spike      (FEDERALBNK 19-Oct 10:25)
#        E1 down  put writers EXIT during a down-spike      (mirror)
#        E3 up    put-writing wave while price rises       E3 down  call-writing wave while price falls
#        E2       writers on BOTH sides while price is flat -> does the stock stay quiet afterwards?
#      Control for E1/E3 = the SAME price move WITHOUT the OI signal (so we measure what OI adds, not the spike itself).
#      Uncertainty = resampling whole SESSIONS (stocks on one day move together, so events are not independent).
#   D) LAG check: inside each E1 event's 5-minute bar, did the OI exit come before, with, or after the price jump (1-min)?
#
#  Tabs: "OI Event Summary", "OI Event List", "OI Event Stocks", "OI Event Notes", "OI Zoom FED"
#  Usage: python oi_universe.py [--selftest]      env: OI_DAYS (40), OI_WORKERS (8), OI_BUDGET_MIN (45), OI_UNIVERSE_LIMIT (test only)
# ==================================================================================
import os, sys, json, time, datetime as dt
import concurrent.futures as cf
import numpy as np, pandas as pd
import oi_probe as P

BUDGET = float(os.environ.get("OI_BUDGET_MIN", "45"))
P.DEADLINE = time.time() + BUDGET * 60
OUT = P.OUT; ARCH = f"{OUT}/archive"; os.makedirs(ARCH, exist_ok=True)
DAYS = int(os.environ.get("OI_DAYS", "40")); BAND = int(os.environ.get("OI_BAND", "10"))
WORKERS = int(os.environ.get("OI_WORKERS", "8")); STEP = 5
MIN_OI_SHARE = float(os.environ.get("OI_MIN_OI_SHARE", "0.3"))     # session counts if band OI >= 30% of that stock's biggest session
MIN_ACTIVE = float(os.environ.get("OI_MIN_ACTIVE", "0.5"))         # stock needs OI changes in >= 50% of 5-min marks
R_MIN = 0.30                                                       # % price move that counts as a "spike" (5 min)
K_LIST = [2.0, 3.0, 4.0]                                           # sigma thresholds for the OI exit
HZ = ["fwd15", "fwd30", "fwd60", "fwdEOD"]
ZOOM = {}                                                          # 1-min arrays for the case stock
CASE = P.CASE
def hm(m): return f"{m // 60:02d}:{m % 60:02d}"

# ───────────────────────────── metadata ─────────────────────────────
def load_meta(universe):
    names = sorted(set(universe) | {CASE})
    body, r = P._discover_body([names[0]])
    if body is None: raise SystemExit("contract list unavailable")
    recs, lots = {}, {}
    for i in range(0, len(names), 40):
        b = dict(body); b["underlyer_list"] = names[i:i + 40]
        rr, err = P._post_meta(b)
        if rr is None or rr.status_code != 200: print("batch failed", i, err or rr.status_code); continue
        j = P.unwrap(rr.json()); recs.update(P.harvest(j))
        for u, v in ((j.get("payload") or {}).get("derivatives") or {}).items():
            try: lots[u] = int(v["default_lot_size"])
            except Exception: pass
    return P.build_chains(recs, set(names)), lots

# ───────────────────────────── per-stock processing ─────────────────────────────
def compact(panel):
    """Lossless shrink: drop minutes where nothing changed and nothing traded (re-create with forward-fill)."""
    p = panel.sort_values(["typ", "strike", "ts"]).reset_index(drop=True)
    g = p.groupby(["typ", "strike", "session"], sort=False)
    chg = (p["oi"] != g["oi"].shift()) | (p["close"] != g["close"].shift()) | (p["volume"].fillna(0) > 0) | g["ts"].transform("first").eq(p["ts"])
    out = p[chg].drop(columns=["session"]).copy()
    for c in ("strike", "close", "volume", "oi"): out[c] = out[c].astype("float32")
    out["typ"] = out["typ"].astype("category"); return out

def save(df, path):
    try: df.to_parquet(path + ".parquet", index=False, compression="zstd"); return os.path.getsize(path + ".parquet")
    except Exception:
        df.to_csv(path + ".csv.gz", index=False); return os.path.getsize(path + ".csv.gz")

def session_marks(u, s, d, sp, ks, i0, lot):
    """One session -> list of 5-min mark dicts (+ 1-min arrays)."""
    piv = d.pivot_table(index="ts", columns=["typ", "strike"], values="oi", aggfunc="last").sort_index()
    sp = sp.sort_values("ts").reset_index(drop=True); idx = pd.DatetimeIndex(sp["ts"])
    piv = piv.reindex(piv.index.union(idx)).ffill().reindex(idx).fillna(0)
    bs = ks[max(0, i0 - BAND): i0 + BAND + 1]
    cc = [("C", k) for k in bs if ("C", k) in piv.columns]; pc = [("P", k) for k in bs if ("P", k) in piv.columns]
    if not cc or not pc: return None
    ce = piv[cc].sum(axis=1).to_numpy(float); pe = piv[pc].sum(axis=1).to_numpy(float); band0 = ce[0] + pe[0]
    if band0 <= 0: return None
    c = sp["close"].to_numpy(float); mod = (idx.hour * 60 + idx.minute).to_numpy()
    def pos(m): return int(np.searchsorted(mod, m, side="right")) - 1
    rows = []
    for m in range(9 * 60 + 15 + STEP * 6, 15 * 60 + 31, STEP):            # 09:45 .. 15:30 (need 30 min of look-back)
        i = pos(m)
        if i < 30 or mod[i] < m - 3: continue
        i5, i30 = pos(m - 5), pos(m - 30)
        r = {"symbol": u, "session": s, "m": m, "spot": c[i], "band0": band0,
             "ret5": (c[i] / c[i5] - 1) * 100, "ret30": (c[i] / c[i30] - 1) * 100,
             "dce5": (ce[i] - ce[i5]) / band0 * 100, "dpe5": (pe[i] - pe[i5]) / band0 * 100,
             "dce30": (ce[i] - ce[i30]) / band0 * 100, "dpe30": (pe[i] - pe[i30]) / band0 * 100}
        for h in (15, 30, 60):
            j = int(np.searchsorted(mod, m + h, side="left"))
            r[f"fwd{h}"] = (c[j] / c[i] - 1) * 100 if j < len(mod) and mod[j] - (m + h) <= 3 else np.nan
            if h == 60:
                r["abs60"] = abs(r["fwd60"]) if r["fwd60"] == r["fwd60"] else np.nan
                seg = c[i: j + 1] if j < len(mod) else None
                r["rng60"] = (seg.max() - seg.min()) / c[i] * 100 if seg is not None and len(seg) > 10 and r["fwd60"] == r["fwd60"] else np.nan
        r["fwdEOD"] = (c[-1] / c[i] - 1) * 100 if mod[-1] >= 15 * 60 + 25 and i < len(c) - 1 else np.nan
        # 1-minute timing inside the last 5 minutes: which minute had the biggest price jump / call exit / put exit?
        a, b = i5 + 1, i + 1
        if b - a >= 3:
            ds = np.abs(np.diff(c[i5: i + 1])); dce1 = np.diff(ce[i5: i + 1]); dpe1 = np.diff(pe[i5: i + 1])
            r["j_price"], r["j_ce"], r["j_pe"] = int(np.argmax(ds)), int(np.argmin(dce1)), int(np.argmin(dpe1))
        rows.append(r)
    return rows, (mod, c, ce, pe, idx)

def process(args):
    u, chains, lot = args
    if P.out_of_time(): return u, None, None, "time budget"
    try:
        ex = chains.get(u)
        if not ex: return u, None, None, "no contracts"
        e = P.nearest_expiry(list(ex.keys())); ch = ex[e]; label = P.exp_label(e)
        ks = sorted(set(ch["CE"]) & set(ch["PE"]))
        if len(ks) < 8: return u, None, None, "few strikes"
        spot, info = P.candles(u, days=DAYS)
        if spot is None: return u, None, None, f"no spot {info.get('err')}"
        opens = spot.groupby("session")["close"].first()
        ii = [int(np.argmin(np.abs(np.array(ks) - o))) for o in opens.values]
        lo, hi = max(0, min(ii) - BAND), min(len(ks) - 1, max(ii) + BAND)
        frames = []
        for k in ks[lo: hi + 1]:
            for t in ("CE", "PE"):
                sym = ch[t].get(k)
                if not sym: continue
                df, inf = P.candles(sym, days=DAYS)
                if df is None: continue
                d = df[["ts", "session", "close", "volume", "oi"]].copy(); d["strike"] = k; d["typ"] = t[0]; frames.append(d)
        if not frames: return u, None, None, "no option candles"
        panel = pd.concat(frames, ignore_index=True)
        arch = compact(panel); size = save(arch, f"{ARCH}/{u}_{label}")
        save(spot[["ts", "open", "high", "low", "close", "volume"]], f"{ARCH}/{u}_spot")
        rows = []; sess_ok = set(panel["session"].unique())
        for s, sp in spot.groupby("session"):
            if s not in sess_ok: continue
            i0 = int(np.argmin(np.abs(np.array(ks) - float(sp["close"].iloc[0]))))
            res = session_marks(u, s, panel[panel["session"] == s], sp, ks, i0, lot)
            if res is None: continue
            rows += res[0]
            if u == CASE: ZOOM[s] = res[1]
        if not rows: return u, None, None, "no marks"
        M = pd.DataFrame(rows); M["lot"] = lot
        meta = {"symbol": u, "expiry": label, "strikes": len(ks[lo: hi + 1]), "contracts": len(frames), "archive_rows": len(arch), "archive_kb": round(size / 1e3), "sessions": M["session"].nunique()}
        return u, M, meta, "ok"
    except Exception as ex_:
        return u, None, None, f"error {type(ex_).__name__}: {str(ex_)[:100]}"

# ───────────────────────────── events + statistics ─────────────────────────────
def prepare(M):
    """Session filter, per-stock z-scores, activity filter, eligibility."""
    M = M.copy()
    mx = M.groupby("symbol")["band0"].transform("max"); M["used"] = M["band0"] >= MIN_OI_SHARE * mx
    U = M[M["used"]]
    sd = U.groupby("symbol")[["ret5", "ret30", "dce5", "dpe5", "dce30", "dpe30"]].std().replace(0, np.nan)
    for c in sd.columns: M[f"z_{c}"] = M[c] / M["symbol"].map(sd[c])
    act = U.assign(a=(U["dce5"] != 0) | (U["dpe5"] != 0)).groupby("symbol")["a"].mean()
    M["active"] = M["symbol"].map(act) >= MIN_ACTIVE
    M["elig"] = M["used"] & M["active"] & (M["m"] >= 9 * 60 + 45) & (M["m"] <= 14 * 60 + 30)
    for h in HZ:
        M[f"mkt_{h}"] = M[M["elig"]].groupby(["session", "m"])[h].transform("mean")
    return M

def cooldown(E, minutes=60):
    keep = []; last = {}
    for idx, r in E.sort_values(["symbol", "session", "m"]).iterrows():
        k = (r["symbol"], r["session"])
        if k in last and r["m"] - last[k] < minutes: continue
        last[k] = r["m"]; keep.append(idx)
    return E.loc[keep]

def boot_diff(ev, ct, col, sign_ev, sign_ct, B=500, seed=5):
    """mean(ev) - mean(ct) with session-cluster bootstrap. Returns (diff, lo, hi, ev_mean, ct_mean)."""
    se = ev.assign(v=sign_ev * ev[col]); sc = ct.assign(v=sign_ct * ct[col])
    se = se.dropna(subset=["v"]); sc = sc.dropna(subset=["v"])
    if len(se) < 5 or len(sc) < 5: return (np.nan,) * 5
    ss = sorted(set(se["session"]) | set(sc["session"])); ix = {s: i for i, s in enumerate(ss)}
    def agg(d):
        g = d.groupby("session")["v"].agg(["sum", "count"]); sm = np.zeros(len(ss)); n = np.zeros(len(ss))
        for s, r in g.iterrows(): sm[ix[s]], n[ix[s]] = r["sum"], r["count"]
        return sm, n
    es, en = agg(se); cs, cn = agg(sc); rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ss), len(ss)), minlength=len(ss))
        if (w * en).sum() > 0 and (w * cn).sum() > 0: out.append((w * es).sum() / (w * en).sum() - (w * cs).sum() / (w * cn).sum())
    d0 = se["v"].mean() - sc["v"].mean()
    return (d0, np.percentile(out, 2.5), np.percentile(out, 97.5), se["v"].mean(), sc["v"].mean()) if len(out) > 50 else (d0, np.nan, np.nan, se["v"].mean(), sc["v"].mean())

def boot_mean(ev, col, sign, mkt, B=500, seed=6):
    """event excess over the same-moment market average, session-cluster bootstrap CI."""
    d = ev.assign(v=sign * (ev[col] - ev[mkt])).dropna(subset=["v"])
    if len(d) < 5: return (np.nan,) * 3
    g = d.groupby("session")["v"].agg(["sum", "count"]); rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        w = rng.integers(0, len(g), len(g)); n = g["count"].to_numpy()[w].sum()
        if n: out.append(g["sum"].to_numpy()[w].sum() / n)
    return d["v"].mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)

def verdict(lo, hi, diff):
    if lo != lo: return "too few events"
    return "OI ADDS an edge" if lo > 0 else ("OI HURTS (reverse)" if hi < 0 else "no evidence")

def directional_rows(name, K, ev_up, ct_up, ev_dn, ct_dn):
    rows = []
    for tag, parts in (("both", [(ev_up, ct_up, 1), (ev_dn, ct_dn, -1)]), ("up", [(ev_up, ct_up, 1)]), ("down", [(ev_dn, ct_dn, -1)])):
        if tag == "both":
            ev = pd.concat([e.assign(fav=s) for e, c, s in parts]); ct = pd.concat([c.assign(fav=s) for e, c, s in parts])
        else:
            e, c, s = parts[0]; ev = e.assign(fav=s); ct = c.assign(fav=s)
        if len(ev) < 5: continue
        for h in HZ:
            sev, sct = ev["fav"].to_numpy(), ct["fav"].to_numpy()
            ev2 = ev.copy(); ct2 = ct.copy(); ev2["val"] = ev2["fav"] * ev2[h]; ct2["val"] = ct2["fav"] * ct2[h]
            d, lo, hi, em, cm = boot_diff(ev2.assign(one=1), ct2.assign(one=1), "val", 1, 1)
            x, xl, xh = boot_mean(ev, h, ev["fav"].to_numpy(), f"mkt_{h}") if False else (np.nan, np.nan, np.nan)
            ev2["vm"] = ev2["fav"] * (ev2[h] - ev2[f"mkt_{h}"]); xm = ev2["vm"].dropna()
            hit = (ev2["val"].dropna() > 0).mean() * 100
            rows.append({"event": name, "K (sigma)": K, "direction": tag, "horizon": h.replace("fwd", ""), "events": int(ev2["val"].notna().sum()),
                         "stocks": ev["symbol"].nunique(), "sessions": ev["session"].nunique(), "event mean fav %": em, "hit %": hit,
                         "same-move-no-OI mean fav %": cm, "OI adds (diff) %": d, "diff CI low": lo, "diff CI high": hi,
                         "excess vs market %": xm.mean() if len(xm) else np.nan, "verdict": verdict(lo, hi, d)})
    return rows

def study(M):
    E = M[M["elig"]]; summ = []; lists = []
    for K in K_LIST:
        eu = cooldown(E[(E["z_dce5"] <= -K) & (E["ret5"] >= R_MIN)]); cu = E[(E["ret5"] >= R_MIN) & (E["z_dce5"] > -1)]
        ed = cooldown(E[(E["z_dpe5"] <= -K) & (E["ret5"] <= -R_MIN)]); cd = E[(E["ret5"] <= -R_MIN) & (E["z_dpe5"] > -1)]
        summ += directional_rows("E1 writers EXIT during spike", K, eu, cu, ed, cd)
        if K == 3.0:
            lists += [eu.assign(event="E1 up"), ed.assign(event="E1 down")]
            lag = []
            for df, jc in ((eu, "j_ce"), (ed, "j_pe")):
                d = df.dropna(subset=["j_price", jc]); lag += list(d[jc] - d["j_price"])
            lagrow = (len(lag), np.mean([x < 0 for x in lag]) * 100 if lag else np.nan, np.mean([x == 0 for x in lag]) * 100 if lag else np.nan, np.mean([x > 0 for x in lag]) * 100 if lag else np.nan)
    e3u = cooldown(E[(E["z_dpe30"] >= 2) & (E["ret30"] >= 0.5)]); c3u = E[(E["ret30"] >= 0.5) & (E["z_dpe30"] < 1)]
    e3d = cooldown(E[(E["z_dce30"] >= 2) & (E["ret30"] <= -0.5)]); c3d = E[(E["ret30"] <= -0.5) & (E["z_dce30"] < 1)]
    summ += directional_rows("E3 writing wave WITH the move", 2.0, e3u, c3u, e3d, c3d)
    lists += [e3u.assign(event="E3 up"), e3d.assign(event="E3 down")]
    both = (E["z_dce30"] >= 1.5) & (E["z_dpe30"] >= 1.5) & (E["ret30"].abs() <= 0.3)
    e2 = cooldown(E[both]); c2 = E[(E["ret30"].abs() <= 0.3) & ~both]; lists.append(e2.assign(event="E2 both-side writing"))
    rows2 = []
    for col, lab in (("abs60", "|move| next 60m %"), ("rng60", "high-low range next 60m %")):
        d, lo, hi, em, cm = boot_diff(e2, c2, col, 1, 1)
        rows2.append({"event": "E2 both-side writing, price flat", "K (sigma)": 1.5, "direction": "n/a", "horizon": lab, "events": int(e2[col].notna().sum()),
                      "stocks": e2["symbol"].nunique(), "sessions": e2["session"].nunique(), "event mean fav %": em, "hit %": np.nan,
                      "same-move-no-OI mean fav %": cm, "OI adds (diff) %": d, "diff CI low": lo, "diff CI high": hi, "excess vs market %": np.nan,
                      "verdict": ("QUIETER after (as theory says)" if hi < 0 else ("MORE volatile after" if lo > 0 else "no evidence")) if lo == lo else "too few events"})
    S = pd.DataFrame(summ + rows2)
    L = pd.concat(lists) if lists else pd.DataFrame()
    return S, L, lagrow

# ───────────────────────────── outputs ─────────────────────────────
def zoom_table(date, a="10:10", b="10:35"):
    z = ZOOM.get(date)
    if not z: return pd.DataFrame()
    mod, c, ce, pe, idx = z; rows = []
    for i in range(1, len(mod)):
        t = hm(mod[i])
        if a <= t <= b: rows.append({"time": t, "spot": c[i], "spot chg%": (c[i] / c[i - 1] - 1) * 100, "band call OI (L)": ce[i] / 1e5, "call OI chg (L)": (ce[i] - ce[i - 1]) / 1e5,
                                    "band put OI (L)": pe[i] / 1e5, "put OI chg (L)": (pe[i] - pe[i - 1]) / 1e5})
    return pd.DataFrame(rows)

NOTES = [["term", "meaning"],
    ["scope", "every F&O stock, nearest expiry, 10 strikes either side of each day's open; sessions where band OI < 30% of the stock's biggest session are dropped (thin far-month contract)"],
    ["z / sigma", "a move measured in multiples of that stock's own typical 5-min (or 30-min) move. K=3 means: 3x a normal OI move"],
    ["E1 up", "band call OI falls by >= K sigma in 5 min while spot rises >= 0.30% (FEDERALBNK 19-Oct 10:25 is this). E1 down is the mirror with put OI and a fall"],
    ["E3", "30-min put-OI rise >= 2 sigma with price up >= 0.5% (and mirror with calls / price down)"],
    ["E2", "30-min call-OI AND put-OI rise >= 1.5 sigma with price flat (|move| <= 0.3%): writers on both sides"],
    ["control", "E1/E3: the SAME price move without the OI signal. E2: flat windows without both-side writing. So 'OI adds' is what OI gives beyond the price move"],
    ["OI adds (diff) %", "event mean minus control mean, in the trade direction. CI = 95% from resampling whole sessions"],
    ["lag", "inside each E1 5-min bar: was the biggest OI exit before, in the same minute as, or after the biggest price jump? (OI is reported with delays, so 'same minute' is common)"],
    ["caution", "~14 sessions only; many stocks but days are shared, so the CI is wide. Treat a pass as a lead worth watching, not a proven rule"]]

def main(selftest=False):
    universe = ([] if selftest else P.universe_from_sheet())
    if selftest: universe = sorted(_fake_universe())
    lim = int(os.environ.get("OI_UNIVERSE_LIMIT") or 0); universe = universe[:lim] if lim else universe
    chains, lots = load_meta(universe); names = [u for u in sorted(set(universe) | {CASE}) if u in chains]
    print(f"{len(names)} stocks with contracts; fetching history ...", flush=True)
    parts, metas, fails = [], [], []; t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for n, (u, Mx, meta, why) in enumerate(ex.map(process, [(u, chains, lots.get(u, 0)) for u in names]), 1):
            if Mx is None: fails.append((u, why))
            else: parts.append(Mx); metas.append(meta)
            if n % 10 == 0: print(f"  {n}/{len(names)} stocks ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {len(parts)} ok, {len(fails)} failed {fails[:8]}")
    if not parts: raise SystemExit("no data")
    M = prepare(pd.concat(parts, ignore_index=True)); M.to_parquet(f"{OUT}/marks.parquet", index=False) if not selftest else None
    S, L, lagrow = study(M)
    mb = sum(os.path.getsize(os.path.join(ARCH, f)) for f in os.listdir(ARCH)) / 1e6
    print(f"archive: {len(os.listdir(ARCH))} files, {mb:.0f} MB | eligible marks {int(M['elig'].sum()):,} | stocks active {M[M.active]['symbol'].nunique()}")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S[S["direction"].isin(["both", "n/a"])].round(3).to_string(index=False))
    print(f"LAG (E1, K=3): n={lagrow[0]} | OI exit BEFORE price jump {lagrow[1]:.0f}% | same minute {lagrow[2]:.0f}% | AFTER {lagrow[3]:.0f}%")
    St = pd.DataFrame(metas).merge(M.groupby("symbol").agg(sessions_used=("used", lambda s: M.loc[s.index[s.values], "session"].nunique()), oi_active=("active", "first")).reset_index().rename(columns={"symbol": "symbol"}), on="symbol", how="left")
    if len(L):
        cols = ["event", "symbol", "session", "m", "spot", "ret5", "ret30", "dce5", "dpe5", "dce30", "dpe30", "fwd15", "fwd30", "fwd60", "fwdEOD"]
        L = L[cols].copy(); L["time"] = L["m"].map(hm); L = L.drop(columns=["m"]).sort_values(["session", "time"], ascending=False)
    Z = zoom_table(os.environ.get("OI_ZOOM_DATE", "2026-10-19") if not selftest else str(max(ZOOM)) if ZOOM else "")
    if not Z.empty and not selftest: print(Z.round(2).to_string(index=False))
    for n, df in (("summary", S), ("events", L), ("stocks", St), ("zoom", Z)): df.to_csv(f"{OUT}/universe_{n}.csv", index=False)
    if selftest: return S, L, lagrow, M
    sh = P.open_sheet()
    lag_note = [["lag result (E1, K=3)", f"n={lagrow[0]}: OI exit before price jump {lagrow[1]:.0f}%, same minute {lagrow[2]:.0f}%, after {lagrow[3]:.0f}%"],
                ["archive", f"{len(os.listdir(ARCH))} files, {mb:.0f} MB in out/oi/archive (artifact + branch oi-archive)"], ["failed stocks", ", ".join(f"{u}:{w}" for u, w in fails[:30])]]
    P.put_tab(sh, "OI Event Summary", list(S.columns), S.round(3).values.tolist())
    if len(L): P.put_tab(sh, "OI Event List", list(L.columns), L.head(3000).round(3).values.tolist())
    P.put_tab(sh, "OI Event Stocks", list(St.columns), St.values.tolist())
    P.put_tab(sh, "OI Event Notes", NOTES[0], NOTES[1:] + lag_note)
    if not Z.empty: P.put_tab(sh, "OI Zoom FED", list(Z.columns), Z.round(3).values.tolist())
    print("sheet tabs written")

# ───────────────────────────── offline self-test ─────────────────────────────
def _fake_universe(): return {"FEDERALBNK": 200.0, "RELIANCE": 1300.0, "ABB": 5500.0, "BAJAJ-AUTO": 9000.0, "SBIN": 800.0, "TCS": 3500.0, "INFY": 1500.0, "ITC": 400.0}

def _fake_transport(method, url, body=None):
    U = _fake_universe()
    if "metacache" in url:
        if method != "POST": return P._R(405, text="")
        if body is None: return P._R(400, text="bad")
        if "underlyer_list" not in body: return P._R(422, text="missing field `underlyer_list`")
        ders = {}
        for u in body["underlyer_list"]:
            sp = U.get(u, 100.0); step = 5 if sp < 1000 else 50
            ks = list(range(int(sp * .8 // step * step), int(sp * 1.2), step))
            ders[u] = json.dumps({"default_lot_size": 1000, "derivatives": {"2026-10-27": {"options": {f"{k}.0": {"CE": {"tradingsymbol": f"{u}26OCT{k}CE"}, "PE": {"tradingsymbol": f"{u}26OCT{k}PE"}} for k in ks}}}})
        return P._R(200, {"success": True, "payload": {"derivatives": ders}})
    if "/candles/" in url:
        sym = url.rsplit("/", 1)[1]; m = P.OPT_RE.match(sym); u = m.group(1) if m else sym
        base = U.get(u, 100.0); rows = []; spx = base; days = [d.date() for d in pd.bdate_range(end="2026-10-19", periods=16)]
        rnd = np.random.default_rng(abs(hash(sym)) % 10**6)
        for di, d in enumerate(days):
            sr = np.random.default_rng(abs(hash((u, di))) % 10**6); jump = int(sr.integers(60, 300)); sgn = 1 if sr.random() < .5 else -1; has = sr.random() < .6
            step_r = sr.normal(0, .0005, 385)
            if has: step_r[jump] += sgn * 0.012
            path = spx * np.cumprod(1 + step_r); spx = path[-1]
            oi = 400000.0 * (0.4 if di < 4 else 1.0) if m else 0
            for i in range(385):
                t = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15 + i)
                if m:
                    k = float(m.group(4)); typ = m.group(5)
                    oi += rnd.choice([0, 0, 1000, -1000, 3000])
                    if has and i == jump: oi *= (0.9 if (typ == "CE" and sgn > 0) else (0.9 if (typ == "PE" and sgn < 0) else 1.02))
                    intr = max(path[i] - k, 0) if typ == "CE" else max(k - path[i], 0); c = intr + 3.0
                    rows.append({"ts": t.isoformat() + "+05:30", "open": c, "high": c, "low": c, "close": c, "volume": float(rnd.integers(0, 3000)), "oi": float(oi)})
                else: rows.append({"ts": t.isoformat() + "+05:30", "open": path[i], "high": path[i] * 1.0004, "low": path[i] * .9996, "close": path[i], "volume": float(rnd.integers(500, 4000))})
        return P._R(200, {"payload": {"candles": rows}})
    return P._R(404, text="nf")

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        P.TRANSPORT = _fake_transport; P.CASE = CASE
        S, L, lagrow, M = main(selftest=True)
        assert len(S) >= 6 and len(M) > 500, "too little output"
        print("SELFTEST OK |", len(S), "summary rows |", len(L), "events")
    else:
        main()
