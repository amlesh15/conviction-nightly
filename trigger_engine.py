# ==================================================================================
# TRIGGER ENGINE (Phase 1) — order-flow PROXY on 1-min bars -> ATM option trigger + paper-trade log
#
#   Flow proxy (no L2 book available from our sources):
#     signed_vol(bar) = volume * CLV,  CLV = (2*close - high - low) / (high - low)   in [-1, +1]
#     imbalance       = sum(signed_vol, last W bars) / sum(volume, last W bars)       in [-1, +1]
#   Normalised against the SAME clock window on the previous sessions (relative volume + z-score).
#
#   A trigger needs: imbalance + relative volume + price confirmation + VWAP side, then gates from the
#   daily rule score, the live S/R flag / room to the next level, and ATM option liquidity.
#   Output: Telegram alert + row in the "Triggers" tab; later filled with 15/30/60-min & EOD outcomes,
#   and summarised in "Trigger Scorecard".  NO ORDERS ARE PLACED. FORCE=true -> rehearsal (prints only).
# ==================================================================================
import os, re, sys, html, datetime as dt
from zoneinfo import ZoneInfo
import concurrent.futures as cf
import numpy as np, pandas as pd

IST = ZoneInfo("Asia/Kolkata")

CFG = dict(
    W=20,                 # flow window, 1-min bars
    IMB_MIN=0.30,         # |imbalance| needed
    RELVOL_MIN=1.5,       # window volume vs same clock window on prior sessions
    RET_MIN=0.15,         # % price move over the window, in the flow direction
    VWAP_CONFIRM=True,    # CALL needs price >= session VWAP, PUT needs <=
    N_PRIOR=7,            # prior sessions used for normalisation
    START_MIN=9*60+30,    # no new triggers before 09:30 IST ...
    END_MIN=15*60+15,     # ... or after 15:15 IST
    MAX_PER_DAY=2,        # per symbol
    COOLDOWN_MIN=60,      # per symbol
    MAX_NEW_PER_RUN=5,
    UNIVERSE_MIN_VOLX=1.5,  # only scan stocks with a recent volume surge (Vol× column); 0 = scan all
    CALL_MIN_SCORE=-10.0, # daily rule score must not be bearish for a CALL
    PUT_MAX_SCORE=10.0,   # ... or bullish for a PUT
    ROOM_MIN=0.25,        # % room needed to the next S/R level in the trade direction
    MIN_PREMIUM=1.0,      # skip near-worthless options
    OPT_MIN_VOL15=1,      # option must have traded in the last 15 minutes
    STALE_MIN=12,         # latest bar must be this fresh
    FETCH_DAYS=12, WORKERS=8,
)
for _k, _v in list(CFG.items()):          # optional overrides: env TRIGGER_<KEY>
    _e = os.environ.get("TRIGGER_" + _k, "")
    if _e != "":
        try: CFG[_k] = (_e.lower() in ("1", "true", "yes")) if isinstance(_v, bool) else type(_v)(_e)
        except Exception: pass

OUT_COLS = ["spot15%", "spot30%", "spot60%", "spotEOD%", "opt15%", "opt30%", "opt60%", "optEOD%"]
HDR = ["date", "time", "symbol", "side", "tag", "option", "strike", "expiry", "spot", "premium", "imb", "rel_vol",
       "ret_w%", "vs_vwap%", "score", "regime", "sr_flag", "volx", "note"] + OUT_COLS

# ───────────────────────────── bars ─────────────────────────────
def _norm(df):
    d = df.copy(); ts = pd.to_datetime(d["ts"])
    if getattr(ts.dt, "tz", None) is not None: ts = ts.dt.tz_convert(IST).dt.tz_localize(None)
    d["ts"] = ts; d["session"] = ts.dt.date; d["mod"] = ts.dt.hour * 60 + ts.dt.minute
    for c in ("open", "high", "low", "close", "volume"):
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)

def _fix_volume(d):
    """If volume looks cumulative within a session (monotonic rising), convert to per-bar."""
    out = []
    for _, g in d.groupby("session", sort=True):
        v = g["volume"].fillna(0).to_numpy(float)
        if len(v) > 30 and (np.diff(v) >= 0).mean() > 0.98 and v[-1] > v[0] * 3:
            g = g.assign(volume=np.clip(np.diff(v, prepend=0.0), 0, None))
        out.append(g)
    return pd.concat(out).reset_index(drop=True) if out else d

def fetch_bars(sym):
    import intraday_fvg as F
    df = None
    for _ in range(3):
        try:
            df = F.fetch_1m(sym, days=CFG["FETCH_DAYS"], seg=getattr(F, "CASH_SEG", 1))
            if df is not None and not df.empty: break
        except Exception: df = None
    if df is None or df.empty: return sym, None
    return sym, _fix_volume(_norm(df))

# ───────────────────────────── flow proxy ─────────────────────────────
def compute_flow(df, asof=None, W=None, n_prior=None):
    W = W or CFG["W"]; n_prior = n_prior or CFG["N_PRIOR"]
    if df is None or df.empty: return None
    asof = asof if asof is not None else df["ts"].iloc[-1]
    d = df[df["ts"] <= asof]
    if d.empty: return None
    today = d["session"].iloc[-1]; cur = d[d["session"] == today]
    if len(cur) < W + 1: return None
    win = cur.tail(W)
    h = win["high"].to_numpy(float); l = win["low"].to_numpy(float); c = win["close"].to_numpy(float)
    rng = h - l; clv = np.where(rng > 0, (2 * c - h - l) / np.where(rng > 0, rng, 1.0), 0.0)
    vol = win["volume"].fillna(0).to_numpy(float); tv = vol.sum()
    if tv <= 0: return None
    imb = float((clv * vol).sum() / tv)
    ret = float((c[-1] / float(win["open"].iloc[0]) - 1) * 100)
    tp = (cur["high"] + cur["low"] + cur["close"]) / 3; v = cur["volume"].fillna(0)
    vwap = float((tp * v).sum() / v.sum()) if v.sum() > 0 else np.nan
    vs_vwap = float((c[-1] / vwap - 1) * 100) if vwap == vwap and vwap > 0 else np.nan
    m0, m1 = int(win["mod"].iloc[0]), int(win["mod"].iloc[-1])
    pv, pi = [], []
    for s in sorted(x for x in d["session"].unique() if x < today)[-n_prior:]:
        g = d[(d["session"] == s) & (d["mod"] >= m0) & (d["mod"] <= m1)]
        if len(g) < 0.8 * W: continue
        gh = g["high"].to_numpy(float); gl = g["low"].to_numpy(float); gc = g["close"].to_numpy(float)
        gr = gh - gl; gclv = np.where(gr > 0, (2 * gc - gh - gl) / np.where(gr > 0, gr, 1.0), 0.0)
        gv = g["volume"].fillna(0).to_numpy(float)
        if gv.sum() > 0: pv.append(gv.sum()); pi.append(float((gclv * gv).sum() / gv.sum()))
    rel = float(tv / np.mean(pv)) if pv and np.mean(pv) > 0 else np.nan
    z = float((imb - np.mean(pi)) / max(np.std(pi), 0.05)) if len(pi) >= 3 else np.nan
    return {"asof": asof, "spot": float(c[-1]), "imb": imb, "ret": ret, "vs_vwap": vs_vwap,
            "rel_vol": rel, "z": z, "n_prior": len(pv)}

def flow_signal(f):
    if not f or not (f["rel_vol"] == f["rel_vol"]) or f["rel_vol"] < CFG["RELVOL_MIN"]: return None
    vw = f["vs_vwap"]; vw_ok = lambda up: (not CFG["VWAP_CONFIRM"]) or (vw == vw and (vw >= 0 if up else vw <= 0))
    if f["imb"] >= CFG["IMB_MIN"] and f["ret"] >= CFG["RET_MIN"] and vw_ok(True): return "CALL"
    if f["imb"] <= -CFG["IMB_MIN"] and f["ret"] <= -CFG["RET_MIN"] and vw_ok(False): return "PUT"
    return None

# ───────────────────────────── sheet context + gates ─────────────────────────────
def _num(x):
    try: return float(str(x).replace(",", "").strip())
    except Exception: return None

def read_dashboard(ws):
    vals = ws.get_all_values()
    if not vals: return {}
    hdr = [h.strip() for h in vals[0]]; ix = {h: i for i, h in enumerate(hdr)}
    symi = ix.get("symbol", 0); strip = [i for i, h in enumerate(hdr) if re.fullmatch(r"\d{2}-\d{2}", h)]
    get = lambda row, name: row[ix[name]] if name in ix and ix[name] < len(row) else ""
    ctx = {}
    for row in vals[1:]:
        if symi >= len(row) or not row[symi].strip(): continue
        sc = None
        for i in reversed(strip):
            sc = _num(row[i]) if i < len(row) else None
            if sc is not None: break
        ctx[row[symi].strip().upper()] = {"score": sc, "volx": _num(get(row, "Vol×")), "sr": get(row, "S/R Flag").strip(),
            "regime": get(row, "Regime").strip(), "ltp": _num(get(row, "LTP")),
            "sup1": _num(get(row, "imb_sup1")), "res1": _num(get(row, "imb_res1"))}
    return ctx

def gate(side, f, c):
    """Daily-context gates. Returns (ok, tag, note)."""
    sc = c.get("score"); sr = c.get("sr", ""); spot = f["spot"]; notes = []
    if side == "CALL":
        if sc is not None and sc < CFG["CALL_MIN_SCORE"]: return False, "", f"daily score {sc:+.0f} is bearish"
        if sr.startswith("Broke S"): return False, "", f"live flag '{sr}' contradicts CALL"
        r1 = c.get("res1")
        if r1 and r1 > spot:
            room = (r1 / spot - 1) * 100; notes.append(f"R1 +{room:.1f}%")
            if room < CFG["ROOM_MIN"]: return False, "", f"only {room:.2f}% room to resistance"
        tag = "strong-aligned" if (sc is not None and sc >= 30) else "neutral-ok"
    else:
        if sc is not None and sc > CFG["PUT_MAX_SCORE"]: return False, "", f"daily score {sc:+.0f} is bullish"
        if sr.startswith("Broke R"): return False, "", f"live flag '{sr}' contradicts PUT"
        s1 = c.get("sup1")
        if s1 and s1 < spot:
            room = (spot / s1 - 1) * 100; notes.append(f"S1 -{room:.1f}%")
            if room < CFG["ROOM_MIN"]: return False, "", f"only {room:.2f}% room to support"
        tag = "strong-aligned" if (sc is not None and sc <= -30) else "neutral-ok"
    if sr: notes.append(f"flag {sr}")
    return True, tag, "; ".join(notes)

# ───────────────────────────── option ─────────────────────────────
def _candles_df(c):
    if not c: return None
    d = pd.DataFrame(c)
    if "ts" not in d.columns or "close" not in d.columns: return None
    return _norm(d)

def fetch_option_df(osym, today):
    import candle_run as CR
    seed = (today - dt.timedelta(days=3)).isoformat()
    return _candles_df(CR.fetch_candles(osym, seed, today.isoformat(), "1M"))

def pick_option(sym, side, today):
    """ATM option for the side. Returns dict (premium/vol15 may be None if option candles unavailable) or None."""
    import candle_run as CR
    ref = CR.fetch_atm_ref(sym)
    if not ref: return None
    atm, exp = ref["atm"], ref["expiry"]; typ = "CE" if side == "CALL" else "PE"
    osym = CR.build_sym(sym, exp, atm, typ); out = {"osym": osym, "strike": atm, "expiry": str(exp), "premium": None, "vol15": None}
    try:
        od = fetch_option_df(osym, today)
        if od is not None and not od.empty:
            od = od[od["session"] == od["session"].iloc[-1]]
            out["premium"] = float(od["close"].iloc[-1])
            if "volume" in od.columns: out["vol15"] = float(od["volume"].fillna(0).tail(15).sum())
    except Exception as e:
        print(f"  option candles failed for {osym}: {e}")
    return out

# ───────────────────────────── outcomes ─────────────────────────────
def _first_at_or_after(d, t):
    g = d[d["ts"] >= t]
    return g.iloc[0] if len(g) else None

def fill_outcomes(table, now, bars, opt_dfs):
    """table = [header, row, row...] from the Triggers tab. Returns {(row_idx0, col_idx0): value} for due, empty cells."""
    hdr = table[0]; ix = {h: i for i, h in enumerate(hdr)}; ups = {}
    for r, row in enumerate(table[1:], start=1):
        row = row + [""] * (len(hdr) - len(row))
        try: trig = dt.datetime.strptime(row[ix["date"]] + " " + row[ix["time"]], "%Y-%m-%d %H:%M")
        except Exception: continue
        if (now.date() - trig.date()).days > 3: continue
        side = row[ix["side"]]; spot0 = _num(row[ix["spot"]]); prem0 = _num(row[ix["premium"]])
        sym = row[ix["symbol"]]; osym = row[ix["option"]]; b = bars.get(sym); od = opt_dfs.get(osym)
        sess_over = (now.date() > trig.date()) or (now.hour * 60 + now.minute >= 15 * 60 + 31)
        for col in OUT_COLS:
            if row[ix[col]] != "": continue
            kind = "spot" if col.startswith("spot") else "opt"; hz = col[len(kind):-1]
            target = None if hz == "EOD" else trig + dt.timedelta(minutes=int(hz))
            if hz == "EOD":
                if not sess_over: continue
            elif now < target: continue
            src = b if kind == "spot" else od
            base = spot0 if kind == "spot" else prem0
            if src is None or src.empty or not base:
                if sess_over or kind == "opt" and (prem0 is None): ups[(r, ix[col])] = "n/a"
                continue
            day = src[src["session"] == trig.date()]
            if day.empty: continue
            bar = day.iloc[-1] if hz == "EOD" else _first_at_or_after(day, target)
            if bar is None:
                if sess_over: ups[(r, ix[col])] = "n/a"
                continue
            px = float(bar["close"]); pct = (px / base - 1) * 100
            if kind == "spot" and side == "PUT": pct = -pct           # favourable direction
            ups[(r, ix[col])] = round(pct, 2)
    return ups

def scorecard(table):
    if len(table) < 2: return pd.DataFrame()
    df = pd.DataFrame([r + [""] * (len(table[0]) - len(r)) for r in table[1:]], columns=table[0])
    for c in OUT_COLS: df[c] = pd.to_numeric(df[c], errors="coerce")
    groups = [("ALL", df)] + [(f"{s}", g) for s, g in df.groupby("side")] + [(f"{s} / {t}", g) for (s, t), g in df.groupby(["side", "tag"])]
    rows = []
    for name, g in groups:
        r = {"group": name, "triggers": len(g)}
        for h in ("30", "60", "EOD"):
            r[f"spot{h} avg%"] = g[f"spot{h}%"].mean(); r[f"opt{h} avg%"] = g[f"opt{h}%"].mean()
        for h in ("30", "EOD"):
            x = g[f"opt{h}%"].dropna(); r[f"opt{h} win%"] = (x > 0).mean() * 100 if len(x) else np.nan
        r["n scored (opt30)"] = int(g["opt30%"].notna().sum()); rows.append(r)
    return pd.DataFrame(rows).round(2)

# ───────────────────────────── alert ─────────────────────────────
def alert_text(side, sym, f, opt, tag, note, c, now):
    e = "🟢" if side == "CALL" else "🔴"; o = opt or {}
    optline = (f"{html.escape(str(o.get('osym', '—')))} · prem {o['premium']:.2f}" if o.get("premium") is not None
               else f"{html.escape(str(o.get('osym', '—')))} · prem n/a")
    sc = c.get("score")
    return (f"{e} <b>{side}</b> trigger · <code>{html.escape(sym)}</code>\n"
            f"Spot {f['spot']:.2f} · {optline}\n"
            f"Flow {f['imb']:+.2f} · vol {f['rel_vol']:.1f}× · {CFG['W']}m {f['ret']:+.2f}% · vsVWAP {f['vs_vwap']:+.2f}%\n"
            f"Daily score {('%+.0f' % sc) if sc is not None else '—'} ({tag}) · {html.escape(c.get('regime') or '—')}"
            + (f"\n{html.escape(note)}" if note else "") + f"\n⏱ {now:%H:%M} IST · paper-trade log, not advice")

# ───────────────────────────── main run ─────────────────────────────
def _triggers_ws(sh, create):
    import gspread
    try: return sh.worksheet("Triggers")
    except gspread.WorksheetNotFound:
        if not create: return None
        ws = sh.add_worksheet(title="Triggers", rows=2000, cols=len(HDR) + 2)
        ws.update(values=[HDR], range_name="A1")
        try: ws.freeze(rows=1)
        except Exception: pass
        return ws

def run(sh, now=None, force=False):
    now = now or dt.datetime.now(IST).replace(tzinfo=None)
    ctx = read_dashboard(sh.worksheet("Rule Dashboard"))
    ws_t = _triggers_ws(sh, create=not force)
    table = ws_t.get_all_values() if ws_t else [HDR]
    if not table: table = [HDR]
    ix = {h: i for i, h in enumerate(table[0])}

    # today's history per symbol (cooldown / daily cap)
    hist = {}
    for row in table[1:]:
        if len(row) > ix["time"] and row[ix["date"]] == now.strftime("%Y-%m-%d"):
            hist.setdefault(row[ix["symbol"]], []).append(row[ix["time"]])

    universe = [s for s, c in ctx.items() if CFG["UNIVERSE_MIN_VOLX"] <= 0 or (c["volx"] or 0) >= CFG["UNIVERSE_MIN_VOLX"]]
    open_syms = {r[ix["symbol"]] for r in table[1:] if len(r) > ix["symbol"] and r[ix["date"]] >= (now - dt.timedelta(days=3)).strftime("%Y-%m-%d")}
    fetch_list = sorted(set(universe) | open_syms)
    print(f"scanning {len(universe)} in-play symbols (+{len(open_syms - set(universe))} open-trigger symbols)")
    bars = {}
    with cf.ThreadPoolExecutor(max_workers=CFG["WORKERS"]) as ex:
        for s, d in ex.map(fetch_bars, fetch_list):
            if d is not None: bars[s] = d

    # ---- new triggers ----
    in_window = force or (CFG["START_MIN"] <= now.hour * 60 + now.minute <= CFG["END_MIN"])
    cands = []
    for s in universe:
        d = bars.get(s)
        if d is None: continue
        f = compute_flow(d)
        if not f: continue
        if not force and (now - f["asof"]).total_seconds() / 60 > CFG["STALE_MIN"]: continue
        side = flow_signal(f)
        if side: cands.append((abs(f["imb"]) * f["rel_vol"], s, side, f))
    cands.sort(reverse=True); newrows = []; sent = 0
    print(f"flow candidates: {len(cands)} | trigger window open: {in_window}")
    for _, s, side, f in cands:
        if not in_window or sent >= CFG["MAX_NEW_PER_RUN"]: break
        c = ctx[s]; ok, tag, note = gate(side, f, c)
        if not ok: print(f"  {s} {side} blocked: {note}"); continue
        h = hist.get(s, [])
        if len(h) >= CFG["MAX_PER_DAY"]: print(f"  {s} blocked: daily cap"); continue
        if h:
            last = dt.datetime.strptime(now.strftime("%Y-%m-%d") + " " + max(h), "%Y-%m-%d %H:%M")
            if (now - last).total_seconds() / 60 < CFG["COOLDOWN_MIN"]: print(f"  {s} blocked: cooldown"); continue
        try: opt = pick_option(s, side, now.date())
        except Exception as e: opt = None; print(f"  {s}: option lookup failed: {e}")
        if opt is None: print(f"  {s} blocked: no ATM reference"); continue
        if opt["premium"] is not None and opt["premium"] < CFG["MIN_PREMIUM"]: print(f"  {s} blocked: premium {opt['premium']:.2f} too low"); continue
        if opt["vol15"] is not None and opt["vol15"] < CFG["OPT_MIN_VOL15"]: print(f"  {s} blocked: option illiquid (vol15={opt['vol15']:.0f})"); continue
        if opt["premium"] is None: note = (note + "; " if note else "") + "option data n/a"
        txt = alert_text(side, s, f, opt, tag, note, c, now); print(("[REHEARSAL] " if force else "") + txt.replace("\n", " | "))
        row = {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"), "symbol": s, "side": side, "tag": tag,
               "option": opt["osym"], "strike": opt["strike"], "expiry": opt["expiry"], "spot": round(f["spot"], 2),
               "premium": round(opt["premium"], 2) if opt["premium"] is not None else "", "imb": round(f["imb"], 2),
               "rel_vol": round(f["rel_vol"], 2), "ret_w%": round(f["ret"], 2), "vs_vwap%": round(f["vs_vwap"], 2),
               "score": c["score"] if c["score"] is not None else "", "regime": c["regime"], "sr_flag": c["sr"],
               "volx": c["volx"] if c["volx"] is not None else "", "note": note}
        newrows.append([row.get(h, "") for h in HDR]); sent += 1
        if not force:
            try:
                import notify
                if notify.enabled(): notify.send(txt, parse_mode="HTML")
            except Exception as e: print("telegram failed:", e)
    if newrows and not force: ws_t.append_rows(newrows, value_input_option="USER_ENTERED")

    # ---- outcomes for earlier triggers ----
    if ws_t is not None and len(table) > 1:
        opt_dfs = {}
        for r in table[1:]:
            o = r[ix["option"]] if len(r) > ix["option"] else ""
            if o and o not in opt_dfs:
                try: opt_dfs[o] = fetch_option_df(o, now.date())
                except Exception: opt_dfs[o] = None
        ups = fill_outcomes(table, now, bars, opt_dfs)
        if ups:
            import gspread
            data = [{"range": gspread.utils.rowcol_to_a1(r + 1, c + 1), "values": [[v]]} for (r, c), v in ups.items()]
            print(f"outcomes: filling {len(ups)} cells")
            if not force:
                ws_t.batch_update(data, value_input_option="USER_ENTERED")
                for (r, c), v in ups.items():
                    table[r] = table[r] + [""] * (len(table[0]) - len(table[r])); table[r][c] = v
                sc = scorecard(table)
                if not sc.empty:
                    try: wsc = sh.worksheet("Trigger Scorecard"); wsc.clear()
                    except Exception: wsc = sh.add_worksheet(title="Trigger Scorecard", rows=60, cols=14)
                    out = sc.astype(object).where(sc.notna(), "")
                    wsc.update(values=[list(out.columns)] + out.values.tolist(), range_name="A1", value_input_option="USER_ENTERED")
    return {"new": len(newrows), "candidates": len(cands)}

def main():
    import live_refresh_local as LR
    force = os.environ.get("FORCE", "").lower() == "true"
    print("REHEARSAL (FORCE): nothing will be written or sent" if force else "live run")
    print(run(LR.open_sheet(), force=force))

if __name__ == "__main__":
    main()
