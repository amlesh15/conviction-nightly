# ==================================================================================
# OI RADAR — end-of-day watcher for UNUSUAL option open-interest changes in the stock-option MONTHLY series (default: the November expiry).
#   Not the intraday churn that follows price — it asks, for every stock and every strike: "is today's OI change out of character, and is it
#   explained by the stock's own move or by the whole market rolling positions?"  Data = NSE's daily F&O file (already cached by the nightly job):
#   OI, day's price, volume for every strike of every stock, so history can start from any past day (default 2026-10-21).
#   Per strike & day we score:
#     size vs the stock's other strikes today (robust z), share of the stock's whole OI change that landed on this one strike, ₹ size of the change,
#     the stock's chain growth vs the market's (rollover removes market-wide growth), clustering over neighbouring strikes, and whether the stock
#     barely moved (an OI move the price does not explain) or moved a lot (OI change then expected).
#   A plain-words reading is attached (call/put writing or buying, far or near the money, lid / floor / range).  The score is a RANK of oddness, not a signal.
#   Tabs: "OI Radar" (latest day, top flags), "OI Radar Stocks" (per-stock chain view), "OI Radar History" (every flag since the start + what the stock did next).
#   Usage: python oi_radar.py [--selftest]     env: GCP_SA_JSON, SHEET_ID; optional RADAR_START (2026-10-21), RADAR_MONTH (2026-11), RADAR_TOP (30), RADAR_TELEGRAM=1
# ==================================================================================
import os, sys, math, html, datetime as dt
import numpy as np, pandas as pd

SELFTEST = "--selftest" in sys.argv
START = os.environ.get("RADAR_START", "2026-10-21"); MONTH = os.environ.get("RADAR_MONTH", "2026-11"); TOP = int(os.environ.get("RADAR_TOP", "30"))
MIN_CR = 2.0          # ignore changes smaller than ₹2 crore of underlying notional (noise on thin strikes)
FLAG_AT = 5           # score needed to be listed as a flag

# ───────────────────────────── panel from NSE F&O files ─────────────────────────────
def panel_from(frames, month):
    """frames: {date: (fo_df, cash_df_or_None)} -> one row per stock-option contract of that month's expiry per day."""
    out = []
    for D, (fo, cash) in sorted(frames.items()):
        if fo is None: continue
        o = fo[fo["FinInstrmTp"] == "STO"].copy()
        if o.empty: continue
        o["exp"] = pd.to_datetime(o["XpryDt"]).dt.strftime("%Y-%m-%d"); o = o[o["exp"].str[:7] == month]
        if o.empty: continue
        o = o.sort_values("exp").drop_duplicates(["TckrSymb", "StrkPric", "OptnTp"], keep="last")           # one expiry per stock (monthly)
        d = pd.DataFrame({"date": D, "sym": o["TckrSymb"].str.strip(), "exp": o["exp"], "strike": o["StrkPric"].astype(float), "typ": o["OptnTp"], "oi": pd.to_numeric(o["OpnIntrst"], errors="coerce").fillna(0.0),
                          "px": pd.to_numeric(o["ClsPric"], errors="coerce"), "vol": pd.to_numeric(o["TtlTradgVol"], errors="coerce") if "TtlTradgVol" in o else np.nan,
                          "lot": pd.to_numeric(o["NewBrdLotQty"], errors="coerce") if "NewBrdLotQty" in o else np.nan,
                          "und": pd.to_numeric(o["UndrlygPric"], errors="coerce") if "UndrlygPric" in o else np.nan})
        if cash is not None and "CLOSE_PRICE" in cash.columns:
            cc = pd.to_numeric(cash["CLOSE_PRICE"], errors="coerce"); cc = cc[~cc.index.duplicated()]
            d["und"] = d["sym"].map(cc).fillna(d["und"])
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# ───────────────────────────── scoring one day ─────────────────────────────
def _mad_z(x):
    x = np.asarray(x, float); m = np.nanmedian(x); s = 1.4826 * np.nanmedian(np.abs(x - m))
    s = s if s > 1e-9 else (np.nanstd(x) or 1.0)
    return (x - m) / s

def score_day(cur, prev):
    """cur / prev: panel rows of two consecutive sessions. Returns (strike_table, stock_table)."""
    k = ["sym", "strike", "typ"]
    d = cur.merge(prev[k + ["oi", "px", "und"]].rename(columns={"oi": "oi0", "px": "px0", "und": "und0"}), on=k, how="left")
    d["oi0"] = d["oi0"].fillna(0.0); d["doi"] = d["oi"] - d["oi0"]
    d["lot"] = d.groupby("sym")["lot"].transform(lambda s: s.fillna(s.median()))
    d["lot"] = d["lot"].fillna(1.0).replace(0, 1.0)
    floor = 5 * d["lot"]; d["dpct"] = d["doi"] / np.maximum(d["oi0"], floor) * 100
    d["cr"] = d["doi"].abs() * d["strike"] / 1e7                                    # ₹ crore of underlying notional that changed
    d["stock_mv"] = (d["und"] / d["und0"] - 1) * 100
    d["pxchg"] = np.where(d["px0"] > 0, (d["px"] / d["px0"] - 1) * 100, np.nan)
    d["otm"] = np.where(d["typ"] == "CE", d["strike"] / d["und"] - 1, 1 - d["strike"] / d["und"]) * 100          # >0 = out of the money
    # stock chain growth vs the market (removes roll-over / market-wide positioning)
    g = d.groupby("sym").agg(oi=("oi", "sum"), oi0=("oi0", "sum"), mv=("stock_mv", "first"), tot=("doi", lambda s: s.abs().sum()))
    g["growth"] = (g["oi"] / g["oi0"].replace(0, np.nan) - 1) * 100; mg = np.nanmedian(g["growth"]); g["excess"] = g["growth"] - mg
    g["excess_z"] = _mad_z(g["excess"].fillna(0).to_numpy())
    cp = d.pivot_table(index="sym", columns="typ", values="oi", aggfunc="sum"); cp0 = d.pivot_table(index="sym", columns="typ", values="oi0", aggfunc="sum")
    g["pcr"] = (cp.get("PE") / cp.get("CE").replace(0, np.nan)).reindex(g.index); g["pcr0"] = (cp0.get("PE") / cp0.get("CE").replace(0, np.nan)).reindex(g.index); g["pcr_chg"] = g["pcr"] - g["pcr0"]
    d = d.merge(g[["tot", "excess", "excess_z", "growth"]].add_prefix("s_"), left_on="sym", right_index=True, how="left")
    d["share"] = d["doi"].abs() / d["s_tot"].replace(0, np.nan) * 100                  # % of the stock's whole OI churn that hit this strike
    # peer z: this strike's % change vs ALL stocks' strikes of the same side and distance from the money (pooled, so a handful of strikes in one stock cannot fake an outlier)
    d["bucket"] = pd.cut(d["otm"], [-1e9, -5, 0, 3, 6, 10, 20, 1e9], labels=False); d["pz"] = 0.0
    adj = (d["dpct"] - np.nanmedian(g["growth"])).clip(-500, 500)
    for (t, bk), ix in d.groupby(["typ", "bucket"]).groups.items():
        if len(ix) >= 30: d.loc[ix, "pz"] = _mad_z(adj.loc[ix].to_numpy())
    # cluster: neighbours (±2 listed strikes) moving the same way by a lot
    d = d.sort_values(["sym", "typ", "strike"]); d["cl"] = 0
    for (s, t), ix in d.groupby(["sym", "typ"]).groups.items():
        sub = d.loc[ix]; sgn = np.sign(sub["doi"].to_numpy()); big = (sub["cr"].to_numpy() >= MIN_CR / 2) & (np.abs(sub["dpct"].to_numpy()) >= 15); n = len(sub); cl = np.zeros(n, int)
        for i in range(n):
            if not big[i]: continue
            cl[i] = int(sum(1 for j in range(max(0, i - 2), min(n, i + 3)) if j != i and big[j] and sgn[j] == sgn[i]))
        d.loc[ix, "cl"] = cl
    # points
    pts = np.zeros(len(d)); why = [[] for _ in range(len(d))]
    def add(mask, p, text):
        for i in np.where(mask.to_numpy() if hasattr(mask, "to_numpy") else mask)[0]: pts[i] += p; why[i].append(text)
    add(d["pz"] >= 4, 3, "far bigger % change than the stock's other strikes"); add((d["pz"] >= 3) & (d["pz"] < 4), 2, "much bigger % change than the stock's other strikes")
    add(d["share"] >= 40, 3, "one strike took a big share of the stock's OI churn"); add((d["share"] >= 25) & (d["share"] < 40), 2, "one strike took a large share of the stock's OI churn")
    add(d["cr"] >= 20, 2, "large ₹ size"); add((d["cr"] >= 5) & (d["cr"] < 20), 1, "sizeable ₹ amount")
    add(d["cl"] >= 1, 1, "neighbouring strikes moved the same way"); add(d["s_excess_z"] >= 2.5, 1, "this stock's whole chain grew more than the market's")
    add(d["stock_mv"].abs() < 1, 1, "stock barely moved, so price does not explain it"); add(d["stock_mv"].abs() >= 3, -1, "stock moved a lot, so some OI change is expected")
    d["score"] = pts; d["why"] = ["; ".join(w) for w in why]
    d = d[(d["cr"] >= MIN_CR) & (d["oi"] + d["oi0"] > 0)]
    d["read"] = d.apply(reading, axis=1)
    return d.sort_values("score", ascending=False), g

def reading(r):
    ce = r["typ"] == "CE"; up = r["doi"] > 0; far = r["otm"] >= 7; near = abs(r["otm"]) <= 2; side = "call" if ce else "put"
    where = "far out of the money" if far else ("near the current price" if near else ("out of the money" if r["otm"] > 0 else "in the money"))
    pm = r["pxchg"]; sm = r["stock_mv"]
    if not up: return f"{side.title()} OI removed {where} ({r['dpct']:.0f}%): positions at {r['strike']:g} are being closed — " + ("a lid is being lifted." if ce else "a floor is being pulled.")
    # premium rose/fell by more than the stock move alone would explain (crude delta)
    delta = (min(max(0.5 - 2.5 * (r["strike"] / r["und"] - 1), 0.05), 0.95) if ce else -min(max(0.5 + 2.5 * (r["strike"] / r["und"] - 1), 0.05), 0.95)) if r["und"] and r["und"] == r["und"] else 0.5
    exp_chg = delta * (sm if sm == sm else 0) * r["und"] / max(r["px0"], 0.05) if r["px0"] and r["px0"] == r["px0"] else 0
    excess = (pm - exp_chg) if pm == pm else np.nan
    if excess == excess and excess <= 0:
        return (f"New {side} WRITING {where} at {r['strike']:g}: sellers expect the stock to stay {'below' if ce else 'above'} it — " + ("a lid is being built." if ce else "a floor is being built.") + (" Far-dated, so a longer-term view." if far else ""))
    if excess == excess and excess > 0:
        return (f"New {side} BUYING {where} at {r['strike']:g}: " + ("a bet on a rise / upside hedge" if ce else "a bet on a fall / downside hedge") + " — the option got more expensive while OI rose.")
    return f"{side.title()} OI added {where} at {r['strike']:g}; price data does not say whether it is writing or buying."

# ───────────────────────────── run ─────────────────────────────
def flags_for(P):
    dates = sorted(P["date"].unique()); out = []; stocks = {}
    for a, b in zip(dates[:-1], dates[1:]):
        s, g = score_day(P[P["date"] == b], P[P["date"] == a]); stocks[b] = g; s = s[s["score"] >= FLAG_AT].copy(); s["date"] = b; out.append(s)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), stocks, dates

def forward_moves(F, P, dates):
    """What the stock did after each flag: close-to-close move to the next 1 / 3 / 5 sessions (known only for old flags)."""
    px = P.drop_duplicates(["date", "sym"]).pivot(index="date", columns="sym", values="und"); idx = list(px.index)
    for h in (1, 3, 5):
        vals = []
        for _, r in F.iterrows():
            i = idx.index(r["date"]) if r["date"] in idx else None
            vals.append(round((px.iloc[i + h][r["sym"]] / px.iloc[i][r["sym"]] - 1) * 100, 2) if i is not None and i + h < len(idx) and r["sym"] in px.columns else "")
        F[f"stock +{h}d %"] = vals
    return F

def tables(F, stocks, P, dates):
    last = dates[-1]; L = F[F["date"] == last].sort_values("score", ascending=False).head(TOP)
    lot = lambda r: r["lot"] if r["lot"] == r["lot"] and r["lot"] > 0 else 1
    H = ["rank", "symbol", "expiry", "strike", "side", "% from price", "OI now (lots)", "OI change (lots)", "OI change %", "₹ cr changed", "option price chg %", "stock move %", "score", "why flagged", "plain reading"]
    exp = P.drop_duplicates("sym").set_index("sym")["exp"].to_dict(); rows = []
    for n, (_, r) in enumerate(L.iterrows(), 1):
        rows.append([n, r["sym"], exp.get(r["sym"], ""), r["strike"], r["typ"], round(r["otm"], 1), round(r["oi"] / lot(r)), round(r["doi"] / lot(r)), round(r["dpct"]), round(r["cr"], 1), "" if r["pxchg"] != r["pxchg"] else round(r["pxchg"], 1), "" if r["stock_mv"] != r["stock_mv"] else round(r["stock_mv"], 2), r["score"], r["why"], r["read"]])
    g = stocks.get(last)
    SH = ["symbol", "chain OI (lakh units)", "chain OI change %", "vs market (pts)", "put/call ratio", "PCR change", "stock move %", "flags today"]; fc = L.groupby("sym").size().to_dict() if len(L) else {}
    srows = []
    if g is not None:
        for sym, r in g.sort_values("excess", ascending=False).iterrows():
            srows.append([sym, round(r["oi"] / 1e5, 1), "" if r["growth"] != r["growth"] else round(r["growth"], 1), "" if r["excess"] != r["excess"] else round(r["excess"], 1), "" if r["pcr"] != r["pcr"] else round(r["pcr"], 2), "" if r["pcr_chg"] != r["pcr_chg"] else round(r["pcr_chg"], 2), "" if r["mv"] != r["mv"] else round(r["mv"], 2), fc.get(sym, 0)])
    FF = forward_moves(F.copy(), P, dates); HH = ["date", "symbol", "strike", "side", "% from price", "₹ cr changed", "OI change %", "score", "plain reading", "stock +1d %", "stock +3d %", "stock +5d %"]
    hrows = [[r["date"], r["sym"], r["strike"], r["typ"], round(r["otm"], 1), round(r["cr"], 1), round(r["dpct"]), r["score"], r["read"], r["stock +1d %"], r["stock +3d %"], r["stock +5d %"]] for _, r in FF.sort_values(["date", "score"], ascending=[False, False]).iterrows()]
    return (H, rows), (SH, srows), (HH, hrows), L

def telegram(L, last):
    tok = (os.environ.get("EMA_TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_TOKEN") or "").strip(); chat = (os.environ.get("RADAR_TELEGRAM_CHAT_ID") or os.environ.get("EMA_TELEGRAM_CHAT_ID") or "").strip()
    if not tok or not chat or L.empty: print("telegram: not configured or nothing to send"); return
    import urllib.parse, urllib.request
    top = L.head(8); txt = f"<b>OI Radar · {last}</b> — most unusual stock-option OI changes (rank of oddness, not a signal)\n\n" + "\n\n".join(f"<b>{html.escape(r['sym'])}</b> {r['strike']:g} {r['typ']} · ₹{r['cr']:.0f}cr · score {r['score']:.0f}\n{html.escape(r['read'])}" for _, r in top.iterrows())
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": txt[:3900], "parse_mode": "HTML", "disable_web_page_preview": "true"}).encode()
        urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=data), timeout=20); print("telegram sent")
    except Exception as e: print("telegram failed:", e)

def main():
    import run_nightly as RN, gspread, live_refresh_local as LR
    G = RN.load_engine(); RN.seed_cache(); frames = {}
    d = pd.to_datetime(START); end = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()
    while d <= end:
        if d.weekday() < 5:
            D = d.strftime("%Y-%m-%d"); fo = G["get_fo_bhav"](D)
            if fo is not None and len(fo): frames[D] = (fo, G["get_cash_bhav"](D))
        d += pd.Timedelta(days=1)
    print(f"F&O files for {len(frames)} sessions: {sorted(frames)[:1]} .. {sorted(frames)[-1:]}")
    if len(frames) < 2: raise SystemExit("need at least two sessions of F&O data")
    P = panel_from(frames, MONTH)
    if P.empty: raise SystemExit(f"no stock options found for expiry month {MONTH}")
    print(f"{P['sym'].nunique()} stocks, {P['date'].nunique()} sessions, {len(P):,} contract-days for the {MONTH} series")
    F, stocks, dates = flags_for(P); print(f"{len(F)} flags in total across {len(dates) - 1} day-pairs")
    (H, rows), (SH, srows), (HH, hrows), L = tables(F, stocks, P, dates)
    for r in rows[:10]: print(" | ".join(str(x) for x in r[:8]), "|", r[-1])
    sh = LR.open_sheet()
    def put(title, header, body):
        def clean(v):
            try:
                import math
                if v is None: return ""
                if hasattr(v, "item"): v = v.item()
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return ""
                if hasattr(v, "isoformat"): return str(v)[:10]
            except Exception: pass
            return v
        vals = [[clean(x) for x in r] for r in [header] + body]
        try: ws = sh.worksheet(title); ws.clear()
        except gspread.WorksheetNotFound: ws = sh.add_worksheet(title=title, rows=len(vals) + 20, cols=len(header) + 2)
        if ws.row_count < len(vals) + 5: ws.resize(rows=len(vals) + 20)
        ws.update(values=vals, range_name="A1", value_input_option="USER_ENTERED")
    put("OI Radar", H, rows); put("OI Radar Stocks", SH, srows); put("OI Radar History", HH, hrows); print("tabs written")
    if os.environ.get("RADAR_TELEGRAM", "").lower() in ("1", "true", "yes"): telegram(L, dates[-1])

# ───────────────────────────── self-test ─────────────────────────────
def _synth():
    rng = np.random.default_rng(7); syms = [f"S{i:02d}" for i in range(30)]; strikes = np.arange(80, 125, 5.0); days = ["2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27"]; frames = {}
    base = {(s, k, t): rng.integers(200, 900) * 5000.0 * (1.6 - abs(k - 100) / 40) for s in syms for k in strikes for t in ("CE", "PE")}
    for n, D in enumerate(days):
        rows = []
        for (s, k, t), v in list(base.items()):
            roll = 1 + 0.08 * n                                     # market-wide roll-over growth, same for everyone (must NOT flag)
            oi = v * roll * (1 + rng.normal(0, 0.03))
            if n == 4 and s == "S07" and k == 115 and t == "CE": oi = v * roll * 6.0          # planted anomaly: one far-OTM call strike x6 while the stock is flat
            rows.append(("STO", s, "2026-11-24", k, t, oi, max(0.5, 100 * 0.04 + (k - 100) * (-.02 if t == "CE" else .02)) + (0.0 if n < 4 else (-0.2 if (s, k, t) == ("S07", 115.0, "CE") else 0)), 100 + n * 0.1 * (1 if s != "S03" else 30), 100.0))
        fo = pd.DataFrame(rows, columns=["FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric", "OptnTp", "OpnIntrst", "ClsPric", "UndrlygPric", "NewBrdLotQty"]); frames[D] = (fo, None)
    return frames

def selftest():
    P = panel_from(_synth(), "2026-11"); assert P["sym"].nunique() == 30 and P["date"].nunique() == 5
    F, stocks, dates = flags_for(P); last = F[F["date"] == dates[-1]].sort_values("score", ascending=False)
    top = last.iloc[0]; assert top["sym"] == "S07" and top["strike"] == 115 and top["typ"] == "CE", top[["sym", "strike", "typ", "score"]].to_dict()
    assert "WRITING" in top["read"] or "writing" in top["read"].lower(), top["read"]
    assert len(F[F["date"] != dates[-1]]) <= 2, f"roll-over days should not flag much: {len(F[F['date'] != dates[-1]])}"
    assert len(last) <= 5, f"too many flags on the anomaly day: {len(last)}"
    (H, rows), (SH, srows), (HH, hrows), L = tables(F, stocks, P, dates); assert rows[0][1] == "S07" and len(srows) == 30
    print("SELFTEST OK — planted anomaly ranks #1:", rows[0][1], rows[0][3], rows[0][4], "| reading:", rows[0][-1][:80])

if __name__ == "__main__":
    selftest() if SELFTEST else main()
