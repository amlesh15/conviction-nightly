# ==================================================================================
# NIFTY "INSTITUTIONAL" SUPPORT / RESISTANCE — inferred from the footprints large players leave, then TESTED type by type.
#   Nobody can see who trades, so we look for the tracks big orders leave in Nifty futures footprint data (last month, 1-minute):
#     absorb    5-min bar with heavy volume, lopsided delta, but price barely moved  (a big passive order soaked it up)
#                 sellers hit, price held -> SUPPORT;  buyers lifted, price capped -> RESISTANCE
#     reject    5-min bar with a long wick on heavy volume (price pushed there and was thrown back): wick low -> SUPPORT, wick high -> RESISTANCE
#     stack     3+ adjacent price levels where one side dominates 3:1 in a 30-min block: buy stack -> SUPPORT, sell stack -> RESISTANCE
#     origin    30-min block with a big one-way move, closing at its extreme, delta agreeing, heavy volume: the block's start is SUPPORT (up-move) / RESISTANCE (down-move)
#   Events of one role are grouped into ~30-point zones. A zone is stronger the more separate DAYS it was defended and the more event types agree.
#   HONEST TEST (the point of this file): for each type alone and for all together, zones built ONLY from earlier sessions are judged on the next session.
#   A support zone is tested only when price comes DOWN into it, a resistance zone only when price comes UP into it. Compared with the same zones
#   shifted 60/120 points. A type that does not beat the shifted zones is marked as not proven.
#   OUTPUT tabs "Nifty Inst Zones", "Nifty Inst Test", "Nifty Inst Events".  Research only.
#   Usage: python nifty_inst.py [--selftest]     env: DHAN_HEADERS_JSON, GCP_SA_JSON, SHEET_ID, NL_DAYS (default 45)
# ==================================================================================
import os, sys, datetime as dt
import numpy as np, pandas as pd
import nifty_levels as NL, nifty_zones as Z

SELFTEST = "--selftest" in sys.argv
TYPES = ("absorb", "reject", "stack", "origin"); TW = {"absorb": 3, "reject": 2, "stack": 1, "origin": 2}; MONTH = NL.MONTH

# ───────────────────────────── event detection ─────────────────────────────
def bars_n(B, minutes):
    g = B.assign(k=(B["mod"] - 555) // minutes).groupby(["date", "k"], sort=True)
    A = g.agg(hi=("hi", "max"), lo=("lo", "min"), vol=("tot", "sum"), buy=("buy", "sum"), sell=("sell", "sum"), o_hi=("hi", "first"), o_lo=("lo", "first"), c_hi=("hi", "last"), c_lo=("lo", "last")).reset_index()
    A["open"] = (A["o_hi"] + A["o_lo"]) / 2; A["close"] = (A["c_hi"] + A["c_lo"]) / 2; A["delta"] = A["buy"] - A["sell"]; A["rng"] = A["hi"] - A["lo"]
    return A

def detect(B, days, stk, ref_days=None):
    """Events from sessions in `days`; thresholds from the medians of `ref_days` (the sessions known at the time). -> DataFrame date,type,role,price,vol"""
    ref = ref_days or days; ev = []
    for mins, kind in ((5, "five"), (30, "thirty")):
        A = bars_n(B, mins); R = A[A["date"].isin(ref)]
        if R.empty: continue
        mv, mr = R["vol"].median(), R["rng"].median(); X = A[A["date"].isin(days)]
        for r in X.itertuples():
            if mr <= 0 or mv <= 0: continue
            if kind == "five":
                if r.vol >= 2.5 * mv and abs(r.delta) >= 0.2 * r.vol and r.rng <= 1.0 * mr:
                    ev.append((r.date, "absorb", "support" if r.delta < 0 else "resistance", (r.hi + r.lo) / 2, r.vol))
                top = r.hi - max(r.open, r.close); bot = min(r.open, r.close) - r.lo
                if r.vol >= 1.5 * mv and r.rng >= 1.2 * mr:
                    if top >= 0.5 * r.rng: ev.append((r.date, "reject", "resistance", r.hi, r.vol))
                    if bot >= 0.5 * r.rng: ev.append((r.date, "reject", "support", r.lo, r.vol))
            else:
                move = r.close - r.open
                if r.rng >= 1.5 * mr and abs(move) >= 0.7 * r.rng and r.vol >= 1.2 * mv and np.sign(move) == np.sign(r.delta):
                    ev.append((r.date, "origin", "support" if move > 0 else "resistance", r.lo if move > 0 else r.hi, r.vol))
    for d, side, p in stk:
        if d in set(days): ev.append((d, "stack", "support" if side == "buy" else "resistance", p, 0.0))
    return pd.DataFrame(ev, columns=["date", "type", "role", "price", "vol"])

def zones_from(E, types, role):
    """Density-peak zones from events of the chosen types and role. Adds days defended / volume."""
    X = E[(E["role"] == role) & (E["type"].isin(types))]
    if X.empty: return []
    old = dict(Z.WT); Z.WT.update(TW)
    try: zs = Z.make_zones([(float(p), t) for p, t in zip(X["price"], X["type"])])
    finally: Z.WT.clear(); Z.WT.update(old)
    for z in zs:
        m = X[(X["price"] - z["ctr"]).abs() <= Z.HALF]; z["role"] = role; z["events"] = len(m); z["days"] = int(m["date"].nunique()); z["vol"] = float(m["vol"].sum()); z["types"] = sorted(m["type"].unique())
        z["score"] = z["days"] * 2 + len(z["types"]) + 0.0
    return [z for z in zs if z["days"] >= 2 or len(z["types"]) >= 2]

# ───────────────────────────── walk-forward test, per type ─────────────────────────────
def role_touches(z, off, sess):
    want = "above" if z["role"] == "support" else "below"      # support is tested when price comes DOWN into it
    return [t for t in Z.touches(z["lo"] + off, z["hi"] + off, sess) if t[1] == want and t[2]]

def walk_forward(B, L, stk, min_hist=8, topn=12):
    days = sorted(B["date"].unique()); res = {}
    for k in range(min_hist, len(days)):
        past = days[max(0, k - MONTH):k]; E = detect(B, past, stk); s = Z.sessions_of(B, [days[k]])
        for tname, types in [(t, (t,)) for t in TYPES] + [("all types", TYPES)]:
            zs = sorted(sum((zones_from(E, types, r) for r in ("support", "resistance")), []), key=lambda z: -z["score"])[:topn]
            for z in zs:
                for name, offs in ((tname, [0]), (tname + " | shifted", Z.SHIFTS)):
                    for o in offs:
                        for t in role_touches(z, o, s): res.setdefault(name, [0, 0])[0 if t[2] == "react" else 1] += 1
    rows = []
    for tname in list(TYPES) + ["all types"]:
        a = res.get(tname, [0, 0]); b = res.get(tname + " | shifted", [0, 0]); ra = a[0] / sum(a) * 100 if sum(a) else None; rb = b[0] / sum(b) * 100 if sum(b) else None
        n = sum(a); verdict = "too few touches" if n < 30 else ("beats shifted" if ra - rb >= 8 else ("no better than shifted" if ra - rb > -8 else "worse than shifted"))
        rows.append([tname, sum(a), "" if ra is None else round(ra, 1), sum(b), "" if rb is None else round(rb, 1), "" if ra is None or rb is None else round(ra - rb, 1), verdict])
    return rows

# ───────────────────────────── output ─────────────────────────────
def main():
    import live_refresh_local as LR
    sid, name, exp = NL.nearest_future(); cs = NL.fetch(sid, int(os.environ.get("NL_DAYS", "45"))); B, L = NL.to_frames(cs); ds = sorted(B["date"].unique())
    print(name, len(cs), "candles;", len(ds), "sessions", ds[0], "->", ds[-1]); stk = NL.stacks(L)
    last = float((B["hi"].iloc[-1] + B["lo"].iloc[-1]) / 2); month = ds[-MONTH:]; E = detect(B, month, stk)
    zs = sorted(sum((zones_from(E, TYPES, r) for r in ("support", "resistance")), []), key=lambda z: -z["score"])[:20]; sess = Z.sessions_of(B, month); rows = []
    for z in sorted(zs, key=lambda z: -z["ctr"]):
        tt = role_touches(z, 0, sess); r_ = sum(t[2] == "react" for t in tt)
        status = ("intact" if (z["ctr"] < last) == (z["role"] == "support") else "price has crossed it (role flipped)")
        rows.append([z["lo"], z["hi"], z["ctr"], z["role"], "below price" if z["ctr"] < last else "above price", round((z["ctr"] / last - 1) * 100, 2), status, z["days"], z["events"], ", ".join(z["types"]),
                     len(tt), r_, int(z["vol"])])
    hdr = ["zone from", "zone to", "centre", "formed as", "where now", "% from last", "status", "days defended", "events", "evidence types", "tests in month", "held", "event volume"]
    sh = LR.open_sheet(); stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    NL.sheet_put(sh, "Nifty Inst Zones", hdr, rows + [[], ["contract", name, "last price", round(last, 1), "sessions", f"{month[0]} -> {month[-1]} ({len(month)})", f"run {stamp}"],
        ["how to read", "Zones are INFERRED from footprint tracks, not seen. 'days defended' = separate sessions with events there. Check 'Nifty Inst Test' for which evidence types actually beat shifted zones before trusting any."]])
    tr = walk_forward(B, L, stk)
    NL.sheet_put(sh, "Nifty Inst Test", ["evidence type", "touches", "held %", "touches (shifted)", "held % (shifted)", "difference (pts)", "verdict"], tr +
                 [[], ["reading", f"Zones from earlier sessions only, judged on the next session. Support tested on a move DOWN into it, resistance on a move UP into it. 'held' = bounced one band width (30 pts) before trading through the band. Shifted = the same zones moved {Z.SHIFTS} pts; its held % is the chance baseline (about 50%). Difference needs to be 8+ points on 30+ touches to count as 'beats shifted'."]])
    NL.sheet_put(sh, "Nifty Inst Events", ["date", "type", "role", "price", "volume"], [[r.date, r.type, r.role, round(r.price, 1), int(r.vol)] for r in E.sort_values(["date", "price"], ascending=[False, False]).itertuples()])
    print("tabs written"); [print(r) for r in rows]; [print(r) for r in tr]

# ───────────────────────────── self-test ─────────────────────────────
def _synth(days=14, seed=2):
    rng = np.random.default_rng(seed); cs = []; px = 23100.0
    for d in range(days):
        t0 = int(pd.Timestamp("2026-10-05").timestamp()) + d * 86400 - 19800
        for m in range(375):
            px += rng.normal(0, 2.0)
            vol = 500; buy = 250; sell = 250; hi = px + 2; lo = px - 2
            if 100 <= m < 105:                         # absorption near 23000: huge sell volume, price pinned
                px = 23000.0 + rng.normal(0, 1); hi, lo = px + 1.5, px - 1.5; vol = 4000; buy = 800; sell = 3200
            lv = [{"price": float(k), "buy_volume": buy // 8, "sell_volume": sell // 8} for k in range(int(lo), int(lo) + 8)]
            cs.append({"t": t0 + (555 + m) * 60, "high": hi, "low": lo, "buy_volume": buy, "sell_volume": sell, "total_volume": vol, "levels": lv})
    return cs

def selftest():
    B, L = NL.to_frames(_synth()); ds = sorted(B["date"].unique()); stk = NL.stacks(L)
    E = detect(B, ds, stk); ab = E[(E["type"] == "absorb")]; print(len(E), "events;", len(ab), "absorb;", ab["role"].value_counts().to_dict())
    assert len(ab) >= 8 and (ab["role"] == "support").all() and (ab["price"].between(22990, 23010)).all()
    zs = zones_from(E, TYPES, "support"); print([(z["ctr"], z["days"], z["types"]) for z in zs][:4]); assert any(abs(z["ctr"] - 23000) <= 20 and z["days"] >= 8 for z in zs)
    rows = walk_forward(B, L, stk, min_hist=6); print(rows); assert rows[-1][0] == "all types"
    # no look-ahead: ruin the future, zones as of an earlier day must not change
    D = ds[9]; past = ds[:9]; z1 = zones_from(detect(B, past, stk), TYPES, "support")
    B2 = B.copy(); B2.loc[B2["date"] > D, ["hi", "lo", "tot"]] = [99999.0, 99990.0, 1.0]; z2 = zones_from(detect(B2, past, stk), TYPES, "support")
    assert [(z["ctr"], z["days"]) for z in z1] == [(z["ctr"], z["days"]) for z in z2]
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
