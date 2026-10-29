# ==================================================================================
# NIFTY ZONES — the strongest 20 support / resistance ZONES for Nifty futures, each one PROVEN by what price actually did there.
#   1 CANDIDATES  prices where something happened (footprint + price): 1-month volume peaks, weekly value-area edges, stacked buy/sell imbalances,
#                 absorption minutes, swing highs/lows (30-min pivots), yesterday's high/low.
#   2 ZONES       candidates within MERGE points become one band of +-HALF points around their weighted centre (a ~30-point band).
#   3 PROOF       replay every 1-minute bar of the month. A TOUCH = price enters the band after being outside it (from above = a support test,
#                 from below = a resistance test), then within 60 minutes either REACTS (moves REACT points back away before breaking REACT points
#                 through the far edge) or BREAKS. Counted per zone with avg bounce size and the last time it worked.
#   4 FILTER      kept only if it has real evidence: Proven (>=3 touches, >=65% reacted), Tested (>=2 touches, >=60%), or Footprint-only (not yet
#                 tested but >=3 different evidence types agree; shown last and labelled). Ranked by reactions - 2*breaks + evidence + recency. Top 20.
#   5 CHECK       walk-forward test: zones built only from earlier sessions are judged on the NEXT session and compared with the same zones shifted
#                 +-60/120 points. This is the honest answer to "does it mean anything".
#   OUTPUT tabs "Nifty Zones" and "Nifty Zone Test".  Research only.
#   Usage: python nifty_zones.py [--selftest]      env: DHAN_HEADERS_JSON, GCP_SA_JSON, SHEET_ID, NL_DAYS (default 45)
# ==================================================================================
import os, sys, datetime as dt
import numpy as np, pandas as pd
import nifty_levels as NL

HALF = 15.0; MERGE = 25.0; REACT = 25.0; LOOK = 60; REARM = 30.0; TOP = 20; SHIFTS = (-120, -60, 60, 120)
SELFTEST = "--selftest" in sys.argv
WT = {"HVN": 2, "weekPOC": 2, "weekVAH": 2, "weekVAL": 2, "buyStack": 1, "sellStack": 1, "absorb": 1, "swingHigh": 2, "swingLow": 2, "prevHigh": 1, "prevLow": 1, "monthPOC": 3}

# ───────────────────────────── candidates ─────────────────────────────
def swings(B, n=3):
    """30-minute pivots across the whole series: [(date, 'swingHigh'|'swingLow', price)]."""
    g = B.assign(blk=(B["mod"] - 555) // 30).groupby(["date", "blk"], sort=True).agg(hi=("hi", "max"), lo=("lo", "min")).reset_index()
    hi = g["hi"].to_numpy(float); lo = g["lo"].to_numpy(float); out = []
    for i in range(n, len(g) - n):
        if hi[i] == hi[i - n:i + n + 1].max(): out.append((g["date"][i], "swingHigh", float(hi[i])))
        if lo[i] == lo[i - n:i + n + 1].min(): out.append((g["date"][i], "swingLow", float(lo[i])))
    return out

def candidates(B, L, upto, pre):
    """Evidence from sessions < upto (list of dates). pre = precomputed stacks/absorption/swings with dates."""
    days = [d for d in sorted(B["date"].unique()) if d < upto]
    if len(days) < 3: return [], days
    month = days[-NL.MONTH:]; items = []
    g = NL.profile(L[L["date"].isin(month)])
    va = NL.value_area(g)
    if va: items.append((va[0], "monthPOC"))
    items += [(p, "HVN") for p in NL.hvns(g)]
    for w in range(0, len(month), 5):
        ds = month[w:w + 5]
        if len(ds) >= 3:
            v = NL.value_area(NL.profile(L[L["date"].isin(ds)]))
            if v: items += [(v[0], "weekPOC"), (v[1], "weekVAH"), (v[2], "weekVAL")]
    ok = set(month)
    items += [(p, s) for d, s, p in pre["swing"] if d in ok]
    items += [(p, s + "Stack") for d, s, p in pre["stack"] if d in ok]
    items += [(p, "absorb") for d, p in pre["absorb"] if d in ok]
    yb = B[B["date"] == days[-1]]; items += [(float(yb["hi"].max()), "prevHigh"), (float(yb["lo"].min()), "prevLow")]
    return items, days

def make_zones(items):
    it = sorted(items); cl = []
    for p, s in it:
        if cl and p - cl[-1]["last"] <= MERGE: cl[-1]["m"].append((p, s)); cl[-1]["last"] = p
        else: cl.append({"m": [(p, s)], "last": p})
    zs = []
    for c in cl:
        w = [WT.get(s, 1) for _, s in c["m"]]; ctr = sum(p * x for (p, _), x in zip(c["m"], w)) / sum(w); src = sorted({s for _, s in c["m"]})
        zs.append({"ctr": round(ctr, 1), "lo": round(ctr - HALF, 1), "hi": round(ctr + HALF, 1), "src": src, "n": len(src)})
    return zs

# ───────────────────────────── proof: replay touches ─────────────────────────────
def touches(zlo, zhi, sess):
    """sess = list of (date, hi_array, lo_array). Returns list of (date, approach 'above'|'below', 'react'|'break'|None, bounce_points)."""
    out = []
    for d, hi, lo in sess:
        side = None; armed = True; n = len(hi); i = 0
        while i < n:
            if lo[i] > zhi: side = "above"; armed = armed or lo[i] > zhi + REARM
            elif hi[i] < zlo: side = "below"; armed = armed or hi[i] < zlo - REARM
            elif armed and side:
                res, mfe = None, 0.0
                for j in range(i, min(n, i + LOOK)):
                    if side == "above":
                        dn = lo[j] <= zlo - REACT; up = hi[j] >= zhi + REACT; mfe = max(mfe, hi[j] - zhi)
                    else:
                        dn = lo[j] <= zlo - REACT; up = hi[j] >= zhi + REACT; mfe = max(mfe, zlo - lo[j])
                    if side == "above": hit_react, hit_break = up, dn
                    else: hit_react, hit_break = dn, up
                    if hit_break: res = "break"; break          # same-bar ambiguity counts as a break (conservative)
                    if hit_react: res = "react"; break
                out.append((d, side, res, mfe)); armed = False
            i += 1
    return out

def sessions_of(B, days):
    return [(d, B.loc[B["date"] == d, "hi"].to_numpy(float), B.loc[B["date"] == d, "lo"].to_numpy(float)) for d in days]

def stats(z, sess, ndays):
    ts = [t for t in touches(z["lo"], z["hi"], sess) if t[2]]; r = [t for t in ts if t[2] == "react"]; b = [t for t in ts if t[2] == "break"]
    sup = [t for t in ts if t[1] == "above"]; res = [t for t in ts if t[1] == "below"]
    rs = sum(t[2] == "react" for t in sup); rr = sum(t[2] == "react" for t in res)
    role = "Support" if rs > rr else ("Resistance" if rr > rs else ("Both" if rs else "Untested"))
    recent = sorted({t[0] for t in r})[-1:] or [""]; last_days = [s[0] for s in sess[-5:]]
    bonus = 2 if recent[0] in last_days else 0
    z.update({"touches": len(ts), "react": len(r), "brk": len(b), "rate": len(r) / len(ts) if ts else None, "bounce": float(np.mean([t[3] for t in r])) if r else 0.0,
              "role": role, "last_ok": recent[0], "score": 3 * len(r) - 2 * len(b) + z["n"] + bonus})
    t_, rate = len(ts), z["rate"]
    z["tier"] = "Proven" if t_ >= 3 and rate >= .65 else ("Tested" if t_ >= 2 and rate >= .60 else ("Footprint-only" if t_ < 2 and z["n"] >= 3 else "Rejected"))
    return z

def top_zones(B, L, upto, pre, top=TOP):
    items, days = candidates(B, L, upto, pre)
    if not items: return [], days
    sess = sessions_of(B, days[-NL.MONTH:]); zs = [stats(z, sess, len(days)) for z in make_zones(items)]
    order = {"Proven": 0, "Tested": 1, "Footprint-only": 2}
    zs = [z for z in zs if z["tier"] != "Rejected"]; zs.sort(key=lambda z: (order[z["tier"]] if z["tier"] == "Footprint-only" else 0, -z["score"]))
    return zs[:top], days

# ───────────────────────────── walk-forward check ─────────────────────────────
def walk_forward(B, L, pre, min_hist=8):
    days = sorted(B["date"].unique()); tot = {"zones": [0, 0], "shifted": [0, 0]}; used = 0
    for k in range(min_hist, len(days)):
        zs, _ = top_zones(B, L, days[k], pre)
        if not zs: continue
        s = sessions_of(B, [days[k]]); used += 1
        for z in zs:
            for name, off in [("zones", 0)] + [("shifted", o) for o in SHIFTS]:
                for t in touches(z["lo"] + off, z["hi"] + off, s):
                    if t[2]: tot[name][0 if t[2] == "react" else 1] += 1
    rows = [[n, r, b, round(100 * r / (r + b), 1) if r + b else ""] for n, (r, b) in tot.items()]
    return rows, used

# ───────────────────────────── output ─────────────────────────────
def main():
    import live_refresh_local as LR
    days_n = int(os.environ.get("NL_DAYS", "45")); sid, name, exp = NL.nearest_future(); print("contract:", name, sid, exp)
    cs = NL.fetch(sid, days_n); B, L = NL.to_frames(cs); ds = sorted(B["date"].unique()); print(len(cs), "candles;", len(ds), "sessions;", ds[0], "->", ds[-1])
    pre = {"swing": swings(B), "stack": NL.stacks(L), "absorb": NL.absorption(B)}
    last = float((B["hi"].iloc[-1] + B["lo"].iloc[-1]) / 2)
    zs, _ = top_zones(B, L, "9999-12-31", pre)
    rows = []
    for z in sorted(zs, key=lambda z: -z["ctr"]):
        rows.append([z["lo"], z["hi"], z["ctr"], "above price (resistance side)" if z["ctr"] > last else "below price (support side)", round((z["ctr"] / last - 1) * 100, 2), z["role"], z["tier"],
                     z["touches"], z["react"], z["brk"], "" if z["rate"] is None else round(100 * z["rate"]), round(z["bounce"]), z["last_ok"], ", ".join(z["src"])])
    hdr = ["zone from", "zone to", "centre", "where", "% from last", "acts as", "evidence", "touches", "reacted", "broke", "react %", "avg bounce pts", "last worked", "what it is made of"]
    sh = LR.open_sheet(); stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    NL.sheet_put(sh, "Nifty Zones", hdr, rows + [[], ["contract", name, "last price", round(last, 1), "sessions", len(ds), f"{ds[0]} -> {ds[-1]}", f"run {stamp}"],
        ["how to read", f"Zone = ~{int(2*HALF)}-point band. A touch = price entering the band; REACTED = it moved {int(REACT)}+ pts back away within {LOOK} min before breaking {int(REACT)}+ pts through; BROKE = the opposite. 'Footprint-only' zones have no proven touches yet."]])
    wf, used = walk_forward(B, L, pre)
    NL.sheet_put(sh, "Nifty Zone Test", ["set", "reacted", "broke", "react %"], wf + [[], ["reading", f"Zones built ONLY from earlier sessions, judged on the next session ({used} sessions tested). 'shifted' = the same zones moved by {SHIFTS} points. If zones are not clearly above shifted, the map adds nothing beyond chance."]])
    print("tabs written"); [print(r) for r in rows]; print(wf)

# ───────────────────────────── self-test ─────────────────────────────
def selftest():
    # price oscillates 23100 <-> 23000, so 23100 should be a proven resistance and 23000 a proven support
    rng = np.random.default_rng(5); cs = []
    for d in range(10):
        t0 = int(pd.Timestamp("2026-10-12").timestamp()) + d * 86400 - 19800; px = 23050.0; direction = 1
        for m in range(375):
            px += direction * 2.5 + rng.normal(0, 0.5)
            if px >= 23100: direction = -1
            if px <= 23000: direction = 1
            lv = [{"price": float(k), "buy_volume": 50, "sell_volume": 50} for k in range(int(px - 2), int(px + 3))]
            cs.append({"t": t0 + (555 + m) * 60, "high": px + 2, "low": px - 2, "buy_volume": 250, "sell_volume": 250, "total_volume": 500, "levels": lv})
    B, L = NL.to_frames(cs); pre = {"swing": swings(B), "stack": NL.stacks(L), "absorb": NL.absorption(B)}
    zs, days = top_zones(B, L, "9999-12-31", pre); print(len(zs), [(z["ctr"], z["tier"], z["touches"], z["react"], z["role"]) for z in zs[:6]])
    assert zs and any(abs(z["ctr"] - 23100) <= 30 and z["role"] == "Resistance" and z["tier"] in ("Proven", "Tested") for z in zs), "resistance not found"
    assert any(abs(z["ctr"] - 23000) <= 30 and z["role"] == "Support" for z in zs), "support not found"
    # touch classification
    hi = np.array([130., 125, 110, 100, 90, 120, 140.]); lo = np.array([125., 118, 100, 90, 80, 100, 130.])
    t = touches(95, 105, [("d", hi, lo)]); print(t)
    # zone from above at 95-105: touched at bar 2 (lo 100 within), then price goes to 80 (<= 95-25=70? no) ...
    up = touches(95, 105, [("d", np.array([140., 130, 112, 108, 135, 140]), np.array([130., 125, 100, 100, 128, 130]))]); assert up and up[0][1] == "above" and up[0][2] == "react", up
    dn = touches(95, 105, [("d", np.array([140., 130, 112, 100, 80, 60]), np.array([130., 125, 100, 90, 60, 50]))]); assert dn and dn[0][2] == "break", dn
    assert len(top_zones(B, L, "9999-12-31", pre, top=3)[0]) <= 3
    wf, used = walk_forward(B, L, pre, min_hist=5); print(wf, used); assert used >= 3 and wf[0][3] != ""
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
