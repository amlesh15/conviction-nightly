# ==================================================================================
# NIFTY ZONES, DATE BY DATE — no look-ahead.
#   For every trading day D from the start date: use ONLY sessions up to and including D, find the zones (same logic as nifty_zones.py:
#   footprint + price evidence -> ~30-point bands -> proof by replaying earlier touches), and STORE them as "zones as of D".
#   Then overlay all the stored days: zones that keep showing up in the same place, day after day, are merged into one persistent zone
#   (days seen, first / last seen, latest track record). The final map = persistent zones still alive on the last day, strongest 20.
#   Forward check: the zones stored as of D are judged on session D+1 (which they never saw) and compared with the same zones shifted.
#   OUTPUT tabs: "Nifty Zones Final" (the map), "Nifty Zones By Date" (the stored daily zones), "Nifty Zone Test" (forward check).
#   Research only.     Usage: python nifty_asof.py [--selftest]
#   env: DHAN_HEADERS_JSON, GCP_SA_JSON, SHEET_ID, ZN_START (default 2026-09-28), NL_DAYS (history to fetch, default 70)
# ==================================================================================
import os, sys, datetime as dt
import numpy as np, pandas as pd
import nifty_levels as NL, nifty_zones as Z

SELFTEST = "--selftest" in sys.argv
JOIN = 20.0          # stored zones whose centres are within this many points are the same zone
TOPN = 20

def next_label(d): return (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

def pre_asof(B, L, D):
    """Evidence computed from data up to and including D only (pivots need bars on both sides, so the last blocks cannot become pivots)."""
    Bt = B[B["date"] <= D]; Lt = L[L["date"] <= D]
    return {"swing": Z.swings(Bt), "stack": NL.stacks(Lt), "absorb": NL.absorption(Bt)}

def zones_asof(B, L, D, top=60):
    zs, days = Z.top_zones(B[B["date"] <= D], L[L["date"] <= D], next_label(D), pre_asof(B, L, D), top=top)
    return zs, days

def overlay(daily, all_dates):
    """daily = [(date, zone)] -> persistent zones."""
    items = sorted(daily, key=lambda x: x[1]["ctr"]); cl = []
    for d, z in items:
        for c in cl:
            if abs(c["mean"] - z["ctr"]) <= JOIN: c["m"].append((d, z)); c["mean"] = float(np.mean([q["ctr"] for _, q in c["m"]])); break
        else: cl.append({"m": [(d, z)], "mean": z["ctr"]})
    out = []
    for c in cl:
        ds = sorted({d for d, _ in c["m"]}); first, last = ds[0], ds[-1]; since = [x for x in all_dates if x >= first]
        lz = [z for d, z in c["m"] if d == last][-1]
        out.append({"ctr": round(c["mean"], 1), "lo": round(c["mean"] - Z.HALF, 1), "hi": round(c["mean"] + Z.HALF, 1), "days": len(ds), "first": first, "last": last,
                    "share": len(ds) / len(since), "alive": last == all_dates[-1], "tier": lz["tier"], "role": lz["role"], "touches": lz["touches"], "react": lz["react"], "brk": lz["brk"],
                    "src": sorted({s for _, z in c["m"] for s in z["src"]}), "score": lz["score"] + len(ds)})
    return out

def forward_test(B, L, stored, dates):
    """Zones stored as of D are judged on session D+1. Returns rows + sessions used."""
    tot = {}; used = 0
    for i, D in enumerate(dates[:-1]):
        nxt = dates[i + 1]; s = Z.sessions_of(B, [nxt]); used += 1
        for z in stored.get(D, []):
            grp = "proven/tested zones" if z["tier"] in ("Proven", "Tested") else "early/footprint-only zones"
            for name, offs in ((grp, [0]), ("all zones shifted", Z.SHIFTS)):
                for o in offs:
                    for t in Z.touches(z["lo"] + o, z["hi"] + o, s):
                        if t[2]: tot.setdefault(name, [0, 0])[0 if t[2] == "react" else 1] += 1
    return [[n, r, b, round(100 * r / (r + b), 1) if r + b else ""] for n, (r, b) in tot.items()], used

def run(B, L, start):
    dates = sorted(B["date"].unique()); asof = [d for d in dates if d >= start]; stored = {}; daily = []
    for D in asof:
        zs, _ = zones_asof(B, L, D); stored[D] = zs; daily += [(D, z) for z in zs]; print(D, len(zs), "zones", flush=True)
    return asof, stored, daily

def main():
    import live_refresh_local as LR
    start = os.environ.get("ZN_START", "2026-09-28"); sid, name, exp = NL.nearest_future(); print("contract:", name, sid, exp)
    cs = NL.fetch(sid, int(os.environ.get("NL_DAYS", "70"))); B, L = NL.to_frames(cs); ds = sorted(B["date"].unique())
    print(len(cs), "candles;", len(ds), "sessions;", ds[0], "->", ds[-1])
    asof, stored, daily = run(B, L, start)
    if not asof: print("no sessions on/after", start); return
    fin = overlay(daily, asof); last = float((B["hi"].iloc[-1] + B["lo"].iloc[-1]) / 2)
    alive = [f for f in fin if f["alive"] or f["last"] >= asof[-min(2, len(asof))]]
    alive.sort(key=lambda f: -f["score"]); alive = alive[:TOPN]
    rows = []
    for f in sorted(alive, key=lambda f: -f["ctr"]):
        rows.append([f["lo"], f["hi"], f["ctr"], "above price (resistance side)" if f["ctr"] > last else "below price (support side)", round((f["ctr"] / last - 1) * 100, 2), f["role"], f["tier"],
                     f["days"], f"{f['first']} -> {f['last']}", round(100 * f["share"]), f["touches"], f["react"], f["brk"], ", ".join(f["src"])])
    hdr = ["zone from", "zone to", "centre", "where", "% from last", "acts as", "evidence (latest day)", "days it appeared", "first -> last seen", "% of days since first seen", "touches", "reacted", "broke", "what it is made of"]
    sh = LR.open_sheet(); stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_anch = len(ds[:ds.index(asof[0])]) if asof[0] in ds else 0
    NL.sheet_put(sh, "Nifty Zones Final", hdr, rows + [[], ["contract", name, "last price", round(last, 1), "as-of days", f"{asof[0]} -> {asof[-1]} ({len(asof)} days)", f"history before start: {n_anch} sessions", f"run {stamp}"],
        ["how to read", "Each as-of day uses only data up to that day. A zone that shows up on many days, in the same place, is persistent. 'evidence' is the zone's proof record on the latest day it appeared."],
        ["caution", "Before the 27 Oct expiry this contract was the FAR month (thin volume, price offset by the carry). Zones from those days are indicative only."]])
    log = [[D, z["lo"], z["hi"], z["ctr"], z["tier"], z["role"], z["touches"], z["react"], z["brk"], ", ".join(z["src"])] for D in asof for z in sorted(stored[D], key=lambda q: -q["ctr"])]
    NL.sheet_put(sh, "Nifty Zones By Date", ["as of", "zone from", "zone to", "centre", "evidence", "acts as", "touches", "reacted", "broke", "made of"], log)
    wf, used = forward_test(B, L, stored, asof)
    NL.sheet_put(sh, "Nifty Zone Test", ["set", "reacted", "broke", "react %"], wf + [[], ["reading", f"Zones stored as of day D, judged on session D+1 ({used} day-pairs). 'shifted' = the same zones moved {Z.SHIFTS} points. If zones are not clearly above shifted, the map adds nothing beyond chance."]])
    print("tabs written"); [print(r) for r in rows]; print(wf)

# ───────────────────────────── self-test ─────────────────────────────
def selftest():
    rng = np.random.default_rng(5); cs = []
    for d in range(12):
        t0 = int(pd.Timestamp("2026-10-05").timestamp()) + d * 86400 - 19800; px = 23050.0; direction = 1
        for m in range(375):
            px += direction * 2.5 + rng.normal(0, 0.5)
            if px >= 23100: direction = -1
            if px <= 23000: direction = 1
            lv = [{"price": float(k), "buy_volume": 50, "sell_volume": 50} for k in range(int(px - 2), int(px + 3))]
            cs.append({"t": t0 + (555 + m) * 60, "high": px + 2, "low": px - 2, "buy_volume": 250, "sell_volume": 250, "total_volume": 500, "levels": lv})
    B, L = NL.to_frames(cs); dates = sorted(B["date"].unique()); start = dates[6]
    asof, stored, daily = run(B, L, start); fin = overlay(daily, asof)
    print([(f["ctr"], f["days"], f["tier"], f["role"]) for f in sorted(fin, key=lambda f: -f["score"])[:5]])
    assert any(abs(f["ctr"] - 23100) <= 30 and f["days"] >= 4 and f["role"] == "Resistance" for f in fin)
    # NO LOOK-AHEAD: wreck the last 3 sessions and the zones stored for an earlier day must not change
    D = dates[8]; z1, _ = zones_asof(B, L, D)
    B2 = B.copy(); L2 = L.copy(); late = B2["date"] > D; B2.loc[late, ["hi", "lo"]] = [30000.0, 29900.0]; L2.loc[L2["date"] > D, "bin"] += 10000
    z2, _ = zones_asof(B2, L2, D); assert [(z["ctr"], z["tier"]) for z in z1] == [(z["ctr"], z["tier"]) for z in z2], "look-ahead leak"
    wf, used = forward_test(B, L, stored, asof); print(wf, used); assert used >= 3
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
