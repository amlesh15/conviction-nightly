# ==================================================================================
# NIFTY OPTION OI — SEVERAL SESSIONS, does the OI pattern hold?   (builds on nifty_oi_day.py; research only)
#   For each session we freeze a snapshot at 09:45 (after the opening OI refresh settles) and look at every 50-pt strike 20-300 pts away from spot:
#       below spot  -> a SUPPORT candidate, judged by PUT OI at that strike;   above spot -> a RESISTANCE candidate, judged by CALL OI.
#   Then, after 09:45, we wait for price to first come within 15 pts of the strike and score the next 60 minutes:
#       HELD   = price moves 40 pts back away from the strike before it trades 40 pts through it     BROKE = the opposite     OPEN = neither
#   Questions answered:  (1) do the strikes with the BIGGEST OI at 09:45 hold more often than the others?
#                        (2) does OI that was BUILT before price arrived (09:20 -> touch) hold more often?
#                        (3) what does OI do at the strike right after the touch (writers add / leave)?
#   Everything uses only data up to the touch for the 'cause' columns (no look-ahead); outcome uses the next 60 min.
#   OUTPUT tabs: "Nifty OI Days" (one row per session), "Nifty OI Walls" (every strike observation), "Nifty OI Pattern" (the answers, with counts).
#   Usage: python nifty_oi_days.py [--selftest]   env: NOI_DATES (default 2026-10-28,2026-10-29,2026-10-30), NOI_LO/NOI_HI (default 22300/23700)
# ==================================================================================
import os, sys, numpy as np, pandas as pd
os.environ.setdefault("NOI_LO", "22300"); os.environ.setdefault("NOI_HI", "23700")
import nifty_oi_day as N

SELFTEST = "--selftest" in sys.argv
T0, END, W, MOVE, LOOK, TOPN = "09:45", "14:30", 15, 40, 60, 3
L = 1e5

def study(D, lot):
    i0, iend, ib = D.pos(T0), D.pos(END), D.pos("09:20"); sp = D.spot.to_numpy(float); hi = D.hi.to_numpy(float); lo = D.lo.to_numpy(float); s0 = sp[i0]; obs = []
    for side, t in (("support", "PE"), ("resistance", "CE")):
        ks = [k for k in D.strikes if k % 50 == 0 and W + 5 <= (s0 - k if side == "support" else k - s0) <= 300 and (t, k) in D.oi.columns]
        oi0 = {k: float(D.oi[(t, k)].iloc[i0]) for k in ks}; rank = {k: r for r, k in enumerate(sorted(ks, key=lambda x: -oi0[x]))}
        gr = {k: float(D.oi[(t, k)].iloc[i0] - D.oi[(t, k)].iloc[ib]) for k in ks}; grank = {k: r for r, k in enumerate(sorted(ks, key=lambda x: -gr[x]))}
        for k in ks:
            j = next((i for i in range(i0 + 1, iend) if (lo[i] <= k + W if side == "support" else hi[i] >= k - W)), None)
            if j is None: obs.append({"session": D.session, "side": side, "strike": k, "touched": "no", "OI rank (1=biggest)": rank[k] + 1, "OI at 09:45 (L)": oi0[k] / L, "OI built 09:20-09:45 (L)": gr[k] / L, "build rank": grank[k] + 1}); continue
            out = "OPEN"
            for i in range(j, min(len(sp) - 1, j + LOOK) + 1):
                up, dn = hi[i] >= k + MOVE, lo[i] <= k - MOVE
                if side == "support": out = "BROKE" if dn else ("HELD" if up else out)
                else: out = "BROKE" if up else ("HELD" if dn else out)
                if out != "OPEN": break
            oit = D.oi[(t, k)]; jj = min(len(oit) - 1, j + 30)
            obs.append({"session": D.session, "side": side, "strike": k, "touched": "yes", "touch time": D.hm(j), "outcome": out, "OI rank (1=biggest)": rank[k] + 1, "OI at 09:45 (L)": oi0[k] / L,
                        "OI built 09:20-09:45 (L)": gr[k] / L, "build rank": grank[k] + 1, "OI built 09:45-touch (L)": float(oit.iloc[j] - oit.iloc[i0]) / L,
                        "OI 30 min after touch (L)": float(oit.iloc[jj] - oit.iloc[j]) / L})
    return pd.DataFrame(obs)

def day_row(D, lot):
    i_lo = int(np.argmin(D.lo.to_numpy())); i_hi = int(np.argmax(D.hi.to_numpy())); i0 = D.pos(T0); s0 = float(D.spot.iloc[i0])
    pw = [(float(D.oi[("PE", k)].iloc[i0]), k) for k in D.strikes if ("PE", k) in D.oi.columns and k < s0 and s0 - k <= 300]; cw = [(float(D.oi[("CE", k)].iloc[i0]), k) for k in D.strikes if ("CE", k) in D.oi.columns and k > s0 and k - s0 <= 300]
    pk = max(pw)[1] if pw else np.nan; ck = max(cw)[1] if cw else np.nan
    return {"session": D.session, "open": round(float(D.spot.iloc[0])), "spot 09:45": round(s0), "day low": round(float(D.lo.min())), "low time": D.hm(i_lo), "day high": round(float(D.hi.max())), "high time": D.hm(i_hi),
            "close": round(float(D.spot.iloc[D.pos('15:30')])), "biggest put OI strike (<=300 below)": pk, "day low vs it": round(float(D.lo.min()) - pk) if pk == pk else "",
            "biggest call OI strike (<=300 above)": ck, "day high vs it": round(float(D.hi.max()) - ck) if ck == ck else ""}

def summarize(O):
    T = O[O["touched"] == "yes"].copy(); T = T[T["outcome"].isin(["HELD", "BROKE"])]; rows = []; lines = []
    def rate(d): n = len(d); h = int((d["outcome"] == "HELD").sum()); return n, h, (100 * h / n if n else np.nan)
    def add(name, d, base):
        n, h, p = rate(d); rows.append({"group": name, "levels scored": n, "held": h, "held %": round(p, 1) if p == p else "", "vs all others %": round(p - base, 1) if p == p and base == base else ""})
    for side in ("support", "resistance", "both"):
        S = T if side == "both" else T[T["side"] == side]
        if not len(S): continue
        for nm, col, lim in (("BIGGEST OI at 09:45", "OI rank (1=biggest)", TOPN), ("MOST OI BUILT 09:20-09:45", "build rank", TOPN)):
            a, b = S[S[col] <= lim], S[S[col] > lim]; base = rate(b)[2]; add(f"{side}: {nm} (top {lim})", a, base); add(f"{side}: all other strikes", b, np.nan)
    n, h, p = rate(T); rows.append({"group": "ALL scored levels", "levels scored": n, "held": h, "held %": round(p, 1) if p == p else "", "vs all others %": ""})
    big = T[T["OI rank (1=biggest)"] <= TOPN]
    if len(big):
        after = big["OI 30 min after touch (L)"]; lines.append(f"At the biggest-OI strikes, OI 30 min after the touch: rose in {int((after > 0).sum())} of {len(big)} (median {after.median():+.1f}L).")
        for o in ("HELD", "BROKE"):
            d = big[big["outcome"] == o]
            if len(d): lines.append(f"   when they {o}: median OI change after touch {d['OI 30 min after touch (L)'].median():+.1f}L, OI built before touch (09:45->touch) {d['OI built 09:45-touch (L)'].median():+.1f}L (n={len(d)})")
    return pd.DataFrame(rows), lines

def main():
    dates = [x.strip() for x in os.environ.get("NOI_DATES", "2026-10-28,2026-10-29,2026-10-30").split(",") if x.strip()]
    panel, spot, latest, lot, e = N.pull()
    have = sorted(panel["session"].unique()); print("sessions in data:", have, "| expiry", e)
    obs, days, missing = [], [], []
    for s in dates:
        if s not in have: missing.append(s); continue
        D = N.Day(panel, spot, s)
        if len(D.idx) < 100: missing.append(s + " (too few bars)"); continue
        days.append(day_row(D, lot)); obs.append(study(D, lot))
    if not obs: raise SystemExit(f"none of {dates} available; data has {have}")
    O = pd.concat(obs, ignore_index=True); DAYS = pd.DataFrame(days); P, lines = summarize(O)
    pd.set_option("display.width", 250); print(DAYS.to_string(index=False)); print(P.to_string(index=False)); print("\n".join(lines))
    from live_refresh_local import open_sheet
    import nifty_levels as NL
    sh = open_sheet(); n_t = int((O["touched"] == "yes").sum())
    notes = [[f"expiry {e}; sessions {', '.join(DAYS['session'])}" + (f"; NOT AVAILABLE: {', '.join(missing)}" if missing else "")],
             [f"{len(O)} strike-levels examined, {n_t} touched after {T0}. HELD = price moved {MOVE} pts away before {MOVE} pts through, within {LOOK} min of the first touch (within {W} pts)."],
             ["Small sample: a difference of a few percentage points means nothing; look at the counts."]] + [[l] for l in lines]
    NL.sheet_put(sh, "Nifty OI Days", list(DAYS.columns), DAYS.values.tolist())
    NL.sheet_put(sh, "Nifty OI Walls", list(O.columns), O.round(2).fillna("").values.tolist())
    NL.sheet_put(sh, "Nifty OI Pattern", list(P.columns), P.values.tolist() + [[]] + notes)
    print("tabs written")

def selftest():
    idx = pd.date_range("2026-10-29 09:15", "2026-10-29 15:30", freq="1min"); n = len(idx)
    spot = np.concatenate([np.linspace(23000, 23000, 30), np.linspace(23000, 22885, 60), np.linspace(22885, 22990, 40), np.full(n - 130, 22990.0)])   # falls to touch 22900/22850 area then recovers
    sp = pd.DataFrame({"ts": idx, "session": "2026-10-29", "open": spot, "high": spot + 1, "low": spot - 1, "close": spot, "volume": 1})
    fr = []
    for k in range(22700, 23300, 50):
        for t in ("CE", "PE"):
            base = 1e6 * (3 if (t, float(k)) == ("PE", 22900.0) else 1)      # 22900 PE is the big wall
            oi = np.concatenate([np.full(30, base * 0.5), np.linspace(base * .5, base, 30), np.full(n - 60, base)])
            fr.append(pd.DataFrame({"ts": idx, "session": "2026-10-29", "close": 30.0, "volume": 400.0, "oi": oi, "strike": float(k), "typ": t}))
    panel = pd.concat(fr, ignore_index=True); D = N.Day(panel, sp, "2026-10-29"); O = study(D, 65); r = day_row(D, 65); P, lines = summarize(O)
    print(O[O["touched"] == "yes"][["side", "strike", "touch time", "outcome", "OI rank (1=biggest)"]].to_string(index=False)); print(P.to_string(index=False)); print(r)
    t = O[(O["touched"] == "yes") & (O["strike"] == 22900)]; assert len(t) == 1 and t.iloc[0]["outcome"] == "HELD" and t.iloc[0]["OI rank (1=biggest)"] == 1, t
    assert r["biggest put OI strike (<=300 below)"] == 22900
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
