# ==================================================================================
# IMBALANCE LEVELS -> DHAN CHART (levels are never removed)
#   1. reads today's imbalance levels from the sheet (Rule Dashboard imb_sup1..3 / imb_res1..3, plus the NIFTY Imbalance tab)
#   2. adds any NEW level to the permanent tab "Imbalance Log" (symbol, level, first seen, last seen, open?). Old levels are never deleted;
#      a level that is no longer on the sheet (price closed through it) stays in the log, marked open = N.
#   3. writes imbalance_levels.pine with EVERY level in the log, all black. Open levels are solid, filled ones dotted.
#   Usage: python make_imbalance_pine.py [--selftest]      env: GCP_SA_JSON, SHEET_ID
# ==================================================================================
import sys, datetime as dt
SELFTEST = "--selftest" in sys.argv
TAB = "Imbalance Log"; HDR = ["symbol", "level", "first seen", "last seen", "open"]; TOL = 0.0005
MAX_PER_SYMBOL = 40

def _num(x):
    try:
        v = float(str(x).replace(",", "").strip()); return None if v != v else v
    except Exception: return None

def merge(log, today_levels, asof):
    """log: list of [symbol, level, first, last, open]; today_levels: {symbol: [levels]}. Returns new log (nothing removed)."""
    out = [list(r) for r in log]; seen = set()
    for sym, lvls in today_levels.items():
        for lv in lvls:
            hit = None
            for i, r in enumerate(out):
                if r[0] == sym and abs(r[1] - lv) <= TOL * max(lv, 1): hit = i; break
            if hit is None: out.append([sym, lv, asof, asof, "Y"]); hit = len(out) - 1
            else: out[hit][3] = asof; out[hit][4] = "Y"
            seen.add(hit)
    for i, r in enumerate(out):
        if i not in seen and r[4] == "Y" and r[3] != asof: r[4] = "N"
    return out

TEMPLATE = r'''//@version=5
indicator("Imbalance Levels", shorttitle="IMB", overlay=true)

// Imbalance (fair value gap) levels from the conviction sheet. Built __ASOF__. __COUNT__ symbols.
// Every level ever drawn is kept. Solid = still unfilled, dotted = price has since closed through it. New levels are added each time the script is rebuilt.

colB = input.color(color.black, "Line colour")

tk = syminfo.ticker
var string op = ""
var string fl = ""

if barstate.isfirst
__BLOCKS__
var array<line> ls = array.new<line>()

if barstate.islast
    for i = 0 to array.size(ls) - 1
        line.delete(array.get(ls, i))
    array.clear(ls)
    a = str.split(op, ",")
    for i = 0 to array.size(a) - 1
        v = str.tonumber(array.get(a, i))
        if not na(v)
            array.push(ls, line.new(bar_index - 1, v, bar_index, v, extend = extend.both, color = colB, width = 1, style = line.style_solid))
    b = str.split(fl, ",")
    for i = 0 to array.size(b) - 1
        v = str.tonumber(array.get(b, i))
        if not na(v)
            array.push(ls, line.new(bar_index - 1, v, bar_index, v, extend = extend.both, color = colB, width = 1, style = line.style_dotted))
'''

def fmt(v): return f"{v:.2f}".rstrip("0").rstrip(".")

def build_pine(log, asof):
    by = {}
    for sym, lv, first, last, op in log: by.setdefault(sym, []).append((first, lv, op))
    blocks = []
    for sym in sorted(by):
        rows = sorted(by[sym], reverse=True)[:MAX_PER_SYMBOL]   # newest first, capped
        o = ",".join(fmt(l) for f, l, p in rows if p == "Y"); f_ = ",".join(fmt(l) for f, l, p in rows if p != "Y")
        assert '"' not in sym and "\\" not in sym
        blocks.append(f'    if tk == "{sym}"\n        op := "{o}"\n        fl := "{f_}"')
    return TEMPLATE.replace("__BLOCKS__", "\n".join(blocks)).replace("__ASOF__", asof).replace("__COUNT__", str(len(by)))

def today_from_sheet(sh):
    ws = sh.worksheet("Rule Dashboard"); v = ws.get_all_values(); h = [x.strip() for x in v[0]]; si = h.index("symbol")
    cols = [h.index(f"imb_{k}{n}") for k in ("sup", "res") for n in (1, 2, 3)]; out = {}
    for row in v[1:]:
        if si >= len(row) or not row[si].strip(): continue
        ls = [_num(row[i]) for i in cols if i < len(row)]; ls = [x for x in ls if x]
        if ls: out[row[si].strip().upper()] = ls
    try:
        lv = {r[0]: _num(r[1]) for r in sh.worksheet("NIFTY Imbalance").get_all_values() if len(r) > 1 and r[0].startswith("imb_")}
        n = [x for x in lv.values() if x]
        if n: out["NIFTY"] = n
    except Exception as e: print("NIFTY tab not read:", e)
    return out

def main():
    import live_refresh_local as LR
    sh = LR.open_sheet(); today = today_from_sheet(sh)
    try: asof = sh.worksheet("Run Info").get_all_values()[2][1][:10]
    except Exception: asof = dt.date.today().isoformat()
    try: ws = sh.worksheet(TAB)
    except Exception: ws = sh.add_worksheet(TAB, rows=5000, cols=6)
    old = [[r[0], _num(r[1]), r[2], r[3], r[4]] for r in ws.get_all_values()[1:] if len(r) >= 5 and _num(r[1])]
    new = merge(old, today, asof); added = len(new) - len(old)
    ws.clear(); ws.update(range_name="A1", values=[HDR] + new, value_input_option="RAW")
    src = build_pine(new, asof); open("imbalance_levels.pine", "w").write(src)
    print(f"as of {asof}: {len(today)} symbols today, {added} NEW levels added, log now {len(new)} levels; pine {len(src)} chars")

def selftest():
    log = merge([], {"A": [100.0, 90.0], "NIFTY": [23000.0]}, "2026-10-29")
    assert len(log) == 3 and all(r[4] == "Y" for r in log)
    log = merge(log, {"A": [100.02, 120.0], "NIFTY": [23000.0]}, "2026-10-30")      # 100.02 same level; 120 new; 90 dropped off the sheet
    d = {(r[0], r[1]): r for r in log}
    assert len(log) == 4 and d[("A", 90.0)][4] == "N" and d[("A", 120.0)][2] == "2026-10-30" and d[("A", 100.0)][3] == "2026-10-30"
    s = build_pine(log, "2026-10-30"); assert 'if tk == "A"' in s and 'op := "120,100"' in s and 'fl := "90"' in s and "__" not in s
    print("SELFTEST OK")

if __name__ == "__main__":
    selftest() if SELFTEST else main()
