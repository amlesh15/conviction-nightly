# ==================================================================================
# ARCHIVE 1-MIN — nightly capture of Sensibull 1-minute candles to a local parquet store.
#
#   WHY: the candle endpoint only serves ~30 days of 1-min history. To ever backtest
#   intraday FVGs / ICT setups across regimes, we must save each day BEFORE it ages out.
#
#   WHAT it does each run:
#     - pulls the F&O universe from your live Rule/Calibrated Dashboard (stays in sync)
#     - fetches 1-min candles (last ~30d) for every symbol
#     - writes ONE parquet per session:  ~/fno_1m/1m_cash_YYYY-MM-DD.parquet
#     - first run backfills the whole 30-day window; later runs just add/refresh recent days
#       (so a missed day self-heals as long as you run again within 30 days)
#
#   RUN:   cd ~/live-refresh && python archive_1m.py
#   LOAD:  import archive_1m as A;  df = A.load_1m("cash", start="2026-07-20")
# ==================================================================================
import os, glob, datetime as dt
import pandas as pd
import concurrent.futures as cf
from zoneinfo import ZoneInfo
import intraday_fvg as F                      # reuse fetch_1m + cookie.txt

try:
    import live_refresh_local as LR           # reuse sheet auth + live universe
except Exception:
    LR = None

STORE   = os.environ.get("ARCHIVE_DIR", os.path.expanduser("~/fno_1m"))
IST     = ZoneInfo("Asia/Kolkata")
FUT_SEG = getattr(F, "FUT_SEG", 1)            # set once the futures segment is confirmed

# fallback universe if the sheet can't be read
_FALLBACK = ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","SBIN","AXISBANK","KOTAKBANK",
    "ITC","LT","BHARTIARTL","HINDUNILVR","BAJFINANCE","MARUTI","TATAMOTORS","TATASTEEL",
    "HINDALCO","JSWSTEEL","SUNPHARMA","CIPLA","ADANIENT","ADANIPORTS","WIPRO","TECHM",
    "HCLTECH","ULTRACEMCO","TITAN","ASIANPAINT","NTPC","POWERGRID","ONGC","COALINDIA",
    "GRASIM","DLF","TRENT","VEDL","BANKBARODA","PIDILITIND","DRREDDY","BAJAJFINSV"]

def universe():
    if LR is not None:
        try:
            sh = LR.open_sheet(); syms = set()
            titles = [w.title for w in sh.worksheets()]
            for nm in LR.LIVE_SHEETS:
                if nm in titles:
                    ws = sh.worksheet(nm); syms |= set(LR.sheet_symbols(ws, LR.header_map(ws)))
            syms = {s for s in syms if s}
            if syms: return sorted(syms)
        except Exception as e:
            print("universe from sheet failed, using fallback:", e)
    return _FALLBACK

def archive(seg=1, tag="cash", days=30, workers=6, refresh_last=2):
    """Fetch 1-min for the whole universe, write one parquet per session. Idempotent."""
    syms = universe()
    print(f"[{tag}] universe = {len(syms)} symbols")
    def one(s):
        try:
            df = F.fetch_1m(s, days=days, seg=seg)
            if df.empty: return None
            df = df.copy(); df.insert(0, "symbol", s); return df
        except Exception as e:
            print(s, "err", e); return None
    parts = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, syms):
            if r is not None: parts.append(r)
    if not parts:
        print(f"[{tag}] nothing fetched — stale cookie? (refresh ~/live-refresh/cookie.txt)"); return
    allbars = pd.concat(parts, ignore_index=True)
    allbars["session"] = allbars["session"].astype(str)
    os.makedirs(STORE, exist_ok=True)
    sessions = sorted(allbars["session"].unique())
    recent = set(sessions[-refresh_last:])          # always re-save the last couple (may have been partial)
    saved = 0
    for sess, g in allbars.groupby("session"):
        path = os.path.join(STORE, f"1m_{tag}_{sess}.parquet")
        if os.path.exists(path) and sess not in recent: continue
        g.sort_values(["symbol","ts"]).to_parquet(path, index=False); saved += 1
    total = len(allbars)
    print(f"[{tag}] {len(sessions)} sessions in window | saved/updated {saved} files | {total:,} bars -> {STORE}")

def load_1m(tag="cash", start=None, end=None):
    """Concatenate stored sessions into one DataFrame (symbol, ts, ohlcv, oi, session)."""
    fs = sorted(glob.glob(os.path.join(STORE, f"1m_{tag}_*.parquet")))
    dfs = []
    for f in fs:
        sess = os.path.basename(f).replace(f"1m_{tag}_","").replace(".parquet","")
        if start and sess < start: continue
        if end and sess > end: continue
        dfs.append(pd.read_parquet(f))
    if not dfs: print("no stored sessions"); return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    print(f"loaded {tag}: {len(df):,} bars | {df['session'].min()} .. {df['session'].max()} | {df['session'].nunique()} sessions")
    return df

if __name__ == "__main__":
    now = dt.datetime.now(IST)
    if now.weekday() >= 5:
        print("weekend — skipping"); raise SystemExit
    archive(seg=1, tag="cash")
    # archive(seg=FUT_SEG, tag="fut")     # enable once the futures segment is confirmed
