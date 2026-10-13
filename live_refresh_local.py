# ==================================================================================
# LIVE REFRESH (LOCAL) — runs on your laptop, updates the 5 live columns every 10 min.
#
#   Primary source: NSE bulk F&O snapshot (ONE call, all stocks, NO cookie) — works from a
#   home/residential IP (it only failed on Colab because that's a datacenter IP NSE blocks).
#   Fallback: Sensibull candles (needs SENSIBULL_COOKIE) if NSE is unreachable.
#
#   ONE-TIME SETUP
#   1) pip install gspread google-auth requests
#   2) Google Sheets access (service account):
#        - console.cloud.google.com -> new project -> enable "Google Sheets API" + "Google Drive API"
#        - Create a Service Account -> Keys -> Add key -> JSON -> download it
#        - Open your Google Sheet -> Share -> add the service-account email (…iam.gserviceaccount.com) as Editor
#        - put the JSON path in SERVICE_JSON below
#   3) put your Sheet URL in SHEET_URL
#   RUN:  python live_refresh_local.py       (leave it running during market hours; it self-stops at 15:30)
# ==================================================================================
import time, requests, datetime as dt, os
from zoneinfo import ZoneInfo
import gspread
from google.oauth2.service_account import Credentials

def _load_cookie():
    """Prefer cookie.txt next to this script (easy daily update); else the constant below."""
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookie.txt")
        if os.path.exists(p):
            c = open(p).read().strip()
            if c: return c
    except Exception: pass
    return SENSIBULL_COOKIE

# ── CONFIG ──
SHEET_URL    = ""   # not used on GitHub; SHEET_ID secret is used
SERVICE_JSON = ""   # not used on GitHub; GCP_SA_JSON secret is used
LIVE_SHEETS  = ["Calibrated Dashboard", "Rule Dashboard"]
LIVE_MAP     = [("Today %","pct"),("High %","high_pct"),("Low %","low_pct")]   # dropped Day High / Day Low prices; kept the %s
SENSIBULL_COOKIE = ""                                    # optional fallback only; leave "" to use NSE only
IST = ZoneInfo("Asia/Kolkata")

# ── Google Sheets ──
def open_sheet():
    import json
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(json.loads(os.environ["GCP_SA_JSON"]), scopes=scopes)
    return gspread.authorize(creds).open_by_key(os.environ["SHEET_ID"])

def header_map(ws):
    hdr = ws.row_values(1)
    return {h.strip().lower(): i+1 for i,h in enumerate(hdr)}

def sheet_symbols(ws, m):
    sc = m.get("symbol",1); col = ws.col_values(sc)[1:]
    return [str(s).strip().upper() for s in col]

# ── NSE bulk F&O snapshot (primary: one call, no cookie) ──
_NSE=None
def _nse():
    global _NSE
    if _NSE: return _NSE
    s=requests.Session()
    s.headers.update({"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
        "Accept":"text/html,application/xhtml+xml,*/*;q=0.8","Accept-Language":"en-US,en;q=0.9"})
    try: s.get("https://www.nseindia.com", timeout=15); s.get("https://www.nseindia.com/market-data/live-equity-market", timeout=15)
    except Exception as e: print("nse warmup:", e)
    _NSE=s; return s

def fetch_nse():
    from urllib.parse import quote
    url="https://www.nseindia.com/api/equity-stockIndices?index="+quote("SECURITIES IN F&O", safe="")
    hdr={"Accept":"application/json, text/plain, */*","Referer":"https://www.nseindia.com/market-data/live-equity-market","X-Requested-With":"XMLHttpRequest"}
    for attempt in range(3):
        s=_nse()
        try:
            r=s.get(url, headers=hdr, timeout=20)
            if r.status_code!=200 or not r.text.strip().startswith("{"):
                print(f"nse attempt {attempt+1}: {r.status_code}"); globals()['_NSE']=None; continue
            out={}
            for d in (r.json() or {}).get("data",[]):
                sym=str(d.get("symbol","")).strip().upper(); pc=d.get("previousClose")
                if not sym or not pc: continue
                hi=d.get("dayHigh"); lo=d.get("dayLow")
                def pctof(x):
                    try: return round((float(x)-pc)/pc*100,2)
                    except: return ""
                out[sym]={"pct":round(float(d.get("pChange")),2) if d.get("pChange") not in (None,"") else "",
                          "high":hi,"low":lo,"high_pct":pctof(hi),"low_pct":pctof(lo)}
            return out
        except Exception as e:
            print(f"nse attempt {attempt+1} err:", e); globals()['_NSE']=None
    return {}

# ── Sensibull fallback (per-symbol daily candles; needs cookie) ──
def fetch_sensibull(symbols):
    ck=_load_cookie()
    if not ck: return {}
    import concurrent.futures as cf
    base="https://oxide.sensibull.com/v1/compute/1/candles/"
    hdr={"cookie":ck,"content-type":"application/json","origin":"https://web.sensibull.com",
         "referer":"https://web.sensibull.com/","user-agent":"Mozilla/5.0"}
    frm=(dt.date.today()-dt.timedelta(days=8)).isoformat(); to=(dt.date.today()+dt.timedelta(days=1)).isoformat()
    body={"from_date":frm,"to_date":to,"interval":"1D","skip_last_ts":False}
    def one(s):
        try:
            r=requests.post(base+s, json=body, headers=hdr, timeout=20)
            c=((r.json() or {}).get("payload") or {}).get("candles") or []
            if len(c)<2: return s,None
            prev=float(c[-2]["close"]); cur=c[-1]; ltp=float(cur["close"]); hi=float(cur["high"]); lo=float(cur["low"])
            return s,{"pct":round((ltp-prev)/prev*100,2),"high":round(hi,2),"low":round(lo,2),
                      "high_pct":round((hi-prev)/prev*100,2),"low_pct":round((lo-prev)/prev*100,2)}
        except Exception: return s,None
    out={}
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for s,v in ex.map(one, list(symbols)):
            if v: out[s]=v
    return out

def fetch_live(symbols):
    # NSE's API is Akamai bot-walled (404 from any script) -> use Sensibull when a cookie is set.
    if _load_cookie():
        sb=fetch_sensibull(symbols)
        if sb: return sb, "Sensibull"
    live=fetch_nse()
    return live, "NSE"

# ── write ──
def write_live(ws, m, symbols, live):
    n=len(symbols)
    for name,key in LIVE_MAP:
        c=m.get(name.lower())
        if not c: continue
        col=[[ (live.get(s,{}).get(key,"")) ] for s in symbols]
        a1=f"{gspread.utils.rowcol_to_a1(2,c)}:{gspread.utils.rowcol_to_a1(1+n,c)}"
        ws.update(range_name=a1, values=col, value_input_option="USER_ENTERED")

def refresh_once(sh):
    total=0; src=""
    tabs=[(nm,sh.worksheet(nm)) for nm in LIVE_SHEETS if nm in [w.title for w in sh.worksheets()]]
    allsyms=set()
    for _,ws in tabs: allsyms|=set(sheet_symbols(ws, header_map(ws)))
    live,src=fetch_live(sorted(allsyms))
    for _,ws in tabs:
        m=header_map(ws); syms=sheet_symbols(ws,m); write_live(ws,m,syms,live); total=len(live)
    return total, src

def in_market_hours():
    now=dt.datetime.now(IST)
    if now.weekday()>=5: return False
    t=now.hour*60+now.minute
    return 9*60+15 <= t <= 15*60+30

def main():
    sh=open_sheet()
    print("connected:", SHEET_URL)
    while True:
        now=dt.datetime.now(IST); t=now.hour*60+now.minute
        if now.weekday()>=5: print("weekend — exiting"); break
        if t > 15*60+30:
            n,src=refresh_once(sh); print(f"final refresh {n} via {src} @ {now:%H:%M} — market closed, exiting"); break
        if t < 9*60+15: print(f"{now:%H:%M} pre-market, waiting…"); time.sleep(60); continue
        try:
            n,src=refresh_once(sh); print(f"refreshed {n} symbols via {src} @ {now:%H:%M:%S}")
        except Exception as e:
            print("refresh error:", e)
        time.sleep(600)

if __name__ == "__main__":
    main()
