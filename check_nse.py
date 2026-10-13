"""Connectivity test: can this machine (GitHub runner) download NSE bhavcopy files?"""
import sys, requests, datetime as dt
UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
ok = False
today = dt.date.today()
for i in range(1, 9):
    d = today - dt.timedelta(days=i)
    if d.weekday() >= 5: continue
    u1 = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"
    u2 = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    try:
        r1 = requests.get(u1, headers=UA, timeout=30); r2 = requests.get(u2, headers=UA, timeout=30)
        print(d, "| cash", r1.status_code, len(r1.content), "| fo", r2.status_code, len(r2.content))
        ok = ok or (r1.status_code == 200 and r2.status_code == 200)
    except Exception as e:
        print(d, "error:", e)
print("RESULT:", "NSE reachable" if ok else "NSE NOT reachable / no files")
sys.exit(0 if ok else 1)
