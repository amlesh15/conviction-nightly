import os, datetime as dt
from zoneinfo import ZoneInfo
import archive_1m as A
now = dt.datetime.now(ZoneInfo("Asia/Kolkata"))
if now.weekday() >= 5 and os.environ.get("FORCE", "").lower() != "true":
    print("weekend - skip")
else:
    A.archive(seg=1, tag="cash", refresh_last=1)
