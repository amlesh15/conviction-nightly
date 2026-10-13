"""Gate before each job. Usage: python preflight.py live|atm|archive  -> writes go=true/false to $GITHUB_OUTPUT.
   - skips outside market hours (unless FORCE=true)
   - checks the Sensibull cookie works; if not, sends ONE Telegram alert per day and skips."""
import os, sys, datetime as dt
from zoneinfo import ZoneInfo
mode = sys.argv[1] if len(sys.argv) > 1 else "live"
force = os.environ.get("FORCE", "").lower() == "true"
IST = ZoneInfo("Asia/Kolkata")

def out(go, why=""):
    print(f"go={go} {why}")
    p = os.environ.get("GITHUB_OUTPUT")
    if p: open(p, "a").write(f"go={'true' if go else 'false'}\n")
    sys.exit(0)

now = dt.datetime.now(IST); t = now.hour * 60 + now.minute
if not force:
    if now.weekday() >= 5: out(False, "weekend")
    if mode in ("live", "atm") and not (9 * 60 + 15 <= t <= 15 * 60 + 40): out(False, f"outside market hours {now:%H:%M}")
    if mode == "archive" and t < 15 * 60 + 45: out(False, "market not closed yet")

ok = False
try:
    import intraday_fvg as F
    df = F.fetch_1m("RELIANCE", days=4)
    ok = df is not None and not df.empty
except Exception as e:
    print("cookie check error:", e)
if ok: out(True, "cookie ok")

# cookie bad -> alert once per day
sd = os.environ.get("STATE_DIR", os.path.expanduser("~/live-refresh/state")); os.makedirs(sd, exist_ok=True)
fp = os.path.join(sd, "cookie_alert.txt"); today = now.strftime("%Y-%m-%d")
last = open(fp).read().strip() if os.path.exists(fp) else ""
if last != today:
    try:
        import notify
        notify.send("⚠️ Sensibull cookie looks expired/invalid.\nLive refresh, ATM and 1-min archive are paused.\n"
                    "Update the SENSIBULL_COOKIE secret in GitHub (Settings > Secrets > Actions).")
        open(fp, "w").write(today)
    except Exception as e:
        print("telegram alert failed:", e)
out(False, "cookie invalid")
