# ==================================================================================
# NOTIFY — send alerts to Telegram.
#
#   Config lives OUTSIDE the code (like cookie.txt), in  ~/live-refresh/telegram.json :
#       { "token": "123456:ABC-your-bot-token", "chat_id": "987654321" }
#   Missing / blank config  ->  alerts are silently OFF (nothing breaks).
#
#   SETUP (one time, ~2 min):
#     1. In Telegram, message @BotFather -> /newbot -> follow prompts -> copy the TOKEN.
#     2. Message your new bot once (say "hi") so it can reply to you.
#     3. Get your chat_id: open  https://api.telegram.org/bot<TOKEN>/getUpdates  in a browser
#        (after step 2) and read  result[].message.chat.id .
#     4. Put both into ~/live-refresh/telegram.json  (see telegram.json.example).
#   TEST:  cd ~/live-refresh && python -c "import notify; print(notify.send('test ✅'))"
# ==================================================================================
import os, json, urllib.parse, urllib.request

CFG = os.path.expanduser("~/live-refresh/telegram.json")

def _cfg():
    try:
        with open(CFG) as f: c = json.load(f)
        if c.get("token") and c.get("chat_id"): return c
    except Exception:
        pass
    return None

def enabled():
    return _cfg() is not None

def send(text, parse_mode=None):
    """Send one Telegram message. parse_mode: None | 'HTML' | 'MarkdownV2'.
       Returns True on success, False if unconfigured/failed."""
    c = _cfg()
    if not c: return False
    url  = f"https://api.telegram.org/bot{c['token']}/sendMessage"
    payload = {"chat_id": c["chat_id"], "text": text, "disable_web_page_preview": "true"}
    if parse_mode: payload["parse_mode"] = parse_mode
    data = urllib.parse.urlencode(payload).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10) as r:
            return r.status == 200
    except Exception as e:
        print("telegram send failed:", e); return False

if __name__ == "__main__":
    print("configured:", enabled())
    if enabled(): print("send test:", send("notify.py test ✅"))
    else: print("no ~/live-refresh/telegram.json — see setup notes at the top of this file")
