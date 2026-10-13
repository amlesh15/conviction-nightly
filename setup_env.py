"""Writes cookie.txt + telegram.json from GitHub secrets so the original scripts work unchanged."""
import os, json
d = os.path.expanduser("~/live-refresh"); os.makedirs(d, exist_ok=True)
ck = "".join((os.environ.get("SENSIBULL_COOKIE", "")).split("\n")).strip()
open(os.path.join(d, "cookie.txt"), "w").write(ck)
tok = os.environ.get("TELEGRAM_TOKEN", "").strip(); cid = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
if tok and cid: json.dump({"token": tok, "chat_id": cid}, open(os.path.join(d, "telegram.json"), "w"))
os.makedirs(os.environ.get("STATE_DIR", os.path.join(d, "state")), exist_ok=True)
print(f"cookie chars: {len(ck)} | telegram configured: {bool(tok and cid)}")
