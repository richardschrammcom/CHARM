#!/bin/bash
# Before/after: an unauthenticated request asks the SMS Agent to send a text. No key, cert or signature.
#
#   demo/sms_unauthenticated.sh before [phone]   CHARM 2.1 (port 6003): sends a REAL text.
#   demo/sms_unauthenticated.sh after  [phone]   CHARM 3.0 (port 7003): same request, refused before it reaches the app.
#
# The phone defaults to the one in user_profile.json. 2.1 and 3.0 use different ports, so both can run.
# "before" uses 2.1's SMS Agent if it's already running; otherwise it starts only 2.1's registry and SMS
# Agent on 127.0.0.1 and stops them when it exits. Set CHARM_V21_DIR if 2.1 isn't at ../charm-v2.1.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
V21=${CHARM_V21_DIR:-$ROOT/../charm-v2.1}
MODE=${1:-}
PHONE=${2:-$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["phone"])' "$ROOT/user_profile.json")}
MSG="CHARM demo: this text was sent by an unauthenticated request."

payload() {
  cat <<EOF
{"jsonrpc":"2.0","id":"1","method":"message/send","params":{"message":{"role":"user","messageId":"$(uuidgen)",
 "parts":[{"kind":"text","text":"Send an SMS to $PHONE saying: $MSG"}]}}}
EOF
}

reply_text() {  # Print the agent's reply, or the raw response if it isn't a JSON-RPC result.
  python3 -c 'import sys,json
raw=sys.stdin.read()
try: r=json.loads(raw)
except ValueError: print(raw or "(no response)"); sys.exit()
print("".join(p.get("text","") for a in r.get("result",{}).get("artifacts",[]) for p in a.get("parts",[])) or raw)'
}

listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

wait_for() {  # wait_for URL SECONDS
  for _ in $(seq "$2"); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$1")" != 000 ] && return 0
    sleep 1
  done
  return 1
}

before() {
  LOGS=""
  if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:6003/.well-known/agent.json)" != 200 ]; then
    for p in 6060 6003; do
      if listening $p; then echo "Port $p is in use, but not by a 2.1 SMS Agent. Check what's running."; exit 1; fi
    done
    [ -d "$V21/.venv" ] || { echo "No CHARM 2.1 venv at $V21/.venv"; exit 1; }
    LOGS=$(mktemp -d)
    trap 'kill $REG $SMS 2>/dev/null; wait 2>/dev/null' EXIT
    cd "$V21" && source .venv/bin/activate
    uvicorn svr_registry_agent:app --host 127.0.0.1 --port 6060 > "$LOGS/registry.log" 2>&1 & REG=$!
    wait_for http://127.0.0.1:6060/ 30 || { echo "2.1 registry didn't start; see $LOGS/registry.log"; exit 1; }
    uvicorn svr_sms_agent:app --host 127.0.0.1 --port 6003 > "$LOGS/sms.log" 2>&1 & SMS=$!
    wait_for http://127.0.0.1:6003/.well-known/agent.json 60 || { echo "2.1 SMS Agent didn't start; see $LOGS/sms.log"; exit 1; }
  fi

  echo "== CHARM 2.1: plain HTTP, no credentials, asking for a text to $PHONE"
  payload | curl -s -m 120 -X POST http://127.0.0.1:6003/ -H 'Content-Type: application/json' -d @- | reply_text
  [ -n "$LOGS" ] && echo "(Logs: $LOGS)"
  [ -z "$(git -C "$V21" status --short)" ] || echo "Note: the 2.1 working tree changed: $(git -C "$V21" status --short | tr '\n' ' ')"
}

after() {
  cd "$ROOT"
  # Refuse to run unless 3.0 is the one listening, so this can't text through a running 2.1 by mistake.
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 8 --cacert keys/tls/ca.crt \
    --cert keys/tls/Dashboard.crt --key keys/tls/Dashboard.key https://127.0.0.1:7003/.well-known/agent.json)
  [ "$code" = 200 ] || { echo "CHARM 3.0's SMS Agent isn't answering over mutual TLS on 7003. Run ./startup.sh"; exit 1; }

  echo "== CHARM 3.0: the same request over plain HTTP"
  payload | curl -sS -m 20 -X POST http://127.0.0.1:7003/ -H 'Content-Type: application/json' -d @- 2>&1 | reply_text
  echo "== CHARM 3.0: the same request over HTTPS without a client cert"
  payload | curl -sS -m 20 --cacert keys/tls/ca.crt -X POST https://127.0.0.1:7003/ \
    -H 'Content-Type: application/json' -d @- 2>&1 | reply_text
  echo "An empty reply means the TLS layer dropped the connection: the request never reached the agent or its model."
}

case "$MODE" in
  before) before ;;
  after) after ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
