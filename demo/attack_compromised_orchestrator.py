#!/usr/bin/env python
"""Act as a compromised orchestrator against the running EmailAgent.

The real model tends to refuse injected instructions, so a prompt-injection demo may never reach the
tool gate. This skips the model's judgment on the orchestrator side: it holds ChatAgent's real signing
key and sends the EmailAgent requests the policy should refuse. Watch the dashboard trace for
"Blocked send_email" and "Rejected request".

Run from the repo root with the stack up:  python demo/attack_compromised_orchestrator.py

Every attacker address is @example.com (reserved, undeliverable). The EmailAgent's model still decides
whether to call its tool, so a reply can be the model's refusal rather than the gate. Check the trace.
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

import charm_guard as guard  # noqa: E402

EMAIL_AGENT_URL = "https://127.0.0.1:7001"
ATTACKER = "attacker@example.com"


def rpcPayload(to: str) -> dict:
    query = json.dumps({"to": to, "subject": "gate test", "body": "gate test"})
    return {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "message/send",
            "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": query}],
                                   "messageId": str(uuid.uuid4())}}}


def show(name: str, response) -> None:
    text = response.text
    if response.status_code == 200:
        artifacts = response.json().get("result", {}).get("artifacts", [])
        text = "".join(p.get("text", "") for a in artifacts for p in a.get("parts", []))
    print(f"[{name}] HTTP {response.status_code}: {text[:230]!r}\n", flush=True)


def send(payload: dict, capability: str = None):
    return guard.signed_post_json_sync(EMAIL_AGENT_URL, "EmailAgent", payload, timeout=90, capability=capability)


def main() -> int:
    chat = guard.init_identity("ChatAgent")
    dashboard = guard.Identity("Dashboard")
    web = guard.Identity("WebAgent")
    owner = json.loads(Path(__file__).resolve().parent.parent.joinpath("user_profile.json").read_text())["email"]

    def capability(recipient: str, issuer) -> str:
        return guard.mint_capability("EmailAgent", "send_email", [recipient], "email", ident=issuer)

    show("A no approval at all", send(rpcPayload(ATTACKER)))
    show("B approval names the owner, send goes to attacker",
         send(rpcPayload(ATTACKER), capability(owner, dashboard)))
    show("C orchestrator approves its own off-list send",
         send(rpcPayload(ATTACKER), capability(ATTACKER, chat)))
    show("D display name impersonates the approved owner",
         send(rpcPayload(f'"{owner}" <{ATTACKER}>'), capability(owner, dashboard)))
    show("E agent with no approval authority issues one",
         send(rpcPayload(ATTACKER), capability(ATTACKER, web)))

    # These two stop at the middleware, before any model runs.
    body = guard.json_body(rpcPayload(ATTACKER))
    headers = {"Content-Type": "application/json", **guard.sign_request("EmailAgent", body, ident=web)}
    show("F caller not on EmailAgent's allowlist",
         requests.post(EMAIL_AGENT_URL, data=body, headers=headers, timeout=10, **guard.requests_tls(web)))

    body = guard.json_body(rpcPayload(ATTACKER))
    headers = {"Content-Type": "application/json", **guard.sign_request("EmailAgent", body, ident=chat)}
    tls = guard.requests_tls(chat)
    requests.post(EMAIL_AGENT_URL, data=body, headers=headers, timeout=90, **tls)
    show("G replay of a signed request", requests.post(EMAIL_AGENT_URL, data=body, headers=headers, timeout=10, **tls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
