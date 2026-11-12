# chat_client.py
# CLI client for the Chat Agent using A2A JSON-RPC protocol.

import requests
import uuid
import os
import json
import argparse
from shared_logger import SharedLogger
import charm_guard as guard

# Logging
logger_name = os.path.splitext(os.path.basename(__file__))[0]
logger = SharedLogger.get_logger(logger_name, console_level_override="WARNING")
logger.info(f"{logger_name} has started.")

AGENT_BASE_URL = "https://127.0.0.1:7002"
guard.init_identity("ChatClient")  # CHARM 3.0: the chat agent only accepts signed callers


def check_agent_availability():
    agent_card_url = f"{AGENT_BASE_URL}/.well-known/agent.json"
    try:
        response = requests.get(agent_card_url, **guard.requests_tls())
        if response.status_code != 200:
            print(f"Error fetching agent card: {response.status_code}. Exiting.")
            exit(1)
        card = response.json()
        print(f"Connected to: {card.get('name', 'Unknown Agent')}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to chat agent. Make sure the server is running on port 7002.")
        exit(1)


def send_request(user_text, context_id=None):
    """Send a request to the agent using A2A JSON-RPC protocol."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": user_text}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }

    # Include context ID for conversation continuity
    if context_id:
        payload["params"]["message"]["contextId"] = context_id

    result = guard.signed_post_json_sync(AGENT_BASE_URL, "ChatAgent", payload, timeout=780)
    if result.status_code != 200:
        print(f"Request failed: {result.status_code}, {result.text}")
        return None, context_id

    rpc_response = result.json()

    # Check for JSON-RPC error
    if "error" in rpc_response:
        error = rpc_response["error"]
        print(f"Agent error: {error.get('message', 'Unknown error')}")
        return None, context_id

    # Extract response
    task_result = rpc_response.get("result", {})

    # Get context ID for conversation continuity
    new_context_id = task_result.get("contextId", context_id)

    # Extract text from artifacts
    artifacts = task_result.get("artifacts", [])
    agent_reply = ""
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                agent_reply += part.get("text", "")

    if agent_reply:
        print(f"\nAgent: {agent_reply}\n")
    else:
        print("No response from agent.")

    return agent_reply, new_context_id


def main():
    parser = argparse.ArgumentParser(description='Chat client for A2A agent')
    parser.parse_args()

    check_agent_availability()

    context_id = None
    print("Enter 'exit' or 'quit' to exit")
    print()

    while True:
        input_text = input("You: ").strip()
        if input_text.lower() in ("exit", "quit"):
            break
        if not input_text:
            continue

        try:
            _, context_id = send_request(input_text, context_id)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
