#!/bin/bash
# List every card in the registry. CHARM 3.0: registry reads must be signed, so this script
# signs as the Dashboard identity (run it from this folder with the venv active).
cd "$(dirname "$0")" && python - <<'PY'
import json, uuid
import charm_guard as g
g.init_identity("Dashboard")
r = g.signed_post_json_sync("https://127.0.0.1:7060/tasks/send", "RegistryAgent",
                            {"id": str(uuid.uuid4()), "tool_name": "list_agent_cards"}, timeout=10)
print(json.dumps(r.json(), indent=2))
PY
