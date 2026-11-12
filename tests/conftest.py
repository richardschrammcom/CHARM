"""Test setup: throwaway keys and profile, so tests never touch the real keys/ directory."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = Path(tempfile.mkdtemp(prefix="charm-guard-test-"))
os.environ["CHARM_KEYS_DIR"] = str(_tmp / "keys")
os.environ["CHARM_USER_PROFILE"] = str(_tmp / "user_profile.json")
os.environ["TRACE_COLLECTOR_URL"] = ""  # empty, not unset, so importing an agent can't load it from .env
(_tmp / "user_profile.json").write_text(json.dumps({"name": "Test User", "email": "owner@example.com",
                                                     "phone": "5135550100"}))

import charm_guard  # noqa: E402

charm_guard.generate_keys()
charm_guard.generate_tls()
