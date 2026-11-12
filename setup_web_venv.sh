#!/bin/bash
# setup_web_venv.sh
# Creates the ISOLATED virtualenv (.venv-web) for the Web Agent's browser-use runner.
# browser-use[core] cannot share the main framework venv (it removes/downgrades
# google-adk and a2a-sdk), so it lives here and is driven via web_browser_runner.py.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-web"

echo "Creating isolated browser venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing browser-use into the isolated venv ..."
"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements-web.txt"

echo "Installing Chromium for browser-use ..."
"$VENV_DIR/bin/browser-use" install

echo
echo "Done. The Web Agent will use $VENV_DIR/bin/python to run web_browser_runner.py."
echo "Set BROWSER_USE_API_KEY in .env to enable the cloud fallback."
