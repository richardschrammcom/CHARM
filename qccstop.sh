#!/bin/bash
# Quick stop: shut down CHARM 2.1 (before) and CHARM 3.0 (after).
# Runs each version's own shutdown.sh from inside its folder.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V21="$HERE/../charm-v2.1"

echo "=== Stopping CHARM 2.1 (before) ==="
(cd "$V21" && ./shutdown.sh) || echo "WARNING: CHARM 2.1 shutdown reported an error"

echo ""
echo "=== Stopping CHARM 3.0 (after) ==="
(cd "$HERE" && ./shutdown.sh) || echo "WARNING: CHARM 3.0 shutdown reported an error"
