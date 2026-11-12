#!/bin/bash
# Quick start: launch CHARM 2.1 (before) and CHARM 3.0 (after) side by side.
# Each version's own startup.sh only touches its own ports, so they don't collide.
# Stop them with ./shutdown.sh in each folder.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V21="$HERE/../charm-v2.1"

echo "=== Starting CHARM 2.1 (before) ==="
(cd "$V21" && ./startup.sh) || echo "WARNING: CHARM 2.1 startup reported an error"

echo ""
echo "=== Starting CHARM 3.0 (after) ==="
(cd "$HERE" && ./startup.sh) || echo "WARNING: CHARM 3.0 startup reported an error"

echo ""
echo "CHARM 2.1 dashboard: http://localhost:9090"
echo "CHARM 3.0 dashboard: http://localhost:9091"
