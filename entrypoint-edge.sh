#!/bin/bash
# entrypoint-edge.sh
# Auto-preprocess data if processed.csv doesn't exist yet, then run the edge.
set -e

PROCESSED="data/processed/processed.csv"

if [ ! -f "$PROCESSED" ]; then
    echo "[Entrypoint] processed.csv not found — running preprocess..."
    python -c "
import sys
sys.path.insert(0, '.')
from data.preprocess import preprocess
preprocess()
print('[Entrypoint] Preprocessing complete.')
"
else
    echo "[Entrypoint] processed.csv found. Skipping preprocess."
fi

exec python -m edge.main
