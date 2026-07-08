#!/bin/bash
# Apply the openpi patches to an openpi install (backs up originals as .orig).
# Usage: ./apply.sh [site-packages dir]
# Default target is the container venv used by the pi0.5 eval/rollout.
set -euo pipefail

SITE="${1:-/opt/venv/openpi/lib/python3.11/site-packages}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$SITE/openpi" ]; then
    echo "openpi not found under $SITE" >&2
    exit 1
fi

for f in models_pytorch/pi0_pytorch.py models/model.py shared/array_typing.py; do
    dst="$SITE/openpi/$f"
    [ -f "$dst.orig" ] || cp "$dst" "$dst.orig"
    cp "$SRC/$f" "$dst"
    echo "patched $dst"
done

echo "Done. All patches are unconditional (originals kept as .orig)."
