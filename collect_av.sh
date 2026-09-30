#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 3 ]]; then
    echo "Usage: bash collect_av.sh TASK CONFIG GPU [--episodes N ...]" >&2
    exit 2
fi
TASK="$1"; CONFIG="$2"; GPU="$3"; shift 3
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec python scripts/collect_av.py --task "$TASK" --task-config "$CONFIG" "$@"
