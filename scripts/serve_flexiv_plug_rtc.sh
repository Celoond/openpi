#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR [server options]" >&2
  exit 2
fi
checkpoint_dir="$1"
shift
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
exec .venv/bin/python openpi_zmq_server.py --config flexiv_plug_rtc \
  --checkpoint "$checkpoint_dir" --addr 'tcp://*:5555' "$@"
