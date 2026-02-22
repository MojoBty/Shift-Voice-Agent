#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Try venv in cursor-movement-using-eyes, then parent, then system python3
for VENV in "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/../.venv"; do
  PY="$VENV/bin/python3"
  if [ -x "$PY" ]; then
    exec "$PY" "$SCRIPT_DIR/native_host.py"
  fi
done
exec /usr/bin/env python3 "$SCRIPT_DIR/native_host.py"
