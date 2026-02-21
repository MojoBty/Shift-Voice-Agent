#!/bin/bash
# Install the native messaging host for the Voice Agent extension.
# Run this after loading the extension to get your extension ID.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_HOST="$SCRIPT_DIR/cursor-movement-using-eyes/native_host_launcher.sh"
MANIFEST_SRC="$SCRIPT_DIR/cursor-movement-using-eyes/com.cursor.voiceagent.json"

# Chrome native host manifests on macOS
CHROME_HOSTS="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"
mkdir -p "$CHROME_HOSTS"

# Get extension ID: load the extension in chrome://extensions first
if [ -z "$1" ]; then
  echo "Usage: ./install-native-host.sh <EXTENSION_ID>"
  echo ""
  echo "To get your extension ID:"
  echo "  1. Open chrome://extensions"
  echo "  2. Enable Developer mode"
  echo "  3. Load unpacked extension from: $SCRIPT_DIR/extension"
  echo "  4. Copy the extension ID (e.g. abcdefghijklmnopqrstuvwxyz123456)"
  echo "  5. Run: ./install-native-host.sh <YOUR_EXTENSION_ID>"
  exit 1
fi

EXTENSION_ID="$1"
MANIFEST_DEST="$CHROME_HOSTS/com.cursor.voiceagent.json"

# Create manifest using Python (no jq required)
python3 - "$NATIVE_HOST" "chrome-extension://${EXTENSION_ID}/" "$MANIFEST_SRC" "$MANIFEST_DEST" << 'PYTHON'
import json, sys
path, origin, src, dest = sys.argv[1:5]
with open(src) as f:
    m = json.load(f)
m["path"] = path
m["allowed_origins"] = [origin]
with open(dest, "w") as f:
    json.dump(m, f, indent=2)
PYTHON

chmod +x "$NATIVE_HOST"

echo "Native host installed at: $MANIFEST_DEST"
echo "Extension ID: $EXTENSION_ID"
