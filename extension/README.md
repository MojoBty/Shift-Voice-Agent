# Voice Agent Chrome Extension

Chrome extension that provides a button to start/stop the voice agent (voiceagent.py).

## Setup

### 1. Load the extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `extension` folder
5. Copy the extension ID (e.g. `abcdefghijklmnop`)

### 2. Install the native host

The voice agent is a Python script. The extension uses Chrome Native Messaging to launch it.

From the project root, run:

```bash
chmod +x install-native-host.sh
./install-native-host.sh <YOUR_EXTENSION_ID>
```

Replace `<YOUR_EXTENSION_ID>` with the ID from step 1.

### 3. Environment

Ensure voiceagent.py works on its own:

- Python 3 with dependencies: `pip install -r cursor-movement-using-eyes/requirements.txt`
- `.env` in `cursor-movement-using-eyes/` with `GROQ_API_KEY` and `ELEVENLABS_API_KEY`
- Microphone access

## Usage

Click the extension icon → **Start Voice Agent** to run the voice assistant. Use **Stop Voice Agent** to end it.
