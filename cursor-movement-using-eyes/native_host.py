#!/usr/bin/env python3
"""
Native messaging host for the Voice Agent Chrome extension.
"""

import json
import os
import signal
import struct
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VOICEAGENT_SCRIPT = SCRIPT_DIR / "voiceagent.py"
UNIFIED_AGENT_SCRIPT = SCRIPT_DIR / "unified_agent.py"
CALIBRATE_SCRIPT = SCRIPT_DIR / "calibrate.py"
BEST_GAZE_CALIB = SCRIPT_DIR / "best-gaze" / "calibration.json"
PID_FILE = Path.home() / ".cursor-voiceagent.pid"
TAB_CONTENT_FILE = Path.home() / ".cursor-voiceagent-tab.txt"


def read_message():
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) == 0:
        return None
    msg_len = struct.unpack("@I", raw_len)[0]
    return json.loads(sys.stdin.buffer.read(msg_len).decode("utf-8"))


def send_message(msg):
    encoded = json.dumps(msg).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("@I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def kill_all_voice_agents():
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if pid > 0:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except ValueError:
            pass
        PID_FILE.unlink(missing_ok=True)
    for _ in range(2):
        for proc_name in ["voiceagent.py", "unified_agent.py", "calibrate.py"]:
            subprocess.run(
                ["pkill", "-9", "-f", proc_name],
                capture_output=True,
                timeout=2,
            )


def start_voice_agent(msg):
    kill_all_voice_agents()

    if not BEST_GAZE_CALIB.exists():
        return {"success": False, "error": "Calibration required. Click Calibrate first."}

    voice_name = (msg.get("voiceName") or "").strip()
    tab_content = msg.get("tabContent") or ""

    if tab_content:
        TAB_CONTENT_FILE.write_text(tab_content, encoding="utf-8")
        tab_file_env = f'export TAB_CONTENT_FILE="{TAB_CONTENT_FILE}" && '
    else:
        if TAB_CONTENT_FILE.exists():
            TAB_CONTENT_FILE.unlink(missing_ok=True)
        tab_file_env = ""

    try:
        python_path = sys.executable
        env_str = f'export VOICE_NAME="{voice_name.replace(chr(34), "")}" && ' if voice_name else ""
        env_str = tab_file_env + env_str
        cmd = f'cd "{SCRIPT_DIR}" && {env_str}exec "{python_path}" "{UNIFIED_AGENT_SCRIPT}"; echo ""; read -p "Press Enter to close..."'
        proc = subprocess.Popen(
            [
                "osascript", "-e",
                f'tell application "Terminal" to do script "{cmd.replace(chr(34), chr(92)+chr(34))}"'
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=2)
        PID_FILE.write_text("-1")
        return {"success": True}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"success": False, "error": "Failed to open Terminal"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def stop_voice_agent():
    kill_all_voice_agents()
    return {"success": True}


def run_calibration():
    """Launch calibrate.py in Terminal for eye tracking calibration."""
    try:
        python_path = sys.executable
        cmd = f'cd "{SCRIPT_DIR}" && exec "{python_path}" "{CALIBRATE_SCRIPT}"; echo ""; read -p "Press Enter to close..."'
        proc = subprocess.Popen(
            [
                "osascript", "-e",
                f'tell application "Terminal" to do script "{cmd.replace(chr(34), chr(92)+chr(34))}"'
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=2)
        return {"success": True}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"success": False, "error": "Failed to open Terminal"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    msg = read_message()
    if msg is None:
        sys.exit(1)
    action = msg.get("action", "")
    if action == "start":
        result = start_voice_agent(msg)
    elif action == "stop":
        result = stop_voice_agent()
    elif action == "calibrate":
        result = run_calibration()
    else:
        result = {"success": False, "error": f"Unknown action: {action}"}
    send_message(result)


if __name__ == "__main__":
    main()
