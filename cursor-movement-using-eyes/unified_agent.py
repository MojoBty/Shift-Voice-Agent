#!/usr/bin/env python3
"""
Unified Voice Agent + Eye Tracking.
Runs voice agent and best-gaze eye tracking in one process.
Calibration (best-gaze/calibration.json) must exist before starting.
"""

import os
import sys
import threading
import time
from pathlib import Path

import pyautogui
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
BEST_GAZE_DIR = SCRIPT_DIR / "best-gaze"
CALIB_PATH = BEST_GAZE_DIR / "calibration.json"

sys.path.insert(0, str(BEST_GAZE_DIR))
import main as bg

# Import voice agent logic
sys.path.insert(0, str(SCRIPT_DIR))
import voiceagent as va

HEY_AGENT_PHRASES = ("hey agent", "hey, agent")
RELEASE_PHRASES = ("release", "unlock")


def main():
    if not CALIB_PATH.exists():
        print("Error: Calibration required. Run Calibrate from the extension popup first.")
        sys.exit(1)

    cal = bg.load_cal()
    if not cal or "iris_x_min" not in cal:
        print("Error: Invalid calibration file. Run Calibrate again.")
        sys.exit(1)

    shared_state = {
        "gaze_locked": False,
        "lock_x": pyautogui.size()[0] // 2,
        "lock_y": pyautogui.size()[1] // 2,
        "running": True,
        "captured_text": "",
    }

    def voice_loop():
        """Run voice agent in background thread (OpenCV GUI must use main thread)."""
        try:
            _run_voice_loop(shared_state)
        except Exception as e:
            print(f"Voice loop error: {e}")
        finally:
            shared_state["running"] = False

    voice_thread = threading.Thread(target=voice_loop, daemon=True)
    voice_thread.start()
    time.sleep(1.0)  # Let voice init (mic adjustment, etc.) start

    # Eye tracking MUST run on main thread - OpenCV GUI fails in threads on macOS
    bg.run_tracking_loop(cal, shared_state)


def _run_voice_loop(shared_state):
    groq_key = os.getenv("GROQ_API_KEY")
    eleven_key = os.getenv("ELEVENLABS_API_KEY")

    if not groq_key:
        print("Error: GROQ_API_KEY not set.")
        sys.exit(1)
    if not eleven_key:
        print("Error: ELEVENLABS_API_KEY not set.")
        sys.exit(1)

    groq_client = va.Groq(api_key=groq_key)
    eleven_client = va.ElevenLabs(api_key=eleven_key)
    va.pygame.mixer.init()
    recognizer = va.sr.Recognizer()
    microphone = va.sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    voices = va.fetch_preferred_voices(eleven_client)
    if not voices:
        voices = va.fetch_all_voices(eleven_client)
    voices = va.voices_for_display(voices, max_=5)

    voice_name = os.getenv("VOICE_NAME", "").strip()
    if voice_name:
        selected = next(
            (
                v
                for v in voices
                if va._short_name(v["name"]).lower() == voice_name.lower()
            ),
            voices[0] if voices else None,
        )
        voice_id = selected["id"] if selected else va.DEFAULT_VOICE_ID
        if selected:
            print(f"  Using voice: {va._short_name(selected['name'])}\n")
    else:
        voice_id = va.offer_voice_choice(
            eleven_client, va.speak, va.listen, recognizer, microphone, voices
        )

    tab_content = None
    tab_file = os.getenv("TAB_CONTENT_FILE")
    if tab_file and os.path.isfile(tab_file):
        try:
            with open(tab_file, encoding="utf-8") as f:
                tab_content = f.read()
        except Exception:
            pass

    system_prompt = va.build_system_prompt(tab_content)
    history = [{"role": "system", "content": system_prompt}]

    def _describe_text(text):
        """Describe the captured text and speak it (called when user presses L)."""
        try:
            prompt = (
                "The user is looking at this text on their screen. "
                "Briefly describe or explain what it says in 1-2 clear sentences, as if speaking to them. "
                "Be concise and natural."
            )
            msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
            reply = va.get_llm_response(groq_client, msgs)
            if reply:
                va.speak(eleven_client, voice_id, reply)
        except Exception as e:
            print(f"Describe error: {e}")

    shared_state["on_lock_describe"] = _describe_text

    va.speak(eleven_client, voice_id, "Voice agent with eye tracking ready. Press L to lock on and hear about what you're looking at. Say 'Hey Agent' to lock with voice, 'release' to unlock. How can I help you?")

    try:
        while shared_state["running"]:
            user_text = va.listen(recognizer, microphone)

            if user_text is None:
                va.speak(
                    eleven_client,
                    voice_id,
                    "Sorry, I didn't catch that. Could you say it again?",
                )
                continue

            text_lower = user_text.lower().strip()

            if text_lower in va.EXIT_PHRASES:
                va.speak(eleven_client, voice_id, "Goodbye!")
                shared_state["running"] = False
                break

            if any(r in text_lower for r in RELEASE_PHRASES) and shared_state["gaze_locked"]:
                shared_state["gaze_locked"] = False
                bg.hide_text_window()
                print("  UNLOCKED")
                va.speak(eleven_client, voice_id, "Released.")
                continue

            if any(h in text_lower for h in HEY_AGENT_PHRASES):
                shared_state["gaze_locked"] = True
                shared_state["lock_x"], shared_state["lock_y"] = pyautogui.position()
                print(f"\n  LOCKED at ({shared_state['lock_x']}, {shared_state['lock_y']})")
                captured = bg.capture_text_at(
                    shared_state["lock_x"], shared_state["lock_y"]
                )
                shared_state["captured_text"] = captured
                bg.show_text_window(captured)

                rest = user_text
                for phrase in HEY_AGENT_PHRASES:
                    if phrase in text_lower:
                        idx = text_lower.index(phrase) + len(phrase)
                        rest = user_text[idx:].strip().strip(" ,")
                        break

                question = rest if rest else "What would you like to know about this text?"
                ctx = f"[User locked on text at cursor:\n\n{captured}\n\nUser asks: {question}]"
                history.append({"role": "user", "content": ctx})

                try:
                    reply = va.get_llm_response(groq_client, history)
                except Exception as e:
                    print(f"LLM error: {e}")
                    va.speak(
                        eleven_client,
                        voice_id,
                        "Sorry, I had trouble. Try again.",
                    )
                    history.pop()
                    continue

                history.append({"role": "assistant", "content": reply})
                va.speak(eleven_client, voice_id, reply)
                continue

            if va.user_wants_to_choose_voice(groq_client, user_text):
                voice_id = va.choose_voice_again(
                    eleven_client,
                    va.speak,
                    va.listen,
                    recognizer,
                    microphone,
                    voices,
                    voice_id,
                )
                va.speak(eleven_client, voice_id, "How can I help you?")
                continue

            history.append({"role": "user", "content": user_text})

            try:
                reply = va.get_llm_response(groq_client, history)
            except Exception as e:
                print(f"LLM error: {e}")
                va.speak(
                    eleven_client,
                    voice_id,
                    "Sorry, I had trouble thinking of a response. Try again.",
                )
                history.pop()
                continue

            history.append({"role": "assistant", "content": reply})
            va.speak(eleven_client, voice_id, reply)

    except KeyboardInterrupt:
        print()
        va.speak(eleven_client, voice_id, "Goodbye!")
    finally:
        shared_state["running"] = False
        bg.hide_text_window()
        time.sleep(0.3)
    print("  Done.")


if __name__ == "__main__":
    main()
