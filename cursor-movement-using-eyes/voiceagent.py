import os
import sys
import tempfile
import threading
import time
import speech_recognition as sr
from groq import Groq
from elevenlabs.client import ElevenLabs
import pygame
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT_BASE = (
    "You are a helpful voice assistant. Keep your responses concise and conversational "
    "since they will be spoken aloud. Avoid markdown, bullet points, or long lists. "
    "Respond naturally as if having a spoken conversation."
)


def build_system_prompt(tab_content=None):
    if not tab_content or not tab_content.strip():
        return SYSTEM_PROMPT_BASE
    return (
        f"{SYSTEM_PROMPT_BASE}\n\n"
        "The user has a webpage open and may ask questions about it. Here is the text content "
        "of the current tab (truncated if long):\n\n---\n"
        f"{tab_content[:30000].strip()}\n"
        "---\n\nUse this content to answer questions about the page when relevant."
    )

EXIT_PHRASES = {"quit", "exit", "stop", "goodbye", "bye"}

PREFERRED_VOICES = ["Bella", "Roger", "Sarah", "Laura", "Charlie"]


def _voice_to_dict(v):
    return {"id": v.voice_id, "name": v.name}


def fetch_preferred_voices(eleven_client):
    response = eleven_client.voices.get_all()
    name_set = {n.lower() for n in PREFERRED_VOICES}
    voices = []
    for v in response.voices:
        if v.name.lower() in name_set:
            voices.append(_voice_to_dict(v))
    order = {n.lower(): i for i, n in enumerate(PREFERRED_VOICES)}
    voices.sort(key=lambda v: order.get(v["name"].lower(), 99))
    return voices


def fetch_all_voices(eleven_client, limit=5):
    """Fallback when preferred voices (Bella, Roger, etc.) are not in the API response."""
    response = eleven_client.voices.get_all()
    voices = [_voice_to_dict(v) for v in response.voices[:limit]]
    return voices


def voices_for_display(voices, max_=5):
    """Cap at 5 and put Roger last so he is not the first option (he's the default)."""
    voices = voices[:max_]
    roger = [v for v in voices if _short_name(v["name"]).lower() == "roger"]
    others = [v for v in voices if _short_name(v["name"]).lower() != "roger"]
    return (others + roger)[:max_]


DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
SKIP_WORDS = {"skip", "no", "nope", "continue", "keep", "this one"}
PICK_WORDS = {"choose", "voices", "options", "yes", "yeah", "sure", "show", "hear", "listen"}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def parse_voice_choice(text, voices):
    if not text:
        return None
    text_lower = text.lower().strip()
    words = text_lower.split()
    words_set = set(words)
    if words_set & SKIP_WORDS:
        return "skip"
    if words_set & PICK_WORDS:
        return "pick"
    if text_lower in SKIP_WORDS:
        return "skip"
    if text_lower in PICK_WORDS:
        return "pick"
    if text_lower in NUMBER_WORDS:
        n = NUMBER_WORDS[text_lower]
        if 1 <= n <= len(voices):
            return n
    if text_lower.isdigit() and 1 <= int(text_lower) <= len(voices):
        return int(text_lower)
    for word in words:
        clean = word.rstrip(".,!?")
        if clean in NUMBER_WORDS:
            n = NUMBER_WORDS[clean]
            if 1 <= n <= len(voices):
                return n
        if clean.isdigit() and 1 <= int(clean) <= len(voices):
            return int(clean)
    for i, v in enumerate(voices, 1):
        if v["name"].lower() == text_lower:
            return i
    return None


def _short_name(name):
    return name.split("-")[0].strip() if name else name


def offer_voice_choice(eleven_client, speak_fn, listen_fn, recognizer, microphone, voices):
    roger = next((v for v in voices if _short_name(v["name"]).lower() == "roger"), None)
    if roger:
        default_id, default_name = roger["id"], roger["name"]
    elif voices:
        default_id, default_name = voices[0]["id"], voices[0]["name"]
    else:
        default_id, default_name = DEFAULT_VOICE_ID, "your assistant"

    short_default = _short_name(default_name)
    speak_fn(eleven_client, default_id, f"Hello! I'm your voice assistant. Would you like to choose a different voice? Say skip to keep {short_default}, or say choose to hear the options.")
    user = listen_fn(recognizer, microphone)

    choice = parse_voice_choice(user, voices) if user else None
    if choice == "skip" or choice is None:
        if not user:
            speak_fn(eleven_client, default_id, "No problem. Keeping this voice.")
        print(f"\n  Using voice: {short_default}\n")
        return default_id

    if choice != "pick":
        speak_fn(eleven_client, default_id, "I didn't catch that. Say skip to keep this voice, or say choose to hear options.")
        return offer_voice_choice(eleven_client, speak_fn, listen_fn, recognizer, microphone, voices)

    if not voices:
        speak_fn(eleven_client, default_id, "Sorry, I couldn't load the voice options. Keeping this voice.")
        print(f"\n  Using voice: {short_default}\n")
        return default_id

    speak_fn(eleven_client, default_id, "Here are the voices. Each will say their name.")
    for i, v in enumerate(voices, 1):
        short = _short_name(v["name"])
        print(f"  {i}. {short}")
        speak_fn(eleven_client, v["id"], f"I'm {short}.")

    n = len(voices)
    speak_fn(eleven_client, default_id, f"Say a number from 1 to {n} to pick a voice, or say skip to keep {short_default}.")
    user = listen_fn(recognizer, microphone)
    parsed = parse_voice_choice(user, voices) if user else None

    if parsed == "skip" or parsed is None:
        print(f"\n  Using voice: {short_default}\n")
        return default_id
    if parsed == "pick":
        parsed = 1
    if isinstance(parsed, int):
        selected = voices[parsed - 1]
        short_selected = _short_name(selected["name"])
        speak_fn(eleven_client, selected["id"], f"Got it. You're talking to {short_selected} now.")
        print(f"\n  Using voice: {short_selected}\n")
        return selected["id"]

    speak_fn(eleven_client, default_id, "Keeping the first voice. How can I help you?")
    print(f"\n  Using voice: {short_default}\n")
    return default_id


def speak(eleven_client, voice_id, text):
    print(f"Agent: {text}")
    audio = eleven_client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
    )
    audio_bytes = b"".join(audio)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    finally:
        pygame.mixer.music.unload()
        os.unlink(tmp_path)


def listen(recognizer, microphone):
    result = [None]
    done = threading.Event()

    def do_recognize():
        with microphone as source:
            audio = recognizer.listen(source, phrase_time_limit=15)
        try:
            result[0] = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            result[0] = None
        except sr.RequestError as e:
            print(f"\nSpeech recognition service error: {e}")
            result[0] = None
        done.set()

    thread = threading.Thread(target=do_recognize, daemon=True)
    thread.start()
    print("Listening... ", end="", flush=True)
    while not done.is_set():
        time.sleep(0.4)
        print(".", end="", flush=True)
    print()
    text = result[0]
    if text is not None:
        print(f"You: {text}", flush=True)
    return text


def get_llm_response(client, history):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history,
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content


def user_wants_to_choose_voice(client, user_text):
    if not user_text or len(user_text.strip()) < 2:
        return False
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You determine if the user wants to change, choose, or pick their voice assistant voice (e.g. 'voice again', 'change voice', 'choose a voice', 'switch voice', 'different voice'). Reply with exactly YES or NO and nothing else.",
                },
                {"role": "user", "content": user_text.strip()},
            ],
            temperature=0,
            max_tokens=10,
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        return "YES" in answer
    except Exception:
        return False


def choose_voice_again(eleven_client, speak_fn, listen_fn, recognizer, microphone, voices, current_voice_id):
    default_voice = next((v for v in voices if v["id"] == current_voice_id), None)
    if default_voice:
        default_id = current_voice_id
        short_default = _short_name(default_voice["name"])
    else:
        default_id = current_voice_id
        short_default = "your current voice"

    speak_fn(eleven_client, default_id, "Here are the voices. Each will say their name.")
    for i, v in enumerate(voices, 1):
        short = _short_name(v["name"])
        print(f"  {i}. {short}")
        speak_fn(eleven_client, v["id"], f"I'm {short}.")
    n = len(voices)
    speak_fn(eleven_client, default_id, f"Say a number from 1 to {n} to pick a voice, or say skip to keep {short_default}.")
    user = listen_fn(recognizer, microphone)
    parsed = parse_voice_choice(user, voices) if user else None

    if parsed == "skip" or parsed is None:
        print(f"\n  Using voice: {short_default}\n")
        return current_voice_id
    if parsed == "pick":
        parsed = 1
    if isinstance(parsed, int):
        selected = voices[parsed - 1]
        short_selected = _short_name(selected["name"])
        speak_fn(eleven_client, selected["id"], f"Got it. You're talking to {short_selected} now.")
        print(f"\n  Using voice: {short_selected}\n")
        return selected["id"]
    print(f"\n  Using voice: {short_default}\n")
    return current_voice_id


def main():
    groq_key = os.getenv("GROQ_API_KEY")
    eleven_key = os.getenv("ELEVENLABS_API_KEY")

    if not groq_key:
        print("Error: GROQ_API_KEY not set.")
        print("Get a free key at https://console.groq.com")
        sys.exit(1)

    if not eleven_key:
        print("Error: ELEVENLABS_API_KEY not set.")
        print("Get a free key at https://elevenlabs.io")
        sys.exit(1)

    groq_client = Groq(api_key=groq_key)
    eleven_client = ElevenLabs(api_key=eleven_key)
    pygame.mixer.init()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    voices = fetch_preferred_voices(eleven_client)
    if not voices:
        voices = fetch_all_voices(eleven_client)
    voices = voices_for_display(voices, max_=5)

    voice_name = os.getenv("VOICE_NAME", "").strip()
    if voice_name:
        selected = next(
            (v for v in voices if _short_name(v["name"]).lower() == voice_name.lower()),
            voices[0] if voices else None,
        )
        voice_id = selected["id"] if selected else DEFAULT_VOICE_ID
        if selected:
            print(f"  Using voice: {_short_name(selected['name'])}\n")
    else:
        voice_id = offer_voice_choice(eleven_client, speak, listen, recognizer, microphone, voices)

    tab_content = None
    tab_file = os.getenv("TAB_CONTENT_FILE")
    if tab_file and os.path.isfile(tab_file):
        try:
            with open(tab_file, encoding="utf-8") as f:
                tab_content = f.read()
        except Exception:
            pass
    system_prompt = build_system_prompt(tab_content)
    history = [{"role": "system", "content": system_prompt}]
    speak(eleven_client, voice_id, "How can I help you?")

    try:
        while True:
            user_text = listen(recognizer, microphone)

            if user_text is None:
                speak(eleven_client, voice_id, "Sorry, I didn't catch that. Could you say it again?")
                continue

            if user_text.lower().strip() in EXIT_PHRASES:
                speak(eleven_client, voice_id, "Goodbye!")
                break

            if user_wants_to_choose_voice(groq_client, user_text):
                voice_id = choose_voice_again(
                    eleven_client, speak, listen, recognizer, microphone, voices, voice_id
                )
                speak(eleven_client, voice_id, "How can I help you?")
                continue

            history.append({"role": "user", "content": user_text})

            try:
                reply = get_llm_response(groq_client, history)
            except Exception as e:
                print(f"LLM error: {e}")
                speak(eleven_client, voice_id, "Sorry, I had trouble thinking of a response. Try again.")
                history.pop()
                continue

            history.append({"role": "assistant", "content": reply})
            speak(eleven_client, voice_id, reply)

    except KeyboardInterrupt:
        print()
        speak(eleven_client, voice_id, "Goodbye!")


if __name__ == "__main__":
    main()
