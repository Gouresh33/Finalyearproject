import speech_recognition as sr
import tempfile
import os
import logging
from voice_config import VOICE_PASSWORDS

def verify_voice_from_audio_bytes(audio_bytes):
    recognizer = sr.Recognizer()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name

        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)

        spoken_text = recognizer.recognize_google(audio).lower().strip()
        logging.info(f"Recognized voice text: {spoken_text}")

        os.remove(temp_path)

        for name, expected in VOICE_PASSWORDS.items():
            if spoken_text == expected.lower().strip():
                return (name, 1)

        return ("Unknown", 0)

    except Exception:
        logging.exception("Voice verification failed")
        return ("Unknown", 0)
