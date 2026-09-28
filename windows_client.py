import io
import sys
import time
import signal
import threading
import wave
import requests
import numpy as np
import noisereduce as nr
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import subprocess
import os
import urllib.parse
import asyncio
import edge_tts
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq()


# ============================================================
# CONFIG
# ============================================================

SERVER_URL = "http://127.0.0.1:8000/chat"

# Buddy states
SLEEPING = False
IS_SPEAKING = False

stop_event = threading.Event()
shutdown_event = threading.Event()


# ============================================================
# SPEECH RECOGNIZER
# ============================================================

recognizer = sr.Recognizer()

recognizer.pause_threshold = 0.35
recognizer.non_speaking_duration = 0.2
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 300


# ============================================================
# CTRL+C
# ============================================================

def force_exit(sig, frame):
    print("\n\n🛑 Buddy stopped.")
    shutdown_event.set()
    stop_event.set()
    try:
        sd.stop()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, force_exit)


# ============================================================
# STOP BUDDY SPEAKING
# ============================================================

def stop_speaking():
    global IS_SPEAKING
    stop_event.set()
    try:
        sd.stop()
    except Exception:
        pass
    IS_SPEAKING = False
    print("\n🛑 Buddy stopped speaking.")


# ============================================================
# STOP COMMAND MONITOR (WITH WHISPER LANGUAGE="en")
# ============================================================

def monitor_stop():
    global IS_SPEAKING
    stop_recognizer = sr.Recognizer()
    stop_recognizer.energy_threshold = 300
    stop_recognizer.pause_threshold = 0.2

    try:
        with sr.Microphone() as stop_source:
            while IS_SPEAKING and not stop_event.is_set():
                try:
                    audio = stop_recognizer.listen(
                        stop_source,
                        timeout=0.4,
                        phrase_time_limit=1.0
                    )

                    wav_buffer = io.BytesIO(audio.get_wav_data())
                    wav_buffer.name = "stop_input.wav"

                    transcription = groq_client.audio.transcriptions.create(
                        file=wav_buffer,
                        model="whisper-large-v3-turbo",
                        response_format="text",
                        language="en"
                    )

                    text = transcription.lower().strip()
                    print(f"\n🎧 Stop monitor: {text}")

                    stop_words = ["stop", "buddy stop", "aagu", "agu", "wait", "chal", "chalu", "apu"]
                    if any(word in text for word in stop_words):
                        stop_speaking()
                        break
                except Exception:
                    pass
    except Exception:
        pass


# ============================================================
# PLAY AUDIO
# ============================================================

def play_audio(audio_bytes):
    global IS_SPEAKING
    if not audio_bytes:
        return

    stop_event.clear()
    IS_SPEAKING = True

    monitor_thread = threading.Thread(
        target=monitor_stop,
        daemon=True
    )
    monitor_thread.start()

    try:
        data, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32"
        )
        sd.play(data, samplerate=sample_rate)

        while IS_SPEAKING:
            if stop_event.is_set():
                sd.stop()
                break
            try:
                if not sd.get_stream().active:
                    break
            except Exception:
                pass
            time.sleep(0.02)
    except Exception as e:
        print(f"\n⚠️ Audio error: {e}")
    finally:
        IS_SPEAKING = False
        try:
            sd.stop()
        except Exception:
            pass


# ============================================================
# ASYNC TTS HELPER
# ============================================================

async def speak_text_async(text, voice="en-IN-NeerjaNeural"):
    communicate = edge_tts.Communicate(text, voice, rate="+10%")
    audio_stream = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    audio_stream.seek(0)
    return audio_stream.read()


def speak_response(text):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    audio_bytes = loop.run_until_complete(speak_text_async(text))
    loop.close()
    play_audio(audio_bytes)


# ============================================================
# ROBUST VOICE CONFIRMATION
# ============================================================

def speak_and_wait_for_confirmation(question_text):
    print(f"\n🤖 Buddy: {question_text}")
    speak_response(question_text)

    print("\n🎙️ Listening for your voice confirmation...")
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=3
            )

            print("⚡ Recognizing confirmation with Whisper...")
            wav_buffer = io.BytesIO(audio.get_wav_data())
            wav_buffer.name = "confirm_input.wav"

            transcription = groq_client.audio.transcriptions.create(
                file=wav_buffer,
                model="whisper-large-v3-turbo",
                response_format="text",
                language="en"
            )

            confirm_text = transcription.strip().lower()
            print(f"👉 You said: {confirm_text}")

            # Expanded yes words to include 's', 'chey', 'yes', 'ok', etc.
            yes_words = ["yes", "s", "yeah", "yep", "ha", "y", "chey", "cheyyi", "ok", "okay", "sure", "sari", "cheshey", "sey", "seyyi"]
            if any(word in confirm_text for word in yes_words):
                return True

            return False
        except Exception as e:
            print(f"⚠️ Confirmation error or timeout: {e}")
            return False


# ============================================================
# LISTEN TO USER
# ============================================================

def listen_to_user(source):
    print("\n🎙️ Listening...")

    try:
        audio = recognizer.listen(
            source,
            timeout=6,
            phrase_time_limit=4
        )

        print("⚡ Recognizing with Groq Whisper...")

        wav_buffer = io.BytesIO(audio.get_wav_data())
        wav_buffer.name = "buddy_input.wav"

        transcription = groq_client.audio.transcriptions.create(
            file=wav_buffer,
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en"
        )

        user_text = transcription.strip()
        print(f"👉 You: {user_text}")
        return user_text

    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        print(f"⚠️ Groq STT error: {e}")
        return None


# ============================================================
# FULL LAPTOP ACCESS & MULTITASKING CONTROLS
# ============================================================

def execute_local_command(user_text):
    text = user_text.lower().strip()
    
    # 1. Open Chrome & YouTube (Compound Command)
    if "chrome" in text and ("youtube" in text or "you tube" in text):
        confirmed = speak_and_wait_for_confirmation("Vikram, Chrome open chesi YouTube ki vellamantaava?")
        if confirmed:
            os.system("start chrome https://www.youtube.com")
            speak_response("YouTube open ayyindi Vikram. YouTube lo emaina search cheyamantava?")
            return "Chrome and YouTube open chesesanu Vikram!"
        return "Ok, cancel chesesanu."

    # 2. Open Chrome
    elif "open chrome" in text or "chrome open" in text:
        confirmed = speak_and_wait_for_confirmation("Vikram, Chrome open cheyamantava?")
        if confirmed:
            os.system("start chrome")
            return "Chrome open avuthundi Vikram!"
        return "Ok, aapesaanu."

    # 3. Open Notepad
    elif "open notepad" in text or "notepad open" in text:
        confirmed = speak_and_wait_for_confirmation("Vikram, Notepad open cheyamantava?")
        if confirmed:
            subprocess.Popen(["notepad.exe"])
            return "Notepad open chesesanu Vikram!"
        return "Ok, cancel chesesanu."

    # 4. Open VS Code
    elif "open vs code" in text or "visual studio code" in text:
        confirmed = speak_and_wait_for_confirmation("Vikram, VS Code open cheyamantava?")
        if confirmed:
            os.system("code")
            return "VS Code open chesesanu Vikram!"
        return "Ok, cancel chesesanu."

    # 5. Open Calculator (New Full Access Feature)
    elif "calculator" in text or "calc" in text:
        confirmed = speak_and_wait_for_confirmation("Vikram, Calculator open cheyamantava?")
        if confirmed:
            subprocess.Popen(["calc.exe"])
            return "Calculator open chesesanu Vikram!"
        return "Ok, cancel chesesanu."

    # 6. Open File Explorer (New Full Access Feature)
    elif "file explorer" in text or "my computer" in text or "folders" in text:
        confirmed = speak_and_wait_for_confirmation("Vikram, File Explorer open cheyamantava?")
        if confirmed:
            os.system("start explorer")
            return "File Explorer open chesesanu Vikram!"
        return "Ok, cancel chesesanu."

    # 7. Open Command Prompt (New Full Access Feature)
    elif "command prompt" in text or "cmd" in text:
        confirmed = speak_and_wait_for_confirmation("Vikram, Command Prompt open cheyamantava?")
        if confirmed:
            subprocess.Popen(["cmd.exe"])
            return "Command Prompt open chesesanu Vikram!"
        return "Ok, cancel chesesanu."

    # 8. Check Battery Status
    elif "battery" in text or "battery status" in text:
        try:
            battery_info = subprocess.check_output(
                "wmic path Win32_Battery get EstimatedChargeRemaining", 
                shell=True
            ).decode()
            return f"Vikram, mee laptop battery details ivi: {battery_info.strip()}"
        except Exception:
            return "Battery status check cheyyadam kudaraledu Vikram."

    return None


# ============================================================
# ASK BUDDY (SERVER)
# ============================================================

def ask_buddy(user_text):
    print("⏳ Buddy is thinking...")
    try:
        response = requests.post(
            SERVER_URL,
            json={"message": user_text},
            timeout=15
        )
        if response.status_code != 200:
            print(f"❌ Server error: {response.status_code}")
            return None, None

        reply_header = response.headers.get("X-Reply", "")
        if reply_header:
            reply_text = urllib.parse.unquote(reply_header)
        else:
            reply_text = ""

        return reply_text, response.content
    except requests.exceptions.Timeout:
        print("⚠️ Buddy took too long to respond.")
        return None, None
    except requests.exceptions.ConnectionError:
        print("❌ Buddy server is offline. Run: python main.py")
        return None, None
    except Exception as e:
        print(f"⚠️ Request error: {e}")
        return None, None


# ============================================================
# COMMAND HANDLER
# ============================================================

def handle_command(text):
    global SLEEPING
    lower = text.lower().strip()

    if lower in ["shutdown buddy", "exit buddy", "close buddy"]:
        shutdown_event.set()
        print("\n👋 Buddy completely closed.")
        return "shutdown"

    if lower in ["bye", "goodbye", "sleep buddy", "buddy sleep"]:
        SLEEPING = True
        print("\n😴 Buddy is sleeping. Say 'Buddy' to wake me.")
        return "sleep"

    if SLEEPING and lower in ["buddy", "hey buddy", "wake buddy"]:
        SLEEPING = False
        print("\n🟢 Buddy is awake!")
        return "wake"

    return None


# ============================================================
# MAIN SESSION
# ============================================================

def start_session():
    global SLEEPING

    print("==================================================")
    print("🚀 BUDDY FULL-ACCESS LAPTOP ASSISTANT (MULTITASKING)")
    print("🎙️ Voice & Whisper: Active")
    print("🧠 Groq: Active")
    print("🔊 TTS: Active")
    print("🛑 Say 'stop' / 'agu' to interrupt Buddy")
    print("😴 Say 'bye' to put Buddy to sleep")
    print("❌ Say 'shutdown buddy' to close")
    print("==================================================")

    with sr.Microphone() as source:
        print("\n🔧 Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        print("✅ Microphone ready.")

        while not shutdown_event.is_set():
            if SLEEPING:
                time.sleep(0.2)
                try:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                    text = recognizer.recognize_google(audio, language="en-IN").lower().strip()
                    if text in ["buddy", "hey buddy", "wake buddy"]:
                        SLEEPING = False
                        print("\n🟢 Buddy awakened!")
                except Exception:
                    pass
                continue

            user_input = listen_to_user(source)
            if not user_input:
                continue

            # 1. Check local laptop control first via Voice Confirmation
            local_response = execute_local_command(user_input)
            if local_response:
                print(f"\n🤖 Buddy: {local_response}")
                speak_response(local_response)
                continue

            # 2. Check general commands
            command = handle_command(user_input)
            if command == "shutdown":
                break
            if command == "sleep":
                continue
            if command == "wake":
                continue

            # 3. Send to Groq server if not local command
            reply_text, audio_bytes = ask_buddy(user_input)
            if not reply_text:
                continue

            print(f"\n🤖 Buddy: {reply_text}")
            if audio_bytes:
                play_audio(audio_bytes)

    print("\n👋 Buddy session ended.")


if __name__ == "__main__":
    try:
        start_session()
    except KeyboardInterrupt:
        force_exit(None, None)