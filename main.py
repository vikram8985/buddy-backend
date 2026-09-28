import os
import time
import sqlite3
import urllib.parse
import io

import uvicorn
import edge_tts

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in your .env file.")

client = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-20b"

# English voice
ENGLISH_VOICE = "en-IN-NeerjaNeural"

# Telugu voice
TELUGU_VOICE = "te-IN-ShrutiNeural"

DATABASE = "buddy_memory.db"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="BUDDY AI")


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


class ChatRequest(BaseModel):
    message: str


def save_message(role: str, content: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (role, content) VALUES (?, ?)",
        (role, content)
    )
    conn.commit()
    conn.close()


def get_memory(limit: int = 4): # Reduced memory limit for speed
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chats ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return rows


def contains_telugu(text: str) -> bool:
    return any(
        "\u0C00" <= char <= "\u0C7F"
        for char in text
    )


def choose_voice(text: str) -> str:
    if contains_telugu(text):
        return TELUGU_VOICE
    return ENGLISH_VOICE


def clean_response(text: str) -> str:
    text = text.strip()
    text = text.replace("|||", " ")
    text = (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("—", "-")
    )
    return text.strip()


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    t_start = time.time()
    user_message = request.message.strip()

    if not user_message:
        return JSONResponse(
            {"reply": "Em matladaledu Vikram."},
            status_code=400
        )

    try:
        save_message("user", user_message)
        memory = get_memory(4)

        # Optimized lighter system prompt for speed
        messages = [
            {
                "role": "system",
                "content": (
                    "You are BUDDY, Vikram's fast AI assistant. "
                    "Match user's language (English, Telugu, or mixed). "
                    "Be concise for short questions, and detailed only when asked. "
                    "Never use |||."
                )
            }
        ]

        for role, content in memory:
            messages.append({"role": role, "content": content})

        # ----------------------------------------------------
        # GROQ
        # ----------------------------------------------------
        t_llm = time.time()

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=150  # Reduced max tokens to prevent long essays & speed up TTS
        )

        llm_time = time.time() - t_llm
        print(f"✅ Groq answered in {llm_time:.2f}s")

        raw_output = completion.choices[0].message.content or ""
        raw_output = clean_response(raw_output)

        if not raw_output:
            raw_output = "Ha Vikram, cheppu! Nenu ready ga unna."

        save_message("assistant", raw_output)
        print(f"🤖 Buddy: {raw_output}")

        selected_voice = choose_voice(raw_output)

        # ----------------------------------------------------
        # TTS IN-MEMORY GENERATION (FAST)
        # ----------------------------------------------------
        t_tts = time.time()

        communicate = edge_tts.Communicate(
            raw_output,
            selected_voice,
            rate="+10%",
            pitch="+0Hz"
        )

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)
        tts_time = time.time() - t_tts
        total_time = time.time() - t_start

        print(f"✅ Voice ready in {tts_time:.2f}s")
        print(f"⚡ Total Server Time: {total_time:.2f}s\n")

        encoded_display = urllib.parse.quote(raw_output)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mp3",
            headers={
                "X-Reply": encoded_display
            }
        )

    except Exception as e:
        print("❌ Error inside main.py:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)