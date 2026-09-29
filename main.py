import os
import time
import sqlite3
import urllib.parse
import io
import httpx

import uvicorn
import edge_tts
from duckduckgo_search import DDGS

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

MODEL = "llama-3.1-8b-instant"

ENGLISH_VOICE = "en-IN-NeerjaNeural"
TELUGU_VOICE = "te-IN-ShrutiNeural"

DATABASE = "buddy_memory.db"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="BUDDY AI")


# ============================================================
# DATABASE & MEMORY
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


def get_memory(limit: int = 4):
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


# ============================================================
# INTEGRATIONS (WEATHER & SEARCH)
# ============================================================

async def get_weather(city: str = "Hyderabad") -> str:
    """Free Weather API without API Keys"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as http_client:
            geo_res = await http_client.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1")
            geo_data = geo_res.json()
            if not geo_data.get("results"):
                return "Weather data not found."
            
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            
            w_res = await http_client.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true")
            w_data = w_res.json()["current_weather"]
            return f"Current weather in {city}: {w_data['temperature']}°C, Wind speed: {w_data['windspeed']} km/h."
    except Exception as e:
        print(f"Weather error: {e}")
        return ""


def search_web(query: str) -> str:
    """Free Live Search & News"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            if results:
                return "\n".join([r['body'] for r in results])
    except Exception as e:
        print(f"Search error: {e}")
    return ""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def contains_telugu(text: str) -> bool:
    return any("\u0C00" <= char <= "\u0C7F" for char in text)


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

        # ----------------------------------------------------
        # WEATHER / SEARCH CHECK
        # ----------------------------------------------------
        lower_msg = user_message.lower()
        extra_context = ""

        if "weather" in lower_msg or "వాతావరణం" in lower_msg:
            extra_context = await get_weather("Hyderabad")
        elif any(k in lower_msg for k in ["news", "search", "వార్తలు", "తాజా", "who is", "what is"]):
            extra_context = search_web(user_message)

        # System prompt with punctuation guidance for natural TTS
        system_content = (
            "You are BUDDY, Vikram's intelligent personal assistant. "
            "Match user's language (English, Telugu, or mixed). "
            "Use natural sentence pauses, commas, and full stops so speech output sounds natural and human-like. "
            "Be concise for short questions, and never use |||."
        )

        if extra_context:
            system_content += f"\nReal-time live information context: {extra_context}"

        messages = [{"role": "system", "content": system_content}]

        for role, content in memory:
            messages.append({"role": role, "content": content})

        # ----------------------------------------------------
        # GROQ AI
        # ----------------------------------------------------
        t_llm = time.time()

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=200
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
        # TTS GENERATION (Tuned Rate for Natural Voice)
        # ----------------------------------------------------
        t_tts = time.time()

        # Telugu ki rate="-3%" వాడడం వల్ల శ్రుతి వాయిస్ చాలా ప్రశాంతంగా, Natural గా మాట్లాడుతుంది
        speech_rate = "-3%" if selected_voice == TELUGU_VOICE else "+0%"

        communicate = edge_tts.Communicate(
            raw_output,
            selected_voice,
            rate=speech_rate,
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
    uvicorn.run(app, host="0.0.0.0", port=10000)