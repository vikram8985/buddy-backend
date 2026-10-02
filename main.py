import os
import time
import sqlite3
import json
import base64
import io
import httpx

import uvicorn
import edge_tts

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in your .env file.")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

ENGLISH_VOICE = "en-IN-NeerjaNeural"
TELUGU_VOICE = "te-IN-ShrutiNeural"

DATABASE = "buddy_memory.db"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="BUDDY AI - Pure Gemini Engine")


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
# INTEGRATIONS (WEATHER ONLY)
# ============================================================

async def get_weather(city: str = "Gorantla") -> str:
    try:
        async with httpx.AsyncClient(timeout=2.0) as http_client:
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def contains_telugu(text: str) -> bool:
    return any("\u0C00" <= char <= "\u0C7F" for char in text)


def choose_voice(text: str) -> str:
    if contains_telugu(text):
        return TELUGU_VOICE
    return ENGLISH_VOICE


# ============================================================
# LIVE CHAT ENDPOINT
# ============================================================
class ChatRequest(BaseModel):
    message: str

    
@app.post("/chat")
async def chat(request: ChatRequest):

    t_start = time.time()
    user_message = request.message.strip()

    if not user_message:
        return JSONResponse({"error": "Em matladaledu Vikram."}, status_code=400)

    try:
        save_message("user", user_message)
        memory = get_memory(4)

        lower_msg = user_message.lower()
        extra_context = ""

        if "weather" in lower_msg or "వాతావరణం" in lower_msg:
            extra_context = await get_weather("Gorantla")

        system_content = (
            "You are BUDDY, Vikram's intelligent personal assistant. "
            "Match user's language (English, Telugu, or mixed). Be crisp, fast, and smart. "
            "Return output STRICTLY in valid JSON format:\n"
            "{\n"
            '  "speech_reply": "Natural short voice reply",\n'
            '  "display_title": "Short title",\n'
            '  "display_text": "Detailed response text"\n'
            "}"
        )

        if extra_context:
            system_content += f"\nLive info context: {extra_context}"

        contents = []
        for role, content in memory:
            role_label = "user" if role == "user" else "model"
            contents.append({
                "role": role_label,
                "parts": [{"text": content}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": f"{system_content}\n\nUser Message: {user_message}"}]
        })

        async with httpx.AsyncClient(timeout=10.0) as client:
            gemini_response = await client.post(
                GEMINI_URL,
                json={
                    "contents": contents,
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.4,
                        "maxOutputTokens": 400
                    }
                }
            )

        if gemini_response.status_code != 200:
            raise Exception(f"Gemini API Error: {gemini_response.text}")

        res_data = gemini_response.json()
        raw_json = res_data["candidates"][0]["content"]["parts"][0]["text"]
        response_data = json.loads(raw_json)

        speech_text = response_data.get("speech_reply", "Ha Vikram, cheppu!")
        display_title = response_data.get("display_title", "Buddy Assistant")
        display_text = response_data.get("display_text", speech_text)

        save_message("assistant", speech_text)

        # TTS Audio Generation
        selected_voice = choose_voice(speech_text)
        speech_rate = "-3%" if selected_voice == TELUGU_VOICE else "+0%"

        communicate = edge_tts.Communicate(
            speech_text,
            selected_voice,
            rate=speech_rate,
            pitch="+0Hz"
        )

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')

        total_time = time.time() - t_start
        print(f"⚡ Total Server Time: {total_time:.2f}s")

        return JSONResponse({
            "speech_text": speech_text,
            "display_title": display_title,
            "display_text": display_text,
            "images": [],
            "audio_base64": audio_base64
        })

    except Exception as e:
        print("❌ Error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
