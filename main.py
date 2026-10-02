import os
import time
import sqlite3
import json
import base64
import io
import httpx

import uvicorn
import edge_tts
from duckduckgo_search import DDGS

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
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

ENGLISH_VOICE = "en-IN-NeerjaNeural"
TELUGU_VOICE = "te-IN-ShrutiNeural"

DATABASE = "buddy_memory.db"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="BUDDY AI - Live Engine")


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
# INTEGRATIONS (WEATHER, SEARCH & IMAGES)
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


def search_web_and_images(query: str):
    """Free Live Search Text & Image URLs"""
    search_text = ""
    image_urls = []
    try:
        with DDGS() as ddgs:
            # 1. Fetch Text Results
            results = list(ddgs.text(query, max_results=2))
            if results:
                search_text = "\n".join([r['body'] for r in results])
            
            # 2. Fetch Image Results
            img_results = list(ddgs.images(query, max_results=4))
            for img in img_results:
                image_urls.append(img['image'])
    except Exception as e:
        print(f"Search error: {e}")
        
    return search_text, image_urls


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
# LIVE CHAT ENDPOINT (JSON + AUDIO + IMAGES)
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

        # Weather / Search Check
        lower_msg = user_message.lower()
        extra_context = ""
        image_urls = []

        if "weather" in lower_msg or "వాతావరణం" in lower_msg:
            extra_context = await get_weather("Hyderabad")
        else:
            # Always perform search for current query context and image extraction
            extra_context, image_urls = search_web_and_images(user_message)

        # System Prompt returning JSON for UI display + speech
        system_content = (
            "You are BUDDY, Vikram's intelligent personal assistant. "
            "Match user's language (English, Telugu, or mixed). "
            "Use natural sentence pauses, commas, and full stops so speech output sounds human. "
            "Return output STRICTLY in JSON format:\n"
            "{\n"
            '  "speech_reply": "Natural voice reply for user",\n'
            '  "display_title": "Clean concise title for UI screen",\n'
            '  "display_text": "Detailed structured response text for display card"\n'
            "}"
        )

        if extra_context:
            system_content += f"\nReal-time live info context: {extra_context}"

        messages = [{"role": "system", "content": system_content}]

        for role, content in memory:
            messages.append({"role": role, "content": content})

        # LLM Completion
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=1000
        )

        raw_json = completion.choices[0].message.content or "{}"
        response_data = json.loads(raw_json)

        speech_text = response_data.get("speech_reply", "Ha Vikram, cheppu!")
        display_title = response_data.get("display_title", "Buddy Assistant")
        display_text = response_data.get("display_text", speech_text)

        save_message("assistant", speech_text)
        print(f"🤖 Buddy Speech: {speech_text}")

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
        print(f"⚡ Total Server Time: {total_time:.2f}s\n")

        # JSON payload matching Flutter UI requirements
        return JSONResponse({
            "speech_text": speech_text,
            "display_title": display_title,
            "display_text": display_text,
            "images": image_urls,
            "audio_base64": audio_base64
        })

    except Exception as e:
        print("❌ Error inside main.py:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)