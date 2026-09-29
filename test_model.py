import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ప్రస్తుతం మీ API Key కి అందుబాటులో ఉన్న మోడల్స్ జాబితా తెస్తుంది
models = client.models.list()

print("=== Active Groq Models ===")
for m in models.data:
    print(f"-> {m.id}")