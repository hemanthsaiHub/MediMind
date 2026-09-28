import json
import os
import urllib.error
import urllib.request

SYSTEM = ("You are MediMind, an educational health-information assistant for a student project. "
          "Explain clearly in plain language, keep answers under 150 words, never diagnose, "
          "and remind the user to consult a qualified doctor for personal medical decisions.")


def ask(question: str) -> dict:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return {"mode": "offline",
                "answer": "The AI assistant is not configured. Add GEMINI_API_KEY (free key from "
                          "aistudio.google.com) to your .env file or Vercel environment variables."}
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"system_instruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}]}
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
        return {"mode": "gemini", "answer": data["candidates"][0]["content"]["parts"][0]["text"].strip()}
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as e:
        return {"mode": "error", "answer": f"The AI service did not respond ({type(e).__name__}). "
                                           "Check your API key and model name, then try again."}
