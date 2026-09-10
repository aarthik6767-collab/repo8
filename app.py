import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

from chatbot_config import SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in the .env file.")

client = genai.Client(api_key=API_KEY)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=message,
            generation_config={
                "system_instruction": SYSTEM_PROMPT,
                "thinking_level": "low",
                "temperature": 0.35,
                "max_output_tokens": 1024,
            },
        )
        reply = (interaction.output_text or "").strip()
        if not reply:
            reply = "I couldn't generate a response. Please try a Foodmate AI-related question."
        return jsonify({"reply": reply})
    except Exception:
        app.logger.exception("Gemini API request failed")
        return jsonify({"error": "I’m having trouble connecting right now. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True)
