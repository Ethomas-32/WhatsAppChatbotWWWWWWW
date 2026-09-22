"""
Flask webhook that receives incoming WhatsApp messages from Twilio,
runs them through the RAG engine, and replies with the generated answer.

Run locally:
    python app.py
Then expose it with ngrok and point your Twilio Sandbox webhook at
https://<your-ngrok-subdomain>.ngrok.io/webhook
"""
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

from rag_engine import answer_question

app = Flask(__name__)


@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "unknown")

    print(f"Message from {sender}: {incoming_msg}")

    if not incoming_msg:
        reply_text = "Sorry, I didn't receive any text. Could you try again?"
    else:
        reply_text = answer_question(incoming_msg)

    print(f"Generated reply ({len(reply_text)} chars): {reply_text}")

    twiml_response = MessagingResponse()
    twiml_response.message(reply_text)
    twiml_str = str(twiml_response)
    print(f"TwiML being returned: {twiml_str}")
    return twiml_str


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, port=5000)