# RAG WhatsApp Chatbot (Python + Hugging Face + Twilio)

A Retrieval-Augmented Generation (RAG) chatbot that answers WhatsApp messages using your own documents (PDFs or text files). Built with a warm, caregiver-style tone and support for multiple languages — the bot automatically replies in whatever language the user writes in.

**Stack:**
- **WhatsApp:** [Twilio WhatsApp Sandbox](https://www.twilio.com/docs/whatsapp/sandbox) (free for development)
- **Embeddings:** `sentence-transformers` (multilingual model), runs locally, no API cost
- **Vector store:** [ChromaDB](https://www.trychroma.com/) (local, persistent, free)
- **LLM:** [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) (free tier)
- **Server:** Flask

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Get the Code](#step-1-get-the-code)
3. [Step 2: Set Up Python Environment](#step-2-set-up-python-environment)
4. [Step 3: Get Your API Keys](#step-3-get-your-api-keys)
5. [Step 4: Add Your Knowledge Base](#step-4-add-your-knowledge-base)
6. [Step 5: Build the Vector Store](#step-5-build-the-vector-store)
7. [Step 6: Run the Server Locally](#step-6-run-the-server-locally)
8. [Step 7: Expose Your Server with ngrok](#step-7-expose-your-server-with-ngrok)
9. [Step 8: Connect Twilio's WhatsApp Sandbox](#step-8-connect-twilios-whatsapp-sandbox)
10. [Step 9: Test It](#step-9-test-it)
11. [Restarting Later](#restarting-later)
12. [Troubleshooting](#troubleshooting)
13. [Project Structure](#project-structure)
14. [Next Steps / Going to Production](#next-steps--going-to-production)

---

## Prerequisites

- Python 3.10+
- A free [Hugging Face](https://huggingface.co) account
- A free [Twilio](https://www.twilio.com/try-twilio) account
- [ngrok](https://ngrok.com/download) installed (to expose your local server to the internet)

---

## Step 1: Get the Code

Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

---

## Step 2: Set Up Python Environment

```bash
python -m venv venv

# Activate it:
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

This installs Flask, the Twilio SDK, sentence-transformers, ChromaDB, pypdf, python-dotenv, and requests.

---

## Step 3: Get Your API Keys

### Hugging Face Token

1. Sign up at **https://huggingface.co**
2. Go to **Settings → Access Tokens** → https://huggingface.co/settings/tokens
3. Click **New token** → Role: **Read** → Create
4. Copy the token (starts with `hf_...`)

**Note on gated models:** Some models (like Meta's Llama family) require accepting a license first. Visit the model's page (e.g. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and click "Agree and access repository" before it'll work via the API.

### Twilio Account & WhatsApp Sandbox

1. Sign up for a free trial at **https://www.twilio.com/try-twilio**
2. Go to the **Twilio Console**: **https://console.twilio.com**
3. In the left sidebar: **Messaging → Try it out → Send a WhatsApp message**
   - Direct link: **https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn**
4. You'll see a **Sandbox number** (e.g. `+1 415 523 8886`) and a **join code** (e.g. `join example-word`)
5. From your own WhatsApp, send that join code as a message to the Sandbox number
6. You should get a confirmation reply — your number is now linked for testing

### Set Up Your `.env` File

Copy the example file and fill in your token:
```bash
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Step 4: Add Your Knowledge Base

Place your documents inside the `data/` folder. Supported formats:
- `.pdf`
- `.txt`

The bot will only be able to answer questions covered by these documents — anything outside their scope triggers a polite fallback message.

---

## Step 5: Build the Vector Store

Run:
```bash
python ingest.py
```

This reads every file in `data/`, splits it into chunks, embeds each chunk using a multilingual embedding model, and stores everything in a local ChromaDB database (`chroma_db/`).

**Re-run this any time you add, remove, or edit files in `data/`.**

Expected output:
```
Loading documents...
Loaded 24 chunks from 'data/'.
Loading embedding model 'paraphrase-multilingual-MiniLM-L12-v2'...
Computing embeddings...
Storing in ChromaDB...
Done. 24 chunks stored in 'chroma_db/'.
```

---

## Step 6: Run the Server Locally

```bash
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
* Debug mode: on
```

Leave this terminal open — the server needs to keep running while you test.

---

## Step 7: Expose Your Server with ngrok

Twilio needs a public URL to reach your local machine.

1. Download ngrok: **https://ngrok.com/download**
2. Extract it somewhere convenient (e.g. `C:\ngrok`)
3. In a **new terminal**, run:
   ```bash
   ngrok http 5000
   ```
4. Copy the `https://...ngrok-free.app` (or `.ngrok.io`) URL it prints — you'll need it in the next step

**Note:** on ngrok's free tier, this URL changes every time you restart ngrok. You'll need to update Twilio's webhook (Step 8) each time that happens.

---

## Step 8: Connect Twilio's WhatsApp Sandbox

1. Go back to the Twilio Console's WhatsApp Sandbox page:
   **[https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn?frameUrl=%2Fconsole%2Fsms%2Fwhatsapp%2Flearn%3Fx-target-region%3Dus1)**
2. Scroll to the **Sandbox Configuration** / webhook section
3. Find the field **"When a message comes in"**
4. Paste your ngrok URL with `/webhook` at the end:
   ```
   https://your-ngrok-url.ngrok-free.app/webhook
   ```
5. Make sure the method is set to **HTTP POST**
6. Click **Save**

---

## Step 9: Test It

1. Open WhatsApp on your phone
2. Message your Twilio Sandbox number with a question related to your documents
3. Watch your Flask terminal — you should see the incoming message logged, followed by the generated reply
4. You should receive a reply on WhatsApp within a few seconds

**Try testing in different languages** — the bot is designed to reply in whatever language you write in.

---

## Restarting Later

Every time you come back to work on this:

```bash
# 1. Activate your virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Start the Flask server
python app.py
```

In a separate terminal:
```bash
# 3. Start ngrok
ngrok http 5000
```

4. **Copy the new ngrok URL** (it changes every restart on the free tier) and update it in Twilio's Sandbox settings (Step 8) — the "When a message comes in" field.

5. If it's been a few days since you last used the Sandbox, you may need to **rejoin** it by sending the join code to the Sandbox number again from WhatsApp.

You only need to re-run `python ingest.py` if you've changed the contents of the `data/` folder.

---

## Troubleshooting

| Problem | Likely Cause / Fix |
|---|---|
| `HF_TOKEN environment variable is not set` | Make sure `.env` exists (copied from `.env.example`) and contains a valid token |
| `Model not supported by provider` | The specific Hugging Face provider doesn't host that model — check `LLM_MODEL` in `rag_engine.py`, or try a different model |
| Bot always returns the fallback "I don't have that information" message | Your knowledge base may not cover the topic, or `NO_MATCH_THRESHOLD` in `rag_engine.py` is too strict. Re-run `ingest.py` if you've changed your docs |
| No reply on WhatsApp at all | Check that ngrok is still running, that the webhook URL in Twilio matches your **current** ngrok URL, and that your Sandbox session hasn't expired (rejoin via the join code) |
| `12300 Invalid Content-Type` in Twilio logs | Your server is returning JSON instead of TwiML/XML — make sure `app.py` is the Twilio version (uses `MessagingResponse`), not an Infobip/JSON version |
| Slow first response | Hugging Face's free tier can have a cold start on the first request; subsequent ones are faster |
| `Failed to send telemetry event` messages in the console | Harmless — this is ChromaDB's analytics pings failing silently. Ignore it |

---

## Project Structure

```
.
├── app.py                 # Flask webhook that connects Twilio to the RAG engine
├── rag_engine.py           # Retrieval (ChromaDB) + generation (Hugging Face) logic
├── ingest.py               # Builds the vector store from documents in data/
├── requirements.txt
├── .env.example             # Template for your API keys
├── data/                   # Your source documents (.pdf / .txt) go here
└── chroma_db/              # Generated vector store (created by ingest.py)
```

---

## Next Steps / Going to Production

- **Beyond the Sandbox:** Twilio's Sandbox is for development only — messages need a rejoin every few days, and it shows a "sandbox" disclaimer to users. For real users, apply for your own **Twilio WhatsApp Sender** (requires WhatsApp Business verification).
- **Rate limits:** Hugging Face's free tier has rate limits, and some models may be slow or unavailable at times. For heavier traffic, consider a paid inference provider.
- **Persistent hosting:** `chroma_db/` is a local folder. If you deploy to a platform with ephemeral storage, either re-run `ingest.py` on each deploy or move to a hosted vector database.
- **Tuning retrieval:** Adjust `CHUNK_SIZE`/`CHUNK_OVERLAP` in `ingest.py` and `NO_MATCH_THRESHOLD` in `rag_engine.py` based on your own testing.
- **Conversation memory:** The current bot answers each message independently with no memory of prior turns. Adding conversation history (e.g. via LangChain) is a natural next step.
