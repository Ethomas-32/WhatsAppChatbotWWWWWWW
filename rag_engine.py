"""
RAG engine: embeds a query, retrieves relevant chunks from Chroma,
and generates an answer using a free Hugging Face-hosted LLM.
"""
import os
import chromadb
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"
# Multilingual embedding model — works well for cross-language retrieval
# (e.g. English docs, questions asked in Spanish/Zulu/etc.)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Hugging Face's router-based Inference Providers API. Omitting a specific
# provider (no "/hf-inference" in the path) lets HF auto-route each request
# to whichever partner provider actually hosts the requested model.
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# Free-tier-friendly instruct model on Hugging Face Inference Providers.
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

TOP_K = 3
# Cosine distance ranges 0 (identical) to 2 (opposite), regardless of embedding
# model, so this threshold stays meaningful even if you swap models later.
# A relevant match is typically well under 0.6; irrelevant matches usually
# land above 0.8-1.0. Tune based on your own testing.
NO_MATCH_THRESHOLD = 0.7

SYSTEM_PROMPT = (
    "You are a warm, caring support assistant, speaking the way a gentle, "
    "attentive caregiver would — patient, reassuring, and never clinical or "
    "curt. Use soft, kind language and show you understand the person may be "
    "anxious or in need of support, while still being clear and helpful. "
    "Answer the user's question using ONLY the context provided below. If the "
    "context doesn't contain the answer, gently let them know you don't have "
    "that information yet, and warmly suggest reaching out to human support "
    "rather than leaving them stuck. Keep answers short and conversational, "
    "suitable for WhatsApp — a few caring sentences, not a long clinical answer.\n\n"
    "CRITICAL LANGUAGE RULE: Before writing anything, identify the language of "
    "the user's QUESTION (the text after 'Question:' below) — not the language "
    "of the context. Your entire reply MUST be written in that exact same "
    "language.\n"
    "- If the question is in English, your reply must be in English. Do NOT "
    "translate it into Afrikaans, Spanish, or any other language.\n"
    "- If the question is in Afrikaans, reply in Afrikaans.\n"
    "- If the question is in Spanish, reply in Spanish.\n"
    "- If the question is in any other language, reply in that same language.\n"
    "Never switch to a different language than the one the user just used, "
    "even if the context passages are in another language — translate any "
    "information you use from the context into the user's language."
)

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have that information just yet — but please don't "
    "worry, our team is happy to help. You can reach them at "
    "support@example.com whenever you're ready."
)

# --- Lazy-loaded singletons so we don't reload the model on every message ---
_embedding_model = None
_collection = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set.")
    return token


def retrieve(query: str, top_k: int = TOP_K):
    """Return the top_k most relevant chunks for the query, with distances."""
    model = _get_embedding_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    chunks = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    return list(zip(chunks, distances))


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Call the Hugging Face-hosted LLM (via the router API) with retrieved context."""
    context = "\n\n".join(context_chunks)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"(Reminder: reply in the exact same language as the Question above. "
        f"If the Question is in English, your reply must be in English — do "
        f"not switch to Afrikaans or any other language.)"
    )

    headers = {
        "Authorization": f"Bearer {_get_hf_token()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }

    response = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def answer_question(query: str) -> str:
    """Full RAG pipeline: retrieve -> (maybe fallback) -> generate."""
    retrieved = retrieve(query)

    if not retrieved:
        return FALLBACK_MESSAGE

    # If even the closest match is too far away, don't bother generating —
    # just admit we don't know rather than letting the LLM hallucinate.
    best_distance = retrieved[0][1]
    if best_distance > NO_MATCH_THRESHOLD:
        return FALLBACK_MESSAGE

    context_chunks = [chunk for chunk, _ in retrieved]
    try:
        return generate_answer(query, context_chunks)
    except requests.exceptions.HTTPError as e:
        print(f"Generation HTTP error: {e.response.status_code} - {e.response.text}")
        return FALLBACK_MESSAGE
    except Exception as e:
        print(f"Generation error: {e}")
        return FALLBACK_MESSAGE