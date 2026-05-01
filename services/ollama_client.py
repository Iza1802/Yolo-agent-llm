import httpx
import json
from services.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_KEEP_ALIVE


def warmup_model():
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
        with httpx.Client(timeout=30) as client:
            client.post(OLLAMA_URL, json=payload)
        print(f"[OLLAMA] Modelo '{OLLAMA_MODEL}' aquecido com sucesso.")
    except Exception as e:
        print(f"[OLLAMA] Aviso: warmup falhou ({e}). O modelo será carregado na primeira pergunta.")


def chat_stream(messages: list):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        with client.stream("POST", OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


def check_ollama() -> dict:
    try:
        tags_url = OLLAMA_URL.replace("/api/chat", "/api/tags")
        with httpx.Client(timeout=5) as client:
            r = client.get(tags_url)
            models = [m["name"] for m in r.json().get("models", [])]
            return {"available": True, "models": models, "active_model": OLLAMA_MODEL}
    except Exception as e:
        return {"available": False, "error": str(e), "active_model": OLLAMA_MODEL}
