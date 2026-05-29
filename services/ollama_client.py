import json
import logging
from urllib.parse import urljoin

import httpx

from services.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_KEEP_ALIVE

logger = logging.getLogger(__name__)

# Base URL derivada da URL de chat — mais robusto do que string.replace()
_OLLAMA_BASE = OLLAMA_URL.split("/api/")[0]


def warmup_model() -> None:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    try:
        with httpx.Client(timeout=30) as client:
            client.post(OLLAMA_URL, json=payload)
        logger.info("[OLLAMA] Modelo '%s' aquecido.", OLLAMA_MODEL)
    except Exception as exc:
        logger.warning("[OLLAMA] Warmup falhou (%s). Modelo carregará na primeira pergunta.", exc)


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
                except json.JSONDecodeError:
                    logger.debug("[OLLAMA] Linha não-JSON ignorada: %r", line[:80])
                    continue

                if not isinstance(data, dict):
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break


def check_ollama() -> dict:
    tags_url = urljoin(_OLLAMA_BASE + "/", "api/tags")
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(tags_url)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Resposta inesperada da API Ollama")
        models = [m["name"] for m in data.get("models", [])]
        return {"available": True, "models": models, "active_model": OLLAMA_MODEL}
    except Exception as exc:
        return {"available": False, "error": str(exc), "active_model": OLLAMA_MODEL}
