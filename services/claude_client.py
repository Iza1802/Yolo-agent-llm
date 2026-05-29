import logging
import anthropic
from services.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

logger = logging.getLogger(__name__)

# Singleton — evita recriar o pool HTTP a cada chamada
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY não configurada")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def chat_stream_claude(messages: list):
    system_parts: list[str] = []
    chat_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant") and content:
            chat_messages.append({"role": role, "content": content})

    if not chat_messages:
        raise ValueError("Nenhuma mensagem de usuário ou assistente foi fornecida ao Claude")
    if chat_messages[0]["role"] != "user":
        raise ValueError("A primeira mensagem deve ter role='user' (requisito da API Anthropic)")

    kwargs: dict = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "messages": chat_messages,
    }
    if system_parts:
        kwargs["system"] = [
            {"type": "text", "text": "\n\n".join(system_parts), "cache_control": {"type": "ephemeral"}}
        ]

    with _get_client().messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def check_claude() -> dict:
    if not ANTHROPIC_API_KEY:
        return {"available": False, "error": "ANTHROPIC_API_KEY não configurada", "active_model": ANTHROPIC_MODEL}
    try:
        _get_client().models.retrieve(ANTHROPIC_MODEL)
        return {"available": True, "active_model": ANTHROPIC_MODEL}
    except Exception as exc:
        logger.warning("[claude] check falhou: %s", exc)
        return {"available": False, "error": str(exc), "active_model": ANTHROPIC_MODEL}
