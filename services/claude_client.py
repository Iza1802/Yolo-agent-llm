import anthropic
from services.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


def chat_stream_claude(messages: list):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    system_parts = []
    chat_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant") and content:
            chat_messages.append({"role": role, "content": content})

    # Garante que começa com mensagem de usuário (requisito da API)
    if not chat_messages or chat_messages[0]["role"] != "user":
        chat_messages.insert(0, {"role": "user", "content": "Olá"})

    system = "\n\n".join(system_parts) if system_parts else None

    kwargs = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "messages": chat_messages,
    }
    if system:
        kwargs["system"] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text


def check_claude() -> dict:
    if not ANTHROPIC_API_KEY:
        return {"available": False, "error": "ANTHROPIC_API_KEY não configurada", "active_model": ANTHROPIC_MODEL}
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        client.models.retrieve(ANTHROPIC_MODEL)
        return {"available": True, "active_model": ANTHROPIC_MODEL}
    except Exception as e:
        return {"available": False, "error": str(e), "active_model": ANTHROPIC_MODEL}
