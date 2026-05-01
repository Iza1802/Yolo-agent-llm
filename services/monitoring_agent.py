from dataclasses import dataclass
from collections import Counter
from services.config import AGENT_EVENT_LIMIT

MAX_HISTORY_MESSAGES = 8


@dataclass(frozen=True)
class AgentProfile:
    name: str
    role: str
    goal: str


AGENT_PROFILE = AgentProfile(
    name="Agente AgroVision",
    role="triagem operacional de eventos",
    goal="Analisar detecções recentes, explicar riscos e sugerir a próxima ação.",
)

SYSTEM_PROMPT = (
    f"Você é o {AGENT_PROFILE.name}, um agente de {AGENT_PROFILE.role}. "
    f"Objetivo: {AGENT_PROFILE.goal} "
    "Trate os dados como monitoramento operacional autorizado de ambiente real. "
    "Responda em português do Brasil, de forma direta e útil. "
    "Use os eventos fornecidos como fonte principal. "
    "Não invente dados que não aparecem no contexto. "
    "Não tente identificar pessoas; fale apenas sobre eventos, riscos e próximas ações. "
    "Quando fizer sentido, organize a resposta em: Leitura, Risco e Recomendação."
)


def build_event_context(events: list) -> str:
    if not events:
        return "Contexto operacional: nenhum evento registrado até o momento."

    recent = events[:AGENT_EVENT_LIMIT]
    total = len(recent)
    latest = recent[0]
    dist = Counter(e["label"] for e in recent)
    avg_conf = sum(e["confidence"] for e in recent) / total

    dist_str = ", ".join(f"{k}: {v}" for k, v in dist.most_common())
    lines = "\n".join(
        f"- #{e['id']} | {e['event_time']} | {e['label']} | conf: {e['confidence']:.2f}"
        for e in recent
    )

    return (
        f"Contexto operacional para o agente:\n"
        f"- Eventos considerados: {total}\n"
        f"- Evento mais recente: {latest['label']} em {latest['event_time']}\n"
        f"- Distribuição recente: {dist_str}\n"
        f"- Confiança média: {avg_conf:.2f}\n"
        f"Eventos recentes:\n{lines}"
    )


def normalize_history(history: list) -> list:
    valid = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        if role in ("user", "assistant") and content:
            valid.append({"role": role, "content": content})
    return valid[-MAX_HISTORY_MESSAGES:]


def build_agent_messages(question: str, history: list, events: list) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": build_event_context(events)},
        *normalize_history(history),
        {"role": "user", "content": question},
    ]


def get_agent_status(events: list) -> dict:
    context = build_event_context(events)
    return {
        "name": AGENT_PROFILE.name,
        "role": AGENT_PROFILE.role,
        "goal": AGENT_PROFILE.goal,
        "events_in_context": min(len(events), AGENT_EVENT_LIMIT),
        "context_preview": context[:500] + ("..." if len(context) > 500 else ""),
    }
