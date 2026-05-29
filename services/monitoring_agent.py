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
    f"Você é o {AGENT_PROFILE.name}, um agente de {AGENT_PROFILE.role}.\n"
    f"Objetivo: {AGENT_PROFILE.goal}\n"
    "Trate os dados como monitoramento operacional autorizado de ambiente real.\n"
    "Responda em português do Brasil, de forma direta e útil.\n"
    "Use os eventos fornecidos como fonte principal.\n"
    "Não invente dados que não aparecem no contexto.\n"
    "Não tente identificar pessoas; fale apenas sobre eventos, riscos e próximas ações.\n"
    "Quando fizer sentido, organize a resposta em: Leitura, Risco e Recomendação.\n"
    "Se dados climáticos, cotações ou notícias do setor estiverem disponíveis no contexto, "
    "use-os para enriquecer a análise."
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
        f"Contexto operacional:\n"
        f"- Eventos considerados: {total}\n"
        f"- Evento mais recente: {latest['label']} em {latest['event_time']}\n"
        f"- Distribuição: {dist_str}\n"
        f"- Confiança média: {avg_conf:.2f}\n"
        f"Eventos:\n{lines}"
    )


def _build_scraping_context(scraping_data: dict | None) -> str:
    if not scraping_data:
        return ""

    parts: list[str] = []

    clima = scraping_data.get("clima", {})
    if "erro" not in clima and clima.get("temperatura_c") is not None:
        parts.append(
            f"Clima atual: {clima['temperatura_c']}°C, "
            f"umidade {clima.get('umidade_pct')}%, "
            f"vento {clima.get('vento_kmh')} km/h, "
            f"prob. chuva {clima.get('prob_chuva_pct')}%."
        )

    cotacoes = scraping_data.get("cotacoes", {})
    soja = cotacoes.get("precos", {}).get("soja")
    if soja and isinstance(soja, dict):
        parts.append(
            f"Cotação da soja (CEPEA): R$ {soja['preco']:.2f}/saca 60kg "
            f"(data: {soja.get('data', 'N/D')})."
        )

    noticias = scraping_data.get("noticias", {}).get("noticias", [])
    if noticias:
        titulos = "; ".join(n["titulo"] for n in noticias[:3] if n.get("titulo"))
        parts.append(f"Notícias recentes do setor: {titulos}.")

    if not parts:
        return ""

    return "Dados externos de contexto:\n" + "\n".join(f"- {p}" for p in parts)


def normalize_history(history: list) -> list:
    valid = []
    for msg in history:
        if isinstance(msg, dict):
            role, content = msg.get("role", ""), msg.get("content", "")
        else:
            role, content = getattr(msg, "role", ""), getattr(msg, "content", "")
        if role in ("user", "assistant") and content:
            valid.append({"role": role, "content": content})
    return valid[-MAX_HISTORY_MESSAGES:]


def build_agent_messages(
    question: str,
    history: list,
    events: list,
    scraping_data: dict | None = None,
) -> list:
    context_parts = [build_event_context(events)]
    scraping_ctx = _build_scraping_context(scraping_data)
    if scraping_ctx:
        context_parts.append(scraping_ctx)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "\n\n".join(context_parts)},
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
