import os
from dotenv import load_dotenv

load_dotenv()


def _float_env(key: str, default: float, min_val: float = None, max_val: float = None) -> float:
    raw = os.getenv(key, str(default))
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(f"[config] {key}='{raw}' não é um número válido")
    if min_val is not None and val < min_val:
        raise ValueError(f"[config] {key}={val} abaixo do mínimo permitido ({min_val})")
    if max_val is not None and val > max_val:
        raise ValueError(f"[config] {key}={val} acima do máximo permitido ({max_val})")
    return val


def _int_env(key: str, default: int, min_val: int = None) -> int:
    raw = os.getenv(key, str(default))
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"[config] {key}='{raw}' não é um inteiro válido")
    if min_val is not None and val < min_val:
        raise ValueError(f"[config] {key}={val} abaixo do mínimo permitido ({min_val})")
    return val


# AI Backend
AI_BACKEND = os.getenv("AI_BACKEND", "claude")

# Anthropic Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = _int_env("OLLAMA_TIMEOUT", 120, min_val=5)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

AGENT_EVENT_LIMIT = _int_env("AGENT_EVENT_LIMIT", 12, min_val=1)

_camera_raw = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE: int | str = int(_camera_raw) if _camera_raw.isdigit() else _camera_raw
CAMERA_RECONNECT_SECONDS = _int_env("CAMERA_RECONNECT_SECONDS", 5, min_val=1)

CONFIDENCE_THRESHOLD = _float_env("CONFIDENCE_THRESHOLD", 0.45, min_val=0.01, max_val=0.99)
MIN_CONSECUTIVE_FRAMES = _int_env("MIN_CONSECUTIVE_FRAMES", 3, min_val=1)
ALERT_COOLDOWN_SECONDS = _int_env("ALERT_COOLDOWN_SECONDS", 20, min_val=1)

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
SAVE_DIR = os.getenv("SAVE_DIR", "static/captures")
DB_PATH = os.getenv("DB_PATH", "detections.db")

TARGET_CLASSES: set[str] = {"person", "car", "motorcycle", "truck", "bus"}
