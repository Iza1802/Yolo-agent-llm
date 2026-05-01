import os
from dotenv import load_dotenv

load_dotenv()

# AI Backend: "claude" (Anthropic API) ou "ollama" (local)
AI_BACKEND = os.getenv("AI_BACKEND", "claude")

# Anthropic Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Ollama (fallback / alternativo)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

AGENT_EVENT_LIMIT = int(os.getenv("AGENT_EVENT_LIMIT", "12"))

CAMERA_SOURCE_RAW = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(CAMERA_SOURCE_RAW) if CAMERA_SOURCE_RAW.isdigit() else CAMERA_SOURCE_RAW
CAMERA_RECONNECT_SECONDS = int(os.getenv("CAMERA_RECONNECT_SECONDS", "5"))

CONFIDENCE_THRESHOLD = 0.45
MIN_CONSECUTIVE_FRAMES = 3
ALERT_COOLDOWN_SECONDS = 20

MODEL_PATH = "yolov8n.pt"
SAVE_DIR = "static/captures"
DB_PATH = "detections.db"

TARGET_CLASSES = {"person", "car", "motorcycle", "truck", "bus"}
