import os
import logging
import threading
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services.config import AI_BACKEND, ANTHROPIC_API_KEY
from services.event_repository import init_db, list_events, count_events
from services.video_monitor import start_monitor, get_last_frame, get_camera_status, generate_mjpeg
from services.ollama_client import warmup_model, chat_stream, check_ollama
from services.claude_client import chat_stream_claude, check_claude
from services.monitoring_agent import build_agent_messages, get_agent_status
from services.scraping_service import scraping_service
from services.schemas import ChatRequest

logger = logging.getLogger(__name__)


def _active_backend() -> str:
    return "claude" if AI_BACKEND == "claude" and ANTHROPIC_API_KEY else "ollama"


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/captures", exist_ok=True)
    os.makedirs("templates", exist_ok=True)

    init_db()
    start_monitor(on_detection=_handle_detection)

    backend = _active_backend()
    if backend == "ollama":
        threading.Thread(target=warmup_model, daemon=True).start()

    logger.info("[APP] AgroVision AI iniciado. Backend: %s", backend)
    yield


def _handle_detection(event_id: str, label: str, confidence: float, image_path: str) -> None:
    from services.event_repository import save_event
    save_event(event_id, label, confidence, image_path)


app = FastAPI(title="AgroVision AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── ROTAS PRINCIPAIS ───────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    events = list_events(20)
    return templates.TemplateResponse("index.html", {"request": request, "events": events})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "AgroVision AI"}


@app.get("/events")
def get_events() -> JSONResponse:
    return JSONResponse(content=list_events(50))


@app.get("/frame")
def get_frame() -> Response:
    frame = get_last_frame()
    if frame is None:
        return JSONResponse({"message": "Sem frame disponível."}, status_code=503)
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return JSONResponse({"message": "Erro ao converter frame."}, status_code=500)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── CÂMERA ────────────────────────────────────────────────────
@app.get("/camera/status")
def camera_status() -> JSONResponse:
    return JSONResponse(content=get_camera_status())


# ── AGENTE ────────────────────────────────────────────────────
@app.get("/agent/status")
def agent_status() -> JSONResponse:
    events = list_events(50)
    return JSONResponse(content=get_agent_status(events))


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    events = list_events(50)
    scraping_data = scraping_service.get_all_data()
    messages = build_agent_messages(req.question, req.history or [], events, scraping_data)
    backend = _active_backend()

    def stream_response():
        try:
            if backend == "claude":
                yield from chat_stream_claude(messages)
            else:
                yield from chat_stream(messages)
        except Exception as exc:
            logger.error("[chat] Erro no backend %s: %s", backend, exc, exc_info=True)
            yield "\n\n[Erro ao processar sua pergunta. Tente novamente.]"

    return StreamingResponse(stream_response(), media_type="text/plain; charset=utf-8")


# ── STATUS AI ─────────────────────────────────────────────────
@app.get("/ai/status")
def ai_status() -> JSONResponse:
    backend = _active_backend()
    info = check_claude() if backend == "claude" else check_ollama()
    info["backend"] = backend
    return JSONResponse(content=info)


@app.get("/ollama/status")
def ollama_status() -> JSONResponse:
    return JSONResponse(content=check_ollama())


# ── SCRAPING ──────────────────────────────────────────────────
@app.get("/scraping/data")
def scraping_data() -> JSONResponse:
    return JSONResponse(content=scraping_service.get_all_data())


@app.get("/scraping/weather")
def scraping_weather() -> JSONResponse:
    return JSONResponse(content=scraping_service.get_weather())


@app.get("/scraping/commodities")
def scraping_commodities() -> JSONResponse:
    return JSONResponse(content=scraping_service.get_commodity_prices())


@app.get("/scraping/news")
def scraping_news() -> JSONResponse:
    return JSONResponse(content=scraping_service.get_agro_news())
