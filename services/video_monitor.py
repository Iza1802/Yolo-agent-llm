import os
import cv2
import time
import uuid
import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Callable, Optional

from ultralytics import YOLO
from services.config import (
    CAMERA_SOURCE, CAMERA_RECONNECT_SECONDS,
    MODEL_PATH, CONFIDENCE_THRESHOLD, SAVE_DIR,
    MIN_CONSECUTIVE_FRAMES, ALERT_COOLDOWN_SECONDS,
    TARGET_CLASSES,
)

logger = logging.getLogger(__name__)
os.makedirs(SAVE_DIR, exist_ok=True)

# Intervalo entre frames processados (segundos). Valor baixo = mais CPU, mais latência de detecção.
_FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "0.05"))


class VideoMonitor:
    def __init__(self, on_detection: Optional[Callable[[str, str, float, str], None]] = None):
        self._on_detection = on_detection
        self._last_frame: Optional[cv2.Mat] = None
        self._frame_lock = threading.Lock()
        self._online = False
        self._connected = False
        self._detection_state: dict[str, int] = defaultdict(int)
        self._last_alert_time: dict[str, float] = defaultdict(float)
        self._model: Optional[YOLO] = None

    def _load_model(self) -> None:
        if self._model is None:
            self._model = YOLO(MODEL_PATH)

    def get_last_frame(self) -> Optional[cv2.Mat]:
        with self._frame_lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def get_status(self) -> dict:
        return {
            "online": self._online,
            "connected": self._connected,
            "has_live_frame": self._last_frame is not None,
            "source_type": "webcam" if CAMERA_SOURCE == 0 else "stream",
        }

    def _should_alert(self, label: str) -> bool:
        return (time.time() - self._last_alert_time[label]) > ALERT_COOLDOWN_SECONDS

    @staticmethod
    def _draw_box(frame: cv2.Mat, x1: int, y1: int, x2: int, y2: int, label: str, conf: float) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2,
        )

    def _process_detections(self, frame: cv2.Mat, results) -> None:
        found_labels: set[str] = set()
        best_conf: dict[str, float] = {}

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = self._model.names[cls_id]
                if label not in TARGET_CLASSES:
                    continue
                found_labels.add(label)
                if conf > best_conf.get(label, 0.0):
                    best_conf[label] = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                self._draw_box(frame, x1, y1, x2, y2, label, conf)

        for label in TARGET_CLASSES:
            self._detection_state[label] = (
                self._detection_state[label] + 1 if label in found_labels else 0
            )

        for label in found_labels:
            if self._detection_state[label] >= MIN_CONSECUTIVE_FRAMES and self._should_alert(label):
                self._emit_event(frame, label, best_conf.get(label, 0.0))

    def _emit_event(self, frame: cv2.Mat, label: str, confidence: float) -> None:
        event_id = str(uuid.uuid4())[:8]
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}_{event_id}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        image_path = f"/static/captures/{filename}"
        self._last_alert_time[label] = time.time()
        logger.info("[ALERTA] %s detectado (conf=%.2f)", label, confidence)
        if self._on_detection:
            try:
                self._on_detection(event_id, label, confidence, image_path)
            except Exception as exc:
                logger.error("[monitor] Callback de detecção falhou: %s", exc, exc_info=True)

    def run(self) -> None:
        self._load_model()
        self._online = True

        while True:
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            if not cap.isOpened():
                logger.warning("[CÂMERA] Falha ao abrir. Tentando em %ds...", CAMERA_RECONNECT_SECONDS)
                self._connected = False
                time.sleep(CAMERA_RECONNECT_SECONDS)
                continue

            self._connected = True
            logger.info("[CÂMERA] Conectada: %s", CAMERA_SOURCE)

            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("[CÂMERA] Stream perdido. Reconectando em %ds...", CAMERA_RECONNECT_SECONDS)
                    self._connected = False
                    break

                results = self._model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                self._process_detections(frame, results)

                with self._frame_lock:
                    self._last_frame = frame.copy()

                time.sleep(_FRAME_INTERVAL)

            cap.release()
            time.sleep(CAMERA_RECONNECT_SECONDS)

    def generate_mjpeg(self):
        while True:
            frame = self.get_last_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            time.sleep(_FRAME_INTERVAL)


# Singleton usado pelo app — criado aqui para evitar importação circular
_monitor: Optional[VideoMonitor] = None


def get_monitor() -> VideoMonitor:
    global _monitor
    if _monitor is None:
        raise RuntimeError("VideoMonitor não inicializado. Chame start_monitor() no lifespan.")
    return _monitor


def start_monitor(on_detection: Optional[Callable] = None) -> VideoMonitor:
    global _monitor
    _monitor = VideoMonitor(on_detection=on_detection)
    thread = threading.Thread(target=_monitor.run, daemon=True)
    thread.start()
    return _monitor


# Compat wrappers para não quebrar importações existentes
def get_last_frame():
    return get_monitor().get_last_frame()


def get_camera_status():
    return get_monitor().get_status()


def generate_mjpeg():
    return get_monitor().generate_mjpeg()
