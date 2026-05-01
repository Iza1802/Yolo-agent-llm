import os
import cv2
import time
import uuid
import threading
from collections import defaultdict
from datetime import datetime

from ultralytics import YOLO
from services.config import (
    CAMERA_SOURCE, CAMERA_RECONNECT_SECONDS,
    MODEL_PATH, CONFIDENCE_THRESHOLD, SAVE_DIR,
    MIN_CONSECUTIVE_FRAMES, ALERT_COOLDOWN_SECONDS,
    TARGET_CLASSES
)
from services.event_repository import save_event

os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

_last_frame = None
_last_frame_lock = threading.Lock()
_camera_online = False
_camera_connected = False

_detection_state = defaultdict(int)
_last_alert_time = defaultdict(lambda: 0.0)


def get_last_frame():
    with _last_frame_lock:
        return _last_frame.copy() if _last_frame is not None else None


def get_camera_status():
    return {
        "online": _camera_online,
        "connected": _camera_connected,
        "has_live_frame": _last_frame is not None,
        "source_type": "webcam" if CAMERA_SOURCE == 0 else "stream"
    }


def _draw_box(frame, x1, y1, x2, y2, label, conf):
    text = f"{label} {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)
    cv2.putText(
        frame, text,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (0, 220, 100), 2
    )


def _should_alert(label: str) -> bool:
    return (time.time() - _last_alert_time[label]) > ALERT_COOLDOWN_SECONDS


def process_stream():
    global _last_frame, _camera_online, _camera_connected

    while True:
        _camera_online = True
        cap = cv2.VideoCapture(CAMERA_SOURCE)

        if not cap.isOpened():
            print(f"[CÂMERA] Falha ao abrir fonte. Tentando novamente em {CAMERA_RECONNECT_SECONDS}s...")
            _camera_connected = False
            time.sleep(CAMERA_RECONNECT_SECONDS)
            continue

        _camera_connected = True
        print(f"[CÂMERA] Conectada com sucesso: {CAMERA_SOURCE}")

        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"[CÂMERA] Stream perdido. Reconectando em {CAMERA_RECONNECT_SECONDS}s...")
                _camera_connected = False
                break

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            found_labels = set()
            best_conf = {}

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = model.names[cls_id]

                    if label not in TARGET_CLASSES:
                        continue

                    found_labels.add(label)
                    if label not in best_conf or conf > best_conf[label]:
                        best_conf[label] = conf

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    _draw_box(frame, x1, y1, x2, y2, label, conf)

            for label in TARGET_CLASSES:
                if label in found_labels:
                    _detection_state[label] += 1
                else:
                    _detection_state[label] = 0

            for label in found_labels:
                if _detection_state[label] >= MIN_CONSECUTIVE_FRAMES and _should_alert(label):
                    event_id = str(uuid.uuid4())[:8]
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}_{event_id}.jpg"
                    filepath = os.path.join(SAVE_DIR, filename)
                    cv2.imwrite(filepath, frame)
                    image_path = f"/static/captures/{filename}"
                    save_event(event_id, label, best_conf.get(label, 0.0), image_path)
                    _last_alert_time[label] = time.time()
                    print(f"[ALERTA] {label} detectado com confiança {best_conf.get(label, 0):.2f}")

            with _last_frame_lock:
                _last_frame = frame.copy()

            time.sleep(0.05)

        cap.release()
        time.sleep(CAMERA_RECONNECT_SECONDS)


def generate_mjpeg():
    while True:
        frame = get_last_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )
        time.sleep(0.05)


def start_monitor():
    thread = threading.Thread(target=process_stream, daemon=True)
    thread.start()
