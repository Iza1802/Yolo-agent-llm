import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from services.config import DB_PATH

logger = logging.getLogger(__name__)


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        logger.error("[db] Erro SQLite: %s", exc, exc_info=True)
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id         TEXT PRIMARY KEY,
                event_time TEXT NOT NULL,
                label      TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_path TEXT NOT NULL
            )
        """)


def save_event(event_id: str, label: str, confidence: float, image_path: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO events (id, event_time, label, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
            (event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, round(confidence, 4), image_path),
        )


def list_events(limit: int = 50) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, event_time, label, confidence, image_path FROM events ORDER BY event_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_events() -> int:
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
