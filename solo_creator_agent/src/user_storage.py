from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import SOLODECK_AUTH_DB_PATH, UPLOAD_DIR


WORKSPACE_DATASET_TYPES = {"imported_records", "todo_items", "done_todos"}


def user_data_db_path() -> Path:
    return SOLODECK_AUTH_DB_PATH


def init_user_storage() -> None:
    db = user_data_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                original_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                mime_type TEXT,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, original_name, sha256)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_datasets (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                dataset_type TEXT NOT NULL,
                source_file_id TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe[:120] or "upload"


def _uploaded_file_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def save_uploaded_file(user_id: int | None, uploaded_file: Any, category: str = "general") -> dict[str, Any] | None:
    if not user_id:
        return None
    init_user_storage()
    original_name = getattr(uploaded_file, "name", "upload")
    mime_type = getattr(uploaded_file, "type", "") or ""
    content = _uploaded_file_bytes(uploaded_file)
    digest = hashlib.sha256(content).hexdigest()

    with sqlite3.connect(user_data_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            """
            SELECT id, original_name, stored_path, mime_type, size_bytes, sha256, created_at
            FROM uploaded_files
            WHERE user_id = ? AND original_name = ? AND sha256 = ?
            """,
            (user_id, original_name, digest),
        ).fetchone()
        if existing:
            return {**dict(existing), "existing": True}

        file_id = uuid4().hex
        user_dir = UPLOAD_DIR / str(user_id) / category
        user_dir.mkdir(parents=True, exist_ok=True)
        stored_path = user_dir / f"{file_id}_{_safe_filename(original_name)}"
        stored_path.write_bytes(content)

        conn.execute(
            """
            INSERT INTO uploaded_files
            (id, user_id, original_name, stored_path, mime_type, size_bytes, sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (file_id, user_id, original_name, str(stored_path), mime_type, len(content), digest, _utc_now()),
        )
        conn.commit()

    return {
        "id": file_id,
        "original_name": original_name,
        "stored_path": str(stored_path),
        "mime_type": mime_type,
        "size_bytes": len(content),
        "sha256": digest,
        "existing": False,
    }


def save_user_dataset(
    user_id: int | None,
    dataset_type: str,
    payload: Any,
    source_file_id: str | None = None,
) -> str | None:
    if not user_id:
        return None
    init_user_storage()
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)
    dataset_id = uuid4().hex
    with sqlite3.connect(user_data_db_path()) as conn:
        existing = conn.execute(
            """
            SELECT id
            FROM user_datasets
            WHERE user_id = ?
              AND dataset_type = ?
              AND COALESCE(source_file_id, '') = COALESCE(?, '')
              AND payload_json = ?
            """,
            (user_id, dataset_type, source_file_id, payload_json),
        ).fetchone()
        if existing:
            return str(existing[0])
        conn.execute(
            """
            INSERT INTO user_datasets
            (id, user_id, dataset_type, source_file_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, user_id, dataset_type, source_file_id, payload_json, _utc_now()),
        )
        conn.commit()
    return dataset_id


def load_latest_user_dataset(user_id: int | None, dataset_type: str, default: Any = None) -> Any:
    if not user_id:
        return default
    init_user_storage()
    with sqlite3.connect(user_data_db_path()) as conn:
        row = conn.execute(
            """
            SELECT payload_json
            FROM user_datasets
            WHERE user_id = ? AND dataset_type = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, dataset_type),
        ).fetchone()
    if not row:
        return default
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return default


def load_user_workspace(user_id: int | None) -> dict[str, Any]:
    return {
        "imported_records": load_latest_user_dataset(user_id, "imported_records", {"contents": [], "revenues": [], "campaigns": []}),
        "todo_items": load_latest_user_dataset(user_id, "todo_items", []),
        "done_todos": load_latest_user_dataset(user_id, "done_todos", {}),
        "recommendation_feedback": load_latest_user_dataset(user_id, "recommendation_feedback", []),
    }


def save_user_workspace(
    user_id: int | None,
    imported_records: dict[str, Any] | None = None,
    todo_items: list[dict[str, Any]] | None = None,
    done_todos: dict[str, Any] | None = None,
    recommendation_feedback: list[dict[str, Any]] | None = None,
) -> None:
    if not user_id:
        return
    if imported_records is not None:
        save_user_dataset(user_id, "imported_records", imported_records)
    if todo_items is not None:
        save_user_dataset(user_id, "todo_items", todo_items)
    if done_todos is not None:
        save_user_dataset(user_id, "done_todos", done_todos)
    if recommendation_feedback is not None:
        save_user_dataset(user_id, "recommendation_feedback", recommendation_feedback)


def list_user_files(user_id: int | None, limit: int = 8) -> list[dict[str, Any]]:
    if not user_id:
        return []
    init_user_storage()
    with sqlite3.connect(user_data_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, original_name, stored_path, mime_type, size_bytes, sha256, created_at
            FROM uploaded_files
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]
