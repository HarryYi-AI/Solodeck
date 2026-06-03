from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .config import SOLODECK_AUTH_DB_PATH

ROOT = Path(__file__).resolve().parents[1]


def auth_db_path() -> Path:
    return SOLODECK_AUTH_DB_PATH


def init_auth_db() -> None:
    db = auth_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                last_login_at TEXT
            )
            """
        )
        conn.commit()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120_000)
    return digest.hex()


def register_user(email: str, password: str, display_name: str = "") -> dict:
    init_auth_db()
    email = _normalize_email(email)
    if not email or "@" not in email and not email.isdigit():
        return {"ok": False, "error": "账号需要是邮箱或手机号。"}
    if len(password) < 6:
        return {"ok": False, "error": "密码至少 6 位。"}
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    now = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(auth_db_path()) as conn:
            cursor = conn.execute(
                "INSERT INTO users (email, password_hash, salt, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (email, password_hash, salt, display_name.strip() or email, now),
            )
            user_id = cursor.lastrowid
            conn.commit()
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "账号已存在，请直接登录。"}
    return {"ok": True, "user_id": user_id, "email": email, "display_name": display_name.strip() or email}


def login_user(email: str, password: str) -> dict:
    init_auth_db()
    email = _normalize_email(email)
    with sqlite3.connect(auth_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None:
            return {"ok": False, "error": "账号不存在。"}
        candidate = _hash_password(password, row["salt"])
        if not hmac.compare_digest(candidate, row["password_hash"]):
            return {"ok": False, "error": "密码不正确。"}
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, row["id"]))
        conn.commit()
        return {"ok": True, "user_id": row["id"], "email": row["email"], "display_name": row["display_name"] or row["email"]}


def user_count() -> int:
    init_auth_db()
    with sqlite3.connect(auth_db_path()) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
