"""Simple role-based authentication for the Streamlit/FastAPI app."""

from __future__ import annotations

import hashlib
import secrets
import time
from typing import Any, Dict, Optional

from storage import AppStorage


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    return hash_password(password, salt).split("$", 1)[1] == digest


class AuthService:
    def __init__(self, storage: AppStorage) -> None:
        self.storage = storage
        self.ensure_default_admin()

    def ensure_default_admin(self) -> None:
        if self.get_user("admin"):
            return
        self.create_user("admin", "admin123", "admin")

    def create_user(self, username: str, password: str, role: str = "analyst") -> None:
        if role not in {"admin", "analyst", "viewer"}:
            raise ValueError("role must be admin, analyst, or viewer")
        with self.storage.connect() as conn:
            conn.execute(
                """
                insert into users(id, username, password_hash, role, created_at)
                values (?, ?, ?, ?, ?)
                """,
                ("user-" + secrets.token_hex(6), username, hash_password(password), role, time.time()),
            )

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        with self.storage.connect() as conn:
            row = conn.execute("select * from users where username = ?", (username,)).fetchone()
        return dict(row) if row else None

    def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.get_user(username)
        if not user:
            return None
        if not verify_password(password, user["password_hash"]):
            return None
        return {"username": user["username"], "role": user["role"], "id": user["id"]}

    def create_session(self, username: str, ttl_days: int = 7) -> str:
        token = "sess-" + secrets.token_urlsafe(32)
        now = time.time()
        expires_at = now + ttl_days * 86400
        with self.storage.connect() as conn:
            conn.execute(
                "insert into sessions(token, username, expires_at, created_at) values (?, ?, ?, ?)",
                (token, username, expires_at, now),
            )
        return token

    def user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        now = time.time()
        with self.storage.connect() as conn:
            row = conn.execute("select * from sessions where token = ? and expires_at > ?", (token, now)).fetchone()
        if not row:
            return None
        user = self.get_user(dict(row)["username"])
        if not user:
            return None
        return {"username": user["username"], "role": user["role"], "id": user["id"]}

    def logout(self, token: str) -> None:
        if not token:
            return
        with self.storage.connect() as conn:
            conn.execute("delete from sessions where token = ?", (token,))


def require_role(user: Dict[str, Any], allowed: set[str]) -> None:
    if user.get("role") not in allowed:
        raise PermissionError("当前用户没有执行该操作的权限。")
