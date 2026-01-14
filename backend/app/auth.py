import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request
from jose import jwt, JWTError
from passlib.context import CryptContext

from .config import settings

# PBKDF2 has no bcrypt 72-byte password limit.
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
)

TOKEN_TTL_SECONDS = 60 * 60 * 24  # 24h


def hash_password(password: str) -> str:
    password = (password or "").strip()
    if not password:
        raise ValueError("Password is empty")
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(password or "", password_hash or "")
    except Exception:
        return False


def create_token(user_id: int) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def _extract_token(req: Request) -> Optional[str]:
    # Authorization: Bearer <token>
    auth = req.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    # fallback (optional): ?token=...
    token = req.query_params.get("token")
    if token:
        return token.strip()

    return None


def require_user(req: Request) -> int:
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token")
        return int(sub)
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")
