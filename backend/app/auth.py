import jwt
import hashlib
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from .config import settings


# =========================
# Password utilities
# =========================

def hash_password(pw: str) -> str:
    """
    Simple SHA256 hash (OK for MVP / demo).
    Replace with bcrypt/argon2 before production.
    """
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def verify_password(pw: str, hashed: str) -> bool:
    return hash_password(pw) == hashed


# =========================
# JWT utilities
# =========================

def create_token(user_id: int) -> str:
    """
    Create JWT token valid for 7 days.
    """
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(days=7),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def require_user(authorization: str = Header(None)) -> int:
    """
    FastAPI dependency for protected endpoints (Authorization header).
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def decode_token(token: str) -> int | None:
    """
    Decode JWT from query string (used for PDF download links).
    Returns user_id or None.
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return int(payload.get("sub"))
    except Exception:
        return None
