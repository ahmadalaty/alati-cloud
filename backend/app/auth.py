import jwt
import hashlib
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from .config import settings

def hash_password(pw: str) -> str:
    # MVP simple hash (not for production)
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def verify_password(pw: str, hashed: str) -> bool:
    return hash_password(pw) == hashed

def create_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "exp": datetime.utcnow() + timedelta(days=7)}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

def require_user(authorization: str = Header(None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return int(payload["sub"])
    except Exception:
        raise HTTPException(401, "Invalid token")
