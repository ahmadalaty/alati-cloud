import os
import time
import secrets
import hashlib
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


def create_token(data: dict) -> str:
    """Create JWT token for login"""
    now = int(time.time())
    payload = {
        **data,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


# ============ API TOKEN FUNCTIONS ============

def generate_api_token() -> str:
    """Generate a random API token (shown only once to user)"""
    return secrets.token_urlsafe(32)


def hash_api_token(token: str) -> str:
    """Hash API token for storage in database"""
    return hashlib.sha256(token.encode()).hexdigest()


def verify_api_token(token: str, token_hash: str) -> bool:
    """Verify a plaintext token against its hash"""
    try:
        return hash_api_token(token) == token_hash
    except Exception:
        return False


# ============ TOKEN EXTRACTION & VALIDATION ============

def _extract_token(req: Request) -> Optional[str]:
    """Extract token from Authorization header or query params"""
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
    """
    Validate token and return user_id.
    Accepts both JWT tokens (from login) and API tokens (from admin issuance).
    """
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    # Try JWT first (from login)
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        sub = payload.get("sub")
        if sub:
            return int(sub)
    except (JWTError, ValueError):
        pass

    # If JWT failed, it might be an API token
    # For now, we'll mark it as invalid
    # (API tokens will be checked separately in the scan endpoint)
    raise HTTPException(status_code=401, detail="Invalid token")


def require_admin(req: Request) -> int:
    """
    Validate that request is from admin (OWNER_EMAIL).
    Works with JWT tokens from login OR email:password format.
    
    Format 1 (JWT from UI login): Authorization: Bearer <jwt_token>
    Format 2 (Basic auth): Authorization: Bearer email:password
    """
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing admin credentials")

    admin_email = (settings.OWNER_EMAIL or "").strip().lower()
    admin_password = (settings.OWNER_PASSWORD or "").strip()
    
    if not admin_email or not admin_password:
        raise HTTPException(status_code=500, detail="Admin not configured")
    
    # Try JWT first (from UI login)
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_email = payload.get("email", "").strip().lower()
        
        # Check if JWT email matches admin email
        if user_email == admin_email:
            return int(payload.get("sub", 0))
    except (JWTError, ValueError):
        pass
    
    # Try email:password format (backward compatibility)
    if ":" in token:
        email, password = token.split(":", 1)
        if email.lower() == admin_email and password == admin_password:
            return 0  # Return dummy user_id for admin
    
    # Neither worked
    raise HTTPException(status_code=401, detail="Invalid admin credentials")