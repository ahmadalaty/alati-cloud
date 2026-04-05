from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str = ""  # Optional - admin can create users without password


class LoginResponse(BaseModel):
    """Response from login endpoint"""
    access_token: str
    is_owner: bool = False
    user_id: int
    email: str


class TokenResponse(BaseModel):
    access_token: str


class APITokenResponse(BaseModel):
    id: int
    user_id: int
    name: Optional[str] = None
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: int


class IssuedTokenResponse(BaseModel):
    """Response when admin issues a new token (only shows the token once)"""
    token: str  # The actual token (only shown once!)
    token_id: int
    user_id: int
    created_at: datetime


class IssueTokenRequest(BaseModel):
    user_id: int
    name: Optional[str] = None  # Friendly name like "Mobile App" or "Testing"


class ScanResult(BaseModel):
    id: int
    eye_mode: str
    left_diagnosis: Optional[str] = None
    right_diagnosis: Optional[str] = None
    status: str
    error: Optional[str] = None