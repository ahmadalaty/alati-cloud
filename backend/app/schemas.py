from pydantic import BaseModel, Field
from typing import Optional


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str


class ScanResult(BaseModel):
    id: int
    eye_mode: str
    left_diagnosis: Optional[str] = None
    right_diagnosis: Optional[str] = None
    status: str
    error: Optional[str] = None
