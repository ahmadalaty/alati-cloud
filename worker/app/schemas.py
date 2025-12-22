from pydantic import BaseModel

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str

class ScanCreateRequest(BaseModel):
    upload_id: str
    eye: str  # "OD" or "OS"

class ScanResponse(BaseModel):
    id: int
    status: str
    result: dict | None = None
    report_url: str | None = None
