import hashlib
import json
import os
import sys
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime

from .config import settings
from .db import Base, engine, get_db, SessionLocal
from .models import User, Scan, APIToken
from .schemas import (
    LoginRequest, 
    TokenResponse, 
    ScanResult,
    RegisterRequest,
    IssuedTokenResponse,
    IssueTokenRequest,
    APITokenResponse,
)
from .auth import (
    hash_password, 
    verify_password, 
    create_token, 
    require_user,
    require_admin,
    generate_api_token,
    hash_api_token,
    verify_api_token,
    _extract_token,
)
from .storage_r2 import new_upload_id, key_for, put_bytes, BUILD_MARKER as STORAGE_MARKER
from .inference import (
    predict_diagnosis,
    predict_debug,
    BUILD_MARKER as INF_MARKER,
    ACTIVE_VARIANT,
)

app = FastAPI(title="Alati Cloud - Eye Disease Screening")
Base.metadata.create_all(bind=engine)

# Mount static files folder - CORRECTED PATH
# From /app/app/main.py -> go up 2 levels -> /app -> then add static -> /app/static
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

print(f"✓ BASE_DIR: {BASE_DIR}", file=sys.stderr)
print(f"✓ STATIC_DIR: {STATIC_DIR}", file=sys.stderr)
print(f"✓ Exists: {os.path.exists(STATIC_DIR)}", file=sys.stderr)

if os.path.exists(STATIC_DIR):
    try:
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        print(f"✓ Static files mounted successfully from: {STATIC_DIR}", file=sys.stderr)
    except Exception as e:
        print(f"✗ Error mounting static files: {e}", file=sys.stderr)
else:
    print(f"✗ Static folder not found at: {STATIC_DIR}", file=sys.stderr)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.on_event("startup")
def startup():
    """Create tables and add missing columns"""
    Base.metadata.create_all(bind=engine)
    
    try:
        with engine.connect() as conn:
            # Add missing user columns
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN is_banned INTEGER DEFAULT 0;"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN usage_limit INTEGER DEFAULT -1;"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN usage_count INTEGER DEFAULT 0;"))
                conn.commit()
            except:
                pass
            
            # Add confirmed diagnosis columns
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN confirmed_left_diagnosis VARCHAR(255);"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN confirmed_right_diagnosis VARCHAR(255);"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN confirmed_at TIMESTAMP WITH TIME ZONE;"))
                conn.commit()
            except:
                pass
    except Exception as e:
        print(f"Error adding columns: {e}")
    
    # Create admin user
    db = SessionLocal()
    try:
        email = (settings.OWNER_EMAIL or "").strip().lower()
        password = (settings.OWNER_PASSWORD or "").strip()
        if email and password:
            password = password[:72]
            user = db.query(User).filter(User.email == email).first()
            if not user:
                user = User(email=email, password_hash=hash_password(password))
                db.add(user)
                db.commit()
            else:
                user.password_hash = hash_password(password)
                db.add(user)
                db.commit()
    finally:
        db.close()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
async def root():
    """Serve login page"""
    login_file = os.path.join(STATIC_DIR, "login_improved.html")
    if os.path.exists(login_file):
        return FileResponse(login_file)
    return HTMLResponse("Alati Cloud is running - but static files not found. Check folder structure.")


# Serve static pages
@app.get("/login")
async def login_page():
    login_file = os.path.join(STATIC_DIR, "login_improved.html")
    if os.path.exists(login_file):
        return FileResponse(login_file)
    raise HTTPException(status_code=404, detail="Login page not found")

@app.get("/scan")
async def scan_page():
    scan_file = os.path.join(STATIC_DIR, "scan_improved.html")
    if os.path.exists(scan_file):
        return FileResponse(scan_file)
    raise HTTPException(status_code=404, detail="Scan page not found")

@app.get("/dashboard")
async def dashboard_page():
    dashboard_file = os.path.join(STATIC_DIR, "admin_dashboard.html")
    if os.path.exists(dashboard_file):
        return FileResponse(dashboard_file)
    raise HTTPException(status_code=404, detail="Dashboard page not found")

@app.get("/results")
async def results_page():
    results_file = os.path.join(STATIC_DIR, "results_page.html")
    if os.path.exists(results_file):
        return FileResponse(results_file)
    raise HTTPException(status_code=404, detail="Results page not found")


# ============ AUTH ENDPOINTS ============

@app.post("/auth/register", response_model=TokenResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    email = (body.email or "").strip().lower()
    if not email or len(email) < 3:
        raise HTTPException(status_code=400, detail="Invalid email")
    
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    password = (body.password or "").strip()[:72]
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    
    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_token({"sub": str(user.id), "email": email})
    return TokenResponse(access_token=token, user_id=user.id, is_admin=False)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Login user"""
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()
    
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    is_admin = email == (settings.OWNER_EMAIL or "").strip().lower()
    token = create_token({"sub": str(user.id), "email": email})
    return TokenResponse(access_token=token, user_id=user.id, is_admin=is_admin)


# ============ USER ENDPOINTS ============

@app.get("/admin/users")
def list_users(req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] List all users with their management data"""
    require_admin(req)
    users = db.query(User).all()
    
    # Convert SQLAlchemy objects to dictionaries for JSON serialization
    return [
        {
            "id": u.id,
            "email": u.email,
            "is_banned": getattr(u, 'is_banned', 0),
            "usage_limit": getattr(u, 'usage_limit', -1),
            "usage_count": getattr(u, 'usage_count', 0),
        }
        for u in users
    ]


@app.patch("/admin/users/{user_id}")
def update_user(user_id: int, is_banned: int = None, usage_limit: int = None, req: Request = None, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Update user settings"""
    require_admin(req)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if is_banned is not None:
        user.is_banned = is_banned
    if usage_limit is not None:
        user.usage_limit = usage_limit
    db.add(user)
    db.commit()
    return {"status": "updated"}


# ============ TOKEN ENDPOINTS ============

@app.post("/admin/tokens/issue", response_model=IssuedTokenResponse)
def issue_token(body: IssueTokenRequest, req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Issue API token for user"""
    require_admin(req)
    user = db.query(User).filter(User.id == body.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    plain_token = generate_api_token()
    token_hash = hash_api_token(plain_token)
    api_token = APIToken(user_id=body.user_id, token_hash=token_hash, name=body.name)
    db.add(api_token)
    db.commit()
    db.refresh(api_token)
    
    return IssuedTokenResponse(token=plain_token, token_id=api_token.id, user_id=api_token.user_id, created_at=api_token.created_at)


@app.get("/admin/tokens", response_model=list[APITokenResponse])
def list_tokens(req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] List all API tokens"""
    require_admin(req)
    return db.query(APIToken).all()


@app.delete("/admin/tokens/{token_id}")
def revoke_token(token_id: int, req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Revoke API token"""
    require_admin(req)
    token = db.query(APIToken).filter(APIToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    token.is_active = 0
    db.add(token)
    db.commit()
    return {"status": "revoked"}


# ============ SCAN ENDPOINTS ============

@app.post("/scan/run", response_model=ScanResult)
async def scan_run(
    req: Request,
    eye_mode: str = Form(...),
    file: UploadFile | None = File(None),
    left_file: UploadFile | None = File(None),
    right_file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """Run a scan with usage limit checking"""
    
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    
    user_id = None
    try:
        from jose import jwt
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = int(payload.get("sub", 0))
    except:
        pass
    
    if not user_id:
        api_token = db.query(APIToken).filter(APIToken.token_hash == hash_api_token(token)).first()
        if not api_token or not api_token.is_active:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = api_token.user_id
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user or getattr(user, 'is_banned', 0):
        raise HTTPException(status_code=403, detail="User banned")
    
    # Check usage limits
    usage_limit = getattr(user, 'usage_limit', -1)
    usage_count = getattr(user, 'usage_count', 0)
    if usage_limit >= 0 and usage_count >= usage_limit:
        raise HTTPException(status_code=429, detail="Usage limit reached")
    
    eye_mode = (eye_mode or "").strip().lower()
    if eye_mode not in ("left", "right", "both"):
        raise HTTPException(400, detail="Invalid eye_mode")

    try:
        if eye_mode == "both":
            if not left_file or not right_file:
                raise HTTPException(400, detail="Both files required")
            
            left_bytes = await left_file.read()
            right_bytes = await right_file.read()
            
            left_key = key_for("left", new_upload_id())
            right_key = key_for("right", new_upload_id())
            
            put_bytes(left_key, left_bytes, left_file.content_type or "image/jpeg")
            put_bytes(right_key, right_bytes, right_file.content_type or "image/jpeg")
            
            left_diag = predict_debug(left_bytes).get("translated") or "Uncertain"
            right_diag = predict_debug(right_bytes).get("translated") or "Uncertain"
            
            scan = Scan(
                user_id=user_id, 
                eye_mode="both", 
                left_key=left_key, 
                right_key=right_key,
                left_diagnosis=left_diag, 
                right_diagnosis=right_diag, 
                status="done",
                error=None
            )
            db.add(scan)
            user.usage_count = getattr(user, 'usage_count', 0) + 1
            db.add(user)
            db.commit()
            db.refresh(scan)
            
            return ScanResult(id=scan.id, eye_mode=scan.eye_mode, left_diagnosis=scan.left_diagnosis, right_diagnosis=scan.right_diagnosis, status=scan.status, error=None)
        
        if not file:
            raise HTTPException(400, detail="File required")
        
        image_bytes = await file.read()
        r2_key = key_for(eye_mode, new_upload_id())
        put_bytes(r2_key, image_bytes, file.content_type or "image/jpeg")
        
        diag = predict_debug(image_bytes).get("translated") or "Uncertain"
        
        scan = Scan(
            user_id=user_id, eye_mode=eye_mode, status="done",
            left_key=r2_key if eye_mode == "left" else None,
            right_key=r2_key if eye_mode == "right" else None,
            left_diagnosis=diag if eye_mode == "left" else None,
            right_diagnosis=diag if eye_mode == "right" else None,
        )
        db.add(scan)
        user.usage_count = getattr(user, 'usage_count', 0) + 1
        db.add(user)
        db.commit()
        db.refresh(scan)
        
        return ScanResult(id=scan.id, eye_mode=scan.eye_mode, left_diagnosis=scan.left_diagnosis, right_diagnosis=scan.right_diagnosis, status=scan.status, error=None)
    
    except HTTPException:
        raise
    except Exception as e:
        scan = Scan(user_id=user_id, eye_mode=eye_mode, status="failed", error=str(e))
        db.add(scan)
        db.commit()
        db.refresh(scan)
        return JSONResponse(status_code=500, content={"id": scan.id, "eye_mode": scan.eye_mode, "status": scan.status, "error": "Scan failed"})


@app.post("/scan/confirm")
def confirm_diagnosis(
    req: Request,
    scan_id: int,
    confirmed_left_diagnosis: str = None,
    confirmed_right_diagnosis: str = None,
    db: Session = Depends(get_db),
):
    """Save professional opinion for diagnosis"""
    
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    
    user_id = None
    try:
        from jose import jwt
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = int(payload.get("sub", 0))
    except:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.user_id == user_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    try:
        if confirmed_left_diagnosis:
            scan.confirmed_left_diagnosis = confirmed_left_diagnosis
        if confirmed_right_diagnosis:
            scan.confirmed_right_diagnosis = confirmed_right_diagnosis
        
        scan.confirmed_at = datetime.utcnow()
        db.add(scan)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")
    
    return {"status": "saved", "scan_id": scan.id}