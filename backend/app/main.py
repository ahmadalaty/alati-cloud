import os
import json
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from .db import Base, engine, get_db, SessionLocal
from .models import User, Scan, ScanStatus
from .schemas import LoginRequest, TokenResponse, ScanCreateRequest, ScanResponse
from .auth import (
    hash_password,
    verify_password,
    create_token,
    require_user,
    decode_token,
)
from .storage import new_upload_id, save_upload, upload_target_path
from .tasks import process_scan

app = FastAPI(title="Alati Cloud API")


# ==========================================================
# DATABASE INIT (DEMO MODE SAFE REBUILD)
# ==========================================================
# This fixes the VARCHAR(2) -> VARCHAR(5) issue for `eye`
# Only rebuilds DB when DEMO_MODE=1
# ==========================================================

if os.getenv("DEMO_MODE", "").strip() == "1":
    Base.metadata.drop_all(bind=engine)

Base.metadata.create_all(bind=engine)


# ==========================================================
# UTILS
# ==========================================================

def make_json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]

    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return make_json_safe(obj.tolist())
    except Exception:
        pass

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return float(obj.detach().cpu().item())
            return make_json_safe(obj.detach().cpu().tolist())
    except Exception:
        pass

    return str(obj)


def decode_db_result(val):
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {"raw": val}
    return {"raw": str(val)}


# ==========================================================
# USER SEEDING
# ==========================================================

def _upsert_user(db: Session, email: str, password: str):
    email = (email or "").strip().lower()
    password = (password or "").strip()
    if not email or not password:
        return

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, password_hash=hash_password(password))
        db.add(user)
    else:
        user.password_hash = hash_password(password)

    db.commit()


@app.on_event("startup")
def seed_users():
    db = SessionLocal()
    try:
        # default demo admin
        _upsert_user(db, "admin@alati.ai", "admin123")

        # owner account from Render env vars
        owner_email = os.getenv("OWNER_EMAIL", "").strip()
        owner_password = os.getenv("OWNER_PASSWORD", "").strip()
        if owner_email and owner_password:
            _upsert_user(db, owner_email, owner_password)
    finally:
        db.close()


# ==========================================================
# BASIC ENDPOINTS
# ==========================================================

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug")
def debug_info():
    try:
        import torch
        torch_version = torch.__version__
        torch_ok = True
    except Exception as e:
        torch_ok = False
        torch_version = str(e)

    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "model_files")

    return {
        "demo_mode": os.getenv("DEMO_MODE", ""),
        "debug_errors": os.getenv("DEBUG_ERRORS", ""),
        "torch_ok": torch_ok,
        "torch_version": torch_version,
        "model_dir": model_dir,
        "resnet18_exists": os.path.exists(os.path.join(model_dir, "alati_dualeye_model_resnet18.pth")),
        "resnet50_exists": os.path.exists(os.path.join(model_dir, "alati_dualeye_model_resnet50.pth")),
        "labels_exists": os.path.exists(os.path.join(model_dir, "labels.json")),
    }


# ==========================================================
# AUTH
# ==========================================================

@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token(user.id)}


# ==========================================================
# UPLOAD & SCAN
# ==========================================================

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_id: int = Depends(require_user),
):
    upload_id = new_upload_id()
    path = await save_upload(upload_id, file)
    return {"upload_id": upload_id, "path": path}


@app.post("/scan/create", response_model=ScanResponse)
def create_scan(
    body: ScanCreateRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(require_user),
):
    image_path = upload_target_path(body.upload_id)
    if not os.path.exists(image_path):
        raise HTTPException(400, "Upload not found")

    scan = Scan(
        user_id=user_id,
        eye=body.eye,  # "left" / "right" now fits DB
        image_path=image_path,
        status=ScanStatus.queued,
    )

    db.add(scan)
    db.commit()
    db.refresh(scan)

    # DEMO_MODE: run inline (no worker dependency)
    if os.getenv("DEMO_MODE", "").strip() == "1":
        try:
            process_scan.run(scan.id)
        except Exception as e:
            if os.getenv("DEBUG_ERRORS", "").strip() == "1":
                raise HTTPException(500, f"AI processing failed: {e}")
            raise HTTPException(500, "AI processing failed")

        db.refresh(scan)

        result = make_json_safe(decode_db_result(scan.result))
        report_url = f"/scan/{scan.id}/report?token=" + create_token(user_id)

        return {
            "id": scan.id,
            "status": scan.status.value,
            "result": result,
            "report_url": report_url,
        }

    process_scan.delay(scan.id)
    return {"id": scan.id, "status": scan.status.value, "result": None, "report_url": None}


@app.get("/scan/{scan_id}", response_model=ScanResponse)
def get_scan(
    scan_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(require_user),
):
    scan = db.get(Scan, scan_id)
    if not scan or scan.user_id != user_id:
        raise HTTPException(404, "Not found")

    result = make_json_safe(decode_db_result(scan.result))
    report_url = f"/scan/{scan.id}/report?token=" + create_token(user_id) if scan.report_path else None

    return {
        "id": scan.id,
        "status": scan.status.value,
        "result": result,
        "report_url": report_url,
    }


# ==========================================================
# PDF REPORT (BROWSER SAFE)
# ==========================================================

@app.get("/scan/{scan_id}/report")
def get_report(
    scan_id: int,
    token: str = Query("", description="JWT token"),
    authorization: str = Depends(lambda: None),
    db: Session = Depends(get_db),
):
    user_id = None

    if token:
        user_id = decode_token(token)
    elif authorization:
        user_id = require_user(authorization)

    if not user_id:
        raise HTTPException(401, "Missing token")

    scan = db.get(Scan, scan_id)
    if not scan or scan.user_id != user_id or not scan.report_path:
        raise HTTPException(404, "Not found")

    from fastapi.responses import FileResponse
    return FileResponse(
        scan.report_path,
        media_type="application/pdf",
        filename=f"alati_report_{scan_id}.pdf",
    )
