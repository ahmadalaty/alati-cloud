from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from .db import Base, engine, get_db, SessionLocal
from .models import User, Scan, ScanStatus
from .schemas import LoginRequest, TokenResponse, ScanCreateRequest, ScanResponse
from .auth import hash_password, verify_password, create_token, require_user
from .storage import new_upload_id, save_upload
from .tasks import process_scan
from .config import settings
import os

app = FastAPI(title="Alati Cloud API")

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


def _upsert_user(db: Session, email: str, password: str, make_admin: bool = False):
    """
    Create the user if missing, otherwise update their password.
    This prevents 'Invalid credentials' confusion in demos.
    """
    email = (email or "").strip().lower()
    password = (password or "").strip()
    if not email or not password:
        return

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, password_hash=hash_password(password))
        # If your User model has is_admin, set it
        if make_admin and hasattr(user, "is_admin"):
            user.is_admin = True
        db.add(user)
        db.commit()
        return

    # Update password every startup so Render password changes always apply
    user.password_hash = hash_password(password)
    if make_admin and hasattr(user, "is_admin"):
        user.is_admin = True
    db.add(user)
    db.commit()


@app.on_event("startup")
def seed():
    """
    Startup bootstrap:
    1) Keep MVP default admin user.
    2) Ensure OWNER user exists in cloud DB based on Render env vars.
    """
    db = SessionLocal()
    try:
        # 1) Default MVP admin (keep for now)
        _upsert_user(db, "admin@alati.ai", "admin123", make_admin=True)

        # 2) Owner account from Render Environment Variables
        owner_email = os.getenv("OWNER_EMAIL", "").strip()
        owner_password = os.getenv("OWNER_PASSWORD", "").strip()
        if owner_email and owner_password:
            _upsert_user(db, owner_email, owner_password, make_admin=True)

    finally:
        db.close()


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token(user.id)}


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
    image_path = os.path.join(settings.LOCAL_STORAGE_DIR, f"{body.upload_id}.jpg")
    if not os.path.exists(image_path):
        raise HTTPException(400, "Upload not found")

    scan = Scan(
        user_id=user_id,
        eye=body.eye,
        image_path=image_path,
        status=ScanStatus.queued,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    process_scan.delay(scan.id)

    return {
        "id": scan.id,
        "status": scan.status.value,
        "result": None,
        "report_url": None,
    }


@app.get("/scan/{scan_id}", response_model=ScanResponse)
def get_scan(
    scan_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(require_user),
):
    scan = db.get(Scan, scan_id)
    if not scan or scan.user_id != user_id:
        raise HTTPException(404, "Not found")

    report_url = (
        f"/scan/{scan.id}/report"
        if scan.report_path and scan.status == ScanStatus.done
        else None
    )

    return {
        "id": scan.id,
        "status": scan.status.value,
        "result": scan.result,
        "report_url": report_url,
    }


@app.get("/scan/{scan_id}/report")
def get_report(
    scan_id: int,
    db: Session = Depends(get_db),
    user_id: int = Depends(require_user),
):
    scan = db.get(Scan, scan_id)
    if not scan or scan.user_id != user_id or not scan.report_path:
        raise HTTPException(404, "Not found")

    from fastapi.responses import FileResponse
    return FileResponse(
        scan.report_path,
        media_type="application/pdf",
        filename=f"alati_report_{scan_id}.pdf",
    )
