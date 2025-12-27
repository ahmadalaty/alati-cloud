import os
import uuid
from fastapi import UploadFile
from .config import settings

def _effective_storage_dir() -> str:
    """
    Render containers reliably allow writing to /tmp.
    If we're on Render, use /tmp/alati to avoid permission issues.
    Otherwise use LOCAL_STORAGE_DIR (local docker uses /data).
    """
    if os.getenv("RENDER", "").lower() == "true":
        return "/tmp/alati"
    return settings.LOCAL_STORAGE_DIR

def new_upload_id() -> str:
    return str(uuid.uuid4())

def upload_target_path(upload_id: str) -> str:
    base = _effective_storage_dir()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{upload_id}.jpg")

async def save_upload(upload_id: str, file: UploadFile) -> str:
    path = upload_target_path(upload_id)
    data = await file.read()
    with open(path, "wb") as f:
        f.write(data)
    return path
