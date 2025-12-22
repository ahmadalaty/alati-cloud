import os, uuid
from fastapi import UploadFile
from .config import settings

def new_upload_id() -> str:
    return str(uuid.uuid4())

def upload_target_path(upload_id: str) -> str:
    os.makedirs(settings.LOCAL_STORAGE_DIR, exist_ok=True)
    return os.path.join(settings.LOCAL_STORAGE_DIR, f"{upload_id}.jpg")

async def save_upload(upload_id: str, file: UploadFile) -> str:
    path = upload_target_path(upload_id)
    data = await file.read()
    with open(path, "wb") as f:
        f.write(data)
    return path
