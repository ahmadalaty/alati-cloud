import uuid
import boto3
from botocore.config import Config
from .config import settings

BUILD_MARKER = "R2_ONLY_STORAGE_v1_2026_01_11"


def new_upload_id() -> str:
    return str(uuid.uuid4())


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def put_bytes(key: str, data: bytes, content_type: str = "image/jpeg") -> None:
    s3 = _s3()
    s3.put_object(
        Bucket=settings.R2_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def get_bytes(key: str) -> bytes:
    s3 = _s3()
    obj = s3.get_object(Bucket=settings.R2_BUCKET, Key=key)
    return obj["Body"].read()


def key_for(side: str, upload_id: str) -> str:
    # side: left/right/single
    side = (side or "single").strip().lower()
    return f"uploads/{side}/{upload_id}.jpg"
