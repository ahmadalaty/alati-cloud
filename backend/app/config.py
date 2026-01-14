from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # Auth
    JWT_SECRET: str
    JWT_ALG: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # R2 (S3 compatible) - REQUIRED
    STORAGE_MODE: str = "r2"  # force r2 only
    R2_ENDPOINT: str
    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_BUCKET: str

    # Model
    MODEL_VARIANT: str = "resnet18"  # resnet18 or resnet50

    # Demo admin seed
    OWNER_EMAIL: str = "admin@alati.ai"
    OWNER_PASSWORD: str = "admin123"

    # Debug
    DEMO_MODE: str = "1"
    DEBUG_ERRORS: str = "1"


settings = Settings()
