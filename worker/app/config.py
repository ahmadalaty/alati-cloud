from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    JWT_SECRET: str
    STORAGE_MODE: str = "local"
    LOCAL_STORAGE_DIR: str = "/data"

settings = Settings()
