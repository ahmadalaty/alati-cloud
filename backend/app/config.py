from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    
    # JWT for login tokens
    JWT_SECRET: str

    # Model
    MODEL_VARIANT: str = "resnet18"
    
    # Admin user (auto-created on startup)
    OWNER_EMAIL: str = ""
    OWNER_PASSWORD: str = ""
    
    # Debug mode
    DEBUG_ERRORS: str = "0"
    
    # API Token secret (for hashing API tokens)
    API_TOKEN_SECRET: str = ""
    
    class Config:
        env_file = ".env"


settings = Settings()