from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings

Base = declarative_base()

def _normalize_db_url(url: str) -> str:
    """
    Force SQLAlchemy to use psycopg v3 driver.
    Render often provides 'postgresql://...' which may default to psycopg2 dialect.
    """
    if not url:
        return url

    url = url.strip()

    # If already explicit, keep it
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql+psycopg2://"):
        # Convert psycopg2 -> psycopg (v3)
        return url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

    # Render/others may give "postgresql://"
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)

    # Sometimes "postgres://"
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)

    return url

DATABASE_URL = _normalize_db_url(settings.DATABASE_URL)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
