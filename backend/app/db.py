import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings

def _normalize_db_url(url: str) -> str:
    """
    Render often provides DATABASE_URL like:
      postgresql://user:pass@host:5432/db
    SQLAlchemy may default to psycopg2 unless driver is specified.
    We force psycopg (v3):
      postgresql+psycopg://...
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url

DATABASE_URL = _normalize_db_url(settings.DATABASE_URL)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
