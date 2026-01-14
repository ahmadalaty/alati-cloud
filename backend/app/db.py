from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings


def _sqlalchemy_db_url(url: str) -> str:
    """
    Force SQLAlchemy to use psycopg v3 driver.
    Render often provides DATABASE_URL like:
      - postgres://...
      - postgresql://...

    We convert to:
      - postgresql+psycopg://...
    """
    if not url:
        raise RuntimeError("DATABASE_URL is empty")

    url = url.strip()

    # Handle Render's common scheme
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    # If driver not specified, force psycopg v3
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    return url


DATABASE_URL = _sqlalchemy_db_url(settings.DATABASE_URL)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
