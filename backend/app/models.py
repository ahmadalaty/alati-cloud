from sqlalchemy import Column, Integer, String, DateTime, Text, func
from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True, nullable=False)

    eye_mode = Column(String(10), nullable=False)  # left/right/both
    left_key = Column(String(512), nullable=True)
    right_key = Column(String(512), nullable=True)

    left_diagnosis = Column(String(255), nullable=True)
    right_diagnosis = Column(String(255), nullable=True)

    status = Column(String(20), nullable=False, default="done")  # done/failed
    error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
