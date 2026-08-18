from sqlalchemy import Column, Integer, String, DateTime, Date, Text, func, Boolean
from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    # User management fields
    is_banned = Column(Integer, default=0, nullable=False)  # 1=banned, 0=active
    usage_limit = Column(Integer, default=-1, nullable=False)  # -1 = unlimited; daily cap for the user's current tier
    usage_count = Column(Integer, default=0, nullable=False)  # scans used since usage_reset_date
    usage_reset_date = Column(Date, nullable=True)  # UTC date usage_count last reset on (lazy daily reset)

    # Billing / subscription (Paddle)
    tier = Column(String(20), default="free", nullable=False)  # 'free' | 'premium'
    paddle_customer_id = Column(String(255), nullable=True, index=True)
    paddle_subscription_id = Column(String(255), nullable=True, index=True)
    subscription_status = Column(String(30), nullable=True)  # active/canceled/paused/past_due etc.


class APIToken(Base):
    __tablename__ = "api_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True, nullable=False)
    
    # Token hash (never store plain tokens)
    token_hash = Column(String(255), unique=True, index=True, nullable=False)
    
    # Friendly name for this token
    name = Column(String(255), nullable=True)
    
    # When created and last used
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Can be revoked
    is_active = Column(Integer, default=1, nullable=False)  # 1=active, 0=revoked


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True, nullable=False)

    eye_mode = Column(String(10), nullable=False)  # left/right/both
    left_key = Column(String(512), nullable=True)
    right_key = Column(String(512), nullable=True)

    # AI Predictions
    left_diagnosis = Column(String(255), nullable=True)
    right_diagnosis = Column(String(255), nullable=True)

    # Ground Truth (confirmed by ophthalmologist)
    confirmed_left_diagnosis = Column(String(255), nullable=True)
    confirmed_right_diagnosis = Column(String(255), nullable=True)

    status = Column(String(20), nullable=False, default="done")  # done/failed
    error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    confirmed_at = Column(DateTime(timezone=True), nullable=True)  # When ophthalmologist confirmed