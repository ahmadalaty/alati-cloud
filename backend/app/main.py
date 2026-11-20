import hashlib
import hmac
import json
import os
import secrets
import smtplib
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy import text, or_
from datetime import datetime, timedelta, timezone
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .config import settings
from .db import Base, engine, get_db, SessionLocal
from .models import User, Scan, APIToken
from .schemas import (
    LoginRequest, 
    TokenResponse, 
    ScanResult,
    RegisterRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    IssuedTokenResponse,
    IssueTokenRequest,
    APITokenResponse,
)
from .auth import (
    hash_password, 
    verify_password, 
    create_token, 
    require_user,
    require_admin,
    generate_api_token,
    hash_api_token,
    verify_api_token,
    _extract_token,
)
from .inference import (
    predict_diagnosis,
    predict_debug,
    model_info,
)
from .report import create_pdf_report
from .inference import MODEL_VERSION

app = FastAPI(title="Alati - Diabetic Retinopathy Screening API")
Base.metadata.create_all(bind=engine)


# The service moved from its Render address to alati.nabeeh.health. Pages
# requested at the old address redirect to the same page on the new one. API
# calls and the Paddle webhook are left alone: Paddle still posts to the old
# URL, and a redirect would drop the POST body.
OLD_HOST = "alati-api.onrender.com"
CANONICAL_HOST = "alati.nabeeh.health"
_REDIRECT_PAGES = ("/", "/login", "/scan", "/dashboard", "/account", "/pricing", "/reset-password")


@app.middleware("http")
async def redirect_old_host(request: Request, call_next):
    host = request.headers.get("host", "").split(":")[0].lower()
    path = request.url.path
    if (host == OLD_HOST and request.method in ("GET", "HEAD")
            and (path in _REDIRECT_PAGES or path.startswith("/legal/"))):
        target = f"https://{CANONICAL_HOST}{path}"
        if request.url.query:
            target += f"?{request.url.query}"
        return RedirectResponse(target, status_code=301)
    return await call_next(request)


# ============ APP CONFIG (env-driven) ============

# Daily scan limits per tier. Free/DEFAULT_USAGE_LIMIT kept as a fallback for existing Render configs.
FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", os.getenv("DEFAULT_USAGE_LIMIT", "5")))
PREMIUM_DAILY_LIMIT = int(os.getenv("PREMIUM_DAILY_LIMIT", "50"))

# Paddle billing config
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET", "").strip()
PADDLE_CLIENT_TOKEN = os.getenv("PADDLE_CLIENT_TOKEN", "").strip()
PADDLE_PRICE_ID_PREMIUM = os.getenv("PADDLE_PRICE_ID_PREMIUM", "").strip()
PADDLE_ENV = os.getenv("PADDLE_ENV", "sandbox").strip().lower()  # 'sandbox' or 'production'
# Paddle-hosted portal (Paddle dashboard > Customer Portal) where subscribers
# cancel, update their card and download invoices.
PADDLE_CUSTOMER_PORTAL_URL = os.getenv(
    "PADDLE_CUSTOMER_PORTAL_URL", "https://customer-portal.paddle.com/cpl_01m0a41008xtamfn2akzr6sxsn"
).strip()

# App URL (used in verification email links).
APP_URL = os.getenv("APP_URL", "").rstrip("/") or "https://alati.nabeeh.health"

# SMTP config (if not set, email verification is disabled — registration still works)
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", "").strip() or SMTP_USER


def email_enabled() -> bool:
    """Check if SMTP is configured. If not, verification is bypassed."""
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


def _send_email(to_email: str, subject: str, text_body: str, html_body: str) -> bool:
    if not email_enabled():
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[email] Failed to send '{subject}' to {to_email}: {e}")
        return False


def send_password_reset_email(to_email: str, token: str) -> bool:
    link = f"{APP_URL}/reset-password?token={token}"
    html_body = f"""
    <html>
    <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #f8f7f3; padding: 2rem;">
      <div style="max-width: 500px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
        <div style="background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 1.5rem 2rem; color: white;">
          <h1 style="margin: 0; font-size: 22px; font-weight: 500;">Reset your Alati password</h1>
        </div>
        <div style="padding: 2rem;">
          <p style="font-size: 15px; color: #2c2c2a; line-height: 1.6;">We received a request to reset the password for this account. The link below works once and expires in 1 hour.</p>
          <div style="text-align: center; margin: 2rem 0;">
            <a href="{link}" style="background: #185fa5; color: white; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block;">Choose a new password</a>
          </div>
          <p style="font-size: 13px; color: #888; line-height: 1.6;">Or copy this link into your browser:</p>
          <p style="font-size: 13px; color: #185fa5; word-break: break-all;">{link}</p>
          <p style="font-size: 12px; color: #888; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;">If you didn't ask for this, you can ignore this email. Your password will not change.</p>
        </div>
      </div>
    </body>
    </html>
    """
    text_body = (f"Reset your Alati password\n\nOpen this link to choose a new password "
                 f"(it works once and expires in 1 hour):\n{link}\n\n"
                 f"If you didn't ask for this, you can ignore this email.")
    return _send_email(to_email, "Reset your Alati password", text_body, html_body)


def send_verification_email(to_email: str, token: str) -> bool:
    """
    Send verification email. Returns True on success, False on failure.
    If SMTP not configured, returns False (caller should auto-verify in that case).
    """
    if not email_enabled():
        return False

    verify_link = f"{APP_URL}/auth/verify?token={token}"

    html_body = f"""
    <html>
    <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #f8f7f3; padding: 2rem;">
      <div style="max-width: 500px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
        <div style="background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 2rem; color: white; text-align: center;">
          <h1 style="margin: 0; font-size: 28px; font-weight: 500;">Welcome to Alati</h1>
          <p style="margin: 8px 0 0; opacity: 0.95; font-size: 14px;">AI-assisted diabetic retinopathy screening</p>
        </div>
        <div style="padding: 2rem;">
          <p style="font-size: 16px; color: #2c2c2a; line-height: 1.6;">Hi,</p>
          <p style="font-size: 16px; color: #2c2c2a; line-height: 1.6;">Thanks for signing up. Please confirm your email address to activate your account.</p>
          <div style="text-align: center; margin: 2rem 0;">
            <a href="{verify_link}" style="background: linear-gradient(135deg, #185fa5 0%, #0c447c 100%); color: white; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block;">Verify Email</a>
          </div>
          <p style="font-size: 13px; color: #888; line-height: 1.6;">Or copy this link into your browser:</p>
          <p style="font-size: 13px; color: #185fa5; word-break: break-all;">{verify_link}</p>
          <p style="font-size: 12px; color: #888; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e0e0e0;">If you didn't create this account, you can safely ignore this email.</p>
        </div>
      </div>
    </body>
    </html>
    """

    text_body = f"Welcome to Alati!\n\nPlease verify your email by clicking this link:\n{verify_link}\n\nIf you didn't create this account, you can ignore this email."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Verify your Alati account"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[email] Failed to send verification to {to_email}: {e}")
        return False


# ============ END APP CONFIG ============


# ============ HTML TEMPLATES ============

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alati - Sign In</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto; background: linear-gradient(135deg, #0c447c 0%, #185fa5 50%, #0f6e56 100%); min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem; }
    
    .container { background: white; border-radius: 12px; max-width: 440px; width: 100%; overflow: hidden; box-shadow: 0 12px 32px rgba(0,0,0,0.12); }
    .header { background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 2rem; color: white; text-align: center; }
    .header-title { font-size: 36px; font-weight: 500; margin-bottom: 8px; }
    .header-subtitle { font-size: 14px; opacity: 0.95; margin: 0; }

    /* Measured performance (v10, retrospective, see /model_info) */
    .stats { margin-top: 1.25rem; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; text-align: left; }
    .stat { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22); border-radius: 8px; padding: 10px 12px; }
    .stat-label { font-size: 10.5px; font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase; opacity: 0.9; margin-bottom: 6px; line-height: 1.3; }
    .stat-nums { display: grid; gap: 3px; font-size: 12px; }
    .stat-nums b { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; margin-right: 3px; }
    .stat-note { grid-column: 1 / -1; font-size: 10.5px; line-height: 1.45; opacity: 0.85; }

    .form-container { padding: 2.5rem; }
    .tabs { display: flex; gap: 0; margin-bottom: 2rem; border-bottom: 2px solid #e0e0e0; }
    .tab-btn { flex: 1; padding: 14px; background: none; border: none; border-bottom: 3px solid transparent; color: #888780; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
    .tab-btn.active { border-bottom-color: #185fa5; color: #185fa5; }
    
    .form-group { margin-bottom: 1.5rem; }
    .form-label { display: block; font-size: 12px; font-weight: 600; color: #2c2c2a; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.3px; }
    input { width: 100%; padding: 12px 14px; border: 1.5px solid #e0e0e0; border-radius: 8px; font-size: 14px; font-family: inherit; transition: all 0.3s; background: #fafaf8; }
    input:focus { outline: none; border-color: #185fa5; background: white; }
    
    .form-options { display: flex; justify-content: flex-end; align-items: center; margin: -0.5rem 0 1.5rem; font-size: 13px; }
    .forgot-link { background: none; border: none; color: #185fa5; cursor: pointer; font-weight: 500; font-size: 13px; padding: 0; }
    .form-intro { font-size: 13px; color: #444441; line-height: 1.6; margin-bottom: 1.5rem; }
    .submit-btn:disabled { opacity: 0.6; cursor: wait; transform: none; }
    
    .submit-btn { width: 100%; padding: 13px; background: linear-gradient(135deg, #185fa5 0%, #0c447c 100%); color: white; border: none; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; transition: all 0.3s; margin-bottom: 1.5rem; }
    .submit-btn:hover { transform: translateY(-2px); }
    
    .signup-link { text-align: center; font-size: 13px; color: #888780; }
    .signup-link button { background: none; border: none; color: #185fa5; font-weight: 600; cursor: pointer; padding: 0; }
    
    .form-section { display: none; }
    .form-section.active { display: block; }
    
    .info-box { background: #eaf3de; border-radius: 8px; padding: 12px; margin-bottom: 1.5rem; font-size: 12px; color: #3b6d11; line-height: 1.5; }
    .error-msg { color: #d32f2f; font-size: 13px; margin-bottom: 1rem; }

    /* Disclaimer / consent */
    .disclaimer-box { background: #fff8e1; border: 1px solid #f0c000; border-radius: 8px; padding: 14px; margin-bottom: 1rem; max-height: 220px; overflow-y: auto; font-size: 12px; color: #3d3200; line-height: 1.6; }
    .disclaimer-box h4 { font-size: 13px; margin-bottom: 6px; color: #6d5700; }
    .disclaimer-box ul { margin: 8px 0 0 18px; }
    .disclaimer-box li { margin-bottom: 4px; }
    .consent-row { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 1.5rem; font-size: 13px; color: #2c2c2a; }
    .consent-row input[type="checkbox"] { width: 18px; height: 18px; flex-shrink: 0; margin-top: 1px; cursor: pointer; accent-color: #185fa5; }
    .consent-row label { cursor: pointer; }

    /* Plans / pricing */
    .plans { border-top: 1px solid #e0e0e0; padding: 1.75rem 2.5rem 2rem; background: #fafaf8; }
    .plans h3 { font-size: 12px; font-weight: 600; color: #2c2c2a; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 12px; }
    .plan-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .plan { background: white; border: 1.5px solid #e0e0e0; border-radius: 8px; padding: 12px 14px; }
    .plan.premium { border-style: dashed; background: #fafaf8; }
    .plan-name { font-size: 13px; font-weight: 600; color: #185fa5; display: flex; align-items: center; justify-content: space-between; gap: 6px; }
    .plan-price { font-size: 22px; font-weight: 600; color: #2c2c2a; margin: 2px 0 8px; font-variant-numeric: tabular-nums; }
    .plan-price span { font-size: 12px; font-weight: 400; color: #888780; }
    .plan-price.soon { font-size: 15px; color: #5f5e5a; padding: 4px 0 3px; }
    .soon-badge { font-size: 10px; font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase; color: #0c447c; background: #e6f0fa; border-radius: 999px; padding: 3px 8px; white-space: nowrap; }
    .plan ul { list-style: none; font-size: 12px; color: #444441; line-height: 1.6; }
    .plan li::before { content: "✓ "; color: #0f6e56; font-weight: 600; }
    .plan.premium li::before { content: "– "; color: #888780; }
    .plans-note { font-size: 11.5px; color: #888780; line-height: 1.5; margin-top: 10px; }
    .plans-note a { color: #185fa5; }

    /* Legal footer */
    .legal-footer { width: 100%; max-width: 640px; padding: 18px 8px 0; text-align: center; font-size: 12px; line-height: 2; }
    .legal-footer a { color: rgba(255,255,255,0.85); text-decoration: none; margin: 0 8px; }
    .legal-footer a:hover { text-decoration: underline; color: white; }
    .legal-footer span { color: rgba(255,255,255,0.4); }
    
    /* Banned account banner */
    .banned-banner { display: none; background: #ffebee; border-left: 4px solid #c62828; border-radius: 6px; padding: 14px 16px; margin-bottom: 1.5rem; }
    .banned-banner.active { display: block; }
    .banned-banner .title { font-size: 14px; font-weight: 600; color: #c62828; margin-bottom: 4px; }
    .banned-banner .text { font-size: 13px; color: #5d1818; line-height: 1.5; }
    
    /* Verification needed banner */
    .verify-banner { display: none; background: #fff8e1; border-left: 4px solid #f57c00; border-radius: 6px; padding: 14px 16px; margin-bottom: 1.5rem; }
    .verify-banner.active { display: block; }
    .verify-banner .title { font-size: 14px; font-weight: 600; color: #e65100; margin-bottom: 4px; }
    .verify-banner .text { font-size: 13px; color: #6d3700; line-height: 1.5; }
    .verify-banner button { background: none; border: none; color: #185fa5; font-weight: 600; cursor: pointer; padding: 0; margin-top: 6px; font-size: 13px; text-decoration: underline; }
    
    /* Success banner (used after registration when email verification is required) */
    .success-banner { display: none; background: #e8f5e9; border-left: 4px solid #2e7d32; border-radius: 6px; padding: 14px 16px; margin-bottom: 1.5rem; }
    .success-banner.active { display: block; }
    .success-banner .title { font-size: 14px; font-weight: 600; color: #1b5e20; margin-bottom: 4px; }
    .success-banner .text { font-size: 13px; color: #1b5e20; line-height: 1.5; }
    
    /* ========== MOBILE RESPONSIVE ========== */
    html, body { overflow-x: hidden; -webkit-text-size-adjust: 100%; }
    
    @media (max-width: 600px) {
      body { padding: 1rem; min-height: 100vh; justify-content: flex-start; }
      .container { max-width: 100%; border-radius: 10px; }
      .header { padding: 1.5rem 1rem; }
      .header-title { font-size: 26px; }
      .header-subtitle { font-size: 13px; }
      .stat { padding: 9px 10px; }
      .stat-nums b { font-size: 17px; }
      .form-container { padding: 1.5rem 1.25rem; }
      .tabs { margin-bottom: 1.5rem; }
      .tab-btn { padding: 12px 8px; font-size: 13px; }
      .form-group { margin-bottom: 1.25rem; }
      .form-label { font-size: 11px; }
      input { padding: 11px 12px; font-size: 16px; }  /* 16px prevents iOS auto-zoom */
      .submit-btn { padding: 14px; font-size: 15px; }
      .form-options { font-size: 12px; flex-wrap: wrap; gap: 8px; }
      .signup-link { font-size: 12px; }
      .info-box { font-size: 11px; padding: 10px; }
      .plans { padding: 1.5rem 1.25rem; }
      .plan-grid { grid-template-columns: 1fr; }
      .banned-banner, .verify-banner, .success-banner { padding: 12px; }
      .banned-banner .title, .verify-banner .title, .success-banner .title { font-size: 13px; }
      .banned-banner .text, .verify-banner .text, .success-banner .text { font-size: 12px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-title">Alati</div>
      <p class="header-subtitle">AI-assisted diabetic retinopathy screening</p>
      <div class="stats" aria-label="Measured performance">
        <div class="stat">
          <div class="stat-label">Referable diabetic retinopathy</div>
          <div class="stat-nums"><span><b>99.4%</b> sensitivity</span><span><b>79.2%</b> specificity</span></div>
        </div>
        <div class="stat">
          <div class="stat-label">Any diabetic retinopathy</div>
          <div class="stat-nums"><span><b>99.8%</b> sensitivity</span><span><b>89.0%</b> specificity</span></div>
        </div>
        <p class="stat-note">Retrospective evaluation on 3,662 retinal photographs from an independent public dataset not used in development; 3,406 were gradable, as low-resolution photos are declined rather than called normal (95% CI: referable 98.9–99.7% / 77.3–81.0%; any DR 99.5–99.9% / 87.4–90.5%). Standard fundus cameras; smartphone cameras not validated. A screening aid, not a replacement for an eye examination.</p>
      </div>
    </div>

    <div class="form-container">
      <div class="tabs">
        <button type="button" class="tab-btn active" data-tab="login" onclick="switchTab('login')">Sign in</button>
        <button type="button" class="tab-btn" data-tab="register" onclick="switchTab('register')">Create account</button>
      </div>

      <form id="login" class="form-section active" onsubmit="handleLogin(event)" novalidate>
        <div id="verify-banner" class="verify-banner">
          <div class="title">Email not verified</div>
          <div class="text" id="verify-text">Please verify your email address before signing in. Check your inbox for the verification link.</div>
          <button type="button" onclick="resendVerification()">Resend verification email</button>
        </div>

        <div id="success-banner" class="success-banner">
          <div class="title" id="success-title">Account created</div>
          <div class="text" id="success-text">Please check your email and click the verification link to activate your account.</div>
        </div>

        <div id="login-error" class="error-msg" style="display: none;"></div>

        <div class="form-group">
          <label class="form-label" for="login-email">Email address</label>
          <input type="email" id="login-email" placeholder="doctor@clinic.com" autocomplete="email">
        </div>

        <div class="form-group">
          <label class="form-label" for="login-password">Password</label>
          <input type="password" id="login-password" placeholder="••••••••" autocomplete="current-password">
        </div>

        <div class="form-options">
          <button type="button" class="forgot-link" onclick="switchTab('forgot')">Forgot password?</button>
        </div>

        <button type="submit" class="submit-btn" id="login-btn">Sign in</button>

        <p class="signup-link">Don't have an account? <button type="button" onclick="switchTab('register')">Create one</button></p>
      </form>

      <form id="forgot" class="form-section" onsubmit="handleForgot(event)" novalidate>
        <p class="form-intro">Enter the email address you registered with. We'll send you a link to choose a new password.</p>
        <div id="forgot-error" class="error-msg" style="display: none;"></div>
        <div id="forgot-done" class="success-banner">
          <div class="title">Check your email</div>
          <div class="text" id="forgot-done-text">If an account exists for that address, a reset link is on its way. The link expires in 1 hour.</div>
        </div>
        <div class="form-group">
          <label class="form-label" for="forgot-email">Email address</label>
          <input type="email" id="forgot-email" placeholder="doctor@clinic.com" autocomplete="email">
        </div>
        <button type="submit" class="submit-btn" id="forgot-btn">Send reset link</button>
        <p class="signup-link">Remembered it? <button type="button" onclick="switchTab('login')">Back to sign in</button></p>
      </form>

      <form id="register" class="form-section" onsubmit="handleRegister(event)" novalidate>
        <div id="register-error" class="error-msg" style="display: none;"></div>

        <div class="form-group">
          <label class="form-label" for="register-name">Full name</label>
          <input type="text" id="register-name" placeholder="Dr. Jane Smith" autocomplete="name" maxlength="120">
        </div>

        <div class="form-group">
          <label class="form-label" for="register-email">Email address</label>
          <input type="email" id="register-email" placeholder="doctor@clinic.com" autocomplete="email">
        </div>

        <div class="form-group">
          <label class="form-label" for="register-password">Password</label>
          <input type="password" id="register-password" placeholder="••••••••" autocomplete="new-password">
        </div>

        <div class="info-box">
          Password must be at least 8 characters
        </div>

        <div class="disclaimer-box">
          <h4>Important — please read before use</h4>
          <p>This tool is an AI-based screening aid for diabetic retinopathy. It is currently in a pre-approval / research-use stage and has not yet received regulatory clearance (e.g., FDA, CE, or local Ministry of Health approval) in any jurisdiction.</p>
          <p style="margin-top: 8px;">This tool is intended solely as a supportive aid to a licensed physician's own clinical judgment. It does not diagnose, and its output must not be used as the sole basis for any clinical decision. All findings should be independently verified by the treating physician.</p>
          <p style="margin-top: 8px;">By creating an account and using this tool, you confirm that:</p>
          <ul>
            <li>You are a licensed medical professional using this tool within your own clinical judgment and responsibility</li>
            <li>You understand this tool has not received regulatory approval and are choosing to use it in its current pre-approval state</li>
            <li>You will not rely on this tool as a sole or final diagnostic determination</li>
            <li>You accept full responsibility for clinical decisions made in your practice, including any use of this tool's output</li>
          </ul>
        </div>

        <div class="consent-row">
          <input type="checkbox" id="register-consent">
          <label for="register-consent">I have read and understood the above, and I agree to these terms.</label>
        </div>

        <button type="submit" class="submit-btn" id="register-btn" style="background: linear-gradient(135deg, #0f6e56 0%, #085041 100%);">Create account</button>

        <p class="signup-link">Already have an account? <button type="button" onclick="switchTab('login')">Sign in</button></p>
      </form>
    </div>

    <div class="plans" id="pricing">
      <h3>Plans</h3>
      <div class="plan-grid">
        <div class="plan">
          <div class="plan-name">Free</div>
          <div class="plan-price">$0</div>
          <ul>
            <li>5 scans per day</li>
            <li>Referable DR result per photo</li>
            <li>Image-quality check</li>
            <li>PDF report and scan history</li>
          </ul>
        </div>
        <div class="plan premium">
          <div class="plan-name">Premium <span class="soon-badge">Coming soon</span></div>
          <div class="plan-price soon">For clinics and screening programmes</div>
          <ul>
            <li>Higher daily scan volume</li>
            <li>Everything in Free</li>
          </ul>
        </div>
      </div>
      <p class="plans-note">Alati is free to use while Premium is in preparation. Need a higher scan volume or an evaluation account? Contact <a href="mailto:ahmadalaty@gmail.com">ahmadalaty@gmail.com</a></p>
    </div>
  </div>

  <div class="legal-footer">
    <a href="/legal/terms">Terms of Service</a><span>·</span>
    <a href="/legal/privacy">Privacy Notice</a><span>·</span>
    <a href="/pricing">Pricing</a><span>·</span>
    <a href="mailto:ahmadalaty@gmail.com">Contact</a><span>·</span>
    <a href="https://nabeeh.health">Nabeeh</a>
  </div>

  <script>
    // Sign-in, account creation and password reset are separate panels; the
    // reset panel has no tab of its own, so neither tab is highlighted on it.
    function switchTab(tab) {
      document.querySelectorAll('.form-section').forEach(el => el.classList.remove('active'));
      document.getElementById(tab).classList.add('active');
      document.querySelectorAll('.tab-btn').forEach(el =>
        el.classList.toggle('active', el.dataset.tab === tab));
      if (tab === 'forgot') {
        document.getElementById('forgot-email').value = document.getElementById('login-email').value;
        document.getElementById('forgot-done').classList.remove('active');
        document.getElementById('forgot-error').style.display = 'none';
      }
    }

    let lastEmailForResend = '';

    function hideAllBanners() {
      document.getElementById('verify-banner').classList.remove('active');
      document.getElementById('success-banner').classList.remove('active');
      document.getElementById('login-error').style.display = 'none';
    }

    function showError(id, text) {
      const el = document.getElementById(id);
      el.textContent = text;
      el.style.display = 'block';
    }

    async function readDetail(res, fallback) {
      try {
        const body = await res.json();
        if (typeof body.detail === 'string') return body.detail;
        return body.message || fallback;
      } catch (e) { return fallback; }
    }

    function setBusy(btnId, busy, label) {
      const btn = document.getElementById(btnId);
      btn.disabled = busy;
      if (label) btn.textContent = label;
    }

    async function handleLogin(ev) {
      if (ev) ev.preventDefault();
      const email = document.getElementById('login-email').value.trim();
      const password = document.getElementById('login-password').value;

      hideAllBanners();
      lastEmailForResend = email;

      if (!email || !password) {
        showError('login-error', 'Please enter your email and password.');
        return;
      }

      setBusy('login-btn', true, 'Signing in…');
      try {
        const res = await fetch('/auth/login', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email, password})
        });

        if (res.ok) {
          const data = await res.json();
          localStorage.setItem('token', data.access_token);
          localStorage.setItem('is_admin', data.is_admin ? 'true' : 'false');
          window.location.href = data.is_admin === true ? '/dashboard' : '/scan';
          return;
        }
        const detail = await readDetail(res, 'Incorrect email or password.');
        if (res.status === 403 && /verif/i.test(detail)) {
          document.getElementById('verify-text').textContent = detail;
          document.getElementById('verify-banner').classList.add('active');
        } else if (res.status === 401) {
          showError('login-error', 'Incorrect email or password.');
        } else {
          showError('login-error', detail);
        }
      } catch (err) {
        console.error('Login error:', err);
        showError('login-error', 'Could not reach the server. Check your connection and try again.');
      }
      setBusy('login-btn', false, 'Sign in');
    }

    async function handleRegister(ev) {
      if (ev) ev.preventDefault();
      const full_name = document.getElementById('register-name').value.trim();
      const email = document.getElementById('register-email').value.trim();
      const password = document.getElementById('register-password').value;
      const consent = document.getElementById('register-consent').checked;

      document.getElementById('register-error').style.display = 'none';
      hideAllBanners();

      if (!full_name || !email || !password) {
        showError('register-error', 'Please fill in your name, email and password.');
        return;
      }
      if (password.length < 8) {
        showError('register-error', 'Password must be at least 8 characters.');
        return;
      }
      if (!consent) {
        showError('register-error', 'Please read and agree to the terms above before creating an account.');
        return;
      }

      setBusy('register-btn', true, 'Creating account…');
      try {
        const res = await fetch('/auth/register', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email, password, full_name})
        });

        if (res.ok) {
          const data = await res.json();
          if (data.status === 'verification_sent') {
            switchTab('login');
            document.getElementById('login-email').value = email;
            document.getElementById('success-title').textContent = 'Account created';
            document.getElementById('success-text').textContent = 'We sent a verification link to ' + email + '. Open it to activate your account, then sign in here. If it is not in your inbox, check the spam folder.';
            document.getElementById('success-banner').classList.add('active');
            lastEmailForResend = email;
          } else if (data.access_token) {
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('is_admin', data.is_admin ? 'true' : 'false');
            window.location.href = '/scan';
            return;
          }
        } else {
          showError('register-error', await readDetail(res, 'Registration failed. Please try again.'));
        }
      } catch (err) {
        console.error('Registration error:', err);
        showError('register-error', 'Could not reach the server. Check your connection and try again.');
      }
      setBusy('register-btn', false, 'Create account');
    }

    async function handleForgot(ev) {
      if (ev) ev.preventDefault();
      const email = document.getElementById('forgot-email').value.trim();
      document.getElementById('forgot-error').style.display = 'none';
      document.getElementById('forgot-done').classList.remove('active');
      if (!email) { showError('forgot-error', 'Please enter your email address.'); return; }

      setBusy('forgot-btn', true, 'Sending…');
      try {
        const res = await fetch('/auth/forgot-password', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email})
        });
        if (res.ok) {
          const data = await res.json();
          document.getElementById('forgot-done-text').textContent = data.message;
          document.getElementById('forgot-done').classList.add('active');
        } else {
          showError('forgot-error', await readDetail(res, 'Could not send the reset link. Please try again.'));
        }
      } catch (e) {
        showError('forgot-error', 'Could not reach the server. Check your connection and try again.');
      }
      setBusy('forgot-btn', false, 'Send reset link');
    }

    async function resendVerification() {
      const email = lastEmailForResend || document.getElementById('login-email').value.trim();
      const text = document.getElementById('verify-text');
      if (!email) { text.textContent = 'Enter your email address above first.'; return; }
      try {
        const res = await fetch('/auth/resend-verification', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email})
        });
        const data = await res.json();
        text.textContent = data.message || data.detail || 'Check your email for the verification link.';
      } catch (e) {
        text.textContent = 'Could not resend the verification email. Please try again.';
      }
    }

    // Already signed in: go straight to the scanner.
    if (localStorage.getItem('token')) {
      fetch('/user/me', {headers: {'Authorization': 'Bearer ' + localStorage.getItem('token')}})
        .then(r => { if (r.ok) window.location.href = localStorage.getItem('is_admin') === 'true' ? '/dashboard' : '/scan'; })
        .catch(() => {});
    }
  </script>
</body>
</html>"""

SCAN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alati - Retinal Scan</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto; background: #f8f7f3; min-height: 100vh; padding: 2rem; color: #2c2c2a; }

    .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow: hidden; }
    .header { background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 2rem 2.5rem; color: white; display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
    .header-left h1 { font-size: 28px; font-weight: 500; margin-bottom: 4px; }
    .header-subtitle { font-size: 14px; opacity: 0.95; }
    .header-btns { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .header-btn { padding: 8px 14px; background: rgba(255,255,255,0.16); color: white; border: 1px solid rgba(255,255,255,0.4); border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; text-decoration: none; font-family: inherit; white-space: nowrap; }
    .header-btn:hover { background: rgba(255,255,255,0.28); }

    .content { padding: 2rem 2.5rem 2.5rem; }
    .form-group { margin-bottom: 1.75rem; }
    .form-label { display: block; font-size: 12px; font-weight: 600; color: #2c2c2a; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.3px; }

    .eye-select { display: flex; gap: 10px; }
    .eye-option { flex: 1; }
    .eye-option input[type="radio"] { position: absolute; opacity: 0; pointer-events: none; }
    .eye-option label { display: block; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; text-align: center; cursor: pointer; font-size: 14px; font-weight: 500; color: #444441; }
    .eye-option input[type="radio"]:checked + label { border-color: #185fa5; background: #f0f6ff; color: #0c447c; }
    .eye-option input[type="radio"]:focus-visible + label { outline: 2px solid #185fa5; outline-offset: 2px; }

    .uploads { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .uploads.both { grid-template-columns: 1fr 1fr; }
    .upload-area { position: relative; border: 2px dashed #d6d6d0; border-radius: 10px; padding: 1.75rem 1rem; text-align: center; cursor: pointer; background: #fcfcfa; min-height: 168px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px; }
    .upload-area:hover, .upload-area.drag { border-color: #185fa5; background: #f5f9fe; }
    .upload-area:focus-visible { outline: 2px solid #185fa5; outline-offset: 2px; }
    .upload-area.has-file { border-style: solid; border-color: #0f6e56; background: #f4faf7; }
    .upload-eye { font-size: 11px; font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase; color: #185fa5; }
    .upload-icon { width: 34px; height: 34px; color: #888780; }
    .upload-text { font-size: 14px; color: #444441; }
    .upload-hint { font-size: 12px; color: #888780; }
    .upload-preview { width: 112px; height: 112px; border-radius: 50%; object-fit: cover; background: #000; display: none; box-shadow: 0 1px 4px rgba(0,0,0,0.2); }
    .upload-area.has-file .upload-preview { display: block; }
    .upload-area.has-file .upload-icon, .upload-area.has-file .upload-hint { display: none; }
    .file-name { font-size: 12px; color: #0f6e56; font-weight: 600; word-break: break-all; display: none; }
    .upload-area.has-file .file-name { display: block; }

    .guidance { font-size: 12.5px; color: #5f5e5a; line-height: 1.55; margin-top: 12px; padding: 10px 12px; background: #f8f7f3; border-radius: 8px; }

    .buttons { display: flex; gap: 10px; justify-content: space-between; align-items: center; margin-top: 1.75rem; flex-wrap: wrap; }
    .usage-note { font-size: 12.5px; color: #888780; }
    .btn-row { display: flex; gap: 10px; }
    .btn { padding: 12px 22px; border: none; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; font-family: inherit; }
    .btn-secondary { background: #ecebe6; color: #2c2c2a; }
    .btn-secondary:hover { background: #e0dfd9; }
    .btn-primary { background: linear-gradient(135deg, #185fa5 0%, #0c447c 100%); color: white; min-width: 130px; }
    .btn-outline { background: white; color: #185fa5; border: 1.5px solid #185fa5; }
    .btn:disabled { opacity: 0.55; cursor: not-allowed; }

    .spinner { display: inline-block; width: 15px; height: 15px; border: 2px solid rgba(255,255,255,0.35); border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: -2px; margin-right: 8px; }
    @keyframes spin { to { transform: rotate(360deg); } }

    .processing { display: none; align-items: center; gap: 14px; padding: 1rem 1.25rem; background: #f0f6ff; border-radius: 8px; margin-top: 1.5rem; }
    .processing.active { display: flex; }
    .big-spinner { width: 30px; height: 30px; flex-shrink: 0; border: 3px solid #d7e5f5; border-top-color: #185fa5; border-radius: 50%; animation: spin 0.8s linear infinite; }
    .processing-text { font-size: 14px; color: #0c447c; font-weight: 600; }
    .processing-sub { font-size: 12.5px; color: #5f5e5a; margin-top: 2px; }

    .alert { display: none; margin-top: 1.5rem; padding: 12px 14px; border-radius: 8px; font-size: 13.5px; line-height: 1.5; background: #fdecea; color: #8a1c1c; border-left: 4px solid #c62828; }
    .alert.active { display: block; }

    #results { display: none; margin-top: 1.75rem; }
    .results-title { font-size: 12px; font-weight: 600; color: #888780; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 10px; }
    .result-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .result-grid.both { grid-template-columns: 1fr 1fr; }
    .result-card { border-radius: 10px; padding: 1rem 1.25rem; border: 1px solid #e0e0e0; border-left-width: 5px; background: #fff; }
    .result-card.green { border-left-color: #2e7d32; background: #f4faf5; }
    .result-card.amber { border-left-color: #ef6c00; background: #fff8ef; }
    .result-card.red { border-left-color: #c62828; background: #fdf3f3; }
    .result-card.grey { border-left-color: #888780; background: #f6f6f4; }
    .result-eye { font-size: 11px; font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase; color: #888780; margin-bottom: 4px; }
    .result-headline { font-size: 18px; font-weight: 600; line-height: 1.3; }
    .result-card.green .result-headline { color: #1b5e20; }
    .result-card.amber .result-headline { color: #a64b00; }
    .result-card.red .result-headline { color: #b71c1c; }
    .result-card.grey .result-headline { color: #444441; }
    .result-detail { font-size: 13px; color: #444441; margin-top: 6px; line-height: 1.55; }
    .result-actions { display: flex; gap: 10px; margin-top: 1rem; flex-wrap: wrap; }
    .result-foot { font-size: 11.5px; color: #888780; margin-top: 12px; line-height: 1.5; }

    html, body { overflow-x: hidden; -webkit-text-size-adjust: 100%; max-width: 100%; }

    @media (max-width: 768px) {
      body { padding: 1rem; }
      .container { max-width: 100%; border-radius: 10px; }
      .header { padding: 1.5rem 1.25rem; flex-direction: column; align-items: stretch; }
      .header-left { text-align: center; }
      .header-left h1 { font-size: 24px; }
      .header-subtitle { font-size: 12.5px; }
      .header-btns { justify-content: center; }
      .content { padding: 1.5rem 1.25rem; }
      .uploads.both, .result-grid.both { grid-template-columns: 1fr; }
      .buttons { flex-direction: column-reverse; align-items: stretch; }
      .btn-row .btn { flex: 1; }
      .usage-note { text-align: center; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-left">
        <h1>Retinal Scan</h1>
        <p class="header-subtitle">Diabetic retinopathy screening from a colour fundus photograph</p>
      </div>
      <div class="header-btns">
        <a class="header-btn" id="admin-btn" href="/dashboard" style="display: none;">Admin dashboard</a>
        <a class="header-btn" href="/account">My account</a>
        <button type="button" class="header-btn" onclick="logout()">Sign out</button>
      </div>
    </div>

    <div class="content">
      <div class="form-group">
        <span class="form-label" id="eye-label">Which eye?</span>
        <div class="eye-select" role="radiogroup" aria-labelledby="eye-label">
          <div class="eye-option">
            <input type="radio" id="left" name="eye" value="left" checked>
            <label for="left">Left eye</label>
          </div>
          <div class="eye-option">
            <input type="radio" id="right" name="eye" value="right">
            <label for="right">Right eye</label>
          </div>
          <div class="eye-option">
            <input type="radio" id="both" name="eye" value="both">
            <label for="both">Both eyes</label>
          </div>
        </div>
      </div>

      <div class="form-group">
        <span class="form-label">Fundus photograph</span>
        <div class="uploads" id="uploads">
          <div class="upload-area" id="zone-a" data-slot="a" tabindex="0" role="button" aria-label="Choose a fundus photograph">
            <div class="upload-eye" id="zone-a-eye" style="display:none;">Left eye</div>
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>
            <img class="upload-preview" alt="Selected photograph">
            <div class="upload-text">Click to choose a photo, or drag it here</div>
            <div class="upload-hint">JPG or PNG, up to 20 MB</div>
            <div class="file-name"></div>
          </div>
          <div class="upload-area" id="zone-b" data-slot="b" tabindex="0" role="button" aria-label="Choose the right-eye photograph" style="display:none;">
            <div class="upload-eye">Right eye</div>
            <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>
            <img class="upload-preview" alt="Selected right-eye photograph">
            <div class="upload-text">Click to choose a photo, or drag it here</div>
            <div class="upload-hint">JPG or PNG, up to 20 MB</div>
            <div class="file-name"></div>
          </div>
        </div>
        <input type="file" id="file-a" accept="image/*" hidden>
        <input type="file" id="file-b" accept="image/*" hidden>
        <p class="guidance">Upload the original colour fundus photograph from a standard fundus camera. Screenshots, downloaded copies and photos sent through messaging apps are often too small or too compressed: Alati can still flag referable disease in them, but will not rule it out. Smartphone fundus cameras and ultra-widefield images are not yet validated. Photos are analysed in memory and are not stored.</p>
      </div>

      <div class="buttons">
        <span class="usage-note" id="usage-note"></span>
        <div class="btn-row">
          <button type="button" class="btn btn-secondary" onclick="resetForm()">Clear</button>
          <button type="button" class="btn btn-primary" id="analyze-btn" onclick="submitScan()">Analyze</button>
        </div>
      </div>

      <div class="processing" id="processing" role="status">
        <div class="big-spinner"></div>
        <div>
          <div class="processing-text">Analysing the photograph…</div>
          <div class="processing-sub">This usually takes 5–15 seconds per eye.</div>
        </div>
      </div>

      <div class="alert" id="error-box" role="alert"></div>

      <div id="results" aria-live="polite">
        <div class="results-title">Result</div>
        <div class="result-grid" id="result-grid"></div>
        <div class="result-actions">
          <button type="button" class="btn btn-outline" id="report-btn" onclick="downloadReport()">Download PDF report</button>
          <button type="button" class="btn btn-secondary" onclick="resetForm(); window.scrollTo({top: 0, behavior: 'smooth'});">New scan</button>
        </div>
        <p class="result-foot">Alati is a screening aid for licensed clinicians, not a diagnosis. Confirm every result clinically. It screens for diabetic retinopathy only and does not exclude other eye disease.</p>
      </div>
    </div>
  </div>

  <script>
    const MAX_BYTES = 20 * 1024 * 1024;
    const token = localStorage.getItem('token');
    if (!token) window.location.replace('/login');
    const files = { a: null, b: null };
    let lastScanId = null;

    if (localStorage.getItem('is_admin') === 'true') {
      document.getElementById('admin-btn').style.display = '';
    }

    function logout() {
      localStorage.removeItem('token');
      localStorage.removeItem('is_admin');
      window.location.href = '/login';
    }

    function escapeHtml(s) {
      if (s == null) return '';
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    // Turns the stored result text into what a clinician should read. The
    // wording of the stored text comes from inference.py.
    function describe(dx) {
      const s = dx || '';
      if (/not gradable/i.test(s)) {
        const reason = s.split('—').slice(1).join('—').trim();
        return { tone: 'grey', headline: 'Image not gradable',
                 detail: (reason ? reason + ' ' : '') + 'No result was produced. This is not a normal result.' };
      }
      if (/not referable/i.test(s)) {
        return { tone: 'amber', headline: 'Diabetic retinopathy, not referable',
                 detail: 'Signs of diabetic retinopathy below the referral threshold. Monitor and rescreen; this is not a normal result.' };
      }
      if (/referable/i.test(s)) {
        return { tone: 'red', headline: 'Referable diabetic retinopathy',
                 detail: 'Ophthalmology review recommended. Severity is not graded; assess the eye clinically.' };
      }
      if (/^normal$/i.test(s.trim())) {
        return { tone: 'green', headline: 'No diabetic retinopathy detected',
                 detail: 'Rescreen at the usual interval. This does not exclude other eye disease.' };
      }
      return { tone: 'grey', headline: s || 'No result', detail: '' };
    }

    function selectedMode() {
      return document.querySelector('input[name="eye"]:checked').value;
    }

    function applyMode() {
      const mode = selectedMode();
      const both = mode === 'both';
      document.getElementById('uploads').classList.toggle('both', both);
      document.getElementById('zone-b').style.display = both ? '' : 'none';
      document.getElementById('zone-a-eye').style.display = both ? '' : 'none';
      document.getElementById('zone-a').setAttribute('aria-label', both ? 'Choose the left-eye photograph' : 'Choose a fundus photograph');
      if (!both) clearSlot('b');
    }
    document.querySelectorAll('input[name="eye"]').forEach(r => r.addEventListener('change', applyMode));

    function showError(text) {
      const box = document.getElementById('error-box');
      box.textContent = text;
      box.classList.add('active');
    }
    function hideError() { document.getElementById('error-box').classList.remove('active'); }

    function setSlot(slot, file) {
      hideError();
      if (!file) return;
      if (!/^image\\//.test(file.type) && !/\\.(jpe?g|png|bmp|tiff?|webp)$/i.test(file.name)) {
        showError('"' + file.name + '" is not an image. Choose a JPG or PNG fundus photograph.');
        return;
      }
      if (file.size > MAX_BYTES) {
        showError('"' + file.name + '" is larger than 20 MB. Choose a smaller file.');
        return;
      }
      files[slot] = file;
      const zone = document.getElementById('zone-' + slot);
      const img = zone.querySelector('.upload-preview');
      if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
      img.dataset.url = URL.createObjectURL(file);
      img.src = img.dataset.url;
      zone.classList.add('has-file');
      zone.querySelector('.upload-text').textContent = 'Click to change the photo';
      zone.querySelector('.file-name').textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
    }

    function clearSlot(slot) {
      files[slot] = null;
      const zone = document.getElementById('zone-' + slot);
      const img = zone.querySelector('.upload-preview');
      if (img.dataset.url) { URL.revokeObjectURL(img.dataset.url); delete img.dataset.url; }
      img.removeAttribute('src');
      zone.classList.remove('has-file');
      zone.querySelector('.upload-text').textContent = 'Click to choose a photo, or drag it here';
      zone.querySelector('.file-name').textContent = '';
      document.getElementById('file-' + slot).value = '';
    }

    ['a', 'b'].forEach(slot => {
      const zone = document.getElementById('zone-' + slot);
      const input = document.getElementById('file-' + slot);
      zone.addEventListener('click', () => input.click());
      zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
      input.addEventListener('change', e => setSlot(slot, e.target.files[0]));
      zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
      zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
      zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('drag');
        if (e.dataTransfer.files.length) setSlot(slot, e.dataTransfer.files[0]);
      });
    });

    async function loadUsage() {
      try {
        const res = await fetch('/user/me', { headers: { 'Authorization': 'Bearer ' + token } });
        if (res.status === 401) { logout(); return; }
        if (!res.ok) return;
        const u = await res.json();
        const note = document.getElementById('usage-note');
        if (u.is_banned) {
          note.textContent = 'Your account is suspended.';
          document.getElementById('analyze-btn').disabled = true;
          showError('Your account is suspended, so new scans are disabled. Contact ahmadalaty@gmail.com for help.');
        } else if (u.usage_limit === -1 || u.usage_limit === null) {
          note.textContent = 'Unlimited scans';
        } else {
          const left = Math.max(0, u.usage_limit - u.usage_count);
          note.textContent = left + ' of ' + u.usage_limit + ' scans left today';
        }
      } catch (e) {}
    }

    async function submitScan() {
      const mode = selectedMode();
      hideError();
      if (mode === 'both' && (!files.a || !files.b)) {
        showError('Choose a photograph for each eye, or select a single eye.');
        return;
      }
      if (mode !== 'both' && !files.a) {
        showError('Choose a fundus photograph first.');
        return;
      }

      const formData = new FormData();
      formData.append('eye_mode', mode);
      if (mode === 'both') {
        formData.append('left_file', files.a);
        formData.append('right_file', files.b);
      } else {
        formData.append('file', files.a);
      }

      const analyzeBtn = document.getElementById('analyze-btn');
      const processing = document.getElementById('processing');
      const results = document.getElementById('results');
      analyzeBtn.disabled = true;
      analyzeBtn.innerHTML = '<span class="spinner"></span>Analysing…';
      processing.classList.add('active');
      results.style.display = 'none';

      try {
        const res = await fetch('/scan/run', {
          method: 'POST',
          headers: { 'Authorization': 'Bearer ' + token },
          body: formData
        });
        let data = {};
        try { data = await res.json(); } catch (e) {}

        if (res.ok) {
          lastScanId = data.id;
          const eyes = [];
          if (data.left_diagnosis) eyes.push(['Left eye', data.left_diagnosis]);
          if (data.right_diagnosis) eyes.push(['Right eye', data.right_diagnosis]);
          const grid = document.getElementById('result-grid');
          grid.classList.toggle('both', eyes.length > 1);
          grid.innerHTML = eyes.map(([eye, dx]) => {
            const d = describe(dx);
            return '<div class="result-card ' + d.tone + '">' +
                   '<div class="result-eye">' + eye + '</div>' +
                   '<div class="result-headline">' + escapeHtml(d.headline) + '</div>' +
                   (d.detail ? '<div class="result-detail">' + escapeHtml(d.detail) + '</div>' : '') +
                   '</div>';
          }).join('');
          results.style.display = 'block';
          results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          loadUsage();
        } else if (res.status === 401) {
          showError('Your session has expired. Please sign in again.');
          setTimeout(logout, 1500);
        } else {
          showError(typeof data.detail === 'string' ? data.detail
                    : 'The scan could not be completed. Please try again with a different photograph.');
        }
      } catch (e) {
        showError('Could not reach the server. Check your connection and try again.');
      } finally {
        processing.classList.remove('active');
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze';
      }
    }

    async function downloadReport() {
      if (!lastScanId) return;
      const btn = document.getElementById('report-btn');
      btn.disabled = true;
      try {
        const res = await fetch('/scan/' + lastScanId + '/report', { headers: { 'Authorization': 'Bearer ' + token } });
        if (!res.ok) throw new Error('status ' + res.status);
        const url = URL.createObjectURL(await res.blob());
        const a = document.createElement('a');
        a.href = url; a.download = 'alati_report_' + lastScanId + '.pdf';
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
      } catch (e) {
        showError('The report could not be downloaded. Please try again.');
      } finally {
        btn.disabled = false;
      }
    }

    function resetForm() {
      clearSlot('a'); clearSlot('b');
      hideError();
      document.getElementById('results').style.display = 'none';
      document.getElementById('processing').classList.remove('active');
      lastScanId = null;
    }

    applyMode();
    if (token) loadUsage();
  </script>
</body>
</html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alati - Admin Dashboard</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto; background: #f8f7f3; min-height: 100vh; padding: 2rem; }
    
    .container { max-width: 1200px; margin: 0 auto; }
    .header { background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 2rem; color: white; border-radius: 12px; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center; }
    .header-left h1 { font-size: 28px; font-weight: 500; }
    .header-btns { display: flex; gap: 1rem; }
    .header-btn { padding: 8px 16px; background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.4); border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.3s; }
    .header-btn:hover { background: rgba(255,255,255,0.3); }
    
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
    .stat-card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .stat-label { font-size: 12px; color: #888; text-transform: uppercase; margin-bottom: 8px; }
    .stat-value { font-size: 32px; font-weight: 500; color: #185fa5; }
    
    .section-title { font-size: 16px; font-weight: 600; margin-top: 2rem; margin-bottom: 1rem; color: #2c2c2a; }
    
    .table-container { background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 2rem; }
    .table-header { padding: 1.5rem; border-bottom: 1px solid #e0e0e0; }
    .table-header h3 { font-size: 16px; }
    
    table { width: 100%; border-collapse: collapse; }
    th { padding: 1rem 1.5rem; text-align: left; font-weight: 600; font-size: 12px; color: #888; text-transform: uppercase; border-bottom: 1px solid #e0e0e0; }
    td { padding: 1rem 1.5rem; border-bottom: 1px solid #f0f0f0; }
    
    .user-email { font-weight: 500; color: #2c2c2a; }
    .status { font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #e8f5e9; color: #2e7d32; }
    .status.banned { background: #ffebee; color: #c62828; }
    .diagnosis { font-weight: 500; }
    .diagnosis.normal { color: #2e7d32; }
    .diagnosis.dr { color: #d32f2f; }
    
    input { padding: 8px; border: 1px solid #e0e0e0; border-radius: 4px; font-size: 12px; }
    button { padding: 8px 12px; background: #185fa5; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; }
    
    .toolbar { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; border-bottom: 1px solid #e0e0e0; gap: 1rem; }
    .search-box { flex: 1; max-width: 320px; position: relative; }
    .search-box input { width: 100%; padding: 10px 12px 10px 36px; font-size: 13px; border-radius: 6px; }
    .search-box::before { content: "🔍"; position: absolute; left: 12px; top: 50%; transform: translateY(-50%); font-size: 14px; pointer-events: none; }
    
    .scroll-container { max-height: 480px; overflow-y: auto; }
    .scroll-container table { width: 100%; }
    .scroll-container thead { position: sticky; top: 0; background: #f8f7f3; z-index: 1; }
    
    .pagination { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; border-top: 1px solid #e0e0e0; }
    .pagination-info { font-size: 13px; color: #888; }
    .pagination-controls { display: flex; gap: 8px; }
    .pagination-controls button { padding: 6px 12px; background: white; color: #185fa5; border: 1px solid #e0e0e0; }
    .pagination-controls button:hover:not(:disabled) { background: #f0f6ff; }
    .pagination-controls button:disabled { opacity: 0.4; cursor: not-allowed; }
    
    /* ========== MOBILE RESPONSIVE ========== */
    html, body { overflow-x: hidden; -webkit-text-size-adjust: 100%; max-width: 100%; }
    
    /* Tables: keep their own horizontal scroll so the PAGE doesn't shift */
    .scroll-container { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .scroll-container table { min-width: 720px; }
    
    @media (max-width: 768px) {
      body { padding: 1rem; }
      .container { max-width: 100%; }
      .header { padding: 1.5rem 1.25rem; flex-direction: column; gap: 1rem; align-items: stretch; border-radius: 10px; }
      .header-left { text-align: center; }
      .header-left h1 { font-size: 22px; }
      .header-btns { justify-content: center; flex-wrap: wrap; gap: 8px; }
      .header-btn { font-size: 11px; padding: 8px 14px; }
      
      .stats { grid-template-columns: 1fr; gap: 1rem; margin-bottom: 1.5rem; }
      .stat-card { padding: 1.25rem; border-radius: 10px; }
      .stat-value { font-size: 26px; }
      
      .section-title { font-size: 15px; }
      
      .toolbar { padding: 0.75rem 1rem; flex-direction: column; align-items: stretch; gap: 8px; }
      .toolbar h3 { font-size: 13px; }
      .search-box { max-width: 100%; }
      .search-box input { padding: 9px 12px 9px 32px; font-size: 13px; }
      .search-box::before { font-size: 12px; left: 10px; }
      
      th { padding: 0.75rem 1rem; font-size: 11px; }
      td { padding: 0.75rem 1rem; font-size: 13px; }
      td input[type="number"] { width: 56px !important; font-size: 12px; padding: 5px; }
      td select { font-size: 12px; padding: 4px !important; }
      td button { padding: 6px 10px; font-size: 11px; margin-left: 2px !important; }
      
      .pagination { padding: 0.75rem 1rem; flex-wrap: wrap; gap: 8px; }
      .pagination-info { font-size: 11px; }
      .pagination-controls button { padding: 6px 10px; font-size: 11px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-left">
        <h1>Admin Dashboard</h1>
      </div>
      <div class="header-btns">
        <button class="header-btn" onclick="goToScan()">Retinal scan</button>
        <button class="header-btn" onclick="logout()">Sign out</button>
      </div>
    </div>

    <div class="stats">
      <div class="stat-card">
        <div class="stat-label">Total Users</div>
        <div class="stat-value" id="user-count">-</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Scans</div>
        <div class="stat-value" id="scan-count">-</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Model in service</div>
        <div class="stat-value" id="model-version">-</div>
        <div style="font-size: 12px; color: #888; margin-top: 4px;" id="model-sub"></div>
      </div>
    </div>

    <h2 class="section-title">Recent Scans</h2>
    <div class="table-container">
      <div class="toolbar">
        <h3 style="font-size: 14px; font-weight: 600;">Scans</h3>
        <div class="search-box">
          <input type="text" id="scans-search" placeholder="Search by email or diagnosis..." oninput="onScansSearch()">
        </div>
      </div>
      <div class="scroll-container">
        <table>
          <thead>
            <tr>
              <th>User Email</th>
              <th>Eye Mode</th>
              <th>Left Diagnosis</th>
              <th>Right Diagnosis</th>
              <th>Date</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="scans-tbody">
            <tr><td colspan="6" style="text-align: center; color: #888;">Loading scans...</td></tr>
          </tbody>
        </table>
      </div>
      <div class="pagination">
        <div class="pagination-info" id="scans-info">-</div>
        <div class="pagination-controls">
          <button id="scans-prev" onclick="changeScansPage(-1)">← Prev</button>
          <button id="scans-next" onclick="changeScansPage(1)">Next →</button>
        </div>
      </div>
    </div>

    <h2 class="section-title">User Management</h2>
    <div class="table-container">
      <div class="toolbar">
        <h3 style="font-size: 14px; font-weight: 600;">All Users</h3>
        <div style="display: flex; gap: 10px; align-items: center;">
          <div class="search-box">
            <input type="text" id="users-search" placeholder="Search by email..." oninput="onUsersSearch()">
          </div>
          <button onclick="resetLegacyAccounts()" style="padding: 8px 14px; background: #f57c00; color: white; border: none; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap;" title="Resets stale Free-tier limits and manually-granted Premium accounts with no real subscription back to the current Free default. Custom accounts and genuine paying Premium subscribers are left alone.">Reset Legacy Accounts</button>
        </div>
      </div>
      <div class="scroll-container">
        <table>
          <thead>
            <tr>
              <th>Email</th>
              <th>Usage</th>
              <th>Limit</th>
              <th>Plan</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="users-tbody">
            <tr><td colspan="6" style="text-align: center; color: #888;">Loading users...</td></tr>
          </tbody>
        </table>
      </div>
      <div class="pagination">
        <div class="pagination-info" id="users-info">-</div>
        <div class="pagination-controls">
          <button id="users-prev" onclick="changeUsersPage(-1)">← Prev</button>
          <button id="users-next" onclick="changeUsersPage(1)">Next →</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    function goToScan() {
      window.location.href = '/scan';
    }
    
    function logout() {
      localStorage.removeItem('token');
      localStorage.removeItem('is_admin');
      window.location.href = '/login';
    }
    
    // Pagination state
    let scansPage = 1, scansQuery = '', scansTotal = 0, scansPageSize = 10;
    let usersPage = 1, usersQuery = '', usersTotal = 0, usersPageSize = 10;
    let scansSearchTimer = null, usersSearchTimer = null;
    let planDefaults = { free: 5, premium: 50 };

    async function loadPlanDefaults() {
      try {
        const res = await fetch('/billing/config');
        const cfg = await res.json();
        if (cfg.free_daily_limit != null) planDefaults.free = cfg.free_daily_limit;
        if (cfg.premium_daily_limit != null) planDefaults.premium = cfg.premium_daily_limit;
      } catch (e) {}
    }

    function onTierChange(id) {
      const tier = document.getElementById('tier-' + id).value;
      if (tier === 'custom') return; // leave the limit field alone — admin sets it directly
      document.getElementById('limit-' + id).value = (tier === 'premium') ? planDefaults.premium : planDefaults.free;
    }

    async function resetLegacyAccounts() {
      if (!confirm("Reset stale Free-tier usage limits and manually-granted Premium accounts (no real subscription) back to the current Free plan default?\\n\\nAccounts set to Custom, or with a genuine paying subscription, are left untouched.")) return;
      const token = localStorage.getItem('token');
      try {
        const res = await fetch('/admin/users/reset-legacy-free', {
          method: 'POST',
          headers: {'Authorization': 'Bearer ' + token}
        });
        if (res.ok) {
          const data = await res.json();
          alert(`Reset ${data.reset_count} account(s) to Free.`);
          loadUsers();
        } else {
          alert('Failed to reset accounts: ' + res.status);
        }
      } catch (e) {
        alert('Error: ' + e.message);
      }
    }
    
    function onScansSearch() {
      clearTimeout(scansSearchTimer);
      scansSearchTimer = setTimeout(() => {
        scansQuery = document.getElementById('scans-search').value.trim();
        scansPage = 1;
        loadScans();
      }, 300);
    }
    
    function onUsersSearch() {
      clearTimeout(usersSearchTimer);
      usersSearchTimer = setTimeout(() => {
        usersQuery = document.getElementById('users-search').value.trim();
        usersPage = 1;
        loadUsers();
      }, 300);
    }
    
    function changeScansPage(delta) {
      const maxPage = Math.max(1, Math.ceil(scansTotal / scansPageSize));
      scansPage = Math.max(1, Math.min(maxPage, scansPage + delta));
      loadScans();
    }
    
    function changeUsersPage(delta) {
      const maxPage = Math.max(1, Math.ceil(usersTotal / usersPageSize));
      usersPage = Math.max(1, Math.min(maxPage, usersPage + delta));
      loadUsers();
    }
    
    async function loadScans() {
      const token = localStorage.getItem('token');
      const tbody = document.getElementById('scans-tbody');
      const info = document.getElementById('scans-info');
      
      try {
        const params = new URLSearchParams({ q: scansQuery, page: scansPage, page_size: scansPageSize });
        const res = await fetch('/admin/scans?' + params.toString(), {
          headers: {'Authorization': 'Bearer ' + token}
        });
        
        if (!res.ok) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #c62828;">Failed to load scans (' + res.status + ')</td></tr>';
          document.getElementById('scan-count').textContent = '!';
          info.textContent = '';
          return;
        }
        
        const data = await res.json();
        const scans = data.items || [];
        scansTotal = data.total || 0;
        
        document.getElementById('scan-count').textContent = scansTotal;
        
        const start = scansTotal === 0 ? 0 : (scansPage - 1) * scansPageSize + 1;
        const end = Math.min(scansTotal, scansPage * scansPageSize);
        info.textContent = scansTotal === 0 ? 'No results' : `Showing ${start}-${end} of ${scansTotal}`;
        
        const maxPage = Math.max(1, Math.ceil(scansTotal / scansPageSize));
        document.getElementById('scans-prev').disabled = (scansPage <= 1);
        document.getElementById('scans-next').disabled = (scansPage >= maxPage);
        
        if (scans.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">' + (scansQuery ? 'No scans match your search' : 'No scans yet') + '</td></tr>';
          return;
        }
        
        tbody.innerHTML = scans.map(s => `
          <tr>
            <td class="user-email">${escapeHtml(s.user_email)}</td>
            <td>${escapeHtml(s.eye_mode)}</td>
            <td><span class="diagnosis ${s.left_diagnosis && s.left_diagnosis.toLowerCase().includes('normal') ? 'normal' : 'dr'}">${escapeHtml(s.left_diagnosis || '-')}</span></td>
            <td><span class="diagnosis ${s.right_diagnosis && s.right_diagnosis.toLowerCase().includes('normal') ? 'normal' : 'dr'}">${escapeHtml(s.right_diagnosis || '-')}</span></td>
            <td>${new Date(s.created_at).toLocaleString()}</td>
            <td><span class="status ${s.status === 'failed' ? 'banned' : ''}">${escapeHtml(s.status)}</span></td>
          </tr>
        `).join('');
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #c62828;">Error: ' + escapeHtml(e.message) + '</td></tr>';
      }
    }
    
    async function loadUsers() {
      const token = localStorage.getItem('token');
      const tbody = document.getElementById('users-tbody');
      const info = document.getElementById('users-info');
      
      try {
        const params = new URLSearchParams({ q: usersQuery, page: usersPage, page_size: usersPageSize });
        const res = await fetch('/admin/users?' + params.toString(), {
          headers: {'Authorization': 'Bearer ' + token}
        });
        
        if (!res.ok) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #c62828;">Failed to load users (' + res.status + ')</td></tr>';
          info.textContent = '';
          return;
        }
        
        const data = await res.json();
        const users = data.items || [];
        usersTotal = data.total || 0;
        
        document.getElementById('user-count').textContent = usersTotal;
        
        const start = usersTotal === 0 ? 0 : (usersPage - 1) * usersPageSize + 1;
        const end = Math.min(usersTotal, usersPage * usersPageSize);
        info.textContent = usersTotal === 0 ? 'No results' : `Showing ${start}-${end} of ${usersTotal}`;
        
        const maxPage = Math.max(1, Math.ceil(usersTotal / usersPageSize));
        document.getElementById('users-prev').disabled = (usersPage <= 1);
        document.getElementById('users-next').disabled = (usersPage >= maxPage);
        
        if (users.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #888;">' + (usersQuery ? 'No users match your search' : 'No users yet') + '</td></tr>';
          return;
        }
        
        tbody.innerHTML = users.map(u => {
          const used = u.usage_count || 0;
          const limit = u.usage_limit;
          const limitDisplay = (limit === -1 || limit === null) ? '∞' : limit;
          const nearLimit = (limit > 0 && used >= limit * 0.8);
          const overLimit = (limit >= 0 && used >= limit);
          const usageColor = overLimit ? '#c62828' : (nearLimit ? '#ef6c00' : '#2e7d32');
          
          const tier = (u.tier || 'free').toLowerCase();
          const subStatus = u.subscription_status;
          const planBadge = tier === 'premium'
            ? `⭐ Premium${subStatus ? ` <span style="color:#888; font-weight:400;">(${escapeHtml(subStatus)})</span>` : ''}`
            : tier === 'custom'
            ? '🔧 Custom'
            : 'Free';

          return `
          <tr id="user-row-${u.id}">
            <td class="user-email">${escapeHtml(u.email)}</td>
            <td style="font-weight: 600; color: ${usageColor};">${used} / ${limitDisplay}</td>
            <td><input type="number" id="limit-${u.id}" value="${limit}" style="width: 70px;" title="Use -1 for unlimited"></td>
            <td>
              <select id="tier-${u.id}" style="padding: 6px; border-radius: 4px;" title="${planBadge}" onchange="onTierChange(${u.id})">
                <option value="free" ${tier === 'free' ? 'selected' : ''}>Free</option>
                <option value="premium" ${tier === 'premium' ? 'selected' : ''}>⭐ Premium</option>
                <option value="custom" ${tier === 'custom' ? 'selected' : ''}>🔧 Custom</option>
              </select>
            </td>
            <td>
              <select id="banned-${u.id}" style="padding: 6px; border-radius: 4px;">
                <option value="0" ${!u.is_banned ? 'selected' : ''}>Active</option>
                <option value="1" ${u.is_banned ? 'selected' : ''}>Banned</option>
              </select>
            </td>
            <td>
              <button onclick="updateUser(${u.id})">Save</button>
              <button onclick="resetUsage(${u.id})" style="background: #f57c00; margin-left: 4px;" title="Reset usage count to 0">Reset</button>
            </td>
          </tr>
          `;
        }).join('');
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #c62828;">Error: ' + escapeHtml(e.message) + '</td></tr>';
      }
    }

    function escapeHtml(s) {
      if (s == null) return '';
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    
    async function updateUser(id) {
      const token = localStorage.getItem('token');
      const newLimit = parseInt(document.getElementById('limit-' + id).value);
      const newBanned = parseInt(document.getElementById('banned-' + id).value);
      const newTier = document.getElementById('tier-' + id).value;

      try {
        const params = new URLSearchParams({ usage_limit: newLimit, is_banned: newBanned, tier: newTier });
        const res = await fetch(`/admin/users/${id}?` + params.toString(), {
          method: 'PATCH',
          headers: {'Authorization': 'Bearer ' + token}
        });
        
        if (res.ok) {
          // Flash green to confirm save
          const row = document.getElementById('user-row-' + id);
          if (row) {
            row.style.transition = 'background 0.3s';
            row.style.background = '#e8f5e9';
            setTimeout(() => { row.style.background = ''; }, 800);
          }
          loadUsers();
        } else {
          alert('Failed to update user: ' + res.status);
        }
      } catch (e) {
        alert('Error: ' + e.message);
      }
    }
    
    async function resetUsage(id) {
      if (!confirm("Reset this user's usage count to 0?")) return;
      const token = localStorage.getItem('token');
      try {
        const res = await fetch(`/admin/users/${id}/reset-usage`, {
          method: 'POST',
          headers: {'Authorization': 'Bearer ' + token}
        });
        if (res.ok) {
          loadUsers();
        } else {
          alert('Failed to reset usage');
        }
      } catch (e) {
        alert('Error: ' + e.message);
      }
    }
    
    if (!localStorage.getItem('token')) window.location.replace('/login');
    else if (localStorage.getItem('is_admin') !== 'true') window.location.replace('/scan');

    async function loadModel() {
      try {
        const m = await (await fetch('/model_info')).json();
        document.getElementById('model-version').textContent = m.model_version || '-';
        const r = (m.measured || {}).referable;
        if (r) document.getElementById('model-sub').textContent =
          'Referable DR: sensitivity ' + r.sensitivity + '%, specificity ' + r.specificity + '% (APTOS, external)';
      } catch (e) {}
    }

    loadPlanDefaults();
    loadModel();
    loadScans();
    loadUsers();
  </script>
</body>
</html>"""


ACCOUNT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alati - My Account</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto; background: #f8f7f3; min-height: 100vh; padding: 2rem; }

    .container { max-width: 900px; margin: 0 auto; }
    .header { background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 2rem; color: white; border-radius: 12px; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center; }
    .header-left h1 { font-size: 28px; font-weight: 500; margin-bottom: 4px; }
    .header-left p { font-size: 14px; opacity: 0.95; }
    .header-btns { display: flex; gap: 1rem; }
    .header-btn { padding: 8px 16px; background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.4); border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; text-decoration: none; transition: all 0.3s; }
    .header-btn:hover { background: rgba(255,255,255,0.3); }
    .header-btn.disabled { opacity: 0.4; cursor: not-allowed; pointer-events: none; }

    .cards { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin-bottom: 2rem; }
    .card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .card-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
    .card-value { font-size: 24px; font-weight: 500; color: #2c2c2a; }
    .card-sub { font-size: 13px; color: #666; margin-top: 6px; }

    .status-pill { display: inline-block; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 600; }
    .status-pill.active { background: #e8f5e9; color: #1b5e20; }
    .status-pill.suspended { background: #ffebee; color: #c62828; }

    .ban-notice { display: none; background: #ffebee; border-left: 4px solid #c62828; border-radius: 6px; padding: 1.25rem 1.5rem; margin-bottom: 2rem; }
    .ban-notice.active { display: block; }
    .ban-notice h3 { color: #c62828; font-size: 16px; margin-bottom: 6px; }
    .ban-notice p { color: #5d1818; font-size: 13px; line-height: 1.5; }

    .usage-bar { background: #e0e0e0; border-radius: 4px; height: 8px; margin-top: 10px; overflow: hidden; }
    .usage-bar-fill { height: 100%; background: linear-gradient(90deg, #2e7d32, #66bb6a); transition: width 0.4s; }
    .usage-bar-fill.warn { background: linear-gradient(90deg, #ef6c00, #ffb74d); }
    .usage-bar-fill.over { background: linear-gradient(90deg, #c62828, #ef5350); }

    .section-title { font-size: 16px; font-weight: 600; margin-bottom: 1rem; color: #2c2c2a; }

    .table-container { background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 2rem; }
    .toolbar { padding: 1rem 1.5rem; border-bottom: 1px solid #e0e0e0; }
    .toolbar h3 { font-size: 14px; font-weight: 600; }

    .scroll-container { max-height: 480px; overflow-y: auto; }
    table { width: 100%; border-collapse: collapse; }
    thead { position: sticky; top: 0; background: #f8f7f3; z-index: 1; }
    th { padding: 1rem 1.5rem; text-align: left; font-weight: 600; font-size: 12px; color: #888; text-transform: uppercase; border-bottom: 1px solid #e0e0e0; }
    td { padding: 1rem 1.5rem; border-bottom: 1px solid #f0f0f0; }

    .diagnosis { font-weight: 600; font-size: 13px; padding: 3px 9px; border-radius: 999px; white-space: nowrap; }
    .diagnosis.normal { color: #1b5e20; background: #e8f5e9; }
    .diagnosis.mild { color: #a64b00; background: #fff3e0; }
    .diagnosis.dr { color: #b71c1c; background: #fdecea; }
    .diagnosis.ungradable { color: #5f5e5a; background: #efefec; }
    .muted { color: #b4b2a9; }
    .link-btn { background: none; border: 1px solid #c9d9ec; color: #185fa5; border-radius: 6px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; font-family: inherit; }
    .link-btn:hover { background: #f0f6ff; }
    .link-btn:disabled { opacity: 0.5; cursor: wait; }
    .premium-soon { margin-top: 14px; padding-top: 12px; border-top: 1px dashed #e0e0e0; }
    .premium-soon-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 13px; font-weight: 600; color: #2c2c2a; margin-bottom: 4px; }
    .premium-soon p { font-size: 12px; color: #666; line-height: 1.5; }
    .premium-soon a { color: #185fa5; }
    .soon-badge { font-size: 10px; font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase; color: #0c447c; background: #e6f0fa; border-radius: 999px; padding: 3px 8px; white-space: nowrap; }

    .pagination { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; border-top: 1px solid #e0e0e0; }
    .pagination-info { font-size: 13px; color: #888; }
    .pagination-controls { display: flex; gap: 8px; }
    .pagination-controls button { padding: 6px 12px; background: white; color: #185fa5; border: 1px solid #e0e0e0; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; }
    .pagination-controls button:hover:not(:disabled) { background: #f0f6ff; }
    .pagination-controls button:disabled { opacity: 0.4; cursor: not-allowed; }
    
    /* ========== MOBILE RESPONSIVE ========== */
    html, body { overflow-x: hidden; -webkit-text-size-adjust: 100%; max-width: 100%; }
    
    /* Tables: keep their own horizontal scroll so the PAGE doesn't shift */
    .scroll-container { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .scroll-container table { min-width: 600px; }
    
    @media (max-width: 768px) {
      body { padding: 1rem; }
      .container { max-width: 100%; }
      .header { padding: 1.5rem 1.25rem; flex-direction: column; gap: 1rem; align-items: stretch; border-radius: 10px; }
      .header-left { text-align: center; }
      .header-left h1 { font-size: 22px; }
      .header-left p { font-size: 12px; word-break: break-all; }
      .header-btns { justify-content: center; flex-wrap: wrap; gap: 8px; }
      .header-btn { font-size: 11px; padding: 8px 14px; }
      
      .ban-notice { padding: 1rem; }
      .ban-notice h3 { font-size: 14px; }
      .ban-notice p { font-size: 12px; }
      
      .cards { grid-template-columns: 1fr; gap: 1rem; margin-bottom: 1.5rem; }
      .card { padding: 1.25rem; border-radius: 10px; }
      .card-value { font-size: 22px; }
      .status-pill { font-size: 13px; padding: 5px 12px; }
      
      .section-title { font-size: 15px; }
      .toolbar { padding: 0.75rem 1rem; }
      .toolbar h3 { font-size: 13px; }
      
      th { padding: 0.75rem 1rem; font-size: 11px; }
      td { padding: 0.75rem 1rem; font-size: 13px; }
      
      .pagination { padding: 0.75rem 1rem; flex-wrap: wrap; gap: 8px; }
      .pagination-info { font-size: 11px; }
      .pagination-controls button { padding: 6px 10px; font-size: 11px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-left">
        <h1 id="user-name">My Account</h1>
        <p id="user-email">-</p>
      </div>
      <div class="header-btns">
        <a class="header-btn" id="admin-btn" href="/dashboard" style="display: none;">Admin dashboard</a>
        <a class="header-btn" id="scan-btn" href="/scan">New scan</a>
        <button class="header-btn" onclick="logout()">Sign out</button>
      </div>
    </div>

    <div class="ban-notice" id="ban-notice">
      <h3>Account suspended</h3>
      <p>Your account is currently suspended. You can view your previous scans below, but you can't run new scans until it is reactivated. Please contact <a href="mailto:ahmadalaty@gmail.com">ahmadalaty@gmail.com</a> for assistance.</p>
    </div>

    <div class="cards">
      <div class="card">
        <div class="card-label">Status</div>
        <span class="status-pill" id="status-pill">Loading...</span>
      </div>
      <div class="card">
        <div class="card-label">Usage Today</div>
        <div class="card-value" id="usage-value">-</div>
        <div class="card-sub" id="usage-sub"></div>
        <div class="usage-bar"><div class="usage-bar-fill" id="usage-bar"></div></div>
      </div>
      <div class="card">
        <div class="card-label">Total Scans</div>
        <div class="card-value" id="total-scans">-</div>
        <div class="card-sub">across all time</div>
      </div>
      <div class="card">
        <div class="card-label">Plan</div>
        <div class="card-value" id="plan-value">-</div>
        <div class="card-sub" id="plan-sub"></div>
        <div id="premium-soon" class="premium-soon" style="display:none;">
          <div class="premium-soon-head"><span>Premium</span><span class="soon-badge">Coming soon</span></div>
          <p>Higher daily scan volumes for clinics and screening programmes. Need more scans now? Email <a href="mailto:ahmadalaty@gmail.com">ahmadalaty@gmail.com</a>.</p>
        </div>
      </div>
    </div>

    <h2 class="section-title">My Scan History</h2>
    <div class="table-container">
      <div class="toolbar">
        <h3>Recent Scans</h3>
      </div>
      <div class="scroll-container">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Eye</th>
              <th>Left Diagnosis</th>
              <th>Right Diagnosis</th>
              <th>Report</th>
            </tr>
          </thead>
          <tbody id="my-scans-tbody">
            <tr><td colspan="5" style="text-align: center; color: #888;">Loading...</td></tr>
          </tbody>
        </table>
      </div>
      <div class="pagination">
        <div class="pagination-info" id="my-scans-info">-</div>
        <div class="pagination-controls">
          <button id="my-prev" onclick="changePage(-1)">← Prev</button>
          <button id="my-next" onclick="changePage(1)">Next →</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    let myPage = 1;
    let myTotal = 0;
    let myPageSize = 10;
    let planLimits = { free: 5, premium: 50 };
    const token = localStorage.getItem('token');
    if (!token) window.location.replace('/login');
    if (localStorage.getItem('is_admin') === 'true') {
      document.getElementById('admin-btn').style.display = '';
    }

    function logout() {
      localStorage.removeItem('token');
      localStorage.removeItem('is_admin');
      window.location.href = '/login';
    }

    function escapeHtml(s) {
      if (s == null) return '';
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    // Stored result text (from inference.py) -> short label and colour.
    function describe(dx) {
      const s = dx || '';
      if (!s) return null;
      if (/not gradable/i.test(s)) return { cls: 'ungradable', label: 'Not gradable' };
      if (/not referable/i.test(s)) return { cls: 'mild', label: 'DR, not referable' };
      if (/referable/i.test(s)) return { cls: 'dr', label: 'Referable DR' };
      if (/^normal$/i.test(s.trim())) return { cls: 'normal', label: 'No DR detected' };
      return { cls: 'ungradable', label: s };
    }

    function dxCell(dx, status) {
      if (status === 'failed') return '<span class="diagnosis ungradable">Scan failed</span>';
      const d = describe(dx);
      if (!d) return '<span class="muted">–</span>';
      return '<span class="diagnosis ' + d.cls + '" title="' + escapeHtml(dx) + '">' + escapeHtml(d.label) + '</span>';
    }

    async function loadPlanLimits() {
      try {
        const cfg = await (await fetch('/billing/config')).json();
        if (cfg.free_daily_limit != null) planLimits.free = cfg.free_daily_limit;
        if (cfg.premium_daily_limit != null) planLimits.premium = cfg.premium_daily_limit;
      } catch (e) {}
    }

    async function loadInfo() {
      try {
        const res = await fetch('/user/me', {
          headers: {'Authorization': 'Bearer ' + token}
        });
        if (!res.ok) {
          if (res.status === 401) { logout(); return; }
          throw new Error('Failed to load account info');
        }
        const u = await res.json();

        document.getElementById('user-name').textContent = u.full_name || 'My Account';
        document.getElementById('user-email').textContent = u.email;

        // Plan card. Paid plans are not on sale yet: free accounts see
        // "Premium - coming soon"; premium/custom were granted by the admin.
        const isPremium = (u.tier === 'premium');
        const isCustom = (u.tier === 'custom');
        document.getElementById('plan-value').textContent = isPremium ? 'Premium' : (isCustom ? 'Custom' : 'Free');
        document.getElementById('plan-sub').textContent = isPremium
          ? planLimits.premium + ' scans per day'
          : isCustom
          ? (u.usage_limit === -1 ? 'Unlimited scans' : u.usage_limit + ' scans per day')
          : planLimits.free + ' scans per day';
        document.getElementById('premium-soon').style.display = (isPremium || isCustom) ? 'none' : 'block';

        const pill = document.getElementById('status-pill');
        if (u.is_banned) {
          pill.textContent = 'Suspended';
          pill.className = 'status-pill suspended';
          document.getElementById('ban-notice').classList.add('active');
          document.getElementById('scan-btn').classList.add('disabled');
        } else {
          pill.textContent = u.is_admin ? 'Active · Administrator' : 'Active';
          pill.className = 'status-pill active';
        }

        const used = u.usage_count || 0;
        const limit = u.usage_limit;
        const isUnlimited = (limit === -1 || limit === null);
        document.getElementById('usage-value').textContent = isUnlimited ? String(used) : (used + ' / ' + limit);
        const bar = document.getElementById('usage-bar');
        bar.classList.remove('warn', 'over');
        if (!isUnlimited) {
          const pct = limit > 0 ? Math.min(100, (used / limit) * 100) : 100;
          bar.style.width = pct + '%';
          if (pct >= 100) { bar.classList.add('over'); document.getElementById('usage-sub').textContent = 'Daily limit reached. Resets at 00:00 UTC.'; }
          else {
            if (pct >= 80) bar.classList.add('warn');
            const left = limit - used;
            document.getElementById('usage-sub').textContent = left + (left === 1 ? ' scan' : ' scans') + ' left today';
          }
        } else {
          bar.style.width = '100%';
          document.getElementById('usage-sub').textContent = 'Unlimited scans';
        }
      } catch (e) {
        console.error(e);
      }
    }

    function changePage(delta) {
      const maxPage = Math.max(1, Math.ceil(myTotal / myPageSize));
      myPage = Math.max(1, Math.min(maxPage, myPage + delta));
      loadMyScans();
    }

    async function downloadReport(id, btn) {
      btn.disabled = true;
      try {
        const res = await fetch('/scan/' + id + '/report', { headers: { 'Authorization': 'Bearer ' + token } });
        if (!res.ok) throw new Error('status ' + res.status);
        const url = URL.createObjectURL(await res.blob());
        const a = document.createElement('a');
        a.href = url; a.download = 'alati_report_' + id + '.pdf';
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        btn.textContent = 'PDF';
      } catch (e) {
        btn.textContent = 'Retry';
      } finally {
        btn.disabled = false;
      }
    }

    async function loadMyScans() {
      const tbody = document.getElementById('my-scans-tbody');
      const info = document.getElementById('my-scans-info');

      try {
        const params = new URLSearchParams({ page: myPage, page_size: myPageSize });
        const res = await fetch('/user/scans?' + params.toString(), {
          headers: {'Authorization': 'Bearer ' + token}
        });

        if (!res.ok) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #c62828;">Could not load your scans (' + res.status + ')</td></tr>';
          return;
        }

        const data = await res.json();
        const scans = data.items || [];
        myTotal = data.total || 0;
        document.getElementById('total-scans').textContent = myTotal;

        const start = myTotal === 0 ? 0 : (myPage - 1) * myPageSize + 1;
        const end = Math.min(myTotal, myPage * myPageSize);
        info.textContent = myTotal === 0 ? 'No scans yet' : `Showing ${start}–${end} of ${myTotal}`;

        const maxPage = Math.max(1, Math.ceil(myTotal / myPageSize));
        document.getElementById('my-prev').disabled = (myPage <= 1);
        document.getElementById('my-next').disabled = (myPage >= maxPage);

        if (scans.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #888;">No scans yet. Choose "New scan" above to start.</td></tr>';
          return;
        }

        const eyeName = { left: 'Left', right: 'Right', both: 'Both' };
        tbody.innerHTML = scans.map(s => `
          <tr>
            <td>${new Date(s.created_at).toLocaleString([], {dateStyle: 'medium', timeStyle: 'short'})}</td>
            <td>${escapeHtml(eyeName[s.eye_mode] || s.eye_mode)}</td>
            <td>${dxCell(s.left_diagnosis, s.status)}</td>
            <td>${dxCell(s.right_diagnosis, s.status)}</td>
            <td>${s.status === 'failed' ? '<span class="muted">–</span>' : `<button class="link-btn" onclick="downloadReport(${s.id}, this)">PDF</button>`}</td>
          </tr>
        `).join('');
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #c62828;">Error: ' + escapeHtml(e.message) + '</td></tr>';
      }
    }

    if (token) {
      loadPlanLimits().then(loadInfo);
      loadMyScans();
    }
  </script>
</body>
</html>"""


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _ensure_daily_reset(user: User, db: Session) -> None:
    """
    Lazy daily reset: if usage_reset_date isn't today (UTC), zero usage_count first.
    No background scheduler needed — this is checked on every scan request, so it
    self-heals even after downtime/deploys instead of relying on a timer firing.
    """
    today = datetime.utcnow().date()
    if user.usage_reset_date != today:
        user.usage_count = 0
        user.usage_reset_date = today
        db.add(user)
        db.commit()
        db.refresh(user)


def _daily_limit_for_tier(tier: str) -> int:
    return PREMIUM_DAILY_LIMIT if (tier or "free").strip().lower() == "premium" else FREE_DAILY_LIMIT


STARTUP_MIGRATIONS = [
    "ALTER TABLE users ADD COLUMN is_banned INTEGER DEFAULT 0;",
    "ALTER TABLE users ADD COLUMN usage_limit INTEGER DEFAULT -1;",
    "ALTER TABLE users ADD COLUMN usage_count INTEGER DEFAULT 0;",
    "ALTER TABLE scans ADD COLUMN confirmed_left_diagnosis VARCHAR(255);",
    "ALTER TABLE scans ADD COLUMN confirmed_right_diagnosis VARCHAR(255);",
    "ALTER TABLE scans ADD COLUMN confirmed_at TIMESTAMP WITH TIME ZONE;",
    "ALTER TABLE users ADD COLUMN email_verified INTEGER DEFAULT 0;",
    "ALTER TABLE users ADD COLUMN verification_token VARCHAR(255);",
    "ALTER TABLE users ADD COLUMN usage_reset_date DATE;",
    "ALTER TABLE users ADD COLUMN tier VARCHAR(20) DEFAULT 'free';",
    "ALTER TABLE users ADD COLUMN paddle_customer_id VARCHAR(255);",
    "ALTER TABLE users ADD COLUMN paddle_subscription_id VARCHAR(255);",
    "ALTER TABLE users ADD COLUMN subscription_status VARCHAR(30);",
    "ALTER TABLE users ADD COLUMN full_name VARCHAR(120);",
    "ALTER TABLE users ADD COLUMN reset_token_hash VARCHAR(64);",
    "ALTER TABLE users ADD COLUMN reset_token_expires TIMESTAMP WITH TIME ZONE;",
]


@app.on_event("startup")
def startup():
    """Create tables and add missing columns"""
    Base.metadata.create_all(bind=engine)

    # Each DDL statement gets its own connection/transaction, so a column that
    # already exists (a normal "already migrated" case) can't abort the
    # transaction and cause every later statement on a shared connection to
    # silently fail too.
    for ddl in STARTUP_MIGRATIONS:
        try:
            with engine.connect() as conn:
                conn.execute(text(ddl))
                conn.commit()
        except Exception:
            pass

    # Auto-grandfather: any existing users with NULL email_verified
    # (i.e., they registered before email verification existed) get marked as verified.
    # This prevents existing users from being locked out when SMTP is enabled later.
    try:
        with engine.connect() as conn:
            conn.execute(text("UPDATE users SET email_verified=1 WHERE email_verified IS NULL"))
            conn.commit()
    except Exception as e:
        print(f"Error in grandfather email_verified update: {e}")
    
    db = SessionLocal()
    try:
        email = (settings.OWNER_EMAIL or "").strip().lower()
        password = (settings.OWNER_PASSWORD or "").strip()
        if email and password:
            password = password[:72]
            user = db.query(User).filter(User.email == email).first()
            if not user:
                user = User(email=email, password_hash=hash_password(password))
                db.add(user)
                db.commit()
            else:
                user.password_hash = hash_password(password)
                db.add(user)
                db.commit()
            
            # Always auto-verify admin + give unlimited usage
            try:
                conn = engine.connect()
                conn.execute(text("UPDATE users SET email_verified=1, usage_limit=-1 WHERE email = :email"), {"email": email})
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Could not update admin flags: {e}")
    finally:
        db.close()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/model_info")
def model_info_route():
    """
    Which model is actually serving, and at what thresholds.

    Unauthenticated deliberately. MODEL_VERSION is an environment variable, so
    the running model can change without a commit, and after the 4 September
    deploy failure there was no way to tell from outside whether the service was
    on v6 or had fallen back to v1. /health returning {"ok": true} answered the
    wrong question. Exposes no patient data and no secrets.
    """
    return model_info()


@app.get("/", response_class=HTMLResponse)
async def root():
    return LOGIN_HTML


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return LOGIN_HTML

@app.get("/scan", response_class=HTMLResponse)
async def scan_page():
    return SCAN_HTML

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    return DASHBOARD_HTML


@app.get("/account", response_class=HTMLResponse)
async def account_page():
    return ACCOUNT_HTML


# ---- ECG reader (trial) -------------------------------------------------
# Self-contained static page: it does its own image processing in the browser
# and never calls this API, so it needs no auth, no DB and no new dependency.
# Kept entirely separate from the retinal pipeline.
_ECG_PAGE = os.path.join(os.path.dirname(__file__), "static", "ecg.html")


# Hidden since 20 Nov 2026 at the owner's request: no links point here and the
# route answers 404 unless ECG_ENABLED=1 is set in the environment. The page
# gives direct treatment instructions and is separate from the validated
# retinal model, so it stays off until it is reviewed.
ECG_ENABLED = os.getenv("ECG_ENABLED", "0").strip() == "1"


@app.get("/ecg")
async def ecg_page():
    if not ECG_ENABLED or not os.path.exists(_ECG_PAGE):
        raise HTTPException(status_code=404, detail="Not found")
    # FileResponse sends ETag/Last-Modified but no Cache-Control, which lets the
    # browser apply heuristic freshness and serve a stale copy for hours without
    # ever revalidating - so a deployed fix to this page never reaches the user.
    # "no-cache" means revalidate every time, not "don't store": the ETag still
    # turns the usual case into a cheap 304.
    return FileResponse(
        _ECG_PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


# ============ AUTH ENDPOINTS ============

@app.post("/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user with email verification (if SMTP configured)"""
    email = (body.email or "").strip().lower()
    if not email or len(email) < 3 or "@" not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists. Sign in, or use \"Forgot password?\" to reset it.")
    
    password = (body.password or "").strip()[:72]
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    full_name = " ".join((body.full_name or "").split())[:120] or None
    
    # Create user with default usage limit set DIRECTLY on the ORM model
    # (more reliable than a follow-up UPDATE, which can silently fail)
    user = User(
        email=email,
        password_hash=hash_password(password),
        full_name=full_name,
        usage_limit=FREE_DAILY_LIMIT,
        usage_count=0,
        usage_reset_date=datetime.utcnow().date(),
        tier="free",
        is_banned=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Set email verification fields via SQL (not in ORM model)
    verification_token = secrets.token_urlsafe(32) if email_enabled() else None
    email_verified = 0 if email_enabled() else 1  # Auto-verify if SMTP not configured
    
    try:
        db.execute(
            text("UPDATE users SET email_verified=:ver, verification_token=:tok WHERE id=:id"),
            {"ver": email_verified, "tok": verification_token, "id": user.id}
        )
        db.commit()
    except Exception as e:
        print(f"Could not set verification fields for user {user.id}: {e}")
    
    # If email is enabled, send verification and DON'T return a login token yet
    if email_enabled() and verification_token:
        sent = send_verification_email(email, verification_token)
        if not sent:
            # SMTP failed — clean up the user and tell them
            db.delete(user)
            db.commit()
            raise HTTPException(status_code=500, detail="We could not send the verification email. Please check the address and try again.")
        
        return {
            "status": "verification_sent",
            "message": "Please check your email to verify your account before logging in.",
        }
    
    # Email verification disabled — return token immediately (backward compatible)
    token = create_token({"sub": str(user.id), "email": email})
    return TokenResponse(access_token=token, user_id=user.id, is_admin=False)


@app.get("/auth/verify", response_class=HTMLResponse)
def verify_email(token: str, db: Session = Depends(get_db)):
    """Verify email via link from verification email"""
    if not token:
        return HTMLResponse(_verification_result_page(False, "Missing verification token"))
    
    # Find user by token (using raw SQL since column isn't on ORM model)
    row = db.execute(
        text("SELECT id, email, email_verified FROM users WHERE verification_token = :tok"),
        {"tok": token}
    ).fetchone()
    
    if not row:
        return HTMLResponse(_verification_result_page(False, "Invalid or expired verification link"))
    
    user_id, email, already_verified = row[0], row[1], row[2]
    
    if already_verified:
        return HTMLResponse(_verification_result_page(True, "Your email is already verified. You can sign in now."))
    
    db.execute(
        text("UPDATE users SET email_verified=1, verification_token=NULL WHERE id=:id"),
        {"id": user_id}
    )
    db.commit()
    
    return HTMLResponse(_verification_result_page(True, "Your email is verified. You can now sign in."))


@app.post("/auth/resend-verification")
def resend_verification(body: RegisterRequest, db: Session = Depends(get_db)):
    """Resend verification email to a registered (but unverified) user"""
    if not email_enabled():
        raise HTTPException(status_code=400, detail="Email verification is not enabled")
    
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Don't reveal whether email exists
        return {"status": "ok", "message": "If this email is registered, a verification link has been sent."}
    
    row = db.execute(text("SELECT email_verified FROM users WHERE id=:id"), {"id": user.id}).fetchone()
    if row and row[0]:
        return {"status": "ok", "message": "Account is already verified. You can log in."}
    
    new_token = secrets.token_urlsafe(32)
    db.execute(
        text("UPDATE users SET verification_token=:tok WHERE id=:id"),
        {"tok": new_token, "id": user.id}
    )
    db.commit()
    
    send_verification_email(email, new_token)
    return {"status": "ok", "message": "If this email is registered, a verification link has been sent."}


RESET_TOKEN_TTL = timedelta(hours=1)


@app.post("/auth/forgot-password")
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Email a one-time reset link. The reply never says whether the address exists."""
    generic = {"status": "ok", "message": "If an account exists for that address, a reset link is on its way. "
                                          "It expires in 1 hour. Check your spam folder if it doesn't arrive."}
    if not email_enabled():
        raise HTTPException(status_code=503, detail=f"Password reset by email is not available right now. "
                                                    f"Contact {LEGAL_CONTACT_EMAIL} for help.")
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first() if email else None
    if not user:
        return generic
    token = secrets.token_urlsafe(32)
    user.reset_token_hash = _sha256(token.encode())
    user.reset_token_expires = datetime.now(timezone.utc) + RESET_TOKEN_TTL
    db.add(user)
    db.commit()
    send_password_reset_email(email, token)
    return generic


@app.post("/auth/reset-password")
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    token = (body.token or "").strip()
    user = db.query(User).filter(User.reset_token_hash == _sha256(token.encode())).first() if token else None
    expires = user.reset_token_expires if user else None
    if expires is not None and expires.tzinfo is None:  # SQLite drops the zone
        expires = expires.replace(tzinfo=timezone.utc)
    if not user or not expires or expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="This reset link is invalid or has expired. Request a new one from the sign-in page.")
    password = (body.password or "").strip()[:72]
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    user.password_hash = hash_password(password)
    user.reset_token_hash = None
    user.reset_token_expires = None
    db.add(user)
    db.commit()
    # Opening the emailed link proves the address, so an unverified account is
    # verified here rather than left locked out.
    try:
        db.execute(text("UPDATE users SET email_verified=1, verification_token=NULL WHERE id=:id"), {"id": user.id})
        db.commit()
    except Exception:
        db.rollback()
    return {"status": "ok", "message": "Your password has been changed. You can now sign in."}


RESET_PASSWORD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alati - Choose a new password</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #f8f7f3; padding: 4rem 1rem; color: #2c2c2a; }
  .card { max-width: 420px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
  .header { background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 1.5rem 2rem; color: white; }
  .header h1 { font-size: 22px; font-weight: 500; }
  form { padding: 2rem; }
  label { display: block; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 8px; }
  input { width: 100%; padding: 12px 14px; border: 1.5px solid #e0e0e0; border-radius: 8px; font-size: 16px; font-family: inherit; background: #fafaf8; margin-bottom: 1.25rem; }
  input:focus { outline: none; border-color: #185fa5; background: white; }
  button { width: 100%; padding: 13px; background: linear-gradient(135deg, #185fa5 0%, #0c447c 100%); color: white; border: none; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; font-family: inherit; }
  button:disabled { opacity: 0.6; cursor: wait; }
  .msg { display: none; font-size: 13.5px; line-height: 1.5; padding: 12px 14px; border-radius: 8px; margin-bottom: 1.25rem; }
  .msg.error { display: block; background: #fdecea; color: #8a1c1c; }
  .msg.ok { display: block; background: #e8f5e9; color: #1b5e20; }
  .back { display: block; text-align: center; margin-top: 1.25rem; font-size: 13px; color: #185fa5; text-decoration: none; font-weight: 600; }
</style></head>
<body>
  <div class="card">
    <div class="header"><h1>Choose a new password</h1></div>
    <form id="f" novalidate>
      <div class="msg" id="msg"></div>
      <label for="pw">New password</label>
      <input type="password" id="pw" autocomplete="new-password" placeholder="At least 8 characters">
      <label for="pw2">Repeat new password</label>
      <input type="password" id="pw2" autocomplete="new-password">
      <button type="submit" id="btn">Save new password</button>
      <a class="back" href="/login">Back to sign in</a>
    </form>
  </div>
  <script>
    const token = new URLSearchParams(location.search).get('token') || '';
    const msg = document.getElementById('msg');
    function say(cls, text) { msg.className = 'msg ' + cls; msg.textContent = text; }
    if (!token) say('error', 'This link is incomplete. Request a new reset link from the sign-in page.');
    document.getElementById('f').addEventListener('submit', async (e) => {
      e.preventDefault();
      const pw = document.getElementById('pw').value, pw2 = document.getElementById('pw2').value;
      if (pw.length < 8) return say('error', 'Password must be at least 8 characters.');
      if (pw !== pw2) return say('error', 'The two passwords do not match.');
      const btn = document.getElementById('btn'); btn.disabled = true;
      try {
        const res = await fetch('/auth/reset-password', { method: 'POST', headers: {'Content-Type': 'application/json'},
                                                          body: JSON.stringify({ token, password: pw }) });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          say('ok', data.message || 'Your password has been changed.');
          document.getElementById('pw').value = ''; document.getElementById('pw2').value = '';
          btn.textContent = 'Go to sign in';
          btn.disabled = false;
          btn.onclick = (ev) => { ev.preventDefault(); location.href = '/login'; };
          return;
        }
        say('error', typeof data.detail === 'string' ? data.detail : 'Could not change the password. Please try again.');
      } catch (err) {
        say('error', 'Could not reach the server. Check your connection and try again.');
      }
      btn.disabled = false;
    });
  </script>
</body></html>"""


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page():
    return RESET_PASSWORD_HTML


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#185fa5"/><stop offset="1" stop-color="#0f6e56"/></linearGradient></defs><rect width="64" height="64" rx="14" fill="url(#g)"/><circle cx="32" cy="32" r="17" fill="none" stroke="#fff" stroke-width="5"/><circle cx="32" cy="32" r="6.5" fill="#fff"/></svg>"""


@app.get("/favicon.svg")
async def favicon_svg():
    return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/favicon.ico")
async def favicon_ico():
    return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


def _verification_result_page(success: bool, message: str) -> str:
    icon = "✅" if success else "⚠️"
    color = "#0f6e56" if success else "#c62828"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Alati - Email Verification</title><link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>
  body {{ font-family: system-ui, sans-serif; background: #f8f7f3; padding: 4rem 1rem; text-align: center; }}
  .card {{ max-width: 440px; margin: 0 auto; background: white; border-radius: 12px; padding: 3rem 2rem; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
  .icon {{ font-size: 64px; margin-bottom: 1rem; }}
  h1 {{ color: {color}; font-size: 22px; margin-bottom: 1rem; }}
  p {{ color: #555; font-size: 15px; line-height: 1.6; margin-bottom: 2rem; }}
  a {{ background: linear-gradient(135deg, #185fa5 0%, #0c447c 100%); color: white; padding: 12px 28px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block; }}
</style></head>
<body>
  <div class="card">
    <div class="icon">{icon}</div>
    <h1>{'Verification Successful' if success else 'Verification Failed'}</h1>
    <p>{message}</p>
    <a href="/login">Go to sign in</a>
  </div>
</body></html>"""


LEGAL_LAST_UPDATED = "November 20, 2026"
LEGAL_CONTACT_EMAIL = "ahmadalaty@gmail.com"
# The legal operator named in the Terms. Change it if a company is registered.
LEGAL_OPERATOR = "Ahmad Alalati, a sole proprietor"


def _legal_page_html(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alati - {title}</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #f8f7f3; margin: 0; padding: 2rem 1rem; }}
  .card {{ max-width: 720px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); overflow: hidden; }}
  .header {{ background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 2rem; color: white; }}
  .header h1 {{ margin: 0 0 4px; font-size: 26px; font-weight: 500; }}
  .header p {{ margin: 0; opacity: 0.9; font-size: 13px; }}
  .content {{ padding: 2rem; color: #2c2c2a; line-height: 1.7; font-size: 15px; }}
  .content h2 {{ font-size: 17px; color: #185fa5; margin: 1.75rem 0 0.5rem; }}
  .content h2:first-child {{ margin-top: 0; }}
  .content p, .content li {{ color: #444441; }}
  .content ul {{ padding-left: 1.25rem; }}
  .footer-nav {{ padding: 1.5rem 2rem; border-top: 1px solid #e0e0e0; font-size: 13px; display: flex; gap: 1.5rem; flex-wrap: wrap; }}
  .footer-nav a {{ color: #185fa5; text-decoration: none; font-weight: 600; }}
  .footer-nav a:hover {{ text-decoration: underline; }}
</style></head>
<body>
  <div class="card">
    <div class="header">
      <h1>{title}</h1>
      <p>Alati · AI-assisted diabetic retinopathy screening · Last updated {LEGAL_LAST_UPDATED}</p>
    </div>
    <div class="content">
      {body_html}
    </div>
    <div class="footer-nav">
      <a href="/legal/terms">Terms of Service</a>
      <a href="/legal/privacy">Privacy Notice</a>
      <a href="/pricing">Pricing</a>
      <a href="/login">← Back to Alati</a>
    </div>
  </div>
</body></html>"""


TERMS_OF_SERVICE_HTML = _legal_page_html("Terms of Service", f"""
  <p>These Terms of Service ("Terms") govern your access to and use of Alati (the "Service"), an AI-based screening aid for diabetic retinopathy. By creating an account or using the Service, you agree to these Terms.</p>
  <p>Alati is a <a href="https://nabeeh.health">Nabeeh</a> product operated by {LEGAL_OPERATOR} ("we", "us", "our").</p>

  <h2>1. Eligibility</h2>
  <p>The Service is intended solely for use by licensed medical professionals acting within their own clinical judgment and scope of practice. By registering, you confirm that you are a licensed medical professional and that you are using the Service accordingly.</p>

  <h2>2. Description of the Service</h2>
  <p>Alati is currently in a pre-approval / research-use stage and has not received regulatory clearance (e.g., FDA, CE, or local Ministry of Health approval) in any jurisdiction. The Service is a supportive aid only — it does not diagnose, and its output must not be used as the sole basis for any clinical decision. All findings must be independently verified by the treating physician. See the disclaimer shown at sign-up for full detail; it is incorporated into these Terms by reference.</p>

  <h2>3. Your Account</h2>
  <p>You are responsible for maintaining the confidentiality of your account credentials and for all activity under your account. Notify us promptly of any unauthorized use.</p>

  <h2>4. Plans &amp; Fees</h2>
  <p>Alati is currently provided free of charge, with a limited number of scans per day; see <a href="/pricing">Pricing</a>. A paid Premium plan with higher limits is in preparation. No payment is taken today. If paid plans are introduced, their prices, billing and refund terms will be published on the Pricing page before any charge is made, and you will not be charged without choosing a paid plan.</p>

  <h2>5. Acceptable Use</h2>
  <p>You agree not to misuse the Service, including by attempting to circumvent usage limits, uploading content you do not have the right to submit, or using the Service in a manner inconsistent with the eligibility and clinical-use restrictions above.</p>

  <h2>6. Intellectual Property</h2>
  <p>The Service, including its underlying software and models, is owned by us and our licensors. These Terms do not grant you any rights to our intellectual property beyond the limited right to use the Service as intended.</p>

  <h2>7. Disclaimer of Warranties</h2>
  <p>The Service is provided "as is" and "as available," without warranties of any kind, express or implied, including as to accuracy, reliability, or fitness for a particular purpose — consistent with its current pre-approval / research-use status.</p>

  <h2>8. Limitation of Liability</h2>
  <p>To the maximum extent permitted by law, we shall not be liable for any indirect, incidental, or consequential damages, or for any clinical decisions made using the Service's output. By using the Service, you accept full responsibility for clinical decisions made in your practice, as set out in the sign-up disclaimer.</p>

  <h2>9. Termination</h2>
  <p>We may suspend or terminate your account for violation of these Terms. You may stop using the Service at any time, and ask us to delete your account.</p>

  <h2>10. Changes to These Terms</h2>
  <p>We may update these Terms from time to time. Continued use of the Service after changes take effect constitutes acceptance of the revised Terms.</p>

  <h2>11. Contact Us</h2>
  <p>Questions about these Terms can be sent to <a href="mailto:{LEGAL_CONTACT_EMAIL}">{LEGAL_CONTACT_EMAIL}</a>.</p>
""")


PRIVACY_NOTICE_HTML = _legal_page_html("Privacy Notice", f"""
  <p>This Privacy Notice explains what information Alati collects, how it is used, and the choices you have.</p>

  <h2>1. Information We Collect</h2>
  <ul>
    <li><strong>Account information:</strong> your name, email address and password (stored as a salted hash, never in plain text).</li>
    <li><strong>Scan results:</strong> the AI-generated diagnosis text for each scan, associated with your account.</li>
    <li><strong>Usage data:</strong> scan counts, timestamps, and your plan.</li>
  </ul>
  <p><strong>We do not store your retinal images.</strong> Uploaded images are held in memory only for the seconds it takes to run the AI screening, and are discarded immediately afterward — they are never written to disk or cloud storage, and are never used to train or fine-tune our AI models.</p>

  <h2>2. How We Use Your Information</h2>
  <p>We use this information to provide and improve the Service, enforce usage limits for your plan, send account-related emails (such as email verification and password resets), and provide support.</p>

  <h2>3. Data Storage &amp; Security</h2>
  <p>Account records and scan diagnosis text are stored in an encrypted-at-rest database. As noted above, the retinal images themselves are never persisted. Access to stored data is restricted to what is needed to operate the Service.</p>

  <h2>4. Third-Party Service Providers</h2>
  <p>We share the minimum necessary information with the following processors to operate the Service:</p>
  <ul>
<li><strong>Our email provider</strong> — delivery of account verification and transactional emails.</li>
  </ul>
  <p>We do not sell your personal information.</p>

  <h2>5. Data Retention</h2>
  <p>We retain account and scan data for as long as your account is active, or as needed to comply with legal obligations. You may request deletion of your account and associated data at any time (see below).</p>

  <h2>6. Your Rights</h2>
  <p>You may request access to, correction of, or deletion of your personal data by contacting us at the address below. We will respond within a reasonable time.</p>

  <h2>7. Local Storage</h2>
  <p>The Service uses your browser's local storage to keep you signed in (an authentication token) — no third-party tracking or advertising cookies are used.</p>

  <h2>8. Changes to This Notice</h2>
  <p>We may update this Privacy Notice from time to time; the "Last updated" date above will reflect the most recent revision.</p>

  <h2>9. Contact Us</h2>
  <p>Questions or requests regarding your data can be sent to <a href="mailto:{LEGAL_CONTACT_EMAIL}">{LEGAL_CONTACT_EMAIL}</a>.</p>
""")


REFUND_POLICY_HTML = _legal_page_html("Refund Policy", f"""
  <p><strong>Alati does not currently charge for any plan, so no payments are taken and no refunds arise.</strong> A paid Premium plan is in preparation; the terms below describe how refunds will work once it launches, and will be confirmed on the <a href="/pricing">Pricing</a> page before any charge is made.</p>

  <h2>1. Free Plan</h2>
  <p>The free plan is provided at no cost, with a limited number of scans per day. No payment, no refund applicable.</p>

  <h2>2. Premium Subscription</h2>
  <p>Premium will be billed monthly on a recurring basis, starting on the date you subscribe, until cancelled. The price will be shown on the Pricing page before you subscribe.</p>

  <h2>3. Cancellations</h2>
  <p>You may cancel your Premium subscription at any time from your account page. Cancelling stops future renewals; you'll keep Premium access through the end of the billing period you already paid for.</p>

  <h2>4. Refunds</h2>
  <p>If you're not satisfied, you may request a full refund within 14 days of a Premium charge, provided you have used fewer than 10 scans during the billing period being refunded. Contact us at the email below to request one. Refund requests outside this window, or above the usage threshold, are considered on a case-by-case basis. Refunds are limited to one per customer, and may be refused where there is evidence of abuse. Approved refunds are processed back to your original payment method.</p>

  <h2>5. How to Request a Refund</h2>
  <p>Email <a href="mailto:{LEGAL_CONTACT_EMAIL}">{LEGAL_CONTACT_EMAIL}</a> with your account email and the date of the charge you'd like refunded.</p>

  <h2>6. Contact Us</h2>
  <p>For any billing questions, reach us at <a href="mailto:{LEGAL_CONTACT_EMAIL}">{LEGAL_CONTACT_EMAIL}</a>.</p>
""")


@app.get("/legal/terms", response_class=HTMLResponse)
async def legal_terms():
    return TERMS_OF_SERVICE_HTML


@app.get("/legal/privacy", response_class=HTMLResponse)
async def legal_privacy():
    return PRIVACY_NOTICE_HTML


@app.get("/legal/refund-policy", response_class=HTMLResponse)
async def legal_refund_policy():
    return REFUND_POLICY_HTML


PRICING_HTML = _legal_page_html("Pricing", f"""
  <p>Alati is AI-assisted diabetic retinopathy screening software for licensed medical professionals. Upload a retinal photograph and get a referable / not-referable result in seconds.</p>

  <h2>Free — $0</h2>
  <ul>
    <li>5 scans per day</li>
    <li>Referable diabetic retinopathy result for each photo</li>
    <li>Automatic image-quality check</li>
    <li>PDF report and scan history</li>
  </ul>

  <h2>Premium — coming soon</h2>
  <ul>
    <li>Higher daily scan volume for clinics and screening programmes</li>
    <li>Everything in Free</li>
  </ul>
  <p>Premium is in preparation and is not on sale yet. Its price and terms will be published here before launch. Until then, Alati is free to use and no payment is taken.</p>

  <h2>Clinics and institutions</h2>
  <p>Need a higher scan volume or an evaluation account now? Email <a href="mailto:{LEGAL_CONTACT_EMAIL}">{LEGAL_CONTACT_EMAIL}</a>.</p>

  <p>Use of the Service is governed by our <a href="/legal/terms">Terms of Service</a> and <a href="/legal/privacy">Privacy Notice</a>.</p>
""")


@app.get("/pricing", response_class=HTMLResponse)
async def pricing_page():
    return PRICING_HTML


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Login user — banned users CAN log in but can't scan (they see their dashboard)"""
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()
    
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # NOTE: We do NOT block banned users from logging in.
    # They can log in, see their suspended status on the dashboard,
    # and view their history. The /scan/run endpoint blocks actual scanning.
    
    # Check email verification (if SMTP configured)
    if email_enabled():
        try:
            row = db.execute(text("SELECT email_verified FROM users WHERE id=:id"), {"id": user.id}).fetchone()
            if row and not row[0]:
                raise HTTPException(
                    status_code=403,
                    detail="Please verify your email before logging in. Check your inbox for the verification link."
                )
        except HTTPException:
            raise
        except Exception:
            pass  # If column doesn't exist yet, skip check
    
    owner_email = (settings.OWNER_EMAIL or "").strip().lower()
    is_admin = email == owner_email
    
    token = create_token({"sub": str(user.id), "email": email})
    return TokenResponse(access_token=token, user_id=user.id, is_admin=is_admin)


# ============ BILLING (PADDLE) ============

@app.get("/billing/config")
def billing_config():
    """Plan limits shown on the account and admin pages. Paid plans are not on
    sale (Premium is "coming soon"), so no checkout settings are exposed; the
    webhook below stays in place, dormant, for when billing returns."""
    return {
        "premium_available": False,
        "free_daily_limit": FREE_DAILY_LIMIT,
        "premium_daily_limit": PREMIUM_DAILY_LIMIT,
    }


def _verify_paddle_signature(raw_body: bytes, signature_header: str) -> bool:
    """
    Verify Paddle's Paddle-Signature header: 'ts=<unix_ts>;h1=<hex hmac-sha256>'
    Signed payload is '<ts>:<raw_body>', HMAC'd with the webhook's notification secret.
    """
    if not PADDLE_WEBHOOK_SECRET or not signature_header:
        return False
    try:
        parts = dict(p.split("=", 1) for p in signature_header.split(";") if "=" in p)
        ts, h1 = parts.get("ts"), parts.get("h1")
        if not ts or not h1:
            return False
        signed_payload = f"{ts}:".encode() + raw_body
        expected = hmac.new(PADDLE_WEBHOOK_SECRET.encode(), signed_payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, h1)
    except Exception:
        return False


@app.post("/billing/webhook")
async def paddle_webhook(req: Request, db: Session = Depends(get_db)):
    """
    Handle Paddle subscription lifecycle events and flip the matching user's tier.
    Users are matched via custom_data.user_id, set when checkout is opened
    (see /account's 'Upgrade to Premium' button), with paddle_subscription_id /
    paddle_customer_id as fallbacks for later events.
    """
    raw_body = await req.body()
    signature = req.headers.get("Paddle-Signature", "")

    if not _verify_paddle_signature(raw_body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload = json.loads(raw_body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = payload.get("event_type", "")
    data = payload.get("data", {}) or {}

    if not event_type.startswith("subscription."):
        return {"status": "ignored"}

    custom_data = data.get("custom_data") or {}
    user = None

    raw_user_id = custom_data.get("user_id")
    if raw_user_id:
        try:
            user = db.query(User).filter(User.id == int(raw_user_id)).first()
        except (TypeError, ValueError):
            user = None

    if not user and data.get("id"):
        user = db.query(User).filter(User.paddle_subscription_id == data.get("id")).first()

    if not user and data.get("customer_id"):
        user = db.query(User).filter(User.paddle_customer_id == data.get("customer_id")).first()

    if not user:
        # Ack anyway so Paddle doesn't retry forever on an event we can't map.
        return {"status": "no_matching_user"}

    status = (data.get("status") or "").strip().lower()
    user.paddle_customer_id = data.get("customer_id") or user.paddle_customer_id
    user.paddle_subscription_id = data.get("id") or user.paddle_subscription_id
    user.subscription_status = status

    if status in ("active", "trialing"):
        user.tier = "premium"
        user.usage_limit = PREMIUM_DAILY_LIMIT
    elif status in ("canceled", "paused"):
        user.tier = "free"
        user.usage_limit = FREE_DAILY_LIMIT
    # other statuses (e.g. past_due) keep the current tier — Paddle handles payment retry/dunning

    db.add(user)
    db.commit()

    return {"status": "ok"}


# ============ USER (SELF) ENDPOINTS — for the regular user dashboard ============

@app.get("/user/me")
def get_my_info(req: Request, db: Session = Depends(get_db)):
    """Return the logged-in user's own info (status, usage, email)"""
    user_id = require_user(req, db)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    _ensure_daily_reset(user, db)

    owner_email = (settings.OWNER_EMAIL or "").strip().lower()
    is_admin = user.email.lower() == owner_email

    return {
        "id": user.id,
        "email": user.email,
        "full_name": getattr(user, 'full_name', None),
        "is_banned": getattr(user, 'is_banned', 0),
        "usage_count": getattr(user, 'usage_count', 0),
        "usage_limit": getattr(user, 'usage_limit', -1),
        "tier": getattr(user, 'tier', 'free') or 'free',
        "subscription_status": getattr(user, 'subscription_status', None),
        "has_subscription": bool(getattr(user, 'paddle_subscription_id', None)),
        "is_admin": is_admin,
    }


@app.get("/user/scans")
def get_my_scans(req: Request, page: int = 1, page_size: int = 10, db: Session = Depends(get_db)):
    """Return the logged-in user's own scan history (paginated)"""
    user_id = require_user(req, db)
    
    page = max(1, page)
    page_size = max(1, min(50, page_size))
    offset = (page - 1) * page_size
    
    query = db.query(Scan).filter(Scan.user_id == user_id)
    total = query.count()
    scans = query.order_by(Scan.created_at.desc()).offset(offset).limit(page_size).all()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": s.id,
                "eye_mode": s.eye_mode,
                "left_diagnosis": s.left_diagnosis,
                "right_diagnosis": s.right_diagnosis,
                "status": s.status,
                "created_at": s.created_at,
            }
            for s in scans
        ],
    }


# ============ ADMIN USER ENDPOINTS ============

@app.get("/admin/users")
def list_users(req: Request, q: str = "", page: int = 1, page_size: int = 10, db: Session = Depends(get_db)):
    """[ADMIN ONLY] List users with pagination + search"""
    require_admin(req)
    
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size
    
    query = db.query(User)
    if q:
        like = f"%{q.lower()}%"
        query = query.filter(User.email.ilike(like))
    
    total = query.count()
    users = query.order_by(User.id.desc()).offset(offset).limit(page_size).all()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": u.id,
                "email": u.email,
                "is_banned": getattr(u, 'is_banned', 0),
                "usage_limit": getattr(u, 'usage_limit', -1),
                "usage_count": getattr(u, 'usage_count', 0),
                "tier": getattr(u, 'tier', 'free') or 'free',
                "subscription_status": getattr(u, 'subscription_status', None),
            }
            for u in users
        ],
    }


@app.get("/admin/scans")
def list_scans(req: Request, q: str = "", page: int = 1, page_size: int = 10, db: Session = Depends(get_db)):
    """[ADMIN ONLY] List scans with user info, pagination + search by user email or diagnosis"""
    require_admin(req)
    
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    offset = (page - 1) * page_size
    
    query = db.query(Scan, User).join(User, User.id == Scan.user_id)
    
    if q:
        like = f"%{q.lower()}%"
        query = query.filter(or_(
            User.email.ilike(like),
            Scan.left_diagnosis.ilike(like),
            Scan.right_diagnosis.ilike(like),
        ))
    
    total = query.count()
    rows = query.order_by(Scan.created_at.desc()).offset(offset).limit(page_size).all()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": s.Scan.id,
                "user_id": s.Scan.user_id,
                "user_email": s.User.email,
                "eye_mode": s.Scan.eye_mode,
                "left_diagnosis": s.Scan.left_diagnosis,
                "right_diagnosis": s.Scan.right_diagnosis,
                "status": s.Scan.status,
                "created_at": s.Scan.created_at,
            }
            for s in rows
        ],
    }


@app.patch("/admin/users/{user_id}")
def update_user(user_id: int, is_banned: int = None, usage_limit: int = None, tier: str = None, req: Request = None, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Update user"""
    require_admin(req)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if is_banned is not None:
        user.is_banned = is_banned
    if tier is not None:
        tier = tier.strip().lower()
        if tier not in ("free", "premium", "custom"):
            raise HTTPException(status_code=400, detail="tier must be 'free', 'premium', or 'custom'")
        user.tier = tier
        # Manual admin override — no Paddle subscription behind it.
        user.subscription_status = None
        user.paddle_subscription_id = None
        # 'custom' has no preset default — whatever usage_limit is given (or
        # already set) is left as-is; free/premium snap to their plan default
        # unless a usage_limit was explicitly provided alongside.
        if usage_limit is None and tier != "custom":
            user.usage_limit = PREMIUM_DAILY_LIMIT if tier == "premium" else FREE_DAILY_LIMIT
    if usage_limit is not None:
        user.usage_limit = usage_limit
    db.add(user)
    db.commit()
    return {"status": "updated"}


@app.post("/admin/users/reset-legacy-free")
def reset_legacy_free_tier(req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Normalize old/inconsistent accounts to the current Free tier default.

    Skips the owner/admin account, accounts set to 'custom' (an intentional
    per-account override), and 'premium' accounts backed by a real Paddle
    subscription. Everything else — free-tier accounts with a stale usage_limit
    (e.g. leftover -1/unlimited from before tiers existed) and 'premium'
    accounts with no real subscription behind them — is reset to Free.
    """
    require_admin(req)
    owner_email = (settings.OWNER_EMAIL or "").strip().lower()

    reset_count = 0
    for user in db.query(User).filter(User.email != owner_email).all():
        tier = (user.tier or "free").strip().lower()
        if tier == "custom":
            continue

        has_real_subscription = bool(user.paddle_subscription_id) or (
            (user.subscription_status or "").strip().lower() in ("active", "trialing")
        )
        if tier == "premium" and has_real_subscription:
            continue

        if user.tier != "free" or user.usage_limit != FREE_DAILY_LIMIT:
            user.tier = "free"
            user.usage_limit = FREE_DAILY_LIMIT
            user.subscription_status = None
            user.paddle_subscription_id = None
            db.add(user)
            reset_count += 1

    db.commit()
    return {"status": "ok", "reset_count": reset_count}


@app.post("/admin/users/{user_id}/reset-usage")
def reset_user_usage(user_id: int, req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Reset a user's usage_count to 0"""
    require_admin(req)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.usage_count = 0
    db.add(user)
    db.commit()
    return {"status": "reset", "user_id": user_id, "usage_count": 0}


# ============ TOKEN ENDPOINTS ============

@app.post("/admin/tokens/issue", response_model=IssuedTokenResponse)
def issue_token(body: IssueTokenRequest, req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Issue API token"""
    require_admin(req)
    user = db.query(User).filter(User.id == body.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    plain_token = generate_api_token()
    token_hash = hash_api_token(plain_token)
    api_token = APIToken(user_id=body.user_id, token_hash=token_hash, name=body.name)
    db.add(api_token)
    db.commit()
    db.refresh(api_token)
    
    return IssuedTokenResponse(token=plain_token, token_id=api_token.id, user_id=api_token.user_id, created_at=api_token.created_at)


@app.get("/admin/tokens", response_model=list[APITokenResponse])
def list_tokens(req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] List tokens"""
    require_admin(req)
    return db.query(APIToken).all()


@app.delete("/admin/tokens/{token_id}")
def revoke_token(token_id: int, req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Revoke token"""
    require_admin(req)
    token = db.query(APIToken).filter(APIToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    token.is_active = 0
    db.add(token)
    db.commit()
    return {"status": "revoked"}


# ============ SCAN ENDPOINTS ============

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


async def _read_image_upload(upload: UploadFile, which: str = "") -> bytes:
    """Read an upload and refuse, with a reason, anything that is not a readable image."""
    label = f"The {which} file" if which else "The file"
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, detail=f"{label} is empty. Choose a fundus photograph.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"{label} is larger than 20 MB. Choose a smaller photograph.")
    try:
        Image.open(BytesIO(data)).close()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, detail=f"{label} is not an image Alati can read. Upload a JPG or PNG fundus photograph.")
    # It is an image; now check the whole file decodes. A half-downloaded photo
    # opens fine and then fails during analysis, so decode it fully here and
    # give the user the real reason.
    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()
        with Image.open(BytesIO(data)) as im:
            im.load()
    except Image.DecompressionBombError:
        raise HTTPException(413, detail=f"{label} has too many pixels to analyse. Upload the photo as exported by the fundus camera.")
    except Exception:
        raise HTTPException(400, detail=f"{label} is damaged or incomplete. Download or export it again, then upload it.")
    return data


@app.post("/scan/run", response_model=ScanResult)
async def scan_run(
    req: Request,
    eye_mode: str = Form(...),
    file: UploadFile | None = File(None),
    left_file: UploadFile | None = File(None),
    right_file: UploadFile | None = File(None),
    enhance: str = Form("0"),
    db: Session = Depends(get_db),
):
    """Run a scan"""
    
    # Parse enhance flag (form values come as strings)
    enhance_flag = str(enhance).strip().lower() in ("1", "true", "yes", "on")
    
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    
    user_id = None
    try:
        from jose import jwt
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = int(payload.get("sub", 0))
    except:
        pass
    
    if not user_id:
        api_token = db.query(APIToken).filter(APIToken.token_hash == hash_api_token(token)).first()
        if not api_token or not api_token.is_active:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = api_token.user_id
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again.")
    if getattr(user, 'is_banned', 0):
        raise HTTPException(status_code=403, detail=f"Your account is suspended, so new scans are disabled. Contact {LEGAL_CONTACT_EMAIL} for help.")

    _ensure_daily_reset(user, db)

    usage_limit = getattr(user, 'usage_limit', -1)
    usage_count = getattr(user, 'usage_count', 0)
    if usage_limit >= 0 and usage_count >= usage_limit:
        raise HTTPException(status_code=429, detail=(
            f"You have used all {usage_limit} scans for today. The limit resets at 00:00 UTC. "
            f"Need more? Contact {LEGAL_CONTACT_EMAIL}."))

    eye_mode = (eye_mode or "").strip().lower()
    if eye_mode not in ("left", "right", "both"):
        raise HTTPException(400, detail="Choose the left eye, the right eye, or both eyes.")

    try:
        if eye_mode == "both":
            if not left_file or not right_file:
                raise HTTPException(400, detail="Choose a photograph for each eye, or select a single eye.")

            left_bytes = await _read_image_upload(left_file, "left-eye")
            right_bytes = await _read_image_upload(right_file, "right-eye")

            # Images are only held in memory for inference and are never
            # persisted anywhere — only the resulting diagnosis is saved.
            _l = predict_debug(left_bytes, enhance=enhance_flag)
            _rr = predict_debug(right_bytes, enhance=enhance_flag)
            left_diag = _l.get("display") or _l.get("translated") or "Other"
            right_diag = _rr.get("display") or _rr.get("translated") or "Other"

            scan = Scan(
                user_id=user_id,
                eye_mode="both",
                left_diagnosis=left_diag,
                right_diagnosis=right_diag,
                status="done",
                error=None
            )
            db.add(scan)
            user.usage_count = getattr(user, 'usage_count', 0) + 1
            db.add(user)
            db.commit()
            db.refresh(scan)
            
            return ScanResult(id=scan.id, eye_mode=scan.eye_mode, left_diagnosis=scan.left_diagnosis, right_diagnosis=scan.right_diagnosis, status=scan.status, error=None)
        
        if not file:
            raise HTTPException(400, detail="Choose a fundus photograph first.")

        image_bytes = await _read_image_upload(file)

        # Image is only held in memory for inference and is never persisted
        # anywhere — only the resulting diagnosis is saved.
        _r = predict_debug(image_bytes, enhance=enhance_flag)
        # "display" carries the severity when the model has trained severity
        # heads; it falls back to the plain label otherwise, so a model without
        # grading never writes a grade into the record.
        diag = _r.get("display") or _r.get("translated") or "Other"

        scan = Scan(
            user_id=user_id, eye_mode=eye_mode, status="done",
            left_diagnosis=diag if eye_mode == "left" else None,
            right_diagnosis=diag if eye_mode == "right" else None,
        )
        db.add(scan)
        user.usage_count = getattr(user, 'usage_count', 0) + 1
        db.add(user)
        db.commit()
        db.refresh(scan)
        
        return ScanResult(id=scan.id, eye_mode=scan.eye_mode, left_diagnosis=scan.left_diagnosis, right_diagnosis=scan.right_diagnosis, status=scan.status, error=None)
    
    except HTTPException:
        raise
    except Exception as e:
        scan = Scan(user_id=user_id, eye_mode=eye_mode, status="failed", error=str(e))
        db.add(scan)
        db.commit()
        db.refresh(scan)
        return JSONResponse(status_code=500, content={"id": scan.id, "eye_mode": scan.eye_mode, "status": scan.status, "error": "Scan failed",
                                                      "detail": "The scan could not be completed. Please try again; if it keeps failing, try a different photograph."})


@app.get("/scan/{scan_id}/report")
def get_scan_report(scan_id: int, req: Request, db: Session = Depends(get_db)):
    """Download a PDF report for a scan. Restricted to the scan's owner or the admin."""
    user_id = require_user(req, db)
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if scan.user_id != user_id:
        owner_email = (settings.OWNER_EMAIL or "").strip().lower()
        requester = db.query(User).filter(User.id == user_id).first()
        if not requester or requester.email.strip().lower() != owner_email:
            raise HTTPException(status_code=403, detail="Not your scan")

    if scan.status != "done":
        raise HTTPException(status_code=404, detail="This scan did not complete, so there is no report.")
    owner = db.query(User).filter(User.id == scan.user_id).first()
    eyes = []
    if scan.left_diagnosis:
        eyes.append(("Left eye", scan.left_diagnosis))
    if scan.right_diagnosis:
        eyes.append(("Right eye", scan.right_diagnosis))

    pdf_path = create_pdf_report(
        scan.id, eyes, scan.eye_mode,
        created_at=scan.created_at,
        clinician=(owner.full_name or owner.email) if owner else None,
        clinician_email=owner.email if owner else None,
        model_version=MODEL_VERSION,
    )
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"alati_report_{scan.id}.pdf")


@app.post("/scan/confirm")
def confirm_diagnosis(
    req: Request,
    scan_id: int,
    confirmed_left_diagnosis: str = None,
    confirmed_right_diagnosis: str = None,
    db: Session = Depends(get_db),
):
    """Save professional opinion"""
    
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    
    user_id = None
    try:
        from jose import jwt
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = int(payload.get("sub", 0))
    except:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.user_id == user_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    try:
        if confirmed_left_diagnosis:
            scan.confirmed_left_diagnosis = confirmed_left_diagnosis
        if confirmed_right_diagnosis:
            scan.confirmed_right_diagnosis = confirmed_right_diagnosis
        
        scan.confirmed_at = datetime.utcnow()
        db.add(scan)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save: {str(e)}")
    
    return {"status": "saved", "scan_id": scan.id}