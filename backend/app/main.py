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
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text, or_
from datetime import datetime

from .config import settings
from .db import Base, engine, get_db, SessionLocal
from .models import User, Scan, APIToken
from .schemas import (
    LoginRequest, 
    TokenResponse, 
    ScanResult,
    RegisterRequest,
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
    BUILD_MARKER as INF_MARKER,
    ACTIVE_VARIANT,
)
from .report import create_pdf_report

app = FastAPI(title="Alati Cloud - Eye Disease Screening")
Base.metadata.create_all(bind=engine)


# ============ APP CONFIG (env-driven) ============

# Daily scan limits per tier. Free/DEFAULT_USAGE_LIMIT kept as a fallback for existing Render configs.
FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", os.getenv("DEFAULT_USAGE_LIMIT", "5")))
PREMIUM_DAILY_LIMIT = int(os.getenv("PREMIUM_DAILY_LIMIT", "50"))

# Paddle billing config
PADDLE_WEBHOOK_SECRET = os.getenv("PADDLE_WEBHOOK_SECRET", "").strip()
PADDLE_CLIENT_TOKEN = os.getenv("PADDLE_CLIENT_TOKEN", "").strip()
PADDLE_PRICE_ID_PREMIUM = os.getenv("PADDLE_PRICE_ID_PREMIUM", "").strip()
PADDLE_ENV = os.getenv("PADDLE_ENV", "sandbox").strip().lower()  # 'sandbox' or 'production'

# App URL (used in verification email links). Falls back to localhost if not set.
APP_URL = os.getenv("APP_URL", "").rstrip("/") or "https://alati-cloud.onrender.com"

# SMTP config (if not set, email verification is disabled — registration still works)
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", "").strip() or SMTP_USER


def email_enabled() -> bool:
    """Check if SMTP is configured. If not, verification is bypassed."""
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASSWORD)


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
          <p style="margin: 8px 0 0; opacity: 0.95; font-size: 14px;">AI-powered retinal disease detection</p>
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
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alati - Sign In</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto; background: linear-gradient(135deg, #0c447c 0%, #185fa5 50%, #0f6e56 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem; }
    
    .container { background: white; border-radius: 12px; max-width: 440px; width: 100%; overflow: hidden; box-shadow: 0 12px 32px rgba(0,0,0,0.12); }
    .header { background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 2rem; color: white; text-align: center; }
    .header-title { font-size: 36px; font-weight: 500; margin-bottom: 8px; }
    .header-subtitle { font-size: 14px; opacity: 0.95; margin: 0; }
    
    .form-container { padding: 2.5rem; }
    .tabs { display: flex; gap: 0; margin-bottom: 2rem; border-bottom: 2px solid #e0e0e0; }
    .tab-btn { flex: 1; padding: 14px; background: none; border: none; border-bottom: 3px solid transparent; color: #888780; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
    .tab-btn.active { border-bottom-color: #185fa5; color: #185fa5; }
    
    .form-group { margin-bottom: 1.5rem; }
    .form-label { display: block; font-size: 12px; font-weight: 600; color: #2c2c2a; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.3px; }
    input { width: 100%; padding: 12px 14px; border: 1.5px solid #e0e0e0; border-radius: 8px; font-size: 14px; font-family: inherit; transition: all 0.3s; background: #fafaf8; }
    input:focus { outline: none; border-color: #185fa5; background: white; }
    
    .form-options { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; font-size: 13px; }
    .remember-label { display: flex; align-items: center; gap: 6px; cursor: pointer; color: #444441; }
    .forgot-link { background: none; border: none; color: #185fa5; cursor: pointer; font-weight: 500; }
    
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

    /* Legal footer */
    .legal-footer { position: fixed; left: 0; right: 0; bottom: 0; padding: 14px; text-align: center; font-size: 12px; }
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
      body { padding: 1rem; min-height: 100vh; }
      .container { max-width: 100%; border-radius: 10px; }
      .header { padding: 1.5rem 1rem; }
      .header-title { font-size: 26px; }
      .header-subtitle { font-size: 13px; }
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
      <p class="header-subtitle">AI-powered retinal disease detection</p>
    </div>

    <div class="form-container">
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('login')">Sign in</button>
        <button class="tab-btn" onclick="switchTab('register')">Create account</button>
      </div>

      <div id="login" class="form-section active">
        <div id="verify-banner" class="verify-banner">
          <div class="title">📧 Email not verified</div>
          <div class="text" id="verify-text">Please verify your email address before signing in. Check your inbox for the verification link.</div>
          <button onclick="resendVerification()">Resend verification email</button>
        </div>
        
        <div id="success-banner" class="success-banner">
          <div class="title">✅ Account created</div>
          <div class="text" id="success-text">Please check your email and click the verification link to activate your account.</div>
        </div>
        
        <div id="login-error" class="error-msg" style="display: none;"></div>
        
        <div class="form-group">
          <label class="form-label">Email address</label>
          <input type="email" id="login-email" placeholder="doctor@clinic.com">
        </div>

        <div class="form-group">
          <label class="form-label">Password</label>
          <input type="password" id="login-password" placeholder="••••••••">
        </div>

        <div class="form-options">
          <label class="remember-label">
            <input type="checkbox">
            <span>Remember me</span>
          </label>
          <button class="forgot-link">Forgot password?</button>
        </div>

        <button class="submit-btn" onclick="handleLogin()">Sign in</button>

        <p class="signup-link">Don't have an account? <button onclick="switchTab('register')">Create one</button></p>
      </div>

      <div id="register" class="form-section">
        <div id="register-error" class="error-msg" style="display: none;"></div>
        
        <div class="form-group">
          <label class="form-label">Full name</label>
          <input type="text" id="register-name" placeholder="Dr. Ahmad Alalati">
        </div>

        <div class="form-group">
          <label class="form-label">Email address</label>
          <input type="email" id="register-email" placeholder="doctor@clinic.com">
        </div>

        <div class="form-group">
          <label class="form-label">Password</label>
          <input type="password" id="register-password" placeholder="••••••••">
        </div>

        <div class="info-box">
          Password must be at least 8 characters
        </div>

        <div class="disclaimer-box">
          <h4>⚠️ Important — Please Read Before Use</h4>
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

        <button class="submit-btn" style="background: linear-gradient(135deg, #0f6e56 0%, #085041 100%);" onclick="handleRegister()">Create account</button>

        <p class="signup-link">Already have an account? <button onclick="switchTab('login')">Sign in</button></p>
      </div>
    </div>
  </div>

  <div class="legal-footer">
    <a href="/legal/terms">Terms of Service</a><span>·</span>
    <a href="/legal/privacy">Privacy Notice</a><span>·</span>
    <a href="/legal/refund-policy">Refund Policy</a>
  </div>

  <script>
    function switchTab(tab) {
      document.querySelectorAll('.form-section').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tab).classList.add('active');
      event.target.classList.add('active');
    }
    
    let lastEmailForResend = '';
    
    function hideAllBanners() {
      document.getElementById('verify-banner').classList.remove('active');
      document.getElementById('success-banner').classList.remove('active');
      document.getElementById('login-error').style.display = 'none';
    }
    
    async function handleLogin() {
      const email = document.getElementById('login-email').value;
      const password = document.getElementById('login-password').value;
      const errorDiv = document.getElementById('login-error');
      
      hideAllBanners();
      lastEmailForResend = email;
      
      if (!email || !password) {
        errorDiv.textContent = 'Please enter email and password';
        errorDiv.style.display = 'block';
        return;
      }
      
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
          
          if (data.is_admin === true) {
            setTimeout(() => window.location.href = '/dashboard', 500);
          } else {
            setTimeout(() => window.location.href = '/account', 500);
          }
        } else {
          let detail = 'Invalid email or password';
          try {
            const error = await res.json();
            detail = error.detail || detail;
          } catch (e) {}
          
          // Detect specific error states from the server message
          const lowDetail = detail.toLowerCase();
          
          if (res.status === 403 && (lowDetail.includes('verify') || lowDetail.includes('verification'))) {
            document.getElementById('verify-text').textContent = detail;
            document.getElementById('verify-banner').classList.add('active');
          } else {
            errorDiv.textContent = detail;
            errorDiv.style.display = 'block';
          }
        }
      } catch (err) {
        console.error('Login error:', err);
        errorDiv.textContent = 'An error occurred. Please try again.';
        errorDiv.style.display = 'block';
      }
    }
    
    async function handleRegister() {
      const email = document.getElementById('register-email').value;
      const password = document.getElementById('register-password').value;
      const consent = document.getElementById('register-consent').checked;
      const errorDiv = document.getElementById('register-error');

      errorDiv.style.display = 'none';
      hideAllBanners();

      if (!email || !password) {
        errorDiv.textContent = 'Please fill in all fields';
        errorDiv.style.display = 'block';
        return;
      }

      if (!consent) {
        errorDiv.textContent = 'Please read and agree to the disclaimer above before creating an account';
        errorDiv.style.display = 'block';
        return;
      }

      try {
        const res = await fetch('/auth/register', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email, password})
        });
        
        if (res.ok) {
          const data = await res.json();
          
          // Two possible success shapes: verification_sent or immediate login token
          if (data.status === 'verification_sent') {
            // Switch to login tab and show success banner
            switchTab('login');
            document.getElementById('success-text').textContent = data.message || 'Please check your email to verify your account.';
            document.getElementById('success-banner').classList.add('active');
            lastEmailForResend = email;
          } else if (data.access_token) {
            // Direct login (email verification not enabled)
            localStorage.setItem('token', data.access_token);
            localStorage.setItem('is_admin', data.is_admin ? 'true' : 'false');
            window.location.href = '/account';
          }
        } else {
          let detail = 'Registration failed';
          try {
            const error = await res.json();
            detail = error.detail || detail;
          } catch (e) {}
          errorDiv.textContent = detail;
          errorDiv.style.display = 'block';
        }
      } catch (err) {
        console.error('Registration error:', err);
        errorDiv.textContent = 'An error occurred. Please try again.';
        errorDiv.style.display = 'block';
      }
    }
    
    async function resendVerification() {
      const email = lastEmailForResend || document.getElementById('login-email').value;
      if (!email) {
        alert('Please enter your email first');
        return;
      }
      try {
        const res = await fetch('/auth/resend-verification', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email, password: 'placeholder'})
        });
        const data = await res.json();
        alert(data.message || 'Check your email for the verification link.');
      } catch (e) {
        alert('Could not resend verification. Please try again.');
      }
    }
  </script>
</body>
</html>"""

SCAN_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alati - Retinal Scan</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto; background: #f8f7f3; min-height: 100vh; padding: 2rem; }
    
    .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow: hidden; }
    .header { background: linear-gradient(135deg, #185fa5 0%, #0f6e56 100%); padding: 2.5rem; color: white; display: flex; justify-content: space-between; align-items: center; }
    .header-left h1 { font-size: 32px; font-weight: 500; margin-bottom: 4px; }
    .header-subtitle { font-size: 14px; opacity: 0.95; }
    .header-btns { display: flex; gap: 1rem; }
    .header-btn { padding: 8px 16px; background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.4); border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.3s; }
    .header-btn:hover { background: rgba(255,255,255,0.3); }
    
    .content { padding: 2.5rem; }
    .form-group { margin-bottom: 2rem; }
    .form-label { display: block; font-size: 12px; font-weight: 600; color: #2c2c2a; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.3px; }
    
    .eye-select { display: flex; gap: 1rem; }
    .eye-option { flex: 1; }
    .eye-option input[type="radio"] { display: none; }
    .eye-option label { display: block; padding: 1rem; border: 2px solid #e0e0e0; border-radius: 8px; text-align: center; cursor: pointer; transition: all 0.3s; }
    .eye-option input[type="radio"]:checked + label { border-color: #185fa5; background: #f0f6ff; }
    
    .upload-area { border: 2px dashed #e0e0e0; border-radius: 8px; padding: 2rem; text-align: center; cursor: pointer; transition: all 0.3s; }
    .upload-area:hover { border-color: #185fa5; background: #f8fafb; }
    .upload-icon { font-size: 32px; margin-bottom: 1rem; }
    .upload-text { font-size: 14px; color: #666; }
    
    .buttons { display: flex; gap: 1rem; justify-content: space-between; margin-top: 2rem; }
    .btn { padding: 12px 24px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
    .btn-secondary { background: #e0e0e0; color: #333; }
    .btn-primary { background: linear-gradient(135deg, #185fa5 0%, #0c447c 100%); color: white; }
    .btn-primary:hover { transform: translateY(-2px); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    
    .results { background: #f8fafb; border-radius: 8px; padding: 1.5rem; margin-top: 2rem; }
    .result-item { margin-bottom: 1.5rem; }
    .result-label { font-size: 12px; font-weight: 600; color: #888; margin-bottom: 4px; text-transform: uppercase; }
    .result-value { font-size: 18px; font-weight: 500; color: #0f6e56; }
    
    /* Upload states */
    .upload-area.has-file { border-color: #0f6e56; background: #e8f5e9; }
    .upload-area.has-file .upload-icon { color: #0f6e56; }
    .file-name { font-size: 13px; color: #0f6e56; font-weight: 600; margin-top: 6px; word-break: break-all; }
    
    /* Loading spinner */
    .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    
    .processing-overlay { display: none; text-align: center; padding: 2rem; background: #f0f6ff; border-radius: 8px; margin-top: 1.5rem; }
    .processing-overlay.active { display: block; }
    .big-spinner { width: 48px; height: 48px; border: 4px solid #e0e0e0; border-top-color: #185fa5; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 1rem; }
    .processing-text { font-size: 16px; color: #185fa5; font-weight: 600; }
    .processing-sub { font-size: 13px; color: #666; margin-top: 4px; }
    
    .done-banner { display: none; background: #e8f5e9; border-left: 4px solid #2e7d32; padding: 1rem 1.5rem; border-radius: 6px; margin-top: 1.5rem; color: #1b5e20; font-weight: 600; }
    .done-banner.active { display: block; animation: slideIn 0.4s ease; }
    @keyframes slideIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    
    /* Enhancement toggle */
    .enhance-toggle { background: #fef9e6; border: 1px solid #f0c000; border-radius: 8px; padding: 1rem; margin-top: 1rem; display: flex; align-items: center; gap: 12px; }
    .enhance-toggle input[type="checkbox"] { width: 20px; height: 20px; cursor: pointer; accent-color: #185fa5; }
    .enhance-toggle-text { flex: 1; }
    .enhance-toggle-title { font-size: 14px; font-weight: 600; color: #2c2c2a; }
    .enhance-toggle-desc { font-size: 12px; color: #666; margin-top: 2px; }
    
    /* ========== MOBILE RESPONSIVE ========== */
    html, body { overflow-x: hidden; -webkit-text-size-adjust: 100%; max-width: 100%; }
    
    @media (max-width: 768px) {
      body { padding: 1rem; }
      .container { max-width: 100%; border-radius: 10px; }
      .header { padding: 1.5rem 1.25rem; flex-direction: column; gap: 1rem; align-items: stretch; }
      .header-left { text-align: center; }
      .header-left h1 { font-size: 24px; }
      .header-subtitle { font-size: 12px; }
      .header-btns { justify-content: center; flex-wrap: wrap; gap: 8px; }
      .header-btn { font-size: 11px; padding: 8px 14px; }
      .content { padding: 1.5rem 1.25rem; }
      .form-label { font-size: 11px; margin-bottom: 8px; }
      .form-group { margin-bottom: 1.5rem; }
      .eye-select { gap: 8px; }
      .eye-option label { padding: 12px 4px; font-size: 13px; }
      .upload-area { padding: 1.5rem 1rem; }
      .upload-icon { font-size: 28px; }
      .upload-text { font-size: 13px; }
      .file-name { font-size: 12px; }
      .enhance-toggle { padding: 12px; gap: 10px; }
      .enhance-toggle input[type="checkbox"] { width: 22px; height: 22px; flex-shrink: 0; }
      .enhance-toggle-title { font-size: 13px; }
      .enhance-toggle-desc { font-size: 11px; }
      .buttons { gap: 8px; }
      .btn { padding: 12px 18px; font-size: 13px; flex: 1; }
      .processing-overlay { padding: 1.5rem 1rem; }
      .big-spinner { width: 40px; height: 40px; }
      .processing-text { font-size: 14px; }
      .processing-sub { font-size: 12px; }
      .done-banner { padding: 12px; font-size: 13px; }
      .results { padding: 1rem; }
      .result-value { font-size: 16px; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-left">
        <h1>Retinal Scan</h1>
        <p class="header-subtitle">Upload and analyze retinal images</p>
      </div>
      <div class="header-btns">
        <button class="header-btn" id="admin-btn" style="display: none;" onclick="goToDashboard()">📊 Admin Dashboard</button>
        <a class="header-btn" href="/account" style="text-decoration: none;">👤 My Account</a>
        <button class="header-btn" onclick="logout()">Logout</button>
      </div>
    </div>

    <div class="content">
      <div class="form-group">
        <label class="form-label">Which eye?</label>
        <div class="eye-select">
          <div class="eye-option">
            <input type="radio" id="left" name="eye" value="left" checked>
            <label for="left">Left Eye</label>
          </div>
          <div class="eye-option">
            <input type="radio" id="right" name="eye" value="right">
            <label for="right">Right Eye</label>
          </div>
          <div class="eye-option">
            <input type="radio" id="both" name="eye" value="both">
            <label for="both">Both Eyes</label>
          </div>
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">Upload Image</label>
        <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
          <div class="upload-icon" id="upload-icon">📷</div>
          <div class="upload-text" id="upload-text">Click to upload retinal image</div>
          <div class="file-name" id="file-name" style="display: none;"></div>
        </div>
        <input type="file" id="file-input" style="display: none;" accept="image/*">
        
        <div class="enhance-toggle">
          <input type="checkbox" id="enhance-checkbox">
          <div class="enhance-toggle-text">
            <div class="enhance-toggle-title">✨ Enhance image before analysis</div>
            <div class="enhance-toggle-desc"><strong style="color: #c62828;">⚠️ May affect diagnosis accuracy — use only when needed.</strong> The AI was trained on unenhanced images, so enhancement can sometimes hurt results. Recommended only for low-quality, dark, or blurry uploads. Try both with/without to compare.</div>
          </div>
        </div>
      </div>

      <div class="buttons">
        <button class="btn btn-secondary" onclick="reset()">Cancel</button>
        <button class="btn btn-primary" id="analyze-btn" onclick="submitScan()">Analyze</button>
      </div>

      <div class="processing-overlay" id="processing">
        <div class="big-spinner"></div>
        <div class="processing-text">Analyzing retinal image...</div>
        <div class="processing-sub">The AI is examining the image — this usually takes 5-15 seconds</div>
      </div>

      <div class="done-banner" id="done-banner">
        ✅ Analysis complete!
      </div>

      <div id="results" style="display: none;">
        <div class="results">
          <div class="result-item">
            <div class="result-label">Diagnosis</div>
            <div class="result-value" id="result-diagnosis">-</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function checkAdmin() {
      const isAdmin = localStorage.getItem('is_admin') === 'true';
      const adminBtn = document.getElementById('admin-btn');
      if (isAdmin) {
        adminBtn.style.display = 'block';
      }
    }
    
    function goToDashboard() {
      window.location.href = '/dashboard';
    }
    
    function logout() {
      localStorage.removeItem('token');
      localStorage.removeItem('is_admin');
      window.location.href = '/login';
    }
    
    // Show filename when file is selected
    document.getElementById('file-input').addEventListener('change', function(e) {
      const file = e.target.files[0];
      const uploadArea = document.getElementById('upload-area');
      const uploadIcon = document.getElementById('upload-icon');
      const uploadText = document.getElementById('upload-text');
      const fileName = document.getElementById('file-name');
      
      if (file) {
        uploadArea.classList.add('has-file');
        uploadIcon.textContent = '✅';
        uploadText.textContent = 'File ready to analyze';
        fileName.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
        fileName.style.display = 'block';
      } else {
        uploadArea.classList.remove('has-file');
        uploadIcon.textContent = '📷';
        uploadText.textContent = 'Click to upload retinal image';
        fileName.style.display = 'none';
      }
    });
    
    async function submitScan() {
      const eye = document.querySelector('input[name="eye"]:checked').value;
      const file = document.getElementById('file-input').files[0];
      const enhance = document.getElementById('enhance-checkbox').checked;
      
      if (!file) {
        alert('Please select an image first');
        return;
      }
      
      const formData = new FormData();
      formData.append('eye_mode', eye);
      formData.append('file', file);
      formData.append('enhance', enhance ? '1' : '0');
      
      const analyzeBtn = document.getElementById('analyze-btn');
      const processing = document.getElementById('processing');
      const doneBanner = document.getElementById('done-banner');
      const results = document.getElementById('results');
      
      // Show processing state
      analyzeBtn.disabled = true;
      analyzeBtn.innerHTML = '<span class="spinner"></span>Analyzing...';
      processing.classList.add('active');
      doneBanner.classList.remove('active');
      results.style.display = 'none';
      
      const token = localStorage.getItem('token');
      
      try {
        const res = await fetch('/scan/run', {
          method: 'POST',
          headers: {'Authorization': 'Bearer ' + token},
          body: formData
        });
        
        processing.classList.remove('active');
        
        if (res.ok) {
          const data = await res.json();
          document.getElementById('result-diagnosis').textContent = data.left_diagnosis || data.right_diagnosis || 'Analysis complete';
          
          // Show done banner first, then results
          doneBanner.classList.add('active');
          setTimeout(() => {
            results.style.display = 'block';
          }, 300);
        } else {
          alert('Scan failed (' + res.status + '). Please try again.');
        }
      } catch (e) {
        processing.classList.remove('active');
        alert('Network error: ' + e.message);
      } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = 'Analyze';
      }
    }
    
    function reset() {
      document.getElementById('file-input').value = '';
      document.getElementById('results').style.display = 'none';
      document.getElementById('done-banner').classList.remove('active');
      document.getElementById('processing').classList.remove('active');
      document.getElementById('upload-area').classList.remove('has-file');
      document.getElementById('upload-icon').textContent = '📷';
      document.getElementById('upload-text').textContent = 'Click to upload retinal image';
      document.getElementById('file-name').style.display = 'none';
    }
    
    checkAdmin();
  </script>
</body>
</html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alati - Admin Dashboard</title>
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
        <button class="header-btn" onclick="goToScan()">🔍 Use AI Scanner</button>
        <button class="header-btn" onclick="logout()">Logout</button>
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
        <div class="stat-label">AI Accuracy</div>
        <div class="stat-value">90%+</div>
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
        <div class="search-box">
          <input type="text" id="users-search" placeholder="Search by email..." oninput="onUsersSearch()">
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
      document.getElementById('limit-' + id).value = (tier === 'premium') ? planDefaults.premium : planDefaults.free;
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
            : 'Free';

          return `
          <tr id="user-row-${u.id}">
            <td class="user-email">${escapeHtml(u.email)}</td>
            <td style="font-weight: 600; color: ${usageColor};">${used} / ${limitDisplay}</td>
            <td><input type="number" id="limit-${u.id}" value="${limit}" style="width: 70px;" title="Use -1 for unlimited"></td>
            <td>
              <select id="tier-${u.id}" style="padding: 6px; border-radius: 4px;" title="${planBadge}" onchange="onTierChange(${u.id})">
                <option value="free" ${tier !== 'premium' ? 'selected' : ''}>Free</option>
                <option value="premium" ${tier === 'premium' ? 'selected' : ''}>⭐ Premium</option>
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
      if (!confirm('Reset this user\\'s usage count to 0?')) return;
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
    
    loadPlanDefaults();
    loadScans();
    loadUsers();
  </script>
</body>
</html>"""


ACCOUNT_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alati - My Account</title>
  <script src="https://cdn.paddle.com/paddle/v2/paddle.js"></script>
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

    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
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

    .diagnosis { font-weight: 500; }
    .diagnosis.normal { color: #2e7d32; }
    .diagnosis.dr { color: #d32f2f; }
    .status { font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #e8f5e9; color: #2e7d32; }
    .status.failed { background: #ffebee; color: #c62828; }

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
        <h1>My Account</h1>
        <p id="user-email">-</p>
      </div>
      <div class="header-btns">
        <a class="header-btn" id="scan-btn" href="/scan">🔍 New Scan</a>
        <button class="header-btn" onclick="logout()">Logout</button>
      </div>
    </div>

    <div class="ban-notice" id="ban-notice">
      <h3>🚫 Account Suspended</h3>
      <p>Your account is currently suspended. You can view your previous scans below, but you can't run new scans until reactivated. Please contact the administrator at <strong>contact@nabeeh.health</strong> for assistance.</p>
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
        <button class="header-btn" id="upgrade-btn" style="display:none; margin-top: 10px; background: linear-gradient(135deg, #0f6e56 0%, #085041 100%); border-color: transparent; color: white;" onclick="openUpgradeCheckout()">⭐ Upgrade to Premium</button>
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
              <th>Status</th>
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
    let currentUser = null;
    let billingConfig = null;

    function logout() {
      localStorage.removeItem('token');
      localStorage.removeItem('is_admin');
      window.location.href = '/login';
    }

    async function initBilling() {
      try {
        const res = await fetch('/billing/config');
        billingConfig = await res.json();
        if (billingConfig.client_token) {
          if (billingConfig.environment === 'sandbox') {
            Paddle.Environment.set('sandbox');
          }
          Paddle.Setup({
            token: billingConfig.client_token,
            eventCallback: function (event) {
              // The actual tier flip happens server-side when /billing/webhook
              // receives the subscription event from Paddle — this just refreshes
              // the UI a couple seconds after the customer finishes paying.
              if (event.name === 'checkout.completed') {
                setTimeout(() => { loadInfo(); }, 3000);
              }
            },
          });
        }
      } catch (e) {
        console.error('Could not load billing config', e);
      }
    }

    function openUpgradeCheckout() {
      if (!billingConfig || !billingConfig.price_id || !billingConfig.client_token) {
        alert('Upgrades are not available yet — billing is not configured.');
        return;
      }
      if (!currentUser) return;
      Paddle.Checkout.open({
        items: [{ priceId: billingConfig.price_id, quantity: 1 }],
        customer: { email: currentUser.email },
        customData: { user_id: String(currentUser.id) },
        settings: { theme: 'light' },
      });
    }

    function escapeHtml(s) {
      if (s == null) return '';
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    async function loadInfo() {
      const token = localStorage.getItem('token');
      if (!token) { window.location.href = '/login'; return; }

      try {
        const res = await fetch('/user/me', {
          headers: {'Authorization': 'Bearer ' + token}
        });
        if (!res.ok) {
          if (res.status === 401) { window.location.href = '/login'; return; }
          throw new Error('Failed to load account info');
        }
        const u = await res.json();
        currentUser = u;

        document.getElementById('user-email').textContent = u.email;

        // Plan card
        const isPremium = (u.tier === 'premium');
        document.getElementById('plan-value').textContent = isPremium ? '⭐ Premium' : 'Free';
        document.getElementById('plan-sub').textContent = isPremium
          ? `${billingConfig ? billingConfig.premium_daily_limit : 50} scans/day`
          : `${billingConfig ? billingConfig.free_daily_limit : 5} scans/day`;
        document.getElementById('upgrade-btn').style.display = isPremium ? 'none' : 'inline-block';

        // Status pill
        const pill = document.getElementById('status-pill');
        if (u.is_banned) {
          pill.textContent = 'Suspended';
          pill.className = 'status-pill suspended';
          // Show ban notice + disable scan button
          document.getElementById('ban-notice').classList.add('active');
          document.getElementById('scan-btn').classList.add('disabled');
        } else {
          pill.textContent = 'Active';
          pill.className = 'status-pill active';
        }

        // Usage display
        const used = u.usage_count || 0;
        const limit = u.usage_limit;
        const isUnlimited = (limit === -1 || limit === null);
        const limitDisplay = isUnlimited ? '∞' : limit;
        document.getElementById('usage-value').textContent = `${used} / ${limitDisplay}`;

        if (!isUnlimited) {
          const pct = Math.min(100, (used / limit) * 100);
          const bar = document.getElementById('usage-bar');
          bar.style.width = pct + '%';
          if (pct >= 100) { bar.classList.add('over'); document.getElementById('usage-sub').textContent = 'Limit reached'; }
          else if (pct >= 80) { bar.classList.add('warn'); document.getElementById('usage-sub').textContent = `${limit - used} scans remaining`; }
          else { document.getElementById('usage-sub').textContent = `${limit - used} scans remaining`; }
        } else {
          document.getElementById('usage-bar').style.width = '100%';
          document.getElementById('usage-sub').textContent = 'Unlimited access';
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

    async function loadMyScans() {
      const token = localStorage.getItem('token');
      const tbody = document.getElementById('my-scans-tbody');
      const info = document.getElementById('my-scans-info');

      try {
        const params = new URLSearchParams({ page: myPage, page_size: myPageSize });
        const res = await fetch('/user/scans?' + params.toString(), {
          headers: {'Authorization': 'Bearer ' + token}
        });

        if (!res.ok) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #c62828;">Failed to load (' + res.status + ')</td></tr>';
          return;
        }

        const data = await res.json();
        const scans = data.items || [];
        myTotal = data.total || 0;
        document.getElementById('total-scans').textContent = myTotal;

        const start = myTotal === 0 ? 0 : (myPage - 1) * myPageSize + 1;
        const end = Math.min(myTotal, myPage * myPageSize);
        info.textContent = myTotal === 0 ? 'No scans yet' : `Showing ${start}-${end} of ${myTotal}`;

        const maxPage = Math.max(1, Math.ceil(myTotal / myPageSize));
        document.getElementById('my-prev').disabled = (myPage <= 1);
        document.getElementById('my-next').disabled = (myPage >= maxPage);

        if (scans.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #888;">No scans yet. Click "New Scan" above to start.</td></tr>';
          return;
        }

        tbody.innerHTML = scans.map(s => `
          <tr>
            <td>${new Date(s.created_at).toLocaleString()}</td>
            <td>${escapeHtml(s.eye_mode)}</td>
            <td><span class="diagnosis ${s.left_diagnosis && s.left_diagnosis.toLowerCase().includes('normal') ? 'normal' : 'dr'}">${escapeHtml(s.left_diagnosis || '-')}</span></td>
            <td><span class="diagnosis ${s.right_diagnosis && s.right_diagnosis.toLowerCase().includes('normal') ? 'normal' : 'dr'}">${escapeHtml(s.right_diagnosis || '-')}</span></td>
            <td><span class="status ${s.status === 'failed' ? 'failed' : ''}">${escapeHtml(s.status)}</span></td>
          </tr>
        `).join('');
      } catch (e) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #c62828;">Error: ' + escapeHtml(e.message) + '</td></tr>';
      }
    }

    initBilling().then(loadInfo);
    loadMyScans();
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


# ============ AUTH ENDPOINTS ============

@app.post("/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user with email verification (if SMTP configured)"""
    email = (body.email or "").strip().lower()
    if not email or len(email) < 3 or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    password = (body.password or "").strip()[:72]
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    
    # Create user with default usage limit set DIRECTLY on the ORM model
    # (more reliable than a follow-up UPDATE, which can silently fail)
    user = User(
        email=email,
        password_hash=hash_password(password),
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
            raise HTTPException(status_code=500, detail="Could not send verification email. Please try again.")
        
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
        return HTMLResponse(_verification_result_page(True, "Your email is already verified. You can log in now."))
    
    db.execute(
        text("UPDATE users SET email_verified=1, verification_token=NULL WHERE id=:id"),
        {"id": user_id}
    )
    db.commit()
    
    return HTMLResponse(_verification_result_page(True, "Email verified! You can now log in."))


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


def _verification_result_page(success: bool, message: str) -> str:
    icon = "✅" if success else "⚠️"
    color = "#0f6e56" if success else "#c62828"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Email Verification</title>
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
    <a href="/login">Go to Login</a>
  </div>
</body></html>"""


LEGAL_LAST_UPDATED = "August 19, 2026"
LEGAL_CONTACT_EMAIL = "ahmadalaty@gmail.com"


def _legal_page_html(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alati - {title}</title>
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
      <p>Alati · AI-powered retinal disease detection · Last updated {LEGAL_LAST_UPDATED}</p>
    </div>
    <div class="content">
      {body_html}
    </div>
    <div class="footer-nav">
      <a href="/legal/terms">Terms of Service</a>
      <a href="/legal/privacy">Privacy Notice</a>
      <a href="/legal/refund-policy">Refund Policy</a>
      <a href="/login">← Back to Alati</a>
    </div>
  </div>
</body></html>"""


TERMS_OF_SERVICE_HTML = _legal_page_html("Terms of Service", f"""
  <p>These Terms of Service ("Terms") govern your access to and use of Alati (the "Service"), an AI-based screening aid for diabetic retinopathy. By creating an account or using the Service, you agree to these Terms.</p>

  <h2>1. Eligibility</h2>
  <p>The Service is intended solely for use by licensed medical professionals acting within their own clinical judgment and scope of practice. By registering, you confirm that you are a licensed medical professional and that you are using the Service accordingly.</p>

  <h2>2. Description of the Service</h2>
  <p>Alati is currently in a pre-approval / research-use stage and has not received regulatory clearance (e.g., FDA, CE, or local Ministry of Health approval) in any jurisdiction. The Service is a supportive aid only — it does not diagnose, and its output must not be used as the sole basis for any clinical decision. All findings must be independently verified by the treating physician. See the disclaimer shown at sign-up for full detail; it is incorporated into these Terms by reference.</p>

  <h2>3. Your Account</h2>
  <p>You are responsible for maintaining the confidentiality of your account credentials and for all activity under your account. Notify us promptly of any unauthorized use.</p>

  <h2>4. Subscription Plans &amp; Billing</h2>
  <p>Alati offers a free tier with a limited number of scans per day, and a paid Premium subscription (currently $29.99/month) offering a higher daily scan limit. Premium subscriptions are billed on a recurring monthly basis through our payment processor, Paddle, until cancelled. See our <a href="/legal/refund-policy">Refund Policy</a> for cancellation and refund terms.</p>

  <h2>5. Acceptable Use</h2>
  <p>You agree not to misuse the Service, including by attempting to circumvent usage limits, uploading content you do not have the right to submit, or using the Service in a manner inconsistent with the eligibility and clinical-use restrictions above.</p>

  <h2>6. Intellectual Property</h2>
  <p>The Service, including its underlying software and models, is owned by Alati and its licensors. These Terms do not grant you any rights to our intellectual property beyond the limited right to use the Service as intended.</p>

  <h2>7. Disclaimer of Warranties</h2>
  <p>The Service is provided "as is" and "as available," without warranties of any kind, express or implied, including as to accuracy, reliability, or fitness for a particular purpose — consistent with its current pre-approval / research-use status.</p>

  <h2>8. Limitation of Liability</h2>
  <p>To the maximum extent permitted by law, Alati shall not be liable for any indirect, incidental, or consequential damages, or for any clinical decisions made using the Service's output. By using the Service, you accept full responsibility for clinical decisions made in your practice, as set out in the sign-up disclaimer.</p>

  <h2>9. Termination</h2>
  <p>We may suspend or terminate your account for violation of these Terms. You may stop using the Service and cancel your subscription at any time.</p>

  <h2>10. Changes to These Terms</h2>
  <p>We may update these Terms from time to time. Continued use of the Service after changes take effect constitutes acceptance of the revised Terms.</p>

  <h2>11. Contact Us</h2>
  <p>Questions about these Terms can be sent to <a href="mailto:{LEGAL_CONTACT_EMAIL}">{LEGAL_CONTACT_EMAIL}</a>.</p>
""")


PRIVACY_NOTICE_HTML = _legal_page_html("Privacy Notice", f"""
  <p>This Privacy Notice explains what information Alati collects, how it is used, and the choices you have.</p>

  <h2>1. Information We Collect</h2>
  <ul>
    <li><strong>Account information:</strong> email address and password (stored as a salted hash, never in plain text).</li>
    <li><strong>Scan results:</strong> the AI-generated diagnosis text for each scan, associated with your account.</li>
    <li><strong>Usage data:</strong> scan counts, timestamps, and subscription/billing status.</li>
  </ul>
  <p><strong>We do not store your retinal images.</strong> Uploaded images are held in memory only for the seconds it takes to run the AI screening, and are discarded immediately afterward — they are never written to disk or cloud storage, and are never used to train or fine-tune our AI models.</p>

  <h2>2. How We Use Your Information</h2>
  <p>We use this information to provide and improve the Service, enforce usage limits for your plan, process subscription billing, send account-related emails (such as email verification), and provide support.</p>

  <h2>3. Data Storage &amp; Security</h2>
  <p>Account records and scan diagnosis text are stored in an encrypted-at-rest database. As noted above, the retinal images themselves are never persisted. Access to stored data is restricted to what is needed to operate the Service.</p>

  <h2>4. Third-Party Service Providers</h2>
  <p>We share the minimum necessary information with the following processors to operate the Service:</p>
  <ul>
    <li><strong>Paddle</strong> — payment processing and subscription billing for Premium accounts (Paddle acts as merchant of record for these transactions).</li>
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
  <p>This Refund Policy applies to paid Premium subscriptions to Alati, billed via Paddle.</p>

  <h2>1. Free Plan</h2>
  <p>The free plan is provided at no cost, with a limited number of scans per day. No payment, no refund applicable.</p>

  <h2>2. Premium Subscription</h2>
  <p>The Premium plan is billed at $29.99 per month on a recurring basis, starting on the date you subscribe, until cancelled.</p>

  <h2>3. Cancellations</h2>
  <p>You may cancel your Premium subscription at any time from your account page. Cancelling stops future renewals; you'll keep Premium access through the end of the billing period you already paid for.</p>

  <h2>4. Refunds</h2>
  <p>If you're not satisfied, you may request a full refund within 7 days of a Premium charge, provided you have used fewer than 10 scans during the billing period being refunded. Contact us at the email below to request one. Refund requests outside this window, or above the usage threshold, are considered on a case-by-case basis. Approved refunds are processed back to your original payment method via Paddle.</p>

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
    """Public config the frontend needs to open Paddle Checkout."""
    return {
        "client_token": PADDLE_CLIENT_TOKEN,
        "price_id": PADDLE_PRICE_ID_PREMIUM,
        "environment": PADDLE_ENV,
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
        "is_banned": getattr(user, 'is_banned', 0),
        "usage_count": getattr(user, 'usage_count', 0),
        "usage_limit": getattr(user, 'usage_limit', -1),
        "tier": getattr(user, 'tier', 'free') or 'free',
        "subscription_status": getattr(user, 'subscription_status', None),
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
        if tier not in ("free", "premium"):
            raise HTTPException(status_code=400, detail="tier must be 'free' or 'premium'")
        user.tier = tier
        # Manual admin override — no Paddle subscription behind it.
        user.subscription_status = None
        user.paddle_subscription_id = None
        if usage_limit is None:
            user.usage_limit = PREMIUM_DAILY_LIMIT if tier == "premium" else FREE_DAILY_LIMIT
    if usage_limit is not None:
        user.usage_limit = usage_limit
    db.add(user)
    db.commit()
    return {"status": "updated"}


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
    if not user or getattr(user, 'is_banned', 0):
        raise HTTPException(status_code=403, detail="User banned")

    _ensure_daily_reset(user, db)

    usage_limit = getattr(user, 'usage_limit', -1)
    usage_count = getattr(user, 'usage_count', 0)
    if usage_limit >= 0 and usage_count >= usage_limit:
        raise HTTPException(status_code=429, detail="Daily usage limit reached. Upgrade to Premium for more scans." if (user.tier or "free") == "free" else "Daily usage limit reached.")
    
    eye_mode = (eye_mode or "").strip().lower()
    if eye_mode not in ("left", "right", "both"):
        raise HTTPException(400, detail="Invalid eye_mode")

    try:
        if eye_mode == "both":
            if not left_file or not right_file:
                raise HTTPException(400, detail="Both files required")
            
            left_bytes = await left_file.read()
            right_bytes = await right_file.read()

            # Images are only held in memory for inference and are never
            # persisted anywhere — only the resulting diagnosis is saved.
            left_diag = predict_debug(left_bytes, enhance=enhance_flag).get("translated") or "Uncertain"
            right_diag = predict_debug(right_bytes, enhance=enhance_flag).get("translated") or "Uncertain"

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
            raise HTTPException(400, detail="File required")
        
        image_bytes = await file.read()

        # Image is only held in memory for inference and is never persisted
        # anywhere — only the resulting diagnosis is saved.
        diag = predict_debug(image_bytes, enhance=enhance_flag).get("translated") or "Uncertain"

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
        return JSONResponse(status_code=500, content={"id": scan.id, "eye_mode": scan.eye_mode, "status": scan.status, "error": "Scan failed"})


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

    parts = []
    if scan.left_diagnosis:
        parts.append(f"Left: {scan.left_diagnosis}")
    if scan.right_diagnosis:
        parts.append(f"Right: {scan.right_diagnosis}")
    diagnosis_full = "; ".join(parts) or "No diagnosis recorded"

    pdf_path = create_pdf_report(scan.id, diagnosis_full, scan.eye_mode)
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