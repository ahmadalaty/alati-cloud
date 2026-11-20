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
from fastapi.staticfiles import StaticFiles
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
    decode_jwt,
)
from .inference import (
    predict_diagnosis,
    predict_debug,
    model_info,
)
from .report import create_pdf_report
from .inference import MODEL_VERSION
from .pages import (LOGIN_HTML, SCAN_HTML, DASHBOARD_HTML, ACCOUNT_HTML, RESET_PASSWORD_HTML,
                    document_page, verification_page, email_html, AI_RELEASE)
from .security import SECURITY_HEADERS, NO_STORE_PREFIXES, rate_limited, client_ip

# The interactive API docs list every route, admin ones included, so they are
# off unless ENABLE_API_DOCS=1 is set.
_API_DOCS = os.getenv("ENABLE_API_DOCS", "0").strip() == "1"
app = FastAPI(title="Alati - Diabetic Retinopathy Screening API",
              docs_url="/docs" if _API_DOCS else None,
              redoc_url="/redoc" if _API_DOCS else None,
              openapi_url="/openapi.json" if _API_DOCS else None)
Base.metadata.create_all(bind=engine)

# Stylesheet and fonts. Only static/assets is served: static/ecg.html stays private.
app.mount("/assets", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static", "assets")), name="assets")


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


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        if name not in response.headers:
            response.headers[name] = value
    if request.url.path.startswith(NO_STORE_PREFIXES):
        response.headers["Cache-Control"] = "no-store"
    return response


def _too_many(detail: str):
    return HTTPException(status_code=429, detail=detail)


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
    html_body = email_html(
        "Reset your Alati password",
        "We received a request to reset the password for this account. The link below works once and expires in 1 hour.",
        "Choose a new password", link,
        "If you didn't ask for this, you can ignore this email. Your password will not change.")
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

    html_body = email_html(
        "Confirm your email address",
        "Thanks for creating an Alati account. Confirm your email address to activate it.",
        "Verify email", verify_link,
        "If you didn't create this account, you can ignore this email.")

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

# Page HTML (design A) lives in pages.py.


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

    # Security clean-up (20 Nov 2026). In April 2026 (commit bb2e18be) the
    # register endpoint gave any account created without a password the
    # literal password "placeholder", which anyone could then sign in with.
    # Such passwords are replaced by an unguessable one at every start; the
    # owner of the account can choose a new password with "Forgot password?".
    db = SessionLocal()
    try:
        fixed = 0
        for u in db.query(User).all():
            if verify_password("placeholder", u.password_hash):
                u.password_hash = hash_password(secrets.token_urlsafe(32))
                db.add(u)
                fixed += 1
        if fixed:
            db.commit()
            print(f"[security] disabled the 'placeholder' password on {fixed} account(s)")
    except Exception as e:
        db.rollback()
        print(f"[security] placeholder-password check failed: {e}")
    finally:
        db.close()

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
    # "release" is the version users see (Alati v1.0); model_version stays the
    # internal build name so deployments can be traced.
    return {"release": AI_RELEASE, **model_info()}


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
def register(body: RegisterRequest, req: Request, db: Session = Depends(get_db)):
    """Register a new user with email verification (if SMTP configured)"""
    if rate_limited("register-ip", client_ip(req), 10, 3600):
        raise _too_many("Too many accounts created from this network. Please try again in an hour.")
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
def resend_verification(body: RegisterRequest, req: Request, db: Session = Depends(get_db)):
    """Resend verification email to a registered (but unverified) user"""
    if not email_enabled():
        raise HTTPException(status_code=400, detail="Email verification is not enabled")

    email = (body.email or "").strip().lower()
    if rate_limited("mail-ip", client_ip(req), 10, 3600) or rate_limited("mail-email", email, 3, 3600):
        raise _too_many("Too many emails requested. Please wait an hour and try again.")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Don't reveal whether email exists
        return {"status": "ok", "message": "If this email is registered, a verification link has been sent."}
    
    row = db.execute(text("SELECT email_verified FROM users WHERE id=:id"), {"id": user.id}).fetchone()
    if row and row[0]:
        # Same reply as for an unknown address, so the endpoint does not reveal accounts.
        return {"status": "ok", "message": "If this email is registered, a verification link has been sent."}
    
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
def forgot_password(body: ForgotPasswordRequest, req: Request, db: Session = Depends(get_db)):
    """Email a one-time reset link. The reply never says whether the address exists."""
    generic = {"status": "ok", "message": "If an account exists for that address, a reset link is on its way. "
                                          "It expires in 1 hour. Check your spam folder if it doesn't arrive."}
    if not email_enabled():
        raise HTTPException(status_code=503, detail=f"Password reset by email is not available right now. "
                                                    f"Contact {LEGAL_CONTACT_EMAIL} for help.")
    email = (body.email or "").strip().lower()
    if rate_limited("mail-ip", client_ip(req), 10, 3600) or rate_limited("mail-email", email, 3, 3600):
        raise _too_many("Too many reset emails requested. Please wait an hour and try again.")
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
def reset_password(body: ResetPasswordRequest, req: Request, db: Session = Depends(get_db)):
    if rate_limited("reset-ip", client_ip(req), 20, 3600):
        raise _too_many("Too many attempts. Please wait an hour and try again.")
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


@app.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page():
    return RESET_PASSWORD_HTML


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="12" fill="#1f4f8a"/><circle cx="32" cy="32" r="16" fill="none" stroke="#fff" stroke-width="6"/><circle cx="32" cy="32" r="6" fill="#fff"/></svg>"""


@app.get("/favicon.svg")
async def favicon_svg():
    return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/favicon.ico")
async def favicon_ico():
    return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


def _verification_result_page(success: bool, message: str) -> str:
    return verification_page(success, message)


LEGAL_LAST_UPDATED = "November 20, 2026"
LEGAL_CONTACT_EMAIL = "ahmadalaty@gmail.com"
# The legal operator named in the Terms. Change it if a company is registered.
LEGAL_OPERATOR = "Ahmad Alalati, a sole proprietor"


def _legal_page_html(title: str, body_html: str) -> str:
    return document_page(title, body_html, LEGAL_LAST_UPDATED)


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
def login(body: LoginRequest, req: Request, db: Session = Depends(get_db)):
    """Login user — banned users CAN log in but can't scan (they see their dashboard)"""
    email = (body.email or "").strip().lower()
    # Password guessing: at most 10 tries per account per 15 minutes and 30 per
    # network per 15 minutes, counted whether or not the password was right.
    if rate_limited("login-email", email, 10, 900) or rate_limited("login-ip", client_ip(req), 30, 900):
        raise _too_many("Too many sign-in attempts. Please wait 15 minutes and try again, or reset your password.")
    user = db.query(User).filter(User.email == email).first()

    # Passwords are stored trimmed and cut at 72 characters (see register), so
    # the attempt is normalised the same way before checking.
    if not user or not verify_password((body.password or "").strip()[:72], user.password_hash):
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
MAX_UPLOAD_PIXELS = 40_000_000          # fundus cameras top out around 24 MP
UPLOAD_FORMATS = ("JPEG", "PNG", "WEBP", "TIFF")   # JPEG also opens multi-picture (MPO) camera files


async def _read_image_upload(upload: UploadFile, which: str = "") -> bytes:
    """Read an upload and refuse, with a reason, anything that is not a readable image."""
    label = f"The {which} file" if which else "The file"
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, detail=f"{label} is empty. Choose a fundus photograph.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"{label} is larger than 20 MB. Choose a smaller photograph.")
    # Only the formats a fundus camera or a clinician would send are decoded at
    # all. Pillow's rarer decoders (PSD, FPX, MSP, GIF...) are where its memory
    # bugs have clustered, so they are never reached.
    try:
        with Image.open(BytesIO(data), formats=UPLOAD_FORMATS) as im:
            w, h = im.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, detail=f"{label} is not an image Alati can read. Upload a JPG or PNG fundus photograph.")
    if w * h > MAX_UPLOAD_PIXELS:
        raise HTTPException(413, detail=f"{label} is {w} x {h} pixels, more than Alati analyses. Upload the photo as exported by the fundus camera.")
    # It is an image; now check the whole file decodes. A half-downloaded photo
    # opens fine and then fails during analysis, so decode it fully here and
    # give the user the real reason.
    try:
        with Image.open(BytesIO(data), formats=UPLOAD_FORMATS) as im:
            im.verify()
        with Image.open(BytesIO(data), formats=UPLOAD_FORMATS) as im:
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
    payload = decode_jwt(token)
    if payload:
        try:
            user_id = int(payload.get("sub", 0))
        except (TypeError, ValueError):
            user_id = None

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

    pdf = create_pdf_report(
        scan.id, eyes, scan.eye_mode,
        created_at=scan.created_at,
        clinician=(owner.full_name or owner.email) if owner else None,
        clinician_email=owner.email if owner else None,
        model_version=AI_RELEASE,
    )
    # Built in memory and never written to disk: the report holds clinical data.
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="alati_report_{scan.id}.pdf"',
                             "Cache-Control": "no-store"})


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
    
    payload = decode_jwt(token)
    try:
        user_id = int((payload or {}).get("sub", 0))
    except (TypeError, ValueError):
        user_id = 0
    if not user_id:
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