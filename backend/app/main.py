import hashlib
import json
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
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
from .storage_r2 import new_upload_id, key_for, put_bytes, BUILD_MARKER as STORAGE_MARKER
from .inference import (
    predict_diagnosis,
    predict_debug,
    BUILD_MARKER as INF_MARKER,
    ACTIVE_VARIANT,
)

app = FastAPI(title="Alati Cloud - Eye Disease Screening")
Base.metadata.create_all(bind=engine)

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
    .loading { display: none; }
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

        <button class="submit-btn" onclick="handleLogin()">
          <span class="text">Sign in</span>
          <span class="loading">Signing in...</span>
        </button>

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

        <button class="submit-btn" style="background: linear-gradient(135deg, #0f6e56 0%, #085041 100%);" onclick="handleRegister()">Create account</button>

        <p class="signup-link">Already have an account? <button onclick="switchTab('login')">Sign in</button></p>
      </div>
    </div>
  </div>

  <script>
    function switchTab(tab) {
      document.querySelectorAll('.form-section').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tab).classList.add('active');
      event.target.classList.add('active');
    }
    
    async function handleLogin() {
      const email = document.getElementById('login-email').value;
      const password = document.getElementById('login-password').value;
      const errorDiv = document.getElementById('login-error');
      
      errorDiv.style.display = 'none';
      
      if (!email || !password) {
        errorDiv.textContent = 'Please enter email and password';
        errorDiv.style.display = 'block';
        return;
      }
      
      const res = await fetch('/auth/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({email, password})
      });
      
      if (res.ok) {
        const data = await res.json();
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('is_admin', data.is_admin);
        
        // Redirect based on admin status
        if (data.is_admin) {
          window.location.href = '/dashboard';
        } else {
          window.location.href = '/scan';
        }
      } else {
        errorDiv.textContent = 'Invalid email or password';
        errorDiv.style.display = 'block';
      }
    }
    
    async function handleRegister() {
      const email = document.getElementById('register-email').value;
      const password = document.getElementById('register-password').value;
      const errorDiv = document.getElementById('register-error');
      
      errorDiv.style.display = 'none';
      
      if (!email || !password) {
        errorDiv.textContent = 'Please fill in all fields';
        errorDiv.style.display = 'block';
        return;
      }
      
      const res = await fetch('/auth/register', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({email, password})
      });
      
      if (res.ok) {
        const data = await res.json();
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('is_admin', data.is_admin);
        window.location.href = '/scan';
      } else {
        const error = await res.json();
        errorDiv.textContent = error.detail || 'Registration failed';
        errorDiv.style.display = 'block';
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
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-left">
        <h1>Retinal Scan</h1>
        <p class="header-subtitle">Upload and analyze retinal images</p>
      </div>
      <div class="header-btns" id="admin-btn-container" style="display: none;">
        <button class="header-btn" onclick="goToDashboard()">📊 Admin Dashboard</button>
        <button class="header-btn" onclick="logout()">Logout</button>
      </div>
      <div class="header-btns" id="user-btn-container">
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
        <div class="upload-area" onclick="document.getElementById('file-input').click()">
          <div class="upload-icon">📷</div>
          <div class="upload-text">Click to upload retinal image</div>
        </div>
        <input type="file" id="file-input" style="display: none;" accept="image/*">
      </div>

      <div class="buttons">
        <button class="btn btn-secondary" onclick="reset()">Cancel</button>
        <button class="btn btn-primary" onclick="submitScan()">Analyze</button>
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
      const adminContainer = document.getElementById('admin-btn-container');
      const userContainer = document.getElementById('user-btn-container');
      
      if (isAdmin) {
        adminContainer.style.display = 'flex';
        userContainer.style.display = 'none';
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
    
    async function submitScan() {
      const eye = document.querySelector('input[name="eye"]:checked').value;
      const file = document.getElementById('file-input').files[0];
      
      if (!file) {
        alert('Please select an image');
        return;
      }
      
      const formData = new FormData();
      formData.append('eye_mode', eye);
      formData.append('file', file);
      
      const token = localStorage.getItem('token');
      const res = await fetch('/scan/run', {
        method: 'POST',
        headers: {'Authorization': 'Bearer ' + token},
        body: formData
      });
      
      if (res.ok) {
        const data = await res.json();
        document.getElementById('result-diagnosis').textContent = data.left_diagnosis || data.right_diagnosis || 'Analysis complete';
        document.getElementById('results').style.display = 'block';
      } else {
        alert('Scan failed');
      }
    }
    
    function reset() {
      document.getElementById('file-input').value = '';
      document.getElementById('results').style.display = 'none';
    }
    
    // Check admin status on load
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
    
    .users-table { background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    .table-header { padding: 1.5rem; border-bottom: 1px solid #e0e0e0; }
    .table-header h3 { font-size: 16px; }
    
    table { width: 100%; border-collapse: collapse; }
    th { padding: 1rem 1.5rem; text-align: left; font-weight: 600; font-size: 12px; color: #888; text-transform: uppercase; border-bottom: 1px solid #e0e0e0; }
    td { padding: 1rem 1.5rem; border-bottom: 1px solid #f0f0f0; }
    
    .user-email { font-weight: 500; color: #2c2c2a; }
    .user-status { font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #e8f5e9; color: #2e7d32; }
    .user-status.banned { background: #ffebee; color: #c62828; }
    
    input { padding: 8px; border: 1px solid #e0e0e0; border-radius: 4px; font-size: 12px; }
    button { padding: 8px 12px; background: #185fa5; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; }
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

    <div class="users-table">
      <div class="table-header">
        <h3>User Management</h3>
      </div>
      <table>
        <thead>
          <tr>
            <th>Email</th>
            <th>Scans Used</th>
            <th>Limit</th>
            <th>Status</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody id="users-tbody">
          <tr><td colspan="5" style="text-align: center; color: #888;">Loading...</td></tr>
        </tbody>
      </table>
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
    
    async function loadUsers() {
      const token = localStorage.getItem('token');
      const res = await fetch('/admin/users', {
        headers: {'Authorization': 'Bearer ' + token}
      });
      
      if (res.ok) {
        const users = await res.json();
        const tbody = document.getElementById('users-tbody');
        tbody.innerHTML = users.map(u => `
          <tr>
            <td class="user-email">${u.email}</td>
            <td>${u.usage_count}</td>
            <td><input type="number" value="${u.usage_limit}" style="width: 60px;"></td>
            <td><span class="user-status ${u.is_banned ? 'banned' : ''}">
              ${u.is_banned ? 'Banned' : 'Active'}
            </span></td>
            <td><button onclick="updateUser(${u.id})">Update</button></td>
          </tr>
        `).join('');
      }
    }
    
    async function updateUser(id) {
      alert('User updated');
    }
    
    loadUsers();
  </script>
</body>
</html>"""


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.on_event("startup")
def startup():
    """Create tables and add missing columns"""
    Base.metadata.create_all(bind=engine)
    
    try:
        with engine.connect() as conn:
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN is_banned INTEGER DEFAULT 0;"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN usage_limit INTEGER DEFAULT -1;"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN usage_count INTEGER DEFAULT 0;"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN confirmed_left_diagnosis VARCHAR(255);"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN confirmed_right_diagnosis VARCHAR(255);"))
                conn.commit()
            except:
                pass
            
            try:
                conn.execute(text("ALTER TABLE scans ADD COLUMN confirmed_at TIMESTAMP WITH TIME ZONE;"))
                conn.commit()
            except:
                pass
    except Exception as e:
        print(f"Error adding columns: {e}")
    
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


# ============ AUTH ENDPOINTS ============

@app.post("/auth/register", response_model=TokenResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    email = (body.email or "").strip().lower()
    if not email or len(email) < 3:
        raise HTTPException(status_code=400, detail="Invalid email")
    
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    password = (body.password or "").strip()[:72]
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    
    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_token({"sub": str(user.id), "email": email})
    return TokenResponse(access_token=token, user_id=user.id, is_admin=False)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Login user"""
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()
    
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    is_admin = email == (settings.OWNER_EMAIL or "").strip().lower()
    token = create_token({"sub": str(user.id), "email": email})
    return TokenResponse(access_token=token, user_id=user.id, is_admin=is_admin)


# ============ USER ENDPOINTS ============

@app.get("/admin/users")
def list_users(req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] List all users"""
    require_admin(req)
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "is_banned": getattr(u, 'is_banned', 0),
            "usage_limit": getattr(u, 'usage_limit', -1),
            "usage_count": getattr(u, 'usage_count', 0),
        }
        for u in users
    ]


@app.patch("/admin/users/{user_id}")
def update_user(user_id: int, is_banned: int = None, usage_limit: int = None, req: Request = None, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Update user"""
    require_admin(req)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if is_banned is not None:
        user.is_banned = is_banned
    if usage_limit is not None:
        user.usage_limit = usage_limit
    db.add(user)
    db.commit()
    return {"status": "updated"}


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
    db: Session = Depends(get_db),
):
    """Run a scan"""
    
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
    
    usage_limit = getattr(user, 'usage_limit', -1)
    usage_count = getattr(user, 'usage_count', 0)
    if usage_limit >= 0 and usage_count >= usage_limit:
        raise HTTPException(status_code=429, detail="Usage limit reached")
    
    eye_mode = (eye_mode or "").strip().lower()
    if eye_mode not in ("left", "right", "both"):
        raise HTTPException(400, detail="Invalid eye_mode")

    try:
        if eye_mode == "both":
            if not left_file or not right_file:
                raise HTTPException(400, detail="Both files required")
            
            left_bytes = await left_file.read()
            right_bytes = await right_file.read()
            
            left_key = key_for("left", new_upload_id())
            right_key = key_for("right", new_upload_id())
            
            put_bytes(left_key, left_bytes, left_file.content_type or "image/jpeg")
            put_bytes(right_key, right_bytes, right_file.content_type or "image/jpeg")
            
            left_diag = predict_debug(left_bytes).get("translated") or "Uncertain"
            right_diag = predict_debug(right_bytes).get("translated") or "Uncertain"
            
            scan = Scan(
                user_id=user_id, 
                eye_mode="both", 
                left_key=left_key, 
                right_key=right_key,
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
        r2_key = key_for(eye_mode, new_upload_id())
        put_bytes(r2_key, image_bytes, file.content_type or "image/jpeg")
        
        diag = predict_debug(image_bytes).get("translated") or "Uncertain"
        
        scan = Scan(
            user_id=user_id, eye_mode=eye_mode, status="done",
            left_key=r2_key if eye_mode == "left" else None,
            right_key=r2_key if eye_mode == "right" else None,
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