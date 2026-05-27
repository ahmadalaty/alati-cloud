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


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@app.on_event("startup")
def startup():
    """Create tables and add missing columns"""
    Base.metadata.create_all(bind=engine)
    
    try:
        with engine.connect() as conn:
            # Add missing user columns
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
            
            # Add confirmed diagnosis columns
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
    
    # Create admin user
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


@app.get("/")
async def ui():
    return HTMLResponse(get_html_ui())


def get_html_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Alati Cloud - Eye Disease Screening</title>
  <style>
    *{box-sizing:border-box;}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#0b1020;color:#e8ecff;}
    .wrap{max-width:1200px;margin:0 auto;padding:28px 16px 48px;}
    .card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:16px;margin:14px 0;}
    h1{font-size:28px;margin:0 0 6px;}
    h2{font-size:20px;margin:12px 0 8px;}
    h3{font-size:16px;margin:10px 0 8px;}
    label{display:block;font-size:13px;opacity:.85;margin:10px 0 6px;}
    input,select,textarea,button{padding:12px;border-radius:12px;border:1px solid rgba(255,255,255,.14);background:rgba(0,0,0,.25);color:#e8ecff;font-size:15px;font-family:inherit;}
    input,select,textarea{width:100%;}
    button{cursor:pointer;background:#355dff;border:0;font-weight:800;width:auto;padding:10px 20px;border-radius:8px;}
    button:hover{background:#2851e8;}
    button.danger{background:#ff6b6b;}
    button.secondary{background:#666;}
    button.success{background:#2ecc71;}
    button:disabled{opacity:.6;}
    .muted{opacity:.8;font-size:13px;}
    .ok{color:#67ffb1;font-weight:800;}
    .bad{color:#ff8a8a;}
    .result{padding:16px;border-radius:12px;background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.12);margin-top:10px;}
    .big{font-size:20px;font-weight:900;}
    .table{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px;}
    .table th{background:rgba(255,255,255,.1);padding:8px;text-align:left;border-bottom:1px solid rgba(255,255,255,.12);}
    .table td{padding:8px;border-bottom:1px solid rgba(255,255,255,.08);}
    .badge{display:inline-block;padding:3px 6px;border-radius:4px;font-size:11px;background:rgba(51,93,255,.3);}
    .badge.confirmed{background:rgba(46,204,113,.3);color:#2ecc71;}
    .badge.pending{background:rgba(255,193,7,.3);color:#ffc107;}
    .badge.banned{background:rgba(255,107,107,.3);color:#ff6b6b;}
    .flex-between{display:flex;justify-content:space-between;align-items:center;gap:12px;}
    .user-row{display:flex;gap:12px;align-items:center;padding:12px;background:rgba(0,0,0,.2);border-radius:8px;margin-bottom:12px;}
    .user-row>div{flex:1;}
    .user-controls{display:flex;gap:8px;align-items:center;}
    .user-controls input,.user-controls select{width:auto;flex:1;max-width:120px;}
    @media(max-width:768px){.user-row{flex-direction:column;}}
  </style>
</head>
<body>
<div class="wrap">
  <h1>🔬 Alati Cloud - Eye Disease Screening</h1>

  <!-- LOGIN -->
  <div class="card" id="authCard">
    <h3>Authentication</h3>
    <div id="loginTab">
      <label>Email</label>
      <input id="email" placeholder="doctor@clinic.com"/>
      <label>Password</label>
      <input id="password" type="password"/>
      <div style="height:10px"></div>
      <button onclick="doLogin()">Login</button>
      <button onclick="showRegister()" class="secondary" style="margin-left:8px;">Register</button>
    </div>
    <div id="registerTab" style="display:none;">
      <label>Email</label>
      <input id="regEmail" placeholder="doctor@clinic.com"/>
      <label>Password</label>
      <input id="regPassword" type="password"/>
      <div style="height:10px"></div>
      <button onclick="doRegister()">Register</button>
      <button onclick="showLogin()" class="secondary" style="margin-left:8px;">Back</button>
    </div>
    <p id="authStatus" class="muted"></p>
  </div>

  <!-- ADMIN DASHBOARD -->
  <div class="card" id="adminCard" style="display:none;border-color:#ffc107;border-width:2px;">
    <div class="flex-between">
      <h2>👑 Owner Dashboard</h2>
      <div>
        <button onclick="switchToTest()" class="success">🧪 Test AI</button>
        <button onclick="showSection('results')">📊 Results</button>
        <button onclick="showSection('users')">👥 Users</button>
        <button onclick="showSection('tokens')">🔑 Tokens</button>
        <button onclick="doLogout()" class="secondary">Logout</button>
      </div>
    </div>

    <!-- RESULTS SECTION -->
    <div id="resultsSection" style="display:none;margin-top:20px;">
      <h3>📊 Beta Test Results</h3>
      <p class="muted">All professional opinions from ophthalmologists</p>
      <div style="margin:12px 0;">
        <button onclick="loadResults()">Refresh Results</button>
        <button onclick="exportResults()" class="secondary">📥 Export as JSON</button>
      </div>
      <table class="table" id="resultsTable">
        <thead>
          <tr>
            <th>ID</th>
            <th>Doctor</th>
            <th>AI Diagnosis</th>
            <th>Professional Opinion</th>
            <th>Match</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody id="resultsList"></tbody>
      </table>
      <div id="analyticsBox" style="margin-top:20px;padding:16px;background:rgba(51,93,255,.1);border-radius:12px;display:none;">
        <h4>Sensitivity & Specificity</h4>
        <div id="analyticsContent"></div>
      </div>
    </div>

    <!-- USERS SECTION - FIXED WITH MANAGEMENT CONTROLS -->
    <div id="usersSection" style="display:none;margin-top:20px;">
      <h3>👥 User Management</h3>
      <div style="margin-bottom:16px;">
        <button onclick="loadUsers()" class="success">🔄 Refresh Users</button>
      </div>
      <div id="usersList"></div>
    </div>

    <!-- TOKENS SECTION -->
    <div id="tokensSection" style="display:none;margin-top:20px;">
      <h3>🔑 API Tokens</h3>
      <div style="margin:12px 0;">
        <h4>Issue New Token</h4>
        <div style="display:flex;gap:12px;margin-bottom:12px;">
          <div style="flex:1;">
            <label>User ID</label>
            <input id="tokenUserId" type="number" placeholder="User ID"/>
          </div>
          <div style="flex:1;">
            <label>Token Name</label>
            <input id="tokenName" placeholder="e.g., Production API"/>
          </div>
        </div>
        <button onclick="issueToken()" style="margin-top:12px;">Issue Token</button>
      </div>
      <h4 style="margin-top:20px;">Active Tokens</h4>
      <table class="table">
        <thead>
          <tr><th>Token ID</th><th>User ID</th><th>Created</th><th>Status</th><th>Action</th></tr>
        </thead>
        <tbody id="tokensList"></tbody>
      </table>
    </div>
  </div>

  <!-- DOCTOR INTERFACE -->
  <div class="card" id="doctorCard" style="display:none;">
    <h2>🏥 Scan & Diagnose</h2>
    <div style="margin:20px 0;">
      <label>Select Eye(s)</label>
      <select id="eyeMode">
        <option value="left">Left Eye Only</option>
        <option value="right">Right Eye Only</option>
        <option value="both">Both Eyes</option>
      </select>
    </div>

    <div id="singleEyeUpload" style="display:block;">
      <label>Upload Image</label>
      <input id="imageFile" type="file" accept="image/*"/>
    </div>

    <div id="bothEyesUpload" style="display:none;">
      <label>Left Eye Image</label>
      <input id="leftFile" type="file" accept="image/*"/>
      <label style="margin-top:12px;">Right Eye Image</label>
      <input id="rightFile" type="file" accept="image/*"/>
    </div>

    <button onclick="runScan()" style="margin-top:16px;width:100%;">Analyze</button>

    <div id="scanResult" style="display:none;margin-top:24px;">
      <h3>Result</h3>
      <div class="result">
        <div style="margin-bottom:12px;">
          <span class="big" id="diagnosisText"></span>
        </div>
        <p id="confidenceText" class="muted"></p>
        
        <h4 style="margin-top:16px;">Professional Opinion</h4>
        <p class="muted">Confirm or correct the AI diagnosis:</p>
        <div id="confirmationUI"></div>
        <button onclick="submitConfirmation()" style="margin-top:12px;width:100%;">Save Opinion</button>
      </div>
    </div>
  </div>
</div>

<script>
let currentToken = localStorage.getItem("token");
let currentUserId = localStorage.getItem("userId");
let currentScanId = null;

function showSection(section) {
  document.getElementById('resultsSection').style.display = section === 'results' ? 'block' : 'none';
  document.getElementById('usersSection').style.display = section === 'users' ? 'block' : 'none';
  document.getElementById('tokensSection').style.display = section === 'tokens' ? 'block' : 'none';
  if (section === 'users') loadUsers();
  if (section === 'tokens') loadTokens();
}

function showLogin() {
  document.getElementById('loginTab').style.display = 'block';
  document.getElementById('registerTab').style.display = 'none';
}

function showRegister() {
  document.getElementById('loginTab').style.display = 'none';
  document.getElementById('registerTab').style.display = 'block';
}

async function doLogin() {
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;
  try {
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({email, password})
    });
    const data = await res.json();
    if (data.access_token) {
      currentToken = data.access_token;
      currentUserId = data.user_id;
      localStorage.setItem('token', currentToken);
      localStorage.setItem('userId', currentUserId);
      document.getElementById('authCard').style.display = 'none';
      document.getElementById('adminCard').style.display = data.is_admin ? 'block' : 'none';
      document.getElementById('doctorCard').style.display = data.is_admin ? 'none' : 'block';
    } else {
      document.getElementById('authStatus').textContent = '❌ Login failed';
    }
  } catch(e) {
    document.getElementById('authStatus').textContent = '❌ Error: ' + e.message;
  }
}

async function doRegister() {
  const email = document.getElementById('regEmail').value;
  const password = document.getElementById('regPassword').value;
  try {
    const res = await fetch('/auth/register', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({email, password})
    });
    const data = await res.json();
    if (data.access_token) {
      currentToken = data.access_token;
      currentUserId = data.user_id;
      localStorage.setItem('token', currentToken);
      localStorage.setItem('userId', currentUserId);
      document.getElementById('authCard').style.display = 'none';
      document.getElementById('doctorCard').style.display = 'block';
    } else {
      document.getElementById('authStatus').textContent = '❌ Registration failed';
    }
  } catch(e) {
    document.getElementById('authStatus').textContent = '❌ Error: ' + e.message;
  }
}

function doLogout() {
  currentToken = null;
  currentUserId = null;
  localStorage.removeItem('token');
  localStorage.removeItem('userId');
  document.getElementById('authCard').style.display = 'block';
  document.getElementById('adminCard').style.display = 'none';
  document.getElementById('doctorCard').style.display = 'none';
  document.getElementById('email').value = '';
  document.getElementById('password').value = '';
}

async function loadUsers() {
  try {
    const res = await fetch('/admin/users', {
      headers: {'Authorization': `Bearer ${currentToken}`}
    });
    const users = await res.json();
    
    if (!Array.isArray(users)) {
      document.getElementById('usersList').innerHTML = '<p class="bad">Error: Invalid response from server</p>';
      return;
    }
    
    let html = '';
    for (const user of users) {
      const isAdmin = user.id === parseInt(currentUserId);
      const isBanned = user.is_banned ? true : false;
      const usageLimit = user.usage_limit || -1;
      const usageCount = user.usage_count || 0;
      
      html += `
        <div class="user-row">
          <div style="flex:2;">
            <strong>${user.email}</strong><br>
            <span class="muted">ID: ${user.id}</span>
          </div>
          <div style="flex:1;">
            <span class="muted">Scans: ${usageCount}/${usageLimit === -1 ? '∞' : usageLimit}</span>
          </div>
          <div class="user-controls">
            <input type="number" id="limit_${user.id}" value="${usageLimit === -1 ? '' : usageLimit}" placeholder="Limit" min="-1"/>
            <select id="ban_${user.id}">
              <option value="0" ${!isBanned ? 'selected' : ''}>Active</option>
              <option value="1" ${isBanned ? 'selected' : ''}>Banned</option>
            </select>
            ${isAdmin ? '<span class="badge">Owner</span>' : `<button onclick="updateUser(${user.id})" class="success" style="width:auto;">Update</button>`}
          </div>
        </div>
      `;
    }
    document.getElementById('usersList').innerHTML = html || '<p class="muted">No users found</p>';
  } catch(e) {
    document.getElementById('usersList').innerHTML = '<p class="bad">Error loading users: ' + e.message + '</p>';
  }
}

async function updateUser(userId) {
  const limitInput = document.getElementById(`limit_${userId}`);
  const banSelect = document.getElementById(`ban_${userId}`);
  
  const usage_limit = limitInput.value === '' ? -1 : parseInt(limitInput.value);
  const is_banned = parseInt(banSelect.value);
  
  try {
    const res = await fetch(`/admin/users/${userId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentToken}`
      },
      body: JSON.stringify({usage_limit, is_banned})
    });
    const data = await res.json();
    if (data.status === 'updated') {
      alert('✓ User updated successfully');
      loadUsers();
    } else {
      alert('❌ Failed to update user');
    }
  } catch(e) {
    alert('Error: ' + e.message);
  }
}

async function loadTokens() {
  try {
    const res = await fetch('/admin/tokens', {
      headers: {'Authorization': `Bearer ${currentToken}`}
    });
    const tokens = await res.json();
    let html = '';
    for (const token of tokens) {
      html += `<tr>
        <td>${token.id}</td>
        <td>${token.user_id}</td>
        <td>${new Date(token.created_at).toLocaleDateString()}</td>
        <td><span class="badge ${token.is_active ? 'confirmed' : 'pending'}">${token.is_active ? 'Active' : 'Inactive'}</span></td>
        <td><button onclick="revokeToken(${token.id})" class="danger" style="width:auto;">Revoke</button></td>
      </tr>`;
    }
    document.getElementById('tokensList').innerHTML = html || '<tr><td colspan="5" class="muted">No tokens</td></tr>';
  } catch(e) {
    alert('Error loading tokens: ' + e.message);
  }
}

async function issueToken() {
  const userId = parseInt(document.getElementById('tokenUserId').value);
  const name = document.getElementById('tokenName').value;
  if (!userId || !name) {
    alert('Please fill all fields');
    return;
  }
  try {
    const res = await fetch('/admin/tokens/issue', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentToken}`
      },
      body: JSON.stringify({user_id: userId, name})
    });
    const data = await res.json();
    alert(`✓ Token issued:\n\n${data.token}`);
    document.getElementById('tokenUserId').value = '';
    document.getElementById('tokenName').value = '';
    loadTokens();
  } catch(e) {
    alert('Error: ' + e.message);
  }
}

async function revokeToken(tokenId) {
  if (!confirm('Revoke this token?')) return;
  try {
    await fetch(`/admin/tokens/${tokenId}`, {
      method: 'DELETE',
      headers: {'Authorization': `Bearer ${currentToken}`}
    });
    alert('✓ Token revoked');
    loadTokens();
  } catch(e) {
    alert('Error: ' + e.message);
  }
}

async function loadResults() {
  try {
    const res = await fetch('/results/all', {
      headers: {'Authorization': `Bearer ${currentToken}`}
    });
    const results = await res.json();
    let html = '';
    for (const r of results) {
      const match = r.ai_diagnosis === r.professional_opinion ? '✓' : '✗';
      html += `<tr>
        <td>${r.id}</td>
        <td>${r.doctor_email}</td>
        <td>${r.ai_diagnosis}</td>
        <td>${r.professional_opinion}</td>
        <td>${match}</td>
        <td>${new Date(r.created_at).toLocaleDateString()}</td>
      </tr>`;
    }
    document.getElementById('resultsList').innerHTML = html || '<tr><td colspan="6" class="muted">No results</td></tr>';
  } catch(e) {
    alert('Error: ' + e.message);
  }
}

function exportResults() {
  alert('Export feature coming soon');
}

async function runScan() {
  const eyeMode = document.getElementById('eyeMode').value;
  const formData = new FormData();
  formData.append('eye_mode', eyeMode);
  
  if (eyeMode === 'both') {
    const leftFile = document.getElementById('leftFile').files[0];
    const rightFile = document.getElementById('rightFile').files[0];
    if (!leftFile || !rightFile) {
      alert('Please select both images');
      return;
    }
    formData.append('left_file', leftFile);
    formData.append('right_file', rightFile);
  } else {
    const imageFile = document.getElementById('imageFile').files[0];
    if (!imageFile) {
      alert('Please select an image');
      return;
    }
    formData.append('file', imageFile);
  }
  
  try {
    const res = await fetch('/scan/run', {
      method: 'POST',
      headers: {'Authorization': `Bearer ${currentToken}`},
      body: formData
    });
    const data = await res.json();
    currentScanId = data.id;
    
    const diagnosis = eyeMode === 'both' 
      ? `Left: ${data.left_diagnosis} | Right: ${data.right_diagnosis}`
      : data.left_diagnosis || data.right_diagnosis;
    
    document.getElementById('diagnosisText').textContent = diagnosis;
    document.getElementById('confidenceText').textContent = 'AI Suggestion - Please verify';
    
    let confirmUI = '';
    if (eyeMode === 'both') {
      confirmUI = `
        <label>Left Eye Opinion</label>
        <input id="confirmLeft" value="${data.left_diagnosis}"/>
        <label style="margin-top:12px;">Right Eye Opinion</label>
        <input id="confirmRight" value="${data.right_diagnosis}"/>
      `;
    } else {
      confirmUI = `
        <label>Your Opinion</label>
        <input id="confirmDiag" value="${diagnosis}"/>
      `;
    }
    document.getElementById('confirmationUI').innerHTML = confirmUI;
    document.getElementById('scanResult').style.display = 'block';
  } catch(e) {
    alert('Scan failed: ' + e.message);
  }
}

async function submitConfirmation() {
  const eyeMode = document.getElementById('eyeMode').value;
  let leftDiag = null, rightDiag = null;
  
  if (eyeMode === 'both') {
    leftDiag = document.getElementById('confirmLeft').value;
    rightDiag = document.getElementById('confirmRight').value;
  } else if (eyeMode === 'left') {
    leftDiag = document.getElementById('confirmDiag').value;
  } else {
    rightDiag = document.getElementById('confirmDiag').value;
  }
  
  try {
    const res = await fetch('/scan/confirm', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${currentToken}`
      },
      body: JSON.stringify({
        scan_id: currentScanId,
        confirmed_left_diagnosis: leftDiag,
        confirmed_right_diagnosis: rightDiag
      })
    });
    const data = await res.json();
    alert('✓ Opinion saved');
  } catch(e) {
    alert('Error saving: ' + e.message);
  }
}

function switchToTest() {
  document.getElementById('adminCard').style.display = 'none';
  document.getElementById('doctorCard').style.display = 'block';
}

document.getElementById('eyeMode').addEventListener('change', function() {
  document.getElementById('singleEyeUpload').style.display = this.value === 'both' ? 'none' : 'block';
  document.getElementById('bothEyesUpload').style.display = this.value === 'both' ? 'block' : 'none';
});

if (currentToken) {
  document.getElementById('authCard').style.display = 'none';
  document.getElementById('adminCard').style.display = 'block';
}
</script>
</body>
</html>
    """


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
    """[ADMIN ONLY] List all users with their management data"""
    require_admin(req)
    users = db.query(User).all()
    
    # Convert SQLAlchemy objects to dictionaries for JSON serialization
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
    """[ADMIN ONLY] Update user settings"""
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
    """[ADMIN ONLY] Issue API token for user"""
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
    """[ADMIN ONLY] List all API tokens"""
    require_admin(req)
    return db.query(APIToken).all()


@app.delete("/admin/tokens/{token_id}")
def revoke_token(token_id: int, req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Revoke API token"""
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
    """Run a scan with usage limit checking"""
    
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
    
    # Check usage limits
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
    """Save professional opinion for diagnosis"""
    
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