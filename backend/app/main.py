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

app = FastAPI(title="Alati Cloud Demo (No Worker, R2-only)")
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
  <title>Alati Cloud - Eye Scan AI</title>
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
    .flex-between{display:flex;justify-content:space-between;align-items:center;gap:12px;}
    .modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.8);z-index:1000;justify-content:center;align-items:center;}
    .modal.show{display:flex;}
    .modal-content{background:#0b1020;border:2px solid #355dff;border-radius:16px;padding:24px;max-width:600px;width:90%;}
    .row{display:flex;gap:12px;}
    .row>div{flex:1;}
    @media(max-width:768px){.row{flex-direction:column;}}
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

    <!-- USERS SECTION -->
    <div id="usersSection" style="display:none;margin-top:20px;">
      <h3>👥 Doctors</h3>
      <button onclick="loadUsers()" style="margin-bottom:12px;">Refresh</button>
      <table class="table">
        <thead>
          <tr><th>Email</th><th>Scans</th><th>Opinions</th></tr>
        </thead>
        <tbody id="usersList"></tbody>
      </table>
    </div>

    <!-- TOKENS SECTION -->
    <div id="tokensSection" style="display:none;margin-top:20px;">
      <h3>🔑 API Tokens</h3>
      <div style="padding:12px;background:rgba(51,93,255,.1);border-radius:8px;margin-bottom:12px;">
        <label>User ID</label>
        <input id="tokenUserId" type="number" placeholder="2"/>
        <label>Token Name</label>
        <input id="tokenName" placeholder="Mobile App"/>
        <button onclick="issueToken()" style="margin-top:8px;">Generate</button>
        <p id="tokenStatus" class="muted" style="margin-top:8px;"></p>
        <div id="tokenDisplay" style="display:none;margin-top:12px;padding:12px;background:rgba(0,0,0,.3);border-radius:8px;">
          <textarea id="tokenValue" readonly style="height:60px;font-family:monospace;font-size:12px;"></textarea>
          <button onclick="copyToken()" style="margin-top:8px;">Copy</button>
        </div>
      </div>
      <h4>Active Tokens</h4>
      <table class="table">
        <thead>
          <tr><th>ID</th><th>User</th><th>Name</th><th>Action</th></tr>
        </thead>
        <tbody id="tokensList"></tbody>
      </table>
    </div>
  </div>

  <!-- SCAN CARD (DOCTORS & OWNER TESTING) -->
  <div class="card" id="scanCard" style="display:none;">
    <div class="flex-between">
      <h3 style="margin:0;">Patient Eye Scan</h3>
      <div>
        <button id="backBtn" onclick="backToAdmin()" class="secondary" style="display:none;margin-right:8px;">← Back to Admin</button>
        <button onclick="doLogout()" class="secondary">Logout</button>
      </div>
    </div>

    <label>Eye</label>
    <select id="eyeMode">
      <option value="left">Left</option>
      <option value="right">Right</option>
      <option value="both">Both</option>
    </select>

    <label>Image</label>
    <input id="singleFile" type="file" accept="image/*"/>

    <div style="height:10px"></div>
    <button onclick="runScan()">Analyze</button>
    <p id="scanStatus" class="muted"></p>

    <!-- AI RESULT -->
    <div class="result" id="resultBox" style="display:none;">
      <div class="big">🤖 AI Analysis Result</div>
      <div id="diagText" style="margin-top:12px;font-size:16px;line-height:1.6;"></div>
      <div style="height:16px"></div>
      <button class="success" onclick="openOpinionModal()">💬 Add Professional Opinion</button>
      <p id="opinionStatus" class="muted" style="margin-top:8px;"></p>
    </div>
  </div>

  <!-- PROFESSIONAL OPINION MODAL -->
  <div id="opinionModal" class="modal">
    <div class="modal-content">
      <h3>Professional Opinion</h3>
      <p class="muted">Enter your professional diagnosis for this patient</p>
      
      <div id="opinionFields" style="margin:16px 0;"></div>
      
      <div style="height:12px"></div>
      <button class="success" onclick="saveProfessionalOpinion()">Save Opinion</button>
      <button class="secondary" onclick="closeOpinionModal()">Cancel</button>
    </div>
  </div>

</div>

<script>
let TOKEN = null;
let IS_OWNER = false;
let OWNER_EMAIL = "";
let OWNER_PASSWORD = "";
let USER_ID = null;
let CURRENT_SCAN_ID = null;
let CURRENT_EYE_MODE = null;

function setStatus(id, msg, ok=null){
  const el = document.getElementById(id);
  if(!el) return;
  el.textContent = msg;
  if(ok===true) el.className="muted ok";
  else if(ok===false) el.className="muted bad";
  else el.className="muted";
}

function showSection(sec){
  document.getElementById("resultsSection").style.display = sec==="results" ? "block" : "none";
  document.getElementById("usersSection").style.display = sec==="users" ? "block" : "none";
  document.getElementById("tokensSection").style.display = sec==="tokens" ? "block" : "none";
  if(sec==="results") loadResults();
  if(sec==="users") loadUsers();
  if(sec==="tokens") loadTokens();
}

function switchToTest(){
  document.getElementById("adminCard").style.display = "none";
  document.getElementById("scanCard").style.display = "block";
  document.getElementById("backBtn").style.display = "block";
}

function backToAdmin(){
  document.getElementById("scanCard").style.display = "none";
  document.getElementById("adminCard").style.display = "block";
  document.getElementById("backBtn").style.display = "none";
}

function showLogin(){
  document.getElementById("loginTab").style.display="block";
  document.getElementById("registerTab").style.display="none";
}

function showRegister(){
  document.getElementById("loginTab").style.display="none";
  document.getElementById("registerTab").style.display="block";
}

async function doRegister(){
  setStatus("authStatus","Registering…");
  const email = document.getElementById("regEmail").value.trim();
  const password = document.getElementById("regPassword").value;
  if(!password){
    setStatus("authStatus","Password required",false);
    return;
  }
  try{
    const r = await fetch("/auth/register",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({email, password})
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || "Failed");
    setStatus("authStatus","Success! Login now. ✅",true);
    setTimeout(() => showLogin(), 1000);
  }catch(e){
    setStatus("authStatus","Error: "+e.message,false);
  }
}

async function doLogin(){
  setStatus("authStatus","Logging in…");
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  try{
    const r = await fetch("/auth/login",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({email,password})
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || "Failed");
    TOKEN = data.access_token;
    IS_OWNER = data.is_owner || false;
    USER_ID = data.user_id;
    setStatus("authStatus","Success! ✅",true);
    document.getElementById("authCard").style.display="none";
    if(IS_OWNER){
      OWNER_EMAIL = email;
      OWNER_PASSWORD = password;
      document.getElementById("adminCard").style.display="block";
      showSection("results");
    } else {
      document.getElementById("scanCard").style.display="block";
    }
  }catch(e){
    TOKEN = null;
    setStatus("authStatus","Failed: "+e.message,false);
  }
}

function doLogout(){
  TOKEN = null;
  IS_OWNER = false;
  document.getElementById("authCard").style.display="block";
  document.getElementById("adminCard").style.display="none";
  document.getElementById("scanCard").style.display="none";
  document.getElementById("loginTab").style.display="block";
  document.getElementById("email").value = "";
  document.getElementById("password").value = "";
}

async function loadResults(){
  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;
  try{
    const r = await fetch("/admin/results/all", {
      headers: {"Authorization": authHeader}
    });
    const data = await r.json();
    displayResults(data);
  }catch(e){
    console.error(e);
  }
}

function displayResults(data){
  const tbody = document.getElementById("resultsList");
  tbody.innerHTML = "";
  
  if(!data.results || data.results.length === 0){
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;opacity:.5;">No results yet</td></tr>';
    return;
  }
  
  data.results.forEach(r => {
    const match = r.ai_diagnosis === r.professional_opinion ? "✅" : "❌";
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${r.scan_id}</td>
      <td>${r.doctor_email}</td>
      <td>${r.ai_diagnosis || "-"}</td>
      <td>${r.professional_opinion || "-"}</td>
      <td>${match}</td>
      <td>${new Date(r.confirmed_at).toLocaleDateString()}</td>
    `;
    tbody.appendChild(row);
  });
  
  if(data.analytics){
    document.getElementById("analyticsBox").style.display="block";
    displayAnalytics(data.analytics);
  }
}

function displayAnalytics(analytics){
  let html = "<table class='table'><tr><th>Diagnosis</th><th>Sensitivity</th><th>Specificity</th><th>Count</th></tr>";
  for(const [diag, stats] of Object.entries(analytics)){
    html += `<tr>
      <td>${diag}</td>
      <td>${(stats.sensitivity*100).toFixed(1)}%</td>
      <td>${(stats.specificity*100).toFixed(1)}%</td>
      <td>${stats.count}</td>
    </tr>`;
  }
  html += "</table>";
  document.getElementById("analyticsContent").innerHTML = html;
}

async function exportResults(){
  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;
  try{
    const r = await fetch("/admin/results/export", {
      headers: {"Authorization": authHeader}
    });
    const data = await r.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "alati_results_" + new Date().toISOString().slice(0,10) + ".json";
    a.click();
  }catch(e){
    alert("Export failed: "+e.message);
  }
}

async function loadUsers(){
  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;
  try{
    const r = await fetch("/admin/users", {
      headers: {"Authorization": authHeader}
    });
    const users = await r.json();
    const tbody = document.getElementById("usersList");
    tbody.innerHTML = "";
    users.forEach(u => {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${u.email}</td><td>${u.usage_count || 0}</td><td id="op-${u.id}">-</td>`;
      tbody.appendChild(row);
      fetch("/admin/user/" + u.id + "/opinions?count=true", {
        headers: {"Authorization": authHeader}
      }).then(r => r.json()).then(d => {
        document.getElementById("op-"+u.id).textContent = d.count || 0;
      }).catch(e => {});
    });
  }catch(e){
    console.error(e);
  }
}

async function loadTokens(){
  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;
  try{
    const r = await fetch("/admin/tokens", {
      headers: {"Authorization": authHeader}
    });
    const tokens = await r.json();
    const tbody = document.getElementById("tokensList");
    tbody.innerHTML = "";
    tokens.forEach(t => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${t.id}</td><td>${t.user_id}</td><td>${t.name || "-"}</td>
        <td>${t.is_active ? '<button class="danger" onclick="revokeToken('+t.id+')">Revoke</button>' : "-"}</td>
      `;
      tbody.appendChild(row);
    });
  }catch(e){
    console.error(e);
  }
}

async function issueToken(){
  const uid = document.getElementById("tokenUserId").value;
  const name = document.getElementById("tokenName").value;
  if(!uid){
    setStatus("tokenStatus","Enter user ID",false);
    return;
  }
  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;
  try{
    const r = await fetch("/admin/tokens/issue", {
      method: "POST",
      headers: {"Authorization": authHeader, "Content-Type": "application/json"},
      body: JSON.stringify({user_id: parseInt(uid), name: name || null})
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail);
    document.getElementById("tokenValue").value = data.token;
    document.getElementById("tokenDisplay").style.display = "block";
    setStatus("tokenStatus","Created! ✅",true);
  }catch(e){
    setStatus("tokenStatus","Error: "+e.message,false);
  }
}

function copyToken(){
  document.getElementById("tokenValue").select();
  document.execCommand("copy");
  alert("Copied!");
}

async function revokeToken(id){
  if(!confirm("Revoke?")) return;
  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;
  try{
    const r = await fetch("/admin/tokens/" + id, {
      method: "DELETE",
      headers: {"Authorization": authHeader}
    });
    if(!r.ok) throw new Error("Failed");
    alert("Revoked!");
    loadTokens();
  }catch(e){
    alert("Error: "+e.message);
  }
}

async function runScan(){
  if(!TOKEN){
    setStatus("scanStatus","Login first",false);
    return;
  }
  const mode = document.getElementById("eyeMode").value;
  const file = document.getElementById("singleFile").files?.[0];
  if(!file){
    setStatus("scanStatus","Select image",false);
    return;
  }
  const fd = new FormData();
  fd.append("eye_mode", mode);
  fd.append("file", file);
  setStatus("scanStatus","Analyzing…");
  document.getElementById("resultBox").style.display="none";
  try{
    const r = await fetch("/scan/run",{
      method:"POST",
      headers:{"Authorization":"Bearer "+TOKEN},
      body: fd
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail);
    if(data.status !== "done"){
      setStatus("scanStatus","Failed",false);
    }else{
      setStatus("scanStatus","Done ✅",true);
      CURRENT_SCAN_ID = data.id;
      CURRENT_EYE_MODE = mode;
      let txt = mode==="left" ? data.left_diagnosis : mode==="right" ? data.right_diagnosis : (data.left_diagnosis || "-") + " (L) / " + (data.right_diagnosis || "-") + " (R)";
      document.getElementById("diagText").textContent = txt || "No diagnosis";
    }
    document.getElementById("resultBox").style.display="block";
  }catch(e){
    setStatus("scanStatus","Error: "+e.message,false);
  }
}

function openOpinionModal(){
  if(!CURRENT_SCAN_ID){
    alert("No scan");
    return;
  }
  let html = "";
  if(CURRENT_EYE_MODE === "left" || CURRENT_EYE_MODE === "both"){
    html += `<div><label>Left Eye Diagnosis</label><input id="opLeft" type="text" placeholder="e.g., Healthy, Diabetic Retinopathy"/></div>`;
  }
  if(CURRENT_EYE_MODE === "right" || CURRENT_EYE_MODE === "both"){
    html += `<div><label>Right Eye Diagnosis</label><input id="opRight" type="text" placeholder="e.g., Healthy, Cataract"/></div>`;
  }
  document.getElementById("opinionFields").innerHTML = html;
  document.getElementById("opinionModal").classList.add("show");
}

function closeOpinionModal(){
  document.getElementById("opinionModal").classList.remove("show");
}

async function saveProfessionalOpinion(){
  if(!CURRENT_SCAN_ID){
    alert("No scan");
    return;
  }
  const body = {
    scan_id: CURRENT_SCAN_ID,
    confirmed_left_diagnosis: document.getElementById("opLeft")?.value || null,
    confirmed_right_diagnosis: document.getElementById("opRight")?.value || null,
  };
  try{
    const r = await fetch("/scan/confirm", {
      method: "POST",
      headers: {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"},
      body: JSON.stringify(body)
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail);
    setStatus("opinionStatus", "✅ Saved!", true);
    closeOpinionModal();
    setTimeout(() => {
      document.getElementById("resultBox").style.display = "none";
      document.getElementById("singleFile").value = "";
      setStatus("opinionStatus", "");
    }, 1500);
  }catch(e){
    alert("Error: "+e.message);
  }
}
</script>
</body>
</html>
"""


# ============ AUTH ENDPOINTS ============

@app.post("/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    email = (body.email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    password = (body.password or "").strip()
    if not password:
        raise HTTPException(status_code=400, detail="Password required")
    
    password = password[:72]
    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {"access_token": create_token(user.id)}


@app.post("/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Login with email and password"""
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()
    
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    is_banned = getattr(user, 'is_banned', 0)
    if is_banned:
        raise HTTPException(status_code=403, detail="User account is banned")
    
    usage_limit = getattr(user, 'usage_limit', -1)
    usage_count = getattr(user, 'usage_count', 0)
    
    owner_email = (settings.OWNER_EMAIL or "").strip().lower()
    is_owner = (email == owner_email)
    
    return {
        "access_token": create_token(user.id),
        "is_owner": is_owner,
        "user_id": user.id,
        "email": user.email,
        "usage_limit": usage_limit,
        "usage_count": usage_count,
    }


# ============ ADMIN ENDPOINTS ============

@app.get("/admin/users")
def list_users(req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] List all users"""
    require_admin(req)
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "usage_count": getattr(u, 'usage_count', 0),
        }
        for u in users
    ]


@app.get("/admin/user/{user_id}/opinions")
def get_user_opinions(user_id: int, count: bool = False, req: Request = None, db: Session = Depends(get_db)):
    """Get opinions for a user"""
    scans = db.query(Scan).filter(
        Scan.user_id == user_id,
        Scan.confirmed_at != None
    ).all()
    
    if count:
        return {"count": len(scans)}
    
    return scans


@app.get("/admin/results/all")
def get_all_results(req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Get all results with analytics"""
    require_admin(req)
    
    scans = db.query(Scan).filter(Scan.confirmed_at != None).all()
    users_map = {u.id: u.email for u in db.query(User).all()}
    
    results = []
    by_diagnosis = {}
    
    for scan in scans:
        diagnoses = []
        if scan.eye_mode == "both":
            diagnoses = [
                (scan.left_diagnosis, scan.confirmed_left_diagnosis),
                (scan.right_diagnosis, scan.confirmed_right_diagnosis),
            ]
        elif scan.eye_mode == "left":
            diagnoses = [(scan.left_diagnosis, scan.confirmed_left_diagnosis)]
        else:
            diagnoses = [(scan.right_diagnosis, scan.confirmed_right_diagnosis)]
        
        for ai_diag, confirmed_diag in diagnoses:
            if not confirmed_diag:
                continue
            
            results.append({
                "scan_id": scan.id,
                "doctor_email": users_map.get(scan.user_id, "Unknown"),
                "ai_diagnosis": ai_diag,
                "professional_opinion": confirmed_diag,
                "confirmed_at": scan.confirmed_at,
            })
            
            key = confirmed_diag
            if key not in by_diagnosis:
                by_diagnosis[key] = {"tp": 0, "fp": 0, "count": 0}
            
            by_diagnosis[key]["count"] += 1
            if ai_diag == confirmed_diag:
                by_diagnosis[key]["tp"] += 1
            else:
                by_diagnosis[key]["fp"] += 1
    
    # Calculate metrics
    analytics = {}
    for diag, stats in by_diagnosis.items():
        tp = stats["tp"]
        fp = stats["fp"]
        total = tp + fp
        
        analytics[diag] = {
            "sensitivity": tp / total if total > 0 else 0,
            "specificity": 0.5,  # Placeholder - would need more data
            "count": total,
            "correct": tp,
            "incorrect": fp,
        }
    
    return {
        "results": results,
        "total": len(results),
        "confirmed_scans": len(scans),
        "analytics": analytics,
    }


@app.get("/admin/results/export")
def export_results(req: Request, db: Session = Depends(get_db)):
    """[ADMIN ONLY] Export results as JSON"""
    require_admin(req)
    return get_all_results(req, db)


@app.put("/admin/users/{user_id}")
def update_user(user_id: int, req: Request, db: Session = Depends(get_db), is_banned: int = None, usage_limit: int = None):
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
                user_id=user_id, eye_mode="both", left_key=left_key, right_key=right_key,
                left_diagnosis=left_diag, right_diagnosis=right_diag, status="done"
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
    
    if confirmed_left_diagnosis:
        scan.confirmed_left_diagnosis = confirmed_left_diagnosis
    if confirmed_right_diagnosis:
        scan.confirmed_right_diagnosis = confirmed_right_diagnosis
    
    scan.confirmed_at = datetime.utcnow()
    db.add(scan)
    db.commit()
    
    return {"status": "saved", "scan_id": scan.id}