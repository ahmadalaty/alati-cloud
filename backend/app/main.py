import hashlib
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

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


def upsert_admin():
    """Create admin user on startup if OWNER_EMAIL and OWNER_PASSWORD are set"""
    db = SessionLocal()
    try:
        email = (settings.OWNER_EMAIL or "").strip().lower()
        password = (settings.OWNER_PASSWORD or "").strip()
        if not email or not password:
            return

        password = password[:72]  # passlib/bcrypt max 72 bytes

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


@app.on_event("startup")
def startup():
    upsert_admin()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug")
def debug():
    return {
        "storage_mode": settings.STORAGE_MODE,
        "model_variant": ACTIVE_VARIANT,
        "storage_marker": STORAGE_MARKER,
        "inference_marker": INF_MARKER,
        "r2_bucket_set": bool(settings.R2_BUCKET),
        "demo_mode": getattr(settings, "DEMO_MODE", ""),
        "note": "Use POST /debug/inference to see raw probabilities + final decision rule.",
    }


@app.post("/debug/inference")
async def debug_inference(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return predict_debug(image_bytes)


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Alati Cloud</title>
  <style>
    *{box-sizing:border-box;}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#0b1020;color:#e8ecff;}
    .wrap{max-width:1200px;margin:0 auto;padding:28px 16px 48px;}
    .card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:16px;margin:14px 0;}
    h1{font-size:28px;margin:0 0 6px;}
    h2{font-size:20px;margin:12px 0 8px;}
    h3{font-size:16px;margin:10px 0 8px;}
    .sub{opacity:.85;margin:0 0 18px;}
    label{display:block;font-size:13px;opacity:.85;margin:10px 0 6px;}
    input,select,textarea,button{padding:12px;border-radius:12px;border:1px solid rgba(255,255,255,.14);background:rgba(0,0,0,.25);color:#e8ecff;font-size:15px;font-family:inherit;}
    input,select,textarea{width:100%;}
    button{cursor:pointer;background:#355dff;border:0;font-weight:800;width:auto;padding:10px 20px;}
    button:hover{background:#2851e8;}
    button.danger{background:#ff6b6b;}
    button.danger:hover{background:#ff5252;}
    button.secondary{background:#666;}
    button.secondary:hover{background:#777;}
    button:disabled{opacity:.6;cursor:not-allowed;}
    .row{display:flex;gap:12px;align-items:flex-end;}
    .row>div{flex:1;}
    .muted{opacity:.8;font-size:13px;}
    .ok{color:#67ffb1;font-weight:800;}
    .bad{color:#ff8a8a;font-weight:800;}
    .result{padding:14px;border-radius:14px;background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.12);margin-top:10px;}
    .big{font-size:18px;font-weight:900;}
    a{color:#9fb3ff;text-decoration:none;}
    a:hover{text-decoration:underline;}
    .tabs{display:flex;gap:10px;margin-top:8px;}
    .tab{flex:1;padding:10px;border-radius:12px;border:1px solid rgba(255,255,255,.14);background:rgba(0,0,0,.20);text-align:center;cursor:pointer;font-weight:800;}
    .tab.active{background:#355dff;border-color:#355dff;}
    .hint{opacity:.75;font-size:12px;margin-top:6px;}
    .diag{white-space:pre-line;font-size:16px;font-weight:900;line-height:1.6;}
    .table{width:100%;border-collapse:collapse;margin-top:12px;}
    .table th{background:rgba(255,255,255,.1);padding:10px;text-align:left;border-bottom:1px solid rgba(255,255,255,.12);}
    .table td{padding:10px;border-bottom:1px solid rgba(255,255,255,.08);}
    .table tr:hover{background:rgba(255,255,255,.04);}
    .badge{display:inline-block;padding:4px 8px;border-radius:6px;font-size:12px;background:rgba(51,93,255,.3);}
    .badge.active{background:rgba(103,255,177,.3);color:#67ffb1;}
    .badge.inactive{background:rgba(255,138,138,.3);color:#ff8a8a;}
    .flex-between{display:flex;justify-content:space-between;align-items:center;}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
    @media(max-width:768px){.grid-2{grid-template-columns:1fr;}}
  </style>
</head>
<body>
<div class="wrap">
  <h1>🔬 Alati Cloud</h1>

  <div class="card" id="authCard">
    <h3 style="margin:0 0 8px;">Authentication</h3>
    <div id="loginTab">
      <label>Email</label>
      <input id="email" placeholder="user@example.com"/>
      <label>Password</label>
      <input id="password" type="password" placeholder="••••••••"/>
      <div style="height:10px"></div>
      <button onclick="doLogin()">Login</button>
      <button onclick="showRegister()" class="secondary" style="margin-left:8px;">Create Account</button>
    </div>
    <div id="registerTab" style="display:none;">
      <label>Email</label>
      <input id="regEmail" placeholder="user@example.com"/>
      <label>Password (optional)</label>
      <input id="regPassword" type="password" placeholder="••••••••"/>
      <div style="height:10px"></div>
      <button onclick="doRegister()">Register</button>
      <button onclick="showLogin()" class="secondary" style="margin-left:8px;">Back to Login</button>
    </div>
    <p id="authStatus" class="muted"></p>
  </div>

  <!-- ADMIN DASHBOARD (only visible for owner) -->
  <div class="card" id="adminCard" style="display:none;border-color:#ffc107;border-width:2px;">
    <div class="flex-between">
      <h2>👑 Admin Dashboard</h2>
      <button onclick="doLogout()" class="secondary">Logout</button>
    </div>

    <!-- Issue Token Section -->
    <div style="margin-top:20px;padding:16px;background:rgba(51,93,255,.1);border-radius:12px;">
      <h3 style="margin-top:0;">Issue New API Token</h3>
      <label>Enter User ID</label>
      <input id="tokenUserId" type="number" placeholder="e.g., 2"/>
      <label>Token Name (optional)</label>
      <input id="tokenName" placeholder="e.g., Mobile App, Testing"/>
      <div style="height:10px"></div>
      <button onclick="issueToken()">Generate Token</button>
      <p id="tokenStatus" class="muted"></p>
      
      <div id="tokenDisplay" style="display:none;margin-top:16px;padding:12px;background:rgba(0,0,0,.3);border-radius:8px;">
        <p style="margin:0 0 8px;"><strong>✅ Token Created!</strong></p>
        <p style="margin:0 0 8px;opacity:.85;font-size:12px;">Save this token now - it won't be shown again:</p>
        <textarea id="tokenValue" readonly style="height:80px;font-family:monospace;font-size:12px;"></textarea>
        <button onclick="copyToken()" style="margin-top:8px;">Copy Token</button>
      </div>
    </div>

    <!-- Manage Tokens -->
    <div style="margin-top:30px;">
      <h3>Active API Tokens</h3>
      <button onclick="loadTokens()" style="margin-bottom:12px;">Refresh</button>
      <table class="table" id="tokensTable">
        <thead>
          <tr>
            <th>Token ID</th>
            <th>User ID</th>
            <th>Name</th>
            <th>Created</th>
            <th>Status</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody id="tokensList"></tbody>
      </table>
    </div>
  </div>

  <!-- SCAN CARD (only for regular users) -->
  <div class="card" id="scanCard" style="display:none;">
    <div class="flex-between">
      <h3 style="margin:0;">Eye Scan</h3>
      <button onclick="doLogout()" class="secondary">Logout</button>
    </div>

    <label>Eye mode</label>
    <select id="eyeMode" onchange="refreshInputs()">
      <option value="left">Left eye</option>
      <option value="right">Right eye</option>
      <option value="both">Both eyes</option>
    </select>

    <label>Source</label>
    <div class="tabs">
      <div class="tab active" id="tabUpload" onclick="setSource('upload')">Upload</div>
      <div class="tab" id="tabCamera" onclick="setSource('camera')">Camera</div>
    </div>
    <div class="hint">Upload = pick photo. Camera = open camera capture.</div>

    <div id="singleBox">
      <label>Image</label>
      <input id="singleFile" type="file" accept="image/*"/>
    </div>

    <div id="bothBox" style="display:none;">
      <div class="row">
        <div>
          <label>Left image</label>
          <input id="leftFile" type="file" accept="image/*"/>
        </div>
        <div>
          <label>Right image</label>
          <input id="rightFile" type="file" accept="image/*"/>
        </div>
      </div>
    </div>

    <div style="height:10px"></div>
    <button onclick="runScan()">Analyze</button>
    <p id="scanStatus" class="muted"></p>

    <div class="result" id="resultBox" style="display:none;">
      <div class="big">Diagnosis</div>
      <div id="diagText" class="diag" style="margin-top:10px;"></div>
    </div>
  </div>

</div>

<script>
let TOKEN = null;
let IS_OWNER = false;
let OWNER_EMAIL = "";
let OWNER_PASSWORD = "";
let SOURCE = "upload";

function setStatus(id, msg, ok=null){
  const el = document.getElementById(id);
  if(!el) return;
  el.textContent = msg;
  if(ok===true) el.className="muted ok";
  else if(ok===false) el.className="muted bad";
  else el.className="muted";
}

function showLogin(){
  document.getElementById("loginTab").style.display="block";
  document.getElementById("registerTab").style.display="none";
}

function showRegister(){
  document.getElementById("loginTab").style.display="none";
  document.getElementById("registerTab").style.display="block";
}

function setSource(src){
  SOURCE = src;
  document.getElementById("tabUpload").classList.toggle("active", src==="upload");
  document.getElementById("tabCamera").classList.toggle("active", src==="camera");
  applyCapture();
}

function applyCapture(){
  const cap = (SOURCE==="camera") ? "environment" : "";
  const inputs = ["singleFile","leftFile","rightFile"];
  for(const id of inputs){
    const el = document.getElementById(id);
    if(!el) continue;
    if(SOURCE==="camera"){
      el.setAttribute("capture", cap);
    } else {
      el.removeAttribute("capture");
    }
    el.value = "";
  }
}

function refreshInputs(){
  const mode = document.getElementById("eyeMode").value;
  document.getElementById("singleBox").style.display = (mode==="both") ? "none" : "block";
  document.getElementById("bothBox").style.display = (mode==="both") ? "block" : "none";
  applyCapture();
}

async function doRegister(){
  setStatus("authStatus","Registering…");
  const email = document.getElementById("regEmail").value.trim();
  const password = document.getElementById("regPassword").value;

  try{
    const r = await fetch("/auth/register",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({email, password})
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || "Registration failed");
    setStatus("authStatus","Registration OK! Now login. ✅",true);
    setTimeout(() => showLogin(), 1000);
  }catch(e){
    setStatus("authStatus","Registration failed: "+e.message,false);
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
    if(!r.ok) throw new Error(data.detail || "Login failed");
    TOKEN = data.access_token;
    IS_OWNER = data.is_owner || false;

    setStatus("authStatus","Login OK ✅",true);
    document.getElementById("authCard").style.display="none";
    
    if(IS_OWNER){
      OWNER_EMAIL = email;
      OWNER_PASSWORD = password;
      document.getElementById("adminCard").style.display="block";
      loadTokens();
    } else {
      document.getElementById("scanCard").style.display="block";
      refreshInputs();
    }
  }catch(e){
    TOKEN = null;
    setStatus("authStatus","Login failed: "+e.message,false);
  }
}

function doLogout(){
  TOKEN = null;
  IS_OWNER = false;
  OWNER_EMAIL = "";
  OWNER_PASSWORD = "";
  document.getElementById("authCard").style.display="block";
  document.getElementById("adminCard").style.display="none";
  document.getElementById("scanCard").style.display="none";
  document.getElementById("loginTab").style.display="block";
  document.getElementById("registerTab").style.display="none";
  document.getElementById("email").value = "";
  document.getElementById("password").value = "";
}

async function loadTokens(){
  if(!OWNER_EMAIL || !OWNER_PASSWORD) return;
  
  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;
  
  try{
    const r = await fetch("/admin/tokens", {
      headers: {"Authorization": authHeader}
    });
    if(!r.ok) throw new Error("Failed to load tokens");
    const tokens = await r.json();
    displayTokens(tokens);
  }catch(e){
    console.error("Error loading tokens:", e);
  }
}

function displayTokens(tokens){
  const tbody = document.getElementById("tokensList");
  tbody.innerHTML = "";
  
  if(!tokens || tokens.length === 0){
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;opacity:.5;">No tokens yet</td></tr>';
    return;
  }
  
  tokens.forEach(token => {
    const row = document.createElement("tr");
    const createdDate = new Date(token.created_at).toLocaleDateString();
    const status = token.is_active ? '<span class="badge active">Active</span>' : '<span class="badge inactive">Revoked</span>';
    
    row.innerHTML = `
      <td>${token.id}</td>
      <td>${token.user_id}</td>
      <td>${token.name || '-'}</td>
      <td>${createdDate}</td>
      <td>${status}</td>
      <td>
        ${token.is_active ? '<button class="danger" onclick="revokeToken(' + token.id + ')">Revoke</button>' : '-'}
      </td>
    `;
    tbody.appendChild(row);
  });
}

async function issueToken(){
  const userId = document.getElementById("tokenUserId").value;
  const tokenName = document.getElementById("tokenName").value;
  
  if(!userId){
    setStatus("tokenStatus","Please enter a user ID",false);
    return;
  }

  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;

  try{
    const r = await fetch("/admin/tokens/issue", {
      method: "POST",
      headers: {
        "Authorization": authHeader,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({user_id: parseInt(userId), name: tokenName || null})
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || "Failed to issue token");
    
    document.getElementById("tokenValue").value = data.token;
    document.getElementById("tokenDisplay").style.display = "block";
    document.getElementById("tokenUserId").value = "";
    document.getElementById("tokenName").value = "";
    setStatus("tokenStatus","Token created! ✅",true);
    
    setTimeout(() => loadTokens(), 500);
  }catch(e){
    setStatus("tokenStatus","Error: "+e.message,false);
  }
}

function copyToken(){
  const textarea = document.getElementById("tokenValue");
  textarea.select();
  document.execCommand("copy");
  alert("Token copied to clipboard!");
}

async function revokeToken(tokenId){
  if(!confirm("Revoke this token?")) return;
  
  const authHeader = "Bearer " + OWNER_EMAIL + ":" + OWNER_PASSWORD;

  try{
    const r = await fetch("/admin/tokens/" + tokenId, {
      method: "DELETE",
      headers: {"Authorization": authHeader}
    });
    if(!r.ok) throw new Error("Failed to revoke token");
    alert("Token revoked!");
    loadTokens();
  }catch(e){
    alert("Error: " + e.message);
  }
}

async function runScan(){
  if(!TOKEN){ setStatus("scanStatus","Please login first.",false); return; }

  const mode = document.getElementById("eyeMode").value;
  const fd = new FormData();
  fd.append("eye_mode", mode);

  if(mode==="both"){
    const lf = document.getElementById("leftFile").files?.[0];
    const rf = document.getElementById("rightFile").files?.[0];
    if(!lf || !rf){ setStatus("scanStatus","Please select both images.",false); return; }
    fd.append("left_file", lf);
    fd.append("right_file", rf);
  }else{
    const f = document.getElementById("singleFile").files?.[0];
    if(!f){ setStatus("scanStatus","Please select an image.",false); return; }
    fd.append("file", f);
  }

  setStatus("scanStatus","Analyzing…");
  document.getElementById("resultBox").style.display="none";

  try{
    const r = await fetch("/scan/run",{
      method:"POST",
      headers:{ "Authorization":"Bearer "+TOKEN },
      body: fd
    });
    const data = await r.json();
    if(!r.ok) throw new Error(data.detail || "Scan failed");

    if(data.status !== "done"){
      setStatus("scanStatus","Failed ❌",false);
      document.getElementById("diagText").textContent = data.error || "Unknown error";
    }else{
      setStatus("scanStatus","Done ✅",true);

      let txt = "";
      if(data.eye_mode === "both"){
        txt = "Left: " + (data.left_diagnosis || "-") + "\\nRight: " + (data.right_diagnosis || "-");
      }else if(data.eye_mode === "left"){
        txt = (data.left_diagnosis || "-");
      }else{
        txt = (data.right_diagnosis || "-");
      }

      document.getElementById("diagText").textContent = txt;
    }

    document.getElementById("resultBox").style.display="block";
  }catch(e){
    setStatus("scanStatus","Error: "+e.message,false);
  }
}
</script>
</body>
</html>
"""


# ============ AUTH ENDPOINTS ============

@app.post("/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user with email and optional password"""
    email = (body.email or "").strip().lower()
    
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    
    # Check if user exists
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Password can be empty initially (admin will issue tokens)
    password = (body.password or "").strip()
    if not password:
        password = "placeholder"  # Set a placeholder password
    
    password = password[:72]  # passlib max 72 bytes
    
    # Create user
    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Return a JWT token so they can test immediately (optional)
    return {"access_token": create_token(user.id)}


@app.post("/auth/login")
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Login with email and password"""
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if this is the owner
    owner_email = (settings.OWNER_EMAIL or "").strip().lower()
    is_owner = (email == owner_email)
    
    return {
        "access_token": create_token(user.id),
        "is_owner": is_owner,
        "user_id": user.id,
        "email": user.email
    }


# ============ ADMIN TOKEN MANAGEMENT ============

@app.post("/admin/tokens/issue", response_model=IssuedTokenResponse)
def issue_token(
    body: IssueTokenRequest, 
    req: Request,
    db: Session = Depends(get_db)
):
    """
    [ADMIN ONLY] Issue a new API token to a user.
    
    Authentication: 
      Authorization: Bearer email:password
      where email/password is OWNER_EMAIL/OWNER_PASSWORD from config
    """
    # Verify admin
    require_admin(req)
    
    # Check user exists
    user = db.query(User).filter(User.id == body.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Generate token
    plain_token = generate_api_token()
    token_hash = hash_api_token(plain_token)
    
    # Store in database
    api_token = APIToken(
        user_id=body.user_id,
        token_hash=token_hash,
        name=body.name,
    )
    db.add(api_token)
    db.commit()
    db.refresh(api_token)
    
    return IssuedTokenResponse(
        token=plain_token,  # Only shown once!
        token_id=api_token.id,
        user_id=api_token.user_id,
        created_at=api_token.created_at,
    )


@app.get("/admin/tokens", response_model=list[APITokenResponse])
def list_tokens(req: Request, db: Session = Depends(get_db)):
    """
    [ADMIN ONLY] List all API tokens in the system.
    
    Authentication:
      Authorization: Bearer email:password
    """
    require_admin(req)
    
    tokens = db.query(APIToken).all()
    return tokens


@app.delete("/admin/tokens/{token_id}")
def revoke_token(token_id: int, req: Request, db: Session = Depends(get_db)):
    """
    [ADMIN ONLY] Revoke (deactivate) an API token.
    
    Authentication:
      Authorization: Bearer email:password
    """
    require_admin(req)
    
    token = db.query(APIToken).filter(APIToken.id == token_id).first()
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    
    token.is_active = 0
    db.add(token)
    db.commit()
    
    return {"status": "revoked", "token_id": token_id}


# ============ SCAN ENDPOINT (SUPPORTS BOTH JWT AND API TOKENS) ============

@app.post("/scan/run", response_model=ScanResult)
async def scan_run(
    req: Request,
    eye_mode: str = Form(...),
    file: UploadFile | None = File(None),
    left_file: UploadFile | None = File(None),
    right_file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    """Run a scan. Accepts both JWT tokens and API tokens."""
    
    # Extract token
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    
    user_id = None
    
    # Try JWT first
    try:
        from jose import jwt
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = int(payload.get("sub", 0))
    except:
        pass
    
    # If JWT failed, try API token
    if not user_id:
        api_token = db.query(APIToken).filter(APIToken.token_hash == hash_api_token(token)).first()
        if not api_token or not api_token.is_active:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = api_token.user_id
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    eye_mode = (eye_mode or "").strip().lower()
    if eye_mode not in ("left", "right", "both"):
        raise HTTPException(400, detail="eye_mode must be left/right/both")

    try:
        if eye_mode == "both":
            if left_file is None or right_file is None:
                raise HTTPException(400, detail="left_file and right_file required")

            left_id = new_upload_id()
            right_id = new_upload_id()

            left_bytes = await left_file.read()
            right_bytes = await right_file.read()

            left_key = key_for("left", left_id)
            right_key = key_for("right", right_id)

            put_bytes(left_key, left_bytes, left_file.content_type or "image/jpeg")
            put_bytes(right_key, right_bytes, right_file.content_type or "image/jpeg")

            left_dbg = predict_debug(left_bytes)
            right_dbg = predict_debug(right_bytes)

            left_diag = left_dbg.get("translated") or "Uncertain"
            right_diag = right_dbg.get("translated") or "Uncertain"

            print(
                "[AI RAW BOTH]",
                "L_sha=", _sha256(left_bytes)[:12],
                "L_final=", left_dbg.get("final_code"), left_dbg.get("final_reason"),
                "L_top3=", left_dbg.get("top3"),
                "| R_sha=", _sha256(right_bytes)[:12],
                "R_final=", right_dbg.get("final_code"), right_dbg.get("final_reason"),
                "R_top3=", right_dbg.get("top3"),
            )

            scan = Scan(
                user_id=user_id,
                eye_mode="both",
                left_key=left_key,
                right_key=right_key,
                left_diagnosis=left_diag,
                right_diagnosis=right_diag,
                status="done",
            )
            db.add(scan)
            db.commit()
            db.refresh(scan)

            return ScanResult(
                id=scan.id,
                eye_mode=scan.eye_mode,
                left_diagnosis=scan.left_diagnosis,
                right_diagnosis=scan.right_diagnosis,
                status=scan.status,
                error=None,
            )

        # single eye
        if file is None:
            raise HTTPException(400, detail="file required for left/right")

        upload_id = new_upload_id()
        image_bytes = await file.read()

        r2_key = key_for(eye_mode, upload_id)
        put_bytes(r2_key, image_bytes, file.content_type or "image/jpeg")

        dbg = predict_debug(image_bytes)
        diag = dbg.get("translated") or "Uncertain"

        print(
            "[AI RAW ONE]",
            "mode=", eye_mode,
            "len=", len(image_bytes),
            "sha=", _sha256(image_bytes)[:12],
            "top_code=", dbg.get("top_code"),
            "top_prob=", dbg.get("top_prob"),
            "top3=", dbg.get("top3"),
            "final_code=", dbg.get("final_code"),
            "reason=", dbg.get("final_reason"),
            "final_diag=", diag,
        )

        scan = Scan(
            user_id=user_id,
            eye_mode=eye_mode,
            left_key=r2_key if eye_mode == "left" else None,
            right_key=r2_key if eye_mode == "right" else None,
            left_diagnosis=diag if eye_mode == "left" else None,
            right_diagnosis=diag if eye_mode == "right" else None,
            status="done",
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)

        return ScanResult(
            id=scan.id,
            eye_mode=scan.eye_mode,
            left_diagnosis=scan.left_diagnosis,
            right_diagnosis=scan.right_diagnosis,
            status=scan.status,
            error=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        scan = Scan(
            user_id=user_id,
            eye_mode=eye_mode,
            status="failed",
            error=f"{type(e).__name__}: {str(e)}",
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)

        detail = scan.error if str(getattr(settings, "DEBUG_ERRORS", "")).strip() == "1" else "Scan failed"
        return JSONResponse(
            status_code=500,
            content={
                "id": scan.id,
                "eye_mode": scan.eye_mode,
                "left_diagnosis": None,
                "right_diagnosis": None,
                "status": scan.status,
                "error": detail,
            },
        )