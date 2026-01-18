import os
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from .config import settings
from .db import Base, engine, get_db, SessionLocal
from .models import User, Scan
from .schemas import LoginRequest, TokenResponse, ScanResult
from .auth import hash_password, verify_password, create_token, require_user
from .storage_r2 import new_upload_id, key_for, put_bytes, BUILD_MARKER as STORAGE_MARKER
from .inference import predict_diagnosis, BUILD_MARKER as INF_MARKER, ACTIVE_VARIANT

app = FastAPI(title="Alati Cloud Demo (No Worker)")
Base.metadata.create_all(bind=engine)


# ----------------------------
# Helpers
# ----------------------------
def diagnosis_to_text(diag):
    if diag is None:
        return None
    if isinstance(diag, dict):
        name = diag.get("name") or diag.get("label") or diag.get("diagnosis") or diag.get("code") or "unknown"
    else:
        name = str(diag)
    return name.replace("_", " ").strip()


def upsert_admin():
    db = SessionLocal()
    try:
        email = (settings.OWNER_EMAIL or "").strip().lower()
        password = (settings.OWNER_PASSWORD or "").strip()
        if not email or not password:
            return

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
        "demo_mode": settings.DEMO_MODE,
    }


# ----------------------------
# UI
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Alati Demo</title>
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#0b1020;color:#e8ecff;}
    .wrap{max-width:760px;margin:0 auto;padding:28px 16px 48px;}
    .card{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:16px;margin:14px 0;}
    h1{font-size:28px;margin:0 0 6px;}
    .sub{opacity:.85;margin:0 0 18px;}
    label{display:block;font-size:13px;opacity:.85;margin:10px 0 6px;}
    input,select,button{width:100%;padding:12px;border-radius:12px;border:1px solid rgba(255,255,255,.14);background:rgba(0,0,0,.25);color:#e8ecff;font-size:15px;}
    button{cursor:pointer;background:#355dff;border:0;font-weight:700;}
    button:disabled{opacity:.6;cursor:not-allowed;}
    .row{display:flex;gap:12px;align-items:flex-end;}
    .row>div{flex:1;}
    .muted{opacity:.8;font-size:13px;}
    .ok{color:#67ffb1;font-weight:700;}
    .bad{color:#ff8a8a;font-weight:700;}
    .result{padding:14px;border-radius:14px;background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.12);margin-top:10px;}
    .big{font-size:18px;font-weight:800;}
    a{color:#9fb3ff;}

    .tabs{display:flex;gap:10px;margin-top:10px;}
    .tab{
      flex:1;
      padding:12px;
      border-radius:12px;
      text-align:center;
      border:1px solid rgba(255,255,255,.14);
      background:rgba(0,0,0,.25);
      cursor:pointer;
      font-weight:700;
      user-select:none;
    }
    .tab.active{
      background:#355dff;
      border-color:#355dff;
    }

    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
    @media (max-width: 640px){
      .grid2{grid-template-columns:1fr;}
    }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Alati Cloud Demo</h1>
  <p class="sub">Login → choose eye → choose input mode → diagnosis only.</p>

  <div class="card">
    <h3 style="margin:10px 0 8px;">1) Login</h3>
    <label>Email</label>
    <input id="email" placeholder="admin@alati.ai"/>
    <label>Password</label>
    <input id="password" type="password" placeholder="••••••••"/>
    <div style="height:10px"></div>
    <button onclick="doLogin()">Login</button>
    <p id="loginStatus" class="muted"></p>
    <p class="muted">Debug: <a href="/debug" target="_blank">/debug</a></p>
  </div>

  <div class="card" id="scanCard" style="display:none;">
    <h3 style="margin:0 0 8px;">2) Scan</h3>

    <label>Eye mode</label>
    <div class="tabs">
      <div class="tab active" id="eye_left" onclick="setEyeMode('left')">Left</div>
      <div class="tab" id="eye_right" onclick="setEyeMode('right')">Right</div>
      <div class="tab" id="eye_both" onclick="setEyeMode('both')">Both</div>
    </div>

    <label style="margin-top:14px;">Input mode</label>
    <div class="tabs">
      <div class="tab active" id="input_capture" onclick="setInputMode('capture')">Capture</div>
      <div class="tab" id="input_upload" onclick="setInputMode('upload')">Upload</div>
    </div>

    <div style="height:12px"></div>

    <!-- Single file input -->
    <div id="singleBox">
      <label id="singleLabel">Image</label>
      <input id="singleFile" type="file" accept="image/*" capture="environment"/>
      <p class="muted" id="singleHint">Capture opens camera on mobile. Upload opens file picker.</p>
    </div>

    <!-- Both eyes input -->
    <div id="bothBox" style="display:none;">
      <div class="grid2">
        <div>
          <label>Left image</label>
          <input id="leftFile" type="file" accept="image/*" capture="environment"/>
        </div>
        <div>
          <label>Right image</label>
          <input id="rightFile" type="file" accept="image/*" capture="environment"/>
        </div>
      </div>
      <p class="muted" id="bothHint">Choose capture or upload above.</p>
    </div>

    <div style="height:10px"></div>
    <button onclick="runScan()">Analyze</button>
    <p id="scanStatus" class="muted"></p>

    <div class="result" id="resultBox" style="display:none;">
      <div class="big">Diagnosis</div>
      <div id="diagText" style="margin-top:8px;white-space:pre-wrap;"></div>
    </div>
  </div>
</div>

<script>
let TOKEN = null;
let EYE_MODE = "left";
let INPUT_MODE = "capture";

function setTab(groupPrefix, value){
  ["left","right","both","capture","upload"].forEach(v=>{
    const el = document.getElementById(groupPrefix + "_" + v);
    if(el) el.classList.remove("active");
  });
  const active = document.getElementById(groupPrefix + "_" + value);
  if(active) active.classList.add("active");
}

function setEyeMode(mode){
  EYE_MODE = mode;
  setTab("eye", mode);
  renderInputs();
}

function setInputMode(mode){
  INPUT_MODE = mode;
  setTab("input", mode);
  renderInputs();
}

function renderInputs(){
  const singleBox = document.getElementById("singleBox");
  const bothBox = document.getElementById("bothBox");
  const singleFile = document.getElementById("singleFile");
  const leftFile = document.getElementById("leftFile");
  const rightFile = document.getElementById("rightFile");

  // Reset selected files when switching modes (prevents accidental reuse)
  singleFile.value = "";
  leftFile.value = "";
  rightFile.value = "";

  // Switch capture attribute
  if(INPUT_MODE === "capture"){
    singleFile.setAttribute("capture","environment");
    leftFile.setAttribute("capture","environment");
    rightFile.setAttribute("capture","environment");
  }else{
    singleFile.removeAttribute("capture");
    leftFile.removeAttribute("capture");
    rightFile.removeAttribute("capture");
  }

  // show correct input layout
  if(EYE_MODE === "both"){
    singleBox.style.display = "none";
    bothBox.style.display = "block";
  }else{
    singleBox.style.display = "block";
    bothBox.style.display = "none";
    document.getElementById("singleLabel").textContent =
      (EYE_MODE === "left" ? "Left image" : "Right image");
  }

  document.getElementById("singleHint").textContent =
    INPUT_MODE === "capture"
      ? "Capture opens the camera on mobile."
      : "Upload opens file picker (camera still possible via phone gallery).";
}

function setStatus(id, msg, ok=null){
  const el = document.getElementById(id);
  el.textContent = msg;
  if(ok===true) el.className="muted ok";
  else if(ok===false) el.className="muted bad";
  else el.className="muted";
}

async function doLogin(){
  setStatus("loginStatus","Logging in…");
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

    setStatus("loginStatus","Login OK ✅",true);
    document.getElementById("scanCard").style.display="block";
    renderInputs();
  }catch(e){
    TOKEN = null;
    setStatus("loginStatus","Login failed: "+e.message,false);
  }
}

async function runScan(){
  if(!TOKEN){ setStatus("scanStatus","Please login first.",false); return; }

  const fd = new FormData();
  fd.append("eye_mode", EYE_MODE);

  if(EYE_MODE === "both"){
    const lf = document.getElementById("leftFile").files?.[0];
    const rf = document.getElementById("rightFile").files?.[0];
    if(!lf || !rf){ setStatus("scanStatus","Please provide BOTH images.",false); return; }
    fd.append("left_file", lf);
    fd.append("right_file", rf);
  }else{
    const f = document.getElementById("singleFile").files?.[0];
    if(!f){ setStatus("scanStatus","Please provide an image.",false); return; }
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


# ----------------------------
# API
# ----------------------------
@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_token(user.id)}


@app.post("/scan/run", response_model=ScanResult)
async def scan_run(
    eye_mode: str = Form(...),
    file: UploadFile | None = File(None),
    left_file: UploadFile | None = File(None),
    right_file: UploadFile | None = File(None),
    user_id: int = Depends(require_user),
    db: Session = Depends(get_db),
):
    eye_mode = (eye_mode or "").strip().lower()
    if eye_mode not in ("left", "right", "both"):
        raise HTTPException(400, detail="eye_mode must be left/right/both")

    try:
        if eye_mode == "both":
            if left_file is None or right_file is None:
                raise HTTPException(400, detail="left_file and right_file are required for both")

            left_id = new_upload_id()
            right_id = new_upload_id()

            left_bytes = await left_file.read()
            right_bytes = await right_file.read()

            left_key = key_for("left", left_id)
            right_key = key_for("right", right_id)

            put_bytes(left_key, left_bytes, left_file.content_type or "image/jpeg")
            put_bytes(right_key, right_bytes, right_file.content_type or "image/jpeg")

            left_diag = diagnosis_to_text(predict_diagnosis(left_bytes))
            right_diag = diagnosis_to_text(predict_diagnosis(right_bytes))

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
            raise HTTPException(400, detail="file is required for left/right")

        upload_id = new_upload_id()
        image_bytes = await file.read()

        r2_key = key_for(eye_mode, upload_id)
        put_bytes(r2_key, image_bytes, file.content_type or "image/jpeg")

        diag = diagnosis_to_text(predict_diagnosis(image_bytes))

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

        detail = scan.error if str(settings.DEBUG_ERRORS).strip() == "1" else "Scan failed"

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
