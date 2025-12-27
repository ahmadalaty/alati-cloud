from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from .db import Base, engine, get_db, SessionLocal
from .models import User, Scan, ScanStatus
from .schemas import LoginRequest, TokenResponse, ScanCreateRequest, ScanResponse
from .auth import hash_password, verify_password, create_token, require_user
from .storage import new_upload_id, save_upload, upload_target_path
from .tasks import process_scan
import os

app = FastAPI(title="Alati Cloud API")

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)


def _upsert_user(db: Session, email: str, password: str, make_admin: bool = False):
    email = (email or "").strip().lower()
    password = (password or "").strip()
    if not email or not password:
        return

    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, password_hash=hash_password(password))
        if make_admin and hasattr(user, "is_admin"):
            user.is_admin = True
        db.add(user)
        db.commit()
        return

    user.password_hash = hash_password(password)
    if make_admin and hasattr(user, "is_admin"):
        user.is_admin = True
    db.add(user)
    db.commit()


@app.on_event("startup")
def seed():
    db = SessionLocal()
    try:
        _upsert_user(db, "admin@alati.ai", "admin123", make_admin=True)

        owner_email = os.getenv("OWNER_EMAIL", "").strip()
        owner_password = os.getenv("OWNER_PASSWORD", "").strip()
        if owner_email and owner_password:
            _upsert_user(db, owner_email, owner_password, make_admin=True)
    finally:
        db.close()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug")
def debug_info():
    """
    SAFE debug endpoint: shows env flags and basic runtime checks.
    DOES NOT return secrets.
    """
    demo_mode = os.getenv("DEMO_MODE", "").strip()
    debug_errors = os.getenv("DEBUG_ERRORS", "").strip()

    # Check torch import
    torch_ok = True
    torch_version = None
    torch_error = None
    try:
        import torch  # noqa
        torch_version = getattr(torch, "__version__", None)
    except Exception as e:
        torch_ok = False
        torch_error = f"{type(e).__name__}: {str(e)}"

    # Check model file presence (both API + worker usually have same paths in repo)
    base_dir = os.path.dirname(__file__)
    model_dir = os.path.join(base_dir, "model_files")
    res18 = os.path.join(model_dir, "alati_dualeye_model_resnet18.pth")
    res50 = os.path.join(model_dir, "alati_dualeye_model_resnet50.pth")
    labels = os.path.join(model_dir, "labels.json")

    return {
        "demo_mode": demo_mode,
        "debug_errors": debug_errors,
        "torch_ok": torch_ok,
        "torch_version": torch_version,
        "torch_error": torch_error,
        "model_dir": model_dir,
        "resnet18_exists": os.path.exists(res18),
        "resnet50_exists": os.path.exists(res50),
        "labels_exists": os.path.exists(labels),
    }


@app.get("/", response_class=HTMLResponse)
def presentation_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Alati Cloud</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #0b1020; color: #e8ecff; }
    .wrap { max-width: 720px; margin: 0 auto; padding: 28px 16px 48px; }
    .card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 16px; margin: 14px 0; }
    h1 { font-size: 28px; margin: 0 0 6px; }
    .sub { opacity: 0.85; margin: 0 0 18px; }
    label { display:block; font-size: 13px; opacity: 0.85; margin: 10px 0 6px; }
    input, select, button { width: 100%; padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.14); background: rgba(0,0,0,0.25); color: #e8ecff; font-size: 15px; }
    button { cursor: pointer; background: #355dff; border: 0; font-weight: 700; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .row { display:flex; gap: 12px; }
    .row > div { flex: 1; }
    .muted { opacity: 0.8; font-size: 13px; }
    pre { white-space: pre-wrap; word-break: break-word; background: rgba(0,0,0,0.35); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.12); }
    a { color: #9fb3ff; }
    .ok { color: #67ffb1; font-weight: 700; }
    .bad { color: #ff8a8a; font-weight: 700; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Alati Cloud</h1>
    <p class="sub">Live fundus AI demo. Login → upload → analyze → report.</p>

    <div class="card">
      <h3 style="margin:0 0 8px;">1) Login</h3>
      <label>Email</label>
      <input id="email" placeholder="ahmadalaty@gmail.com" />
      <label>Password</label>
      <input id="password" type="password" placeholder="••••••••" />
      <div style="height:10px"></div>
      <button id="btnLogin" onclick="doLogin()">Login</button>
      <p id="loginStatus" class="muted"></p>
    </div>

    <div class="card">
      <h3 style="margin:0 0 8px;">2) Upload fundus image</h3>
      <label>Select image</label>
      <input id="file" type="file" accept="image/*" />
      <div class="row">
        <div>
          <label>Eye</label>
          <select id="eye">
            <option value="left">Left</option>
            <option value="right">Right</option>
          </select>
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="btnAnalyze" onclick="doAnalyze()" disabled>Analyze</button>
        </div>
      </div>
      <p class="muted">Tip: On iPhone/Android, choose “Take Photo” for live demo.</p>
    </div>

    <div class="card">
      <h3 style="margin:0 0 8px;">3) Result</h3>
      <p id="resultStatus" class="muted">Waiting…</p>
      <pre id="resultBox">—</pre>
      <p id="reportLink" class="muted"></p>
      <p class="muted">Debug: <a href="/debug" target="_blank">/debug</a></p>
    </div>

    <p class="muted">API docs: <a href="/docs" target="_blank">/docs</a> • Health: <a href="/health" target="_blank">/health</a></p>
  </div>

<script>
let TOKEN = null;

function setStatus(el, msg, ok=null) {
  el.textContent = msg;
  if (ok === true) el.className = "muted ok";
  else if (ok === false) el.className = "muted bad";
  else el.className = "muted";
}

async function doLogin() {
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;
  const st = document.getElementById("loginStatus");
  setStatus(st, "Logging in…");

  try {
    const r = await fetch("/auth/login", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({email, password})
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Login failed");
    TOKEN = data.access_token;
    setStatus(st, "Login OK ✅", true);
    document.getElementById("btnAnalyze").disabled = false;
  } catch (e) {
    TOKEN = null;
    document.getElementById("btnAnalyze").disabled = true;
    setStatus(st, "Login failed: " + e.message, false);
  }
}

async function doAnalyze() {
  const fileInput = document.getElementById("file");
  const eye = document.getElementById("eye").value;
  const rs = document.getElementById("resultStatus");
  const rb = document.getElementById("resultBox");
  const rl = document.getElementById("reportLink");

  rl.textContent = "";
  rb.textContent = "—";

  if (!TOKEN) { setStatus(rs, "Please login first.", false); return; }
  if (!fileInput.files || !fileInput.files[0]) { setStatus(rs, "Please select an image.", false); return; }

  setStatus(rs, "Uploading image…");

  try {
    const form = new FormData();
    form.append("file", fileInput.files[0]);

    const up = await fetch("/upload", {
      method: "POST",
      headers: { "Authorization": "Bearer " + TOKEN },
      body: form
    });

    const upText = await up.text();
    let upData;
    try { upData = JSON.parse(upText); } catch(e) { throw new Error(upText); }
    if (!up.ok) throw new Error(upData.detail || "Upload failed");

    setStatus(rs, "Running AI…");

    const sc = await fetch("/scan/create", {
      method: "POST",
      headers: {
        "Content-Type":"application/json",
        "Authorization":"Bearer " + TOKEN
      },
      body: JSON.stringify({ upload_id: upData.upload_id, eye })
    });

    const scText = await sc.text();

    // Try parse JSON; if not JSON, show raw text (this is your current issue)
    let scData = null;
    try { scData = JSON.parse(scText); } catch(e) {
      throw new Error(scText);
    }

    if (!sc.ok) throw new Error(scData.detail || "Scan failed");

    if (scData.status === "done") {
      setStatus(rs, "Done ✅", true);
      rb.textContent = JSON.stringify(scData.result, null, 2);
      if (scData.report_url) {
        rl.innerHTML = `Report: <a href="${scData.report_url}" target="_blank">Download PDF</a>`;
      }
      return;
    }

    setStatus(rs, "Queued… (non-demo mode)");
  } catch (e) {
    setStatus(rs, "Error: " + e.message, false);
  }
}
</script>
</body>
</html>
"""


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token(user.id)}


@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user_id: int = Depends(require_user),
):
    upload_id = new_upload_id()
    path = await save_upload(upload_id, file)
    return {"upload_id": upload_id, "path": path}


@app.post("/scan/create", response_model=ScanResponse)
def create_scan(
    body: ScanCreateRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(require_user),
):
    image_path = upload_target_path(body.upload_id)
    if not os.path.exists(image_path):
        raise HTTPException(400, "Upload not found")

    scan = Scan(
        user_id=user_id,
        eye=body.eye,
        image_path=image_path,
        status=ScanStatus.queued,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    if os.getenv("DEMO_MODE", "").strip() == "1":
        try:
            process_scan.run(scan.id)
        except Exception as e:
            if os.getenv("DEBUG_ERRORS", "").strip() == "1":
                raise HTTPException(500, f"AI processing failed: {type(e).__name__}: {str(e)}")
            raise HTTPException(500, "AI processing failed")

        db.refresh(scan)
        report_url = f"/scan/{scan.id}/report" if scan.report_path and scan.status == ScanStatus.done else None
        return {"id": scan.id, "status": scan.status.value, "result": scan.result, "report_url": report_url}

    process_scan.delay(scan.id)
    return {"id": scan.id, "status": scan.status.value, "result": None, "report_url": None}


@app.get("/scan/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: int, db: Session = Depends(get_db), user_id: int = Depends(require_user)):
    scan = db.get(Scan, scan_id)
    if not scan or scan.user_id != user_id:
        raise HTTPException(404, "Not found")

    report_url = f"/scan/{scan.id}/report" if scan.report_path and scan.status == ScanStatus.done else None
    return {"id": scan.id, "status": scan.status.value, "result": scan.result, "report_url": report_url}


@app.get("/scan/{scan_id}/report")
def get_report(scan_id: int, db: Session = Depends(get_db), user_id: int = Depends(require_user)):
    scan = db.get(Scan, scan_id)
    if not scan or scan.user_id != user_id or not scan.report_path:
        raise HTTPException(404, "Not found")

    from fastapi.responses import FileResponse
    return FileResponse(scan.report_path, media_type="application/pdf", filename=f"alati_report_{scan_id}.pdf")
