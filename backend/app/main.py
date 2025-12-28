import os
import json
import time
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from .db import Base, engine, get_db, SessionLocal
from .models import User, Scan, ScanStatus
from .schemas import LoginRequest, TokenResponse, ScanCreateRequest, ScanResponse
from .auth import hash_password, verify_password, create_token, require_user
from .storage import new_upload_id, save_upload, upload_target_path
from .tasks import process_scan

app = FastAPI(title="Alati Cloud API")
Base.metadata.create_all(bind=engine)


# ----------------------------
# Helpers
# ----------------------------

def make_json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return make_json_safe(obj.tolist())
    except Exception:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return float(obj.detach().cpu().item())
            return make_json_safe(obj.detach().cpu().tolist())
    except Exception:
        pass
    return str(obj)


def decode_db_result(val):
    # result might be json string OR dict OR null
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {"raw": val}
    return {"raw": str(val)}


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


# ----------------------------
# Startup seed
# ----------------------------
@app.on_event("startup")
def seed():
    db = SessionLocal()
    try:
        # default demo admin
        _upsert_user(db, "admin@alati.ai", "admin123", make_admin=True)

        # owner account from env
        owner_email = os.getenv("OWNER_EMAIL", "").strip()
        owner_password = os.getenv("OWNER_PASSWORD", "").strip()
        if owner_email and owner_password:
            _upsert_user(db, owner_email, owner_password, make_admin=True)
    finally:
        db.close()


# ----------------------------
# Health / Debug
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug")
def debug_info():
    demo_mode = os.getenv("DEMO_MODE", "").strip()
    debug_errors = os.getenv("DEBUG_ERRORS", "").strip()

    torch_ok = True
    torch_version = None
    torch_error = None
    try:
        import torch
        torch_version = getattr(torch, "__version__", None)
    except Exception as e:
        torch_ok = False
        torch_error = f"{type(e).__name__}: {str(e)}"

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


# ----------------------------
# UI
# ----------------------------
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
    .wrap { max-width: 860px; margin: 0 auto; padding: 28px 16px 48px; }
    .top { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; }
    h1 { font-size: 30px; margin: 0; letter-spacing: 0.2px; }
    .sub { opacity: 0.85; margin: 6px 0 0; }
    .card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 18px; padding: 16px; margin: 14px 0; }
    label { display:block; font-size: 13px; opacity: 0.85; margin: 10px 0 6px; }
    input, select, button { width: 100%; padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.14); background: rgba(0,0,0,0.25); color: #e8ecff; font-size: 15px; }
    button { cursor: pointer; background: #355dff; border: 0; font-weight: 800; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .row { display:flex; gap: 12px; }
    .row > div { flex: 1; }
    .muted { opacity: 0.8; font-size: 13px; }
    .ok { color: #67ffb1; font-weight: 800; }
    .bad { color: #ff8a8a; font-weight: 800; }
    .grid { display:grid; grid-template-columns: 1.2fr 0.8fr; gap: 12px; }
    @media (max-width: 860px){ .grid { grid-template-columns: 1fr; } }

    .pill { display:inline-flex; align-items:center; gap:8px; padding: 8px 10px; border-radius: 999px;
            background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); font-size: 13px; }
    .pill strong { font-weight: 900; }
    .triage { font-size: 14px; font-weight: 900; padding: 8px 10px; border-radius: 12px; display:inline-block; }
    .triage.refer { background: rgba(255, 66, 66, 0.18); border: 1px solid rgba(255, 66, 66, 0.35); }
    .triage.routine { background: rgba(103, 255, 177, 0.14); border: 1px solid rgba(103, 255, 177, 0.30); }

    .resultCard { background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 14px; }
    .bar { height: 10px; border-radius: 999px; background: rgba(255,255,255,0.10); overflow:hidden; border: 1px solid rgba(255,255,255,0.10); }
    .bar > div { height: 100%; background: rgba(53,93,255,0.9); width: 0%; transition: width 250ms ease; }
    .probRow { display:flex; align-items:center; justify-content:space-between; gap:12px; margin: 10px 0; }
    .probName { font-weight: 700; }
    .probVal { font-variant-numeric: tabular-nums; opacity: 0.9; }
    .small { font-size: 12px; opacity: 0.8; }
    a { color: #9fb3ff; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <h1>Alati Cloud</h1>
        <div class="sub">Fundus AI demo: login → upload → analyze → results (no PDF needed).</div>
      </div>
      <div class="pill"><strong>Mode:</strong> Cloud</div>
    </div>

    <div class="grid">
      <div class="card">
        <h3 style="margin:0 0 10px;">1) Login</h3>
        <label>Email</label>
        <input id="email" placeholder="ahmadalaty@gmail.com" />
        <label>Password</label>
        <input id="password" type="password" placeholder="••••••••" />
        <div style="height:10px"></div>
        <button id="btnLogin" onclick="doLogin()">Login</button>
        <p id="loginStatus" class="muted"></p>
      </div>

      <div class="card">
        <h3 style="margin:0 0 10px;">2) Upload</h3>

        <label>Workflow</label>
        <select id="mode">
          <option value="single">Single eye</option>
          <option value="both">Both eyes (Left + Right)</option>
        </select>

        <div id="singleBlock">
          <label>Eye</label>
          <select id="eyeSingle">
            <option value="left">Left</option>
            <option value="right">Right</option>
          </select>

          <label>Fundus image</label>
          <input id="fileSingle" type="file" accept="image/*" />
        </div>

        <div id="bothBlock" style="display:none;">
          <label>Left image</label>
          <input id="fileLeft" type="file" accept="image/*" />
          <label>Right image</label>
          <input id="fileRight" type="file" accept="image/*" />
        </div>

        <div style="height:12px"></div>
        <button id="btnAnalyze" onclick="doAnalyze()" disabled>Analyze</button>

        <p class="muted" style="margin-top:10px;">
          Debug: <a href="/debug" target="_blank">/debug</a> • Docs: <a href="/docs" target="_blank">/docs</a>
        </p>
      </div>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px;">3) Results</h3>
      <p id="resultStatus" class="muted">Waiting…</p>
      <div id="resultsContainer" style="display:grid; gap: 12px;"></div>
      <p class="small">Tip: For best demo results, use clear centered fundus images with minimal glare.</p>
    </div>
  </div>

<script>
let TOKEN = null;

document.getElementById("mode").addEventListener("change", () => {
  const m = document.getElementById("mode").value;
  document.getElementById("singleBlock").style.display = (m === "single") ? "block" : "none";
  document.getElementById("bothBlock").style.display = (m === "both") ? "block" : "none";
});

function setStatus(el, msg, ok=null) {
  el.textContent = msg;
  if (ok === true) el.className = "muted ok";
  else if (ok === false) el.className = "muted bad";
  else el.className = "muted";
}

function labelNice(s) {
  if (!s) return "";
  return s.replaceAll("_"," ").replace(/\b\w/g, c => c.toUpperCase());
}

function triageClass(t) {
  return (String(t||"").toLowerCase() === "refer") ? "refer" : "routine";
}

function fmtPct(x) {
  const v = Number(x);
  if (!isFinite(v)) return "-";
  return (v*100).toFixed(1) + "%";
}

function renderResultCard(title, data) {
  // data shape:
  // { model_variant, eye, probs, top_label, top_prob, triage }
  const probs = data && data.probs ? data.probs : {};
  const entries = Object.entries(probs).sort((a,b)=> (Number(b[1]) - Number(a[1])));

  let html = `
    <div class="resultCard">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
        <div class="pill"><strong>${title}</strong></div>
        <div class="pill"><strong>Model:</strong> ${data.model_variant || "-"}</div>
        <div class="pill"><strong>Top:</strong> ${labelNice(data.top_label)} (${fmtPct(data.top_prob)})</div>
        <div class="triage ${triageClass(data.triage)}">${data.triage || "—"}</div>
      </div>
      <div style="height:10px"></div>
  `;

  for (const [k,v] of entries) {
    const pct = Math.max(0, Math.min(100, Number(v)*100));
    html += `
      <div class="probRow">
        <div class="probName">${labelNice(k)}</div>
        <div class="probVal">${fmtPct(v)}</div>
      </div>
      <div class="bar"><div style="width:${pct}%;"></div></div>
    `;
  }

  html += `</div>`;
  return html;
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

async function uploadOne(fileObj) {
  const form = new FormData();
  form.append("file", fileObj);

  const up = await fetch("/upload", {
    method: "POST",
    headers: { "Authorization": "Bearer " + TOKEN },
    body: form
  });

  const upText = await up.text();
  let upData;
  try { upData = JSON.parse(upText); } catch(e) { throw new Error(upText); }
  if (!up.ok) throw new Error(upData.detail || "Upload failed");
  return upData.upload_id;
}

async function createScan(upload_id, eye) {
  const sc = await fetch("/scan/create", {
    method: "POST",
    headers: {
      "Content-Type":"application/json",
      "Authorization":"Bearer " + TOKEN
    },
    body: JSON.stringify({ upload_id, eye })
  });

  const scText = await sc.text();
  let scData;
  try { scData = JSON.parse(scText); } catch(e) { throw new Error(scText); }
  if (!sc.ok) throw new Error(scData.detail || "Scan failed");
  return scData; // {id,status,result,report_url}
}

async function pollScan(scan_id, timeoutMs=60000, intervalMs=1200) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const r = await fetch(`/scan/${scan_id}`, {
      headers: { "Authorization": "Bearer " + TOKEN }
    });
    const t = await r.text();
    let data;
    try { data = JSON.parse(t); } catch(e) { throw new Error(t); }
    if (!r.ok) throw new Error(data.detail || "Failed to fetch scan");
    if (data.status === "done" || data.status === "failed") return data;
    await new Promise(res => setTimeout(res, intervalMs));
  }
  return { status: "queued", result: null };
}

async function doAnalyze() {
  const rs = document.getElementById("resultStatus");
  const container = document.getElementById("resultsContainer");
  container.innerHTML = "";

  if (!TOKEN) { setStatus(rs, "Please login first.", false); return; }

  const mode = document.getElementById("mode").value;

  try {
    setStatus(rs, "Uploading image(s)…");

    if (mode === "single") {
      const eye = document.getElementById("eyeSingle").value;
      const f = document.getElementById("fileSingle");
      if (!f.files || !f.files[0]) { setStatus(rs, "Please select an image.", false); return; }

      const upload_id = await uploadOne(f.files[0]);
      setStatus(rs, "Running AI…");

      const scan = await createScan(upload_id, eye);

      if (scan.status !== "done") {
        setStatus(rs, "Queued… (waiting for worker)", true);
        const final = await pollScan(scan.id);
        if (final.status === "failed") throw new Error(final.result?.error || "AI failed");
        setStatus(rs, "Done ✅", true);
        container.innerHTML = renderResultCard(labelNice(eye) + " eye", final.result || {});
      } else {
        setStatus(rs, "Done ✅", true);
        container.innerHTML = renderResultCard(labelNice(eye) + " eye", scan.result || {});
      }

      return;
    }

    // both eyes
    const fl = document.getElementById("fileLeft");
    const fr = document.getElementById("fileRight");
    if (!fl.files || !fl.files[0]) { setStatus(rs, "Please select LEFT image.", false); return; }
    if (!fr.files || !fr.files[0]) { setStatus(rs, "Please select RIGHT image.", false); return; }

    const left_upload = await uploadOne(fl.files[0]);
    const right_upload = await uploadOne(fr.files[0]);

    setStatus(rs, "Running AI (Left)…");
    const left_scan = await createScan(left_upload, "left");

    setStatus(rs, "Running AI (Right)…");
    const right_scan = await createScan(right_upload, "right");

    // poll both
    setStatus(rs, "Queued… (waiting for worker)", true);

    const left_final = (left_scan.status === "done") ? left_scan : await pollScan(left_scan.id);
    const right_final = (right_scan.status === "done") ? right_scan : await pollScan(right_scan.id);

    if (left_final.status === "failed") throw new Error("Left failed: " + (left_final.result?.error || "AI failed"));
    if (right_final.status === "failed") throw new Error("Right failed: " + (right_final.result?.error || "AI failed"));

    setStatus(rs, "Done ✅", true);

    container.innerHTML =
      renderResultCard("Left eye", left_final.result || {}) +
      renderResultCard("Right eye", right_final.result || {});

  } catch (e) {
    setStatus(rs, "Error: " + e.message, false);
  }
}
</script>
</body>
</html>
"""


# ----------------------------
# API endpoints
# ----------------------------
@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token(user.id)}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...), user_id: int = Depends(require_user)):
    upload_id = new_upload_id()
    path = await save_upload(upload_id, file)
    return {"upload_id": upload_id, "path": path}


@app.post("/scan/create", response_model=ScanResponse)
def create_scan(body: ScanCreateRequest, db: Session = Depends(get_db), user_id: int = Depends(require_user)):
    image_path = upload_target_path(body.upload_id)
    if not os.path.exists(image_path):
        raise HTTPException(400, "Upload not found")

    scan = Scan(user_id=user_id, eye=body.eye, image_path=image_path, status=ScanStatus.queued)
    db.add(scan)
    db.commit()
    db.refresh(scan)

    # For cloud demo: always async via worker
    process_scan.delay(scan.id)
    return {"id": scan.id, "status": scan.status.value, "result": None, "report_url": None}


@app.get("/scan/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: int, db: Session = Depends(get_db), user_id: int = Depends(require_user)):
    scan = db.get(Scan, scan_id)
    if not scan or scan.user_id != user_id:
        raise HTTPException(404, "Not found")

    safe_result = make_json_safe(decode_db_result(scan.result))
    report_url = f"/scan/{scan.id}/report" if scan.report_path and scan.status == ScanStatus.done else None
    return {"id": scan.id, "status": scan.status.value, "result": safe_result, "report_url": report_url}


@app.get("/scan/{scan_id}/report")
def get_report(scan_id: int, db: Session = Depends(get_db), user_id: int = Depends(require_user)):
    scan = db.get(Scan, scan_id)
    if not scan or scan.user_id != user_id or not scan.report_path:
        raise HTTPException(404, "Not found")

    from fastapi.responses import FileResponse
    return FileResponse(scan.report_path, media_type="application/pdf", filename=f"alati_report_{scan_id}.pdf")
