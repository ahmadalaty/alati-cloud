import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

BUILD_MARKER = "BACKEND_PDF_REPORT_v2_2026_11_20"

INK = HexColor("#2c2c2a")
MUTED = HexColor("#6b6a64")
RULE = HexColor("#e0e0e0")
BLUE = HexColor("#185fa5")
TEAL = HexColor("#0f6e56")

# Same reading of the stored result text as the scan page (main.py SCAN_HTML).
_TONES = {
    "green": (HexColor("#1b5e20"), HexColor("#eef7ef")),
    "amber": (HexColor("#a64b00"), HexColor("#fff6ea")),
    "red": (HexColor("#b71c1c"), HexColor("#fdf0f0")),
    "grey": (HexColor("#444441"), HexColor("#f3f3f1")),
}


def describe(dx: str) -> Tuple[str, str, str]:
    """Stored result text -> (tone, headline, explanation)."""
    s = (dx or "").strip()
    low = s.lower()
    if "not gradable" in low:
        reason = s.split("—", 1)[1].strip() if "—" in s else ""
        return ("grey", "Image not gradable",
                (reason + " " if reason else "") + "No result was produced. This is not a normal result; "
                "retake the photograph and scan again.")
    if "not referable" in low:
        return ("amber", "Diabetic retinopathy, not referable",
                "Signs of diabetic retinopathy below the referral threshold. Monitor and rescreen; "
                "this is not a normal result.")
    if "referable" in low:
        return ("red", "Referable diabetic retinopathy",
                "Ophthalmology review recommended. Severity is not graded; assess the eye clinically.")
    if low == "normal":
        return ("green", "No diabetic retinopathy detected",
                "Rescreen at the usual interval. This does not exclude other eye disease.")
    return ("grey", s or "No result", "")


def _wrap(c, text, x, y, width, font="Helvetica", size=10, leading=14, color=INK):
    c.setFont(font, size)
    c.setFillColor(color)
    for line in simpleSplit(text, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def create_pdf_report(scan_id: int, eyes: List[Tuple[str, str]], eye_mode: str,
                      created_at: Optional[datetime] = None, clinician: Optional[str] = None,
                      clinician_email: Optional[str] = None, model_version: Optional[str] = None) -> str:
    out_dir = "/tmp/alati_reports"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"scan_{scan_id}.pdf")

    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle(f"Alati screening report - scan {scan_id}")
    c.setAuthor("Alati")
    W, H = A4
    L, R = 56, W - 56
    width = R - L

    # Header band
    c.setFillColor(BLUE)
    c.rect(0, H - 96, W, 96, stroke=0, fill=1)
    c.setFillColor(TEAL)
    c.rect(0, H - 100, W, 4, stroke=0, fill=1)
    c.setFillColor(HexColor("#ffffff"))
    c.setFont("Helvetica-Bold", 22)
    c.drawString(L, H - 52, "Alati")
    c.setFont("Helvetica", 11)
    c.drawString(L, H - 72, "Diabetic retinopathy screening report")
    c.setFont("Helvetica", 9)
    c.drawRightString(R, H - 52, f"Report no. {scan_id}")
    c.drawRightString(R, H - 66, "alati.nabeeh.health")

    # Scan details
    when = created_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when_txt = when.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    examined = {"left": "Left eye", "right": "Right eye", "both": "Both eyes"}.get(eye_mode, eye_mode)
    who = clinician or "-"
    if clinician_email and clinician and clinician != clinician_email:
        who = f"{clinician} ({clinician_email})"
    rows = [
        ("Date of scan", when_txt),
        ("Requested by", who),
        ("Eye(s) examined", examined),
        ("Screening model", f"Alati {model_version or ''}".strip() + ", with automatic image-quality check"),
        ("Patient reference", "_______________________________  (not recorded by Alati)"),
    ]
    y = H - 136
    for label, value in rows:
        c.setFont("Helvetica", 9)
        c.setFillColor(MUTED)
        c.drawString(L, y, label.upper())
        _wrap(c, value, L + 120, y, width - 120, size=10)
        y -= 18
    y -= 4
    c.setStrokeColor(RULE)
    c.line(L, y, R, y)
    y -= 26

    # Results
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(INK)
    c.drawString(L, y, "Result")
    y -= 14
    for eye_label, dx in eyes or [("Result", "No result recorded")]:
        tone, headline, detail = describe(dx)
        fg, bg = _TONES[tone]
        detail_lines = simpleSplit(detail, "Helvetica", 10, width - 28) if detail else []
        box_h = 50 + 14 * len(detail_lines)
        c.setFillColor(bg)
        c.roundRect(L, y - box_h, width, box_h, 6, stroke=0, fill=1)
        c.setFillColor(fg)
        c.rect(L, y - box_h, 4, box_h, stroke=0, fill=1)
        c.setFont("Helvetica", 8.5)
        c.setFillColor(MUTED)
        c.drawString(L + 16, y - 16, eye_label.upper())
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(fg)
        c.drawString(L + 16, y - 34, headline)
        ty = y - 50
        c.setFont("Helvetica", 10)
        c.setFillColor(INK)
        for line in detail_lines:
            c.drawString(L + 16, ty, line)
            ty -= 14
        y -= box_h + 12

    # About the model
    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(INK)
    c.drawString(L, y, "About this result")
    y -= 16
    y = _wrap(c, "Alati gives a referral decision for diabetic retinopathy from a colour fundus photograph. "
                 "On 3,662 photographs from an independent public dataset that was not used to develop it "
                 "(APTOS 2019, Aravind Eye Hospital, India; 3,406 gradable), it detected referable diabetic "
                 "retinopathy with 99.4% sensitivity (95% CI 98.9-99.7%) and 79.2% specificity "
                 "(95% CI 77.3-81.0%). That evaluation was retrospective and used standard fundus cameras. "
                 "Low-resolution photos are declined rather than called normal. Smartphone fundus cameras, "
                 "ultra-widefield images and severity grading are not validated.",
              L, y, width, size=9.5, leading=13.5, color=INK)
    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(INK)
    c.drawString(L, y, "Important")
    y -= 16
    y = _wrap(c, "Alati is a screening aid for licensed clinicians and is used under their clinical judgement. "
                 "It has not received regulatory clearance in any jurisdiction. It does not make a diagnosis, "
                 "screens for diabetic retinopathy only, and does not exclude other eye disease. Every result "
                 "must be confirmed clinically by the treating physician.",
              L, y, width, size=9.5, leading=13.5, color=INK)

    # Footer
    c.setStrokeColor(RULE)
    c.line(L, 54, R, 54)
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED)
    c.drawString(L, 40, f"Generated {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')} by Alati, a Nabeeh product.")
    c.drawRightString(R, 40, "The photograph itself is not stored by Alati.")

    c.showPage()
    c.save()
    return path
