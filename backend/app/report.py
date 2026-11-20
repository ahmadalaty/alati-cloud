from datetime import datetime, timezone
from io import BytesIO
from typing import List, Optional, Tuple

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

BUILD_MARKER = "BACKEND_PDF_REPORT_v3_DESIGN_A_2026_11_20"

# Design A: white page, one dark blue, hairline rules, figures in a monospace face.
INK = HexColor("#17202b")
INK_2 = HexColor("#2b3540")
MUTED = HexColor("#6b7684")
RULE = HexColor("#dfe4ea")
RULE_LIGHT = HexColor("#edf0f3")
BLUE = HexColor("#1f4f8a")
WHITE = HexColor("#ffffff")

# Same reading of the stored result text as the scan page (pages.py SCAN_HTML).
_TONES = {
    "green": (HexColor("#1f6b3a"), HexColor("#e6f3ea")),
    "amber": (HexColor("#8a4b00"), HexColor("#fdf1df")),
    "red": (HexColor("#a01818"), HexColor("#fbe9e9")),
    "grey": (HexColor("#4a5562"), HexColor("#eef0f3")),
}


def describe(dx: str) -> Tuple[str, str, str, str]:
    """Stored result text -> (tone, headline, action, explanation)."""
    s = (dx or "").strip()
    low = s.lower()
    if "not gradable" in low:
        reason = s.split("—", 1)[1].strip() if "—" in s else ""
        return ("grey", "Image not gradable", "Retake or re-upload",
                (reason + " " if reason else "") + "No result was produced. This is not a normal result.")
    if "not referable" in low:
        return ("amber", "Diabetic retinopathy, not referable", "Monitor and rescreen",
                "Signs of diabetic retinopathy below the referral threshold. This is not a normal result.")
    if "referable" in low:
        return ("red", "Referable diabetic retinopathy", "Refer to ophthalmology",
                "Ophthalmology review recommended. Severity is not graded; assess the eye clinically.")
    if low == "normal":
        return ("green", "No diabetic retinopathy detected", "Rescreen as usual",
                "Rescreen at the usual interval. This does not exclude other eye disease.")
    return ("grey", s or "No result", "", "")


def _wrap(c, text, x, y, width, font="Helvetica", size=10, leading=14, color=INK_2):
    c.setFont(font, size)
    c.setFillColor(color)
    for line in simpleSplit(text, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def create_pdf_report(scan_id: int, eyes: List[Tuple[str, str]], eye_mode: str,
                      created_at: Optional[datetime] = None, clinician: Optional[str] = None,
                      clinician_email: Optional[str] = None, model_version: Optional[str] = None) -> bytes:
    """The report as PDF bytes. Built in memory; nothing is written to disk."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"Alati screening report {scan_id}")
    c.setAuthor("Alati")
    W, H = A4
    L, R = 56, W - 56
    width = R - L

    # Header: eye mark, wordmark, report title; report number on the right.
    c.setStrokeColor(BLUE)
    c.setLineWidth(3.2)
    c.circle(L + 8, H - 62, 7.4, stroke=1, fill=0)
    c.setFillColor(BLUE)
    c.circle(L + 8, H - 62, 2.6, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(L + 24, H - 68, "Alati")
    c.setFont("Helvetica", 10.5)
    c.setFillColor(INK_2)
    c.drawString(L + 78, H - 67, "Diabetic retinopathy screening report")
    c.setFont("Courier", 9.5)
    c.setFillColor(MUTED)
    c.drawRightString(R, H - 60, f"REPORT NO. {scan_id}")
    c.drawRightString(R, H - 73, "alati.nabeeh.health")
    c.setStrokeColor(RULE)
    c.setLineWidth(1)
    c.line(L, H - 90, R, H - 90)

    # Scan details, label / value rows with hairlines.
    when = created_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    when_txt = when.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    examined = {"left": "Left eye", "right": "Right eye", "both": "Both eyes"}.get(eye_mode, eye_mode)
    who = clinician or "-"
    if clinician_email and clinician and clinician != clinician_email:
        who = f"{clinician} ({clinician_email})"
    rows = [
        ("Date of scan", when_txt, True),
        ("Requested by", who, False),
        ("Eye(s) examined", examined, False),
        ("AI version", f"Alati {model_version or ''}".strip() + " with automatic image-quality check", True),
        ("Patient reference", "______________________________  (not recorded by Alati)", False),
    ]
    y = H - 116
    for label, value, mono in rows:
        c.setFont("Helvetica", 9)
        c.setFillColor(MUTED)
        c.drawString(L, y, label)
        _wrap(c, value, L + 120, y, width - 120, font="Courier" if mono else "Helvetica", size=10)
        c.setStrokeColor(RULE_LIGHT)
        c.line(L, y - 7, R, y - 7)
        y -= 21

    # Result table: Eye | Finding (with explanation) | Action chip
    y -= 16
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(INK)
    c.drawString(L, y, "Result")
    y -= 10
    c.setFillColor(HexColor("#fafbfc"))
    c.setStrokeColor(RULE)
    c.rect(L, y - 20, width, 20, stroke=1, fill=1)
    c.setFont("Helvetica", 8.5)
    c.setFillColor(MUTED)
    c.drawString(L + 10, y - 13, "Eye")
    c.drawString(L + 70, y - 13, "Finding")
    c.drawRightString(R - 10, y - 13, "Action")
    y -= 20
    for eye_label, dx in eyes or [("-", "No result recorded")]:
        tone, headline, action, detail = describe(dx)
        fg, bg = _TONES[tone]
        detail_lines = simpleSplit(detail, "Helvetica", 9.5, width - 70 - 150) if detail else []
        row_h = 34 + 13 * len(detail_lines)
        c.setFillColor(WHITE)
        c.setStrokeColor(RULE)
        c.rect(L, y - row_h, width, row_h, stroke=1, fill=1)
        c.setFont("Courier", 9.5)
        c.setFillColor(HexColor("#4a5562"))
        c.drawString(L + 10, y - 20, eye_label.replace(" eye", "").upper())
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(fg)
        c.drawString(L + 70, y - 20, headline)
        ty = y - 34
        c.setFont("Helvetica", 9.5)
        c.setFillColor(INK_2)
        for line in detail_lines:
            c.drawString(L + 70, ty, line)
            ty -= 13
        if action:
            c.setFont("Helvetica-Bold", 8.5)
            tw = c.stringWidth(action, "Helvetica-Bold", 8.5)
            c.setFillColor(bg)
            c.roundRect(R - 10 - tw - 14, y - 26, tw + 14, 16, 2, stroke=0, fill=1)
            c.setFillColor(fg)
            c.drawString(R - 10 - tw - 7, y - 21, action)
        y -= row_h

    # About the result, and the limits.
    y -= 26
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(INK)
    c.drawString(L, y, "About this result")
    y -= 16
    y = _wrap(c, "Alati gives a referral decision for diabetic retinopathy from a colour fundus photograph. "
                 "On 3,662 photographs from an independent public dataset that was not used to develop it "
                 "(APTOS 2019, Aravind Eye Hospital, India; 3,406 gradable), it detected referable diabetic "
                 "retinopathy with 99.4% sensitivity (95% CI 98.9-99.7%) and 79.2% specificity "
                 "(95% CI 77.3-81.0%). That evaluation was retrospective and used standard fundus cameras. "
                 "Low-resolution and heavily compressed photos are declined rather than called normal. "
                 "Smartphone fundus cameras, ultra-widefield images and severity grading are not validated.",
              L, y, width, size=9.5, leading=13.5)
    y -= 12
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(INK)
    c.drawString(L, y, "Important")
    y -= 16
    _wrap(c, "Alati is a screening aid for licensed clinicians and is used under their clinical judgement. "
             "It has not received regulatory clearance in any jurisdiction. It does not make a diagnosis, "
             "screens for diabetic retinopathy only, and does not exclude other eye disease. Every result "
             "must be confirmed clinically by the treating physician.",
          L, y, width, size=9.5, leading=13.5)

    # Footer
    c.setStrokeColor(RULE)
    c.line(L, 54, R, 54)
    c.setFont("Courier", 8)
    c.setFillColor(MUTED)
    c.drawString(L, 40, f"Generated {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')} · Alati, a Nabeeh product")
    c.setFont("Helvetica", 8)
    c.drawRightString(R, 40, "The photograph itself is not stored by Alati.")

    c.showPage()
    c.save()
    return buf.getvalue()
