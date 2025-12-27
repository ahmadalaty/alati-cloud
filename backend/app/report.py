import os
import json
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def create_pdf_report(scan_id: int, result, out_dir: str = "/data/reports") -> str:
    """
    Creates a simple PDF report and returns the file path.

    result can be a dict OR a JSON string. We normalize it.
    """
    # normalize result to dict
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            result = {"raw": result}
    if not isinstance(result, dict):
        result = {"raw": str(result)}

    # output path
    filename = f"alati_report_{scan_id}.pdf"
    path = os.path.join(out_dir, filename)
    _ensure_dir(path)

    # build pdf
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4

    y = h - 60
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, f"Alati Report - Scan #{scan_id}")

    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    y -= 30
    triage = result.get("triage", "")
    top_label = result.get("top_label", result.get("top", ""))

    if triage:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Triage: {triage}")
        y -= 22

    if top_label:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Top label: {top_label}")
        y -= 22

    probs = result.get("probs", None)
    if probs and isinstance(probs, dict):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Probabilities:")
        y -= 18

        c.setFont("Helvetica", 11)
        # sort by probability desc if possible
        try:
            items = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
        except Exception:
            items = list(probs.items())

        for k, v in items:
            if y < 80:
                c.showPage()
                y = h - 60
                c.setFont("Helvetica", 11)

            c.drawString(60, y, f"- {k}: {v}")
            y -= 16
    else:
        # fallback: dump result
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Result:")
        y -= 18

        c.setFont("Helvetica", 10)
        txt = json.dumps(result, indent=2, ensure_ascii=False)
        for line in txt.splitlines():
            if y < 80:
                c.showPage()
                y = h - 60
                c.setFont("Helvetica", 10)
            c.drawString(60, y, line[:110])
            y -= 12

    c.showPage()
    c.save()

    return path
