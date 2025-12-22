import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from .config import settings

def make_report(scan_id: int, result: dict) -> str:
    os.makedirs(settings.LOCAL_STORAGE_DIR, exist_ok=True)
    path = os.path.join(settings.LOCAL_STORAGE_DIR, f"report_{scan_id}.pdf")

    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 800, f"Alati Report - Scan #{scan_id}")

    c.setFont("Helvetica", 12)
    c.drawString(72, 770, f"Triage: {result.get('triage')}")
    c.drawString(72, 750, f"Top label: {result.get('top_label')}")
    c.drawString(72, 730, f"Probabilities: {result.get('probs')}")

    c.showPage()
    c.save()
    return path
