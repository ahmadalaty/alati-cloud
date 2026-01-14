import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BUILD_MARKER = "BACKEND_PDF_FULL_DIAG_v1_2026_01_11"


def create_pdf_report(scan_id: int, diagnosis_full: str, eye: str) -> str:
    out_dir = "/tmp/alati_reports"
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"scan_{scan_id}.pdf")

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, height - 90, "Alati AI Report")

    c.setFont("Helvetica", 12)
    c.drawString(72, height - 130, f"Scan ID: {scan_id}")
    c.drawString(72, height - 150, f"Eye: {eye}")
    c.drawString(72, height - 170, f"Diagnosis: {diagnosis_full}")

    c.setFont("Helvetica", 10)
    c.drawString(72, height - 210, "Disclaimer: AI-assisted screening only. Not a final medical diagnosis.")

    c.showPage()
    c.save()

    return path
