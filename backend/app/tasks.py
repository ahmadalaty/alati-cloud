from celery import Celery
from sqlalchemy.orm import Session
from .config import settings
from .db import SessionLocal
from .models import Scan, ScanStatus
from .inference import run_inference
from .report import make_report

celery_app = Celery("alati", broker=settings.REDIS_URL, backend=settings.REDIS_URL)

@celery_app.task
def process_scan(scan_id: int):
    db: Session = SessionLocal()
    scan = None
    try:
        scan = db.get(Scan, scan_id)
        if not scan:
            return

        scan.status = ScanStatus.running
        db.commit()

        result = run_inference(scan.image_path)
        report_path = make_report(scan.id, result)

        scan.result = result
        scan.report_path = report_path
        scan.status = ScanStatus.done
        db.commit()

    except Exception:
        if scan:
            scan.status = ScanStatus.failed
            db.commit()
        raise
    finally:
        db.close()
