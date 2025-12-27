import json
from celery import Celery
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal
from .models import Scan, ScanStatus
from .inference import run_inference
from .report import create_pdf_report

celery_app = Celery(
    "alati",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# Important for Celery 6+ warning
celery_app.conf.broker_connection_retry_on_startup = True


def _json_safe(obj):
    # minimal safe conversion (no numpy/torch objects)
    try:
        import numpy as np
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return float(obj.detach().cpu().item())
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]

    return obj


@celery_app.task(name="app.tasks.process_scan")
def process_scan(scan_id: int):
    db: Session = SessionLocal()
    try:
        scan = db.get(Scan, scan_id)
        if not scan:
            return

        # run inference
        result = run_inference(scan.image_path, scan.eye)

        # make sure result is JSON-friendly
        result = _json_safe(result)

        # ✅ Store result as JSON string (fixes Postgres dict-adaptation error)
        scan.result = json.dumps(result, ensure_ascii=False)

        # generate report
        report_path = create_pdf_report(scan_id=scan.id, result=result)
        scan.report_path = report_path

        scan.status = ScanStatus.done
        db.add(scan)
        db.commit()

    except Exception as e:
        # mark failed
        try:
            scan = db.get(Scan, scan_id)
            if scan:
                scan.status = ScanStatus.failed
                scan.result = json.dumps({"error": f"{type(e).__name__}: {str(e)}"}, ensure_ascii=False)
                db.add(scan)
                db.commit()
        except Exception:
            pass
        raise
    finally:
        db.close()
