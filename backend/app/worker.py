import logging
import time

from sqlalchemy import select

from .config import get_settings
from .database import Base, SessionLocal, engine
from .models import Analysis, AnalysisStatus
from .services import process_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def run() -> None:
    settings = get_settings()
    if not settings.demo_mode and not settings.model_path.exists():
        raise RuntimeError("A trained MODEL_PATH is required outside demo mode")
    if not settings.demo_mode and not settings.token_encryption_key:
        raise RuntimeError("TOKEN_ENCRYPTION_KEY is required outside demo mode")
    Base.metadata.create_all(engine)
    while True:
        with SessionLocal() as db:
            analysis_id = db.scalar(
                select(Analysis.id)
                .where(Analysis.status == AnalysisStatus.QUEUED)
                .order_by(Analysis.created_at)
                .limit(1)
            )
        if analysis_id:
            process_analysis(analysis_id)
        else:
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    run()
