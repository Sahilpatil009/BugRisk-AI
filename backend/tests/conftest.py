import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///./test-bugrisk.db")
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("INLINE_WORKER", "false")


@pytest.fixture(scope="session", autouse=True)
def initialized_test_database():
    """Make every test module runnable without relying on suite execution order."""
    from app.database import Base, SessionLocal, engine
    from app.seed import seed_demo

    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed_demo(db)
    yield
