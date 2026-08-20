import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test-bugrisk.db")
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("INLINE_WORKER", "false")
