from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    app_name: str = "BugRisk AI API"
    database_url: str = "sqlite:///./bugrisk.db"
    frontend_url: str = "http://localhost:3000"
    backend_url: str = "http://localhost:8000"
    session_secret: str = Field(default="local-demo-secret-change-in-production", min_length=32)
    token_encryption_key: str | None = None
    github_client_id: str | None = None
    github_client_secret: str | None = None
    demo_mode: bool = True
    inline_worker: bool = True
    model_path: Path = Field(default=Path("../ml/artifacts/bugrisk_model.joblib"))
    worker_poll_seconds: float = 2.0
    cookie_same_site: Literal["lax", "strict", "none"] = "lax"
    llm_rewrite_url: str | None = None
    llm_api_key: str | None = None

    @property
    def secure_cookies(self) -> bool:
        return self.backend_url.startswith("https://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
