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
    github_app_id: str | None = None
    github_app_private_key: str | None = None
    github_app_private_key_file: Path | None = None
    github_webhook_secret: str | None = None
    demo_mode: bool = True
    inline_worker: bool = True
    model_path: Path = Field(default=Path("../ml/artifacts/bugrisk_model.joblib"))
    worker_poll_seconds: float = 2.0
    analysis_stale_seconds: int = Field(default=900, ge=60, le=86_400)
    analysis_max_attempts: int = Field(default=3, ge=1, le=10)
    webhook_max_body_bytes: int = Field(default=1_048_576, ge=1_024, le=10_485_760)
    webhook_rate_limit: int = Field(default=120, ge=1, le=10_000)
    webhook_rate_window_seconds: int = Field(default=60, ge=1, le=3_600)
    cookie_same_site: Literal["lax", "strict", "none"] = "lax"
    llm_rewrite_url: str | None = None
    llm_api_key: str | None = None
    github_scopes: str = "read:user user:email repo"
    fusion_model_path: Path | None = None

    @property
    def secure_cookies(self) -> bool:
        return self.backend_url.startswith("https://")

    @property
    def normalized_github_app_private_key(self) -> str | None:
        if self.github_app_private_key:
            return self.github_app_private_key.replace("\\n", "\n")
        if self.github_app_private_key_file:
            try:
                return self.github_app_private_key_file.read_text(encoding="utf-8")
            except OSError:
                return None
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
