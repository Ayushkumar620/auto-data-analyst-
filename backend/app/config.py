from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "auto-data-analyst"
    environment: str = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 100
    database_url: str = "sqlite:///./auto_data_analyst.db"
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    jwt_secret: str = "dev-secret-please-change-in-production-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / settings.upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
