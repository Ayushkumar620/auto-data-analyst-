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
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/auto_data_analyst"
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / settings.upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
