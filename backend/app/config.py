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
    jwt_secret: str = "dev-secret-please-change-in-production-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours
    cors_origins: list[str] = ["*"]

    # LLM settings
    llm_api_key: str = ""
    llm_provider: str = "openai"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    # SMTP email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_from_email: str = ""
    smtp_from_name: str = "Auto Data Analyst Agent"

    # Phone / SMS settings
    sms_provider: str = "auto"
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    sms_gateway_url: str = ""
    sms_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


settings = Settings()
BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / settings.upload_dir
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
