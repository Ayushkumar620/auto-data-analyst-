from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    pass


def _engine_kwargs(database_url: str) -> dict:
    kwargs = {"future": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return kwargs


engine = create_engine(settings.database_url, **_engine_kwargs(settings.database_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_db_and_tables(database_url: str | None = None) -> None:
    resolved_url = database_url or settings.database_url
    engine_to_use = create_engine(resolved_url, **_engine_kwargs(resolved_url))
    Base.metadata.create_all(bind=engine_to_use)
