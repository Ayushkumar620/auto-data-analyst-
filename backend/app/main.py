from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="Auto Data Analyst Agent",
    version="1.0.0",
    description="AI-powered data analysis platform",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
    }


@app.get("/api/v1/health")
def health_v1() -> dict:
    return health()


from backend.app.api.v1.chat import router as chat_router
from backend.app.api.v1.datasets import router as datasets_router
from backend.app.api.v1.forecasting import router as forecasting_router
from backend.app.api.v1.insights import router as insights_router
from backend.app.api.v1.reports import router as reports_router

app.include_router(datasets_router, prefix=settings.api_v1_prefix)
app.include_router(insights_router, prefix=settings.api_v1_prefix)
app.include_router(chat_router, prefix=settings.api_v1_prefix)
app.include_router(forecasting_router, prefix=settings.api_v1_prefix)
app.include_router(reports_router, prefix=settings.api_v1_prefix)
