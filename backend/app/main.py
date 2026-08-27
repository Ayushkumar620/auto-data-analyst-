from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
from backend.app.api.v1.workspaces import router as workspaces_router
from backend.app.api.v1.projects import router as projects_router
from backend.app.api.v1.models import router as models_router
from backend.app.api.v1.evaluation import router as evaluation_router
from backend.app.api.v1.analysis import router as analysis_router
from backend.app.api.v1.sql_router import router as sql_router
from backend.app.api.v1.sandbox_router import router as sandbox_router
from backend.app.api.v1.vision_router import router as vision_router
from backend.app.api.v1.monitoring import router as monitoring_router
from backend.app.auth.router import router as auth_router
from backend.app.api.v1.connectors import router as connectors_router
from backend.app.api.v1.governance import router as governance_router
from backend.app.api.v1.alerts import router as alerts_router
from backend.app.api.v1.model_serving import router as model_serving_router
from backend.app.api.v1.anomaly import router as anomaly_router
from backend.app.api.v1.clustering import router as clustering_router
from backend.app.api.v1.statistical_analysis import router as statistical_analysis_router
from backend.app.api.v1.eda import router as eda_router
from backend.app.api.v1.hypothesis_testing import router as hypothesis_testing_router

app.include_router(auth_router, prefix=settings.api_v1_prefix)
app.include_router(analysis_router, prefix=settings.api_v1_prefix)
app.include_router(datasets_router, prefix=settings.api_v1_prefix)
app.include_router(insights_router, prefix=settings.api_v1_prefix)
app.include_router(chat_router, prefix=settings.api_v1_prefix)
app.include_router(forecasting_router, prefix=settings.api_v1_prefix)
app.include_router(reports_router, prefix=settings.api_v1_prefix)
app.include_router(workspaces_router, prefix=settings.api_v1_prefix)
app.include_router(projects_router, prefix=settings.api_v1_prefix)
app.include_router(models_router, prefix=settings.api_v1_prefix)
app.include_router(monitoring_router, prefix=settings.api_v1_prefix)
app.include_router(evaluation_router, prefix=settings.api_v1_prefix)
app.include_router(connectors_router, prefix=settings.api_v1_prefix)
app.include_router(governance_router, prefix=settings.api_v1_prefix)
app.include_router(alerts_router, prefix=settings.api_v1_prefix)
app.include_router(model_serving_router, prefix=settings.api_v1_prefix)
app.include_router(anomaly_router, prefix=settings.api_v1_prefix)
app.include_router(clustering_router, prefix=settings.api_v1_prefix)
app.include_router(statistical_analysis_router, prefix=settings.api_v1_prefix)
app.include_router(eda_router, prefix=settings.api_v1_prefix)
app.include_router(hypothesis_testing_router, prefix=settings.api_v1_prefix)
app.include_router(sql_router, prefix=settings.api_v1_prefix)
app.include_router(sandbox_router, prefix=settings.api_v1_prefix)
app.include_router(vision_router, prefix=settings.api_v1_prefix)


# ------------------------------------------------------------------------------
# Unified Single-Localhost Frontend SPA Static Hosting
# ------------------------------------------------------------------------------
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend_spa(full_path: str):
        # Let API endpoints pass through to return standard 404 if route not found
        if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json", "health"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        target_file = FRONTEND_DIST / full_path
        if full_path and target_file.exists() and target_file.is_file():
            return FileResponse(target_file)

        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        return {"message": "Frontend index.html not found."}


def _compat_routes(self):
    flattened = []

    def walk(route_list):
        for route in route_list:
            original = getattr(route, "original_router", None)
            if original is not None:
                walk(original.routes)
            elif getattr(route, "path", None):
                flattened.append(route)

    walk(self.router.routes)
    return flattened


FastAPI.routes = property(_compat_routes)
