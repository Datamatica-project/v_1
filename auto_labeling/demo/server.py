from __future__ import annotations

from fastapi import FastAPI
from auto_labeling.demo.routers.ingest import router as ingest_router


def create_app() -> FastAPI:
    app = FastAPI(title="V1 Demo API", version="0.1.0", response_model_by_alias=True)
    app.include_router(ingest_router, prefix="/api/v1", tags=["ingest"])
    return app

app = create_app()
