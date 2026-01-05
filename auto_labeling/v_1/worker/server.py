from __future__ import annotations

from fastapi import FastAPI

from auto_labeling.v_1.api.routers.health import router as health_router
from auto_labeling.v_1.worker.routers.loop_worker import router as loop_worker_router
from auto_labeling.v_1.worker.routers.export_round0_worker import router as export_round0_worker_router
from auto_labeling.v_1.worker.routers.logs_worker import router as logs_worker_router
from auto_labeling.v_1.worker.routers.events_worker import router as events_worker_router
from auto_labeling.v_1.worker.routers.export import router as events

def create_app() -> FastAPI:
    app = FastAPI(
        title="V1 Auto Labeling Worker",
        version="1.0.0",
        response_model_by_alias=True,
    )

    app.include_router(health_router, tags=["health"])
    app.include_router(loop_worker_router, tags=["worker"])
    app.include_router(export_round0_worker_router, tags=["export"])
    app.include_router(events, tags=["export"])
    app.include_router(logs_worker_router, tags=["logs"])
    app.include_router(events_worker_router, tags=["events"])
    return app

app = create_app()