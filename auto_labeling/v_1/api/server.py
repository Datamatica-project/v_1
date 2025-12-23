from fastapi import FastAPI

from auto_labeling.v_1.api.routers.health import router as health_router
from auto_labeling.v_1.api.routers.ingest import router as ingest_router
from auto_labeling.v_1.api.routers.loop import router as loop_router
from auto_labeling.v_1.api.routers.export import router as export_router
from auto_labeling.v_1.api.routers.export_round0 import router as export_round0_router
from auto_labeling.v_1.api.routers.events import router as events_router
from auto_labeling.v_1.api.routers.logs import router as logs_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="V1 Auto Labeling API",
        version="1.0.0",
        response_model_by_alias=True,
    )

    app.include_router(health_router, tags=["health"])
    app.include_router(ingest_router, prefix="/api/v1", tags=["ingest"])
    app.include_router(loop_router, prefix="/api/v1", tags=["loop"])
    app.include_router(export_router, prefix="/api/v1", tags=["export"])
    app.include_router(export_round0_router, prefix="/api/v1", tags=["export"])
    app.include_router(events_router, prefix="/api/v1", tags=["events"])
    app.include_router(logs_router, prefix="/api/v1", tags=["logs"])
    return app


app = create_app()
