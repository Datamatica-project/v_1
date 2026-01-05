# demo/server.py
from __future__ import annotations

from fastapi import FastAPI

# routers
from auto_labeling.demo.routers.ingest import router as ingest_router
from auto_labeling.demo.routers.infer import router as infer_router
from auto_labeling.demo.routers.results import router as results_router


def create_app() -> FastAPI:
    """
    V1 Demo API Application Factory

    포함 라우터:
    - /api/v1/gt/*           : GT ingest / register
    - /api/v1/unlabeled/*    : unlabeled ingest
    - /api/demo/run          : PASS/FAIL inference 실행
    - /api/demo/run/status  : job 상태 조회
    - /api/demo/results/*   : 결과 조회 / preview / download
    """
    app = FastAPI(
        title="V1 Demo API",
        version="0.1.0",
        response_model_by_alias=True,
    )

    # Ingest (GT / Unlabeled)
    app.include_router(
        ingest_router,
        prefix="/api/v1",
        tags=["ingest"],
    )

    # Demo Inference (PASS/FAIL + result/pass)
    app.include_router(
        infer_router,
        prefix="",  # infer.py에서 이미 /api/demo prefix 사용
        tags=["demo-infer"],
    )

    # Demo Results (PASS gallery / preview / download)
    app.include_router(
        results_router,
        prefix="",  # results.py에서 이미 /api/demo/results prefix 사용
        tags=["demo-results"],
    )

    return app


# ASGI entrypoint
app = create_app()
