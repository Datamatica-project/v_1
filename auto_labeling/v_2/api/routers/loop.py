"""
Ensemble Loop API Router (앙상블 방식, v2)

API Server → Worker Server 프록시 역할
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
import httpx
import os

from auto_labeling.v_2.api.dto import (
    EnsembleLoopRequest,
    EnsembleLoopResponse,
    LoopStatusResponse,
)

router = APIRouter(prefix="/api/v2/loop", tags=["Loop (v2 Ensemble)"])

# Worker Server URL
WORKER_URL = os.getenv("WORKER_BASE_URL", "http://v1-worker:8011").rstrip("/")
WORKER_TIMEOUT = 30.0


@router.post("/start", response_model=EnsembleLoopResponse)
async def start_ensemble_loop(request: EnsembleLoopRequest):
    """
    앙상블 Loop 시작 (Worker로 프록시)

    Request:
    ```json
    {
        "models": ["yolo", "model2", "model3"],
        "configOverride": {
            "maxRounds": 3,
            "confThreshold": 0.5,
            "failThreshold": 0.01,
            "minFailCount": 100,
            "patience": 2,
            "iouThreshold": 0.5
        }
    }
    ```

    Response:
    ```json
    {
        "loopId": "loop_abc123",
        "runId": "run_20250106_120000_xyz789",
        "status": "STARTED",
        "message": "Ensemble loop started"
    }
    ```

    처리:
    1. Worker Server로 요청 프록시
    2. Worker는 별도 Thread에서 Loop 실행
    3. 즉시 loopId, runId 반환
    4. Worker는 진행 중 이벤트를 API Server로 콜백
    """
    worker_url = f"{WORKER_URL}/api/v2/loop/start"

    try:
        async with httpx.AsyncClient(timeout=WORKER_TIMEOUT) as client:
            response = await client.post(
                worker_url,
                json=request.model_dump(by_alias=True)
            )

            response.raise_for_status()
            data = response.json()

            return EnsembleLoopResponse(
                loop_id=data.get("loopId"),
                run_id=data.get("runId"),
                status=data.get("status"),
                message=data.get("message")
            )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail={
                "message": f"Worker error: {e.response.text}",
                "url": worker_url,
                "status_code": e.response.status_code
            }
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "message": f"Worker unreachable: {str(e)}",
                "url": worker_url
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/status/{loop_id}", response_model=LoopStatusResponse)
async def get_loop_status(loop_id: str):
    """
    Loop 상태 조회 (Worker로 프록시)

    Response:
    ```json
    {
        "loopId": "loop_abc123",
        "runId": "run_20250106_120000_xyz789",
        "status": "RUNNING",
        "stats": {
            "currentRound": 1,
            "totalRounds": 3,
            "roundHistory": [
                {
                    "round": 0,
                    "total": 1000,
                    "passThree": 650,
                    "passTwo": 200,
                    "fail": 100,
                    "miss": 50,
                    "failMissRatio": 0.15
                }
            ],
            "latestFailMissRatio": 0.15,
            "results": {
                "note": "Results are built by /api/v2/results/* endpoints",
                "suggested": {
                    "roundPreview": "/api/v2/results/round/{round_number}/preview?loopId=xxx",
                    "finalPreview": "/api/v2/results/final/preview?loopId=xxx"
                }
            }
        }
    }
    ```
    """
    worker_url = f"{WORKER_URL}/api/v2/loop/status/{loop_id}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(worker_url)

            response.raise_for_status()
            data = response.json()

            return LoopStatusResponse(
                loop_id=data.get("loopId"),
                run_id=data.get("runId"),
                status=data.get("status"),
                stats=data.get("stats", {})
            )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"Loop not found: {loop_id}"
            )
        raise HTTPException(
            status_code=e.response.status_code,
            detail={
                "message": f"Worker error: {e.response.text}",
                "url": worker_url
            }
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "message": f"Worker unreachable: {str(e)}",
                "url": worker_url
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
