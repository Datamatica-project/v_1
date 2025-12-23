from __future__ import annotations

from fastapi import APIRouter, Query, HTTPException, Path as PathParam
import os
import requests
from typing import Any, Dict

router = APIRouter(prefix="/logs")

WORKER_BASE_URL = os.getenv("WORKER_BASE_URL", "http://localhost:8011").rstrip("/")
DEFAULT_TIMEOUT_SEC = 10


def _get_json(url: str, params: dict | None = None) -> dict:
    """
    Worker 서비스에 HTTP GET 프록시 요청을 보내고 JSON을 반환한다.

    - Worker가 죽었거나 네트워크 문제가 있으면 502를 반환한다.
    - Worker가 4xx/5xx를 반환하면 그대로 502로 래핑한다(게이트웨이 오류).
    """
    try:
        r = requests.get(url, params=(params or {}), timeout=DEFAULT_TIMEOUT_SEC)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"worker request failed: {e}")


@router.get(
    "/jobs",
    summary="Job 목록 조회(Worker 프록시)",
    description=(
        "Worker가 기록한 job 상태 파일 목록을 조회합니다.\n\n"
        "### 동작\n"
        "- 본 API는 `WORKER_BASE_URL`의 `/logs/jobs`를 그대로 프록시합니다.\n"
        "- Worker가 응답하지 않거나 오류가 발생하면 502로 응답합니다.\n\n"
        "### 사용 목적\n"
        "- Loop/Export 등 비동기 작업의 실행 이력과 상태를 FE/Backend에서 조회하기 위함입니다."
    ),
    responses={
        200: {"description": "조회 성공"},
        502: {"description": "Worker 연결 실패 또는 Worker 오류"},
    },
)
def list_jobs(
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="최대 조회 개수 (1~500, 기본 50). 최신 job부터 반환되는 것을 권장합니다.",
        examples=[50],
    )
) -> Dict[str, Any]:
    url = f"{WORKER_BASE_URL}/logs/jobs"
    return _get_json(url, params={"limit": limit})


@router.get(
    "/jobs/{job_id}",
    summary="Job 단건 조회(Worker 프록시)",
    description=(
        "특정 jobId에 대한 상태/결과/에러 정보를 조회합니다.\n\n"
        "### 동작\n"
        "- 본 API는 `WORKER_BASE_URL`의 `/logs/jobs/{jobId}`를 그대로 프록시합니다.\n"
        "- Worker가 응답하지 않거나 오류가 발생하면 502로 응답합니다.\n\n"
        "### jobId 예시\n"
        "- job_20251219_172000_export_final\n"
        "- job_20251219_173500_loop_run"
    ),
    responses={
        200: {"description": "조회 성공"},
        502: {"description": "Worker 연결 실패 또는 Worker 오류"},
    },
)
def read_job(
    job_id: str = PathParam(
        ...,
        description="조회할 job 식별자(jobId). 비동기 작업 실행 시 생성됩니다.",
        examples=["job_20251219_172000_export_final"],
    )
) -> Dict[str, Any]:
    url = f"{WORKER_BASE_URL}/logs/jobs/{job_id}"
    return _get_json(url)


@router.get(
    "",
    summary="로그 호환 엔드포인트(Worker 프록시)",
    description=(
        "기존 호환성을 위해 유지되는 엔드포인트입니다.\n\n"
        "### 동작\n"
        "- 내부적으로는 `/logs/jobs`와 동일하게 동작합니다.\n"
        "- 레거시 클라이언트가 `GET /logs`만 호출하는 경우를 지원합니다."
    ),
    responses={
        200: {"description": "조회 성공"},
        502: {"description": "Worker 연결 실패 또는 Worker 오류"},
    },
)
def get_logs_compat(
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="최대 조회 개수 (1~500, 기본 50). `/logs/jobs`와 동일.",
        examples=[50],
    )
) -> Dict[str, Any]:
    url = f"{WORKER_BASE_URL}/logs/jobs"
    return _get_json(url, params={"limit": limit})
