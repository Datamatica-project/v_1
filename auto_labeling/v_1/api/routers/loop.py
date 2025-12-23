# auto_labeling/v_1/api/routers/loop.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Body, Path as PathParam
import os
import requests
from typing import Any, Dict

from auto_labeling.v_1.api.dto.loop import RunLoopRequest, RunLoopResponse, JobStatusResponse

router = APIRouter(prefix="/loop")

# Worker 주소: API 서버는 프록시 역할만 수행하고 실제 실행은 worker가 담당
WORKER_BASE_URL = os.getenv("WORKER_BASE_URL", "http://v1-worker:8011").rstrip("/")

# (connect timeout, read timeout)
DEFAULT_TIMEOUT = (3.0, 30.0)

# Worker 측 API prefix (worker는 /api/v1 하위에 라우팅이 구성되어 있다고 가정)
WORKER_V1_PREFIX = "/api/v1"
WORKER_LOOP_PREFIX = f"{WORKER_V1_PREFIX}/loop"


def _raise_worker_502(prefix: str, *, url: str, err: Exception, resp: requests.Response | None = None) -> None:
    """
    Worker 연결/응답 오류를 502(Bad Gateway)로 변환한다.

    의미:
    - API 서버 자체 오류가 아니라, Worker가 죽었거나 네트워크/응답 형식 문제가 있다는 뜻.
    """
    detail: Dict[str, Any] = {"message": f"{prefix}: {err}", "url": url}
    if resp is not None:
        detail["status_code"] = resp.status_code
        detail["body_preview"] = (resp.text or "")[:1000]
    raise HTTPException(status_code=502, detail=detail)


def _post_json(url: str, payload: dict) -> dict:
    """Worker에 POST(JSON) 요청 → JSON 응답을 dict로 반환"""
    try:
        r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        _raise_worker_502("worker POST failed", url=url, err=e)

    if not r.ok:
        _raise_worker_502("worker POST bad status", url=url, err=RuntimeError("bad status"), resp=r)

    try:
        return r.json()
    except Exception as e:
        _raise_worker_502("worker POST invalid json", url=url, err=e, resp=r)


def _get_json(url: str) -> dict:
    """Worker에 GET 요청 → JSON 응답을 dict로 반환"""
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        _raise_worker_502("worker GET failed", url=url, err=e)

    if not r.ok:
        _raise_worker_502("worker GET bad status", url=url, err=RuntimeError("bad status"), resp=r)

    try:
        return r.json()
    except Exception as e:
        _raise_worker_502("worker GET invalid json", url=url, err=e, resp=r)


def _normalize_keys(d: dict) -> dict:
    """
    Worker 응답의 키 네이밍이 job_id 또는 jobId로 혼재할 수 있어
    API 서버에서 양쪽 키를 모두 맞춰준다.
    """
    if "job_id" in d and "jobId" not in d:
        d["jobId"] = d["job_id"]
    if "jobId" in d and "job_id" not in d:
        d["job_id"] = d["jobId"]
    return d


@router.get(
    "/run",
    response_model=RunLoopResponse,
    summary="Loop 실행(Worker 프록시)",
    description=(
        "Auto-Labeling V1 Loop 실행을 Worker에게 요청합니다.\n\n"
        "### 이 API의 의미\n"
        "- API 서버는 **프록시** 역할만 수행합니다.\n"
        "- 실제 Loop 실행(학습/추론/분류/teacher/pseudo/export)은 Worker에서 진행됩니다.\n\n"
        "### 요청(선택값 처리)\n"
        "- RunLoopRequest의 대부분 필드는 **비워도 됩니다(Optional)**.\n"
        "- 미지정 시 Worker/서버 기본 설정(cfg, registry 등)을 사용합니다.\n\n"
        "### 응답\n"
        "- jobId를 반환하며, 이후 `GET /api/v1/loop/status/{jobId}`로 진행 상태를 조회합니다.\n\n"
        "### 오류(502)\n"
        "- Worker가 죽었거나 네트워크/응답 형식 문제가 발생하면 502(Bad Gateway)를 반환합니다."
    ),
    responses={
        200: {"description": "요청 성공 (jobId 반환)"},
        502: {"description": "Worker 연결 실패 또는 Worker 오류"},
    },
)
def run_loop(
    req: RunLoopRequest = Body(
        default_factory=RunLoopRequest,
        description=(
            "Loop 실행 요청 본문(JSON).\n\n"
            "대부분 필드는 선택 사항이며, 비워도 됩니다.\n"
            "미지정 시 Worker가 기본 설정을 사용합니다.\n\n"
            "예:\n"
            "- cfgPath: 설정 YAML 경로(선택)\n"
            "- studentWeight: 초기 student weight(선택)\n"
            "- teacherWeight: teacher weight override(선택)\n"
            "- exportRoot: export 출력 루트(선택)\n"
            "- baseModel/gtEpochs/gtImgsz/gtBatch: GT 초기학습 파라미터(선택)\n"
        ),
        examples=[
            {
                "cfgPath": "auto_labeling/v_1/configs/v1_loop.yaml",
                "studentWeight": "models/user_yolo/v1_000/best.pt",
                "teacherWeight": "models/teacher/weights/yolov11x_teacher.pt",
                "exportRoot": "data/exports/run_001",
                "baseModel": "models/pretrained/yolov11x.pt",
                "gtEpochs": 30,
                "gtImgsz": 640,
                "gtBatch": 8,
            }
        ],
    )
):
    url = f"{WORKER_BASE_URL}{WORKER_LOOP_PREFIX}/run"
    payload = req.model_dump(by_alias=True, exclude_none=True)

    data = _post_json(url, payload)
    data = _normalize_keys(data)
    return RunLoopResponse(**data)


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Loop 상태 조회(Worker 프록시)",
    description=(
        "Loop 실행 jobId의 진행 상태/통계를 Worker로부터 조회합니다.\n\n"
        "### 응답 의미\n"
        "- status: QUEUED | RUNNING | DONE | FAILED 등\n"
        "- stats: 진행률, round 번호, failRatio, 경로 등 Worker가 제공하는 통계/상태 정보(dict)\n\n"
        "### 오류(502)\n"
        "- Worker 연결 실패 또는 Worker 내부 오류 시 502를 반환합니다."
    ),
    responses={
        200: {"description": "조회 성공"},
        502: {"description": "Worker 연결 실패 또는 Worker 오류"},
    },
)
def get_status(
    job_id: str = PathParam(
        ...,
        description="조회할 job 식별자(jobId). /loop/run 응답에서 반환됩니다.",
        examples=["job_20251219_172000_loop_run"],
    )
):
    url = f"{WORKER_BASE_URL}{WORKER_LOOP_PREFIX}/status/{job_id}"

    data = _get_json(url)
    data = _normalize_keys(data)
    return JobStatusResponse(**data)
