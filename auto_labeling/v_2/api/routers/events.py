"""
이벤트 API Router (앙상블 방식, v2)

Worker가 Loop 진행 중 이벤트를 API Server로 콜백
Spring Boot 프론트엔드가 폴링으로 이벤트 조회
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

from fastapi import APIRouter, HTTPException, Body, Query

from auto_labeling.v_2.api.dto.event import EventIngestRequest
from auto_labeling.v_2.api.dto.base import CamelModel

router = APIRouter(prefix="/api/v2/events", tags=["Events (v2 Ensemble)"])

# 이벤트 로그 저장 경로
V2_ROOT = Path(__file__).resolve().parents[2]
EVENTS_ROOT = Path(os.getenv("V2_EVENTS_ROOT", str(V2_ROOT.parent / "data" / "logs" / "events"))).resolve()
EVENTS_ROOT.mkdir(parents=True, exist_ok=True)


def _now_iso_z() -> str:
    """ISO 8601 형식 현재 시각"""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_ts() -> str:
    """파일명 충돌 방지용 타임스탬프 (microsecond 포함)"""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _dump_event(event: EventIngestRequest) -> dict:
    """Pydantic v1/v2 호환 dump"""
    if hasattr(event, "model_dump"):
        return event.model_dump(by_alias=True, exclude_none=True)
    return event.dict(by_alias=True, exclude_none=True)


def _safe_join(base: Path, name: str) -> Path:
    """
    Path traversal 방지용 safe join
    runId에 ../ 등이 들어가도 EVENTS_ROOT 밖으로 나갈 수 없음
    """
    base_r = base.resolve()
    p = (base_r / name).resolve()

    if base_r == p:
        return p

    if base_r not in p.parents:
        raise HTTPException(status_code=400, detail="Invalid runId")

    return p


def _event_files(run_dir: Path) -> List[Path]:
    """
    run_dir 아래 *.json 파일을 mtime 내림차순으로 정렬
    """
    files = [p for p in run_dir.glob("*.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _read_json(p: Path) -> Optional[dict]:
    """JSON 파일 읽기"""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ========================================
# POST: 이벤트 수신 (Worker → API 콜백)
# ========================================
@router.post("")
def ingest_event(event: EventIngestRequest = Body(...)):
    """
    Worker가 Loop 진행 중 이벤트 콜백

    Request:
    ```json
    {
        "eventType": "LOOP_STARTED",
        "runId": "run_20250106_120000_xyz789",
        "jobId": "job_20250106_120000",
        "message": "Loop started",
        "payload": {...}
    }
    ```

    Response:
    ```json
    {
        "resultCode": "SUCCESS",
        "message": "Event received"
    }
    ```

    저장 위치:
    - `logs/events/{runId}/{timestamp}_{eventType}.json`
    """
    # 필수 필드 검증
    if not getattr(event, "event_type", None):
        return {
            "resultCode": "FAIL",
            "errorCode": "EVT_4001",
            "message": "eventType is required"
        }

    if not getattr(event, "run_id", None):
        return {
            "resultCode": "FAIL",
            "errorCode": "EVT_4002",
            "message": "runId is required"
        }

    run_id = str(event.run_id).strip()
    event_type = str(event.event_type).strip()
    ts = _safe_ts()

    # 이벤트 저장
    out_dir = _safe_join(EVENTS_ROOT, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{ts}_{event_type}.json"
    out_file.write_text(
        json.dumps(
            {
                "receivedAt": _now_iso_z(),
                "event": _dump_event(event)
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return {
        "resultCode": "SUCCESS",
        "message": "Event received"
    }


# ========================================
# GET: Run 목록 조회
# ========================================
@router.get("/runs")
def list_event_runs(
    limit: int = Query(50, ge=1, le=500, description="최대 조회 개수")
):
    """
    이벤트 Run 목록 조회 (최근순)

    Response:
    ```json
    {
        "resultCode": "SUCCESS",
        "data": [
            {
                "runId": "run_20250106_120000_xyz789",
                "runPath": "/path/to/events/run_xxx",
                "updatedAt": "2025-01-06T12:30:00Z"
            }
        ]
    }
    ```
    """
    if not EVENTS_ROOT.exists():
        return {
            "resultCode": "SUCCESS",
            "message": "No events directory",
            "data": []
        }

    runs = [p for p in EVENTS_ROOT.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    runs = runs[:limit]

    items = []
    for run_dir in runs:
        items.append({
            "runId": run_dir.name,
            "runPath": str(run_dir.resolve()),
            "updatedAt": datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        })

    return {
        "resultCode": "SUCCESS",
        "message": "ok",
        "data": items
    }


# ========================================
# GET: 특정 Run의 이벤트 목록 조회
# ========================================
@router.get("/get")
def get_events(
    runId: str = Query(..., description="Run 식별자"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(200, ge=1, le=5000, description="Pagination limit")
):
    """
    특정 Run의 이벤트 목록 조회 (Pagination)

    Response:
    ```json
    {
        "resultCode": "SUCCESS",
        "data": {
            "runId": "run_20250106_120000_xyz789",
            "total": 10,
            "offset": 0,
            "limit": 200,
            "items": [
                {
                    "receivedAt": "2025-01-06T12:00:00Z",
                    "event": {...},
                    "_fileName": "..."
                }
            ]
        }
    }
    ```
    """
    run_dir = _safe_join(EVENTS_ROOT, runId)

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {runId}")

    files = _event_files(run_dir)
    total = len(files)

    s = int(offset)
    e = min(s + int(limit), total)
    page = files[s:e]

    rows = []
    for p in page:
        j = _read_json(p)
        if j is not None:
            j["_fileName"] = p.name
            rows.append(j)

    return {
        "resultCode": "SUCCESS",
        "message": "ok",
        "data": {
            "runId": runId,
            "total": total,
            "offset": s,
            "limit": limit,
            "items": rows
        }
    }


# ========================================
# GET: 최신 이벤트 조회 (Spring Boot 폴링용)
# ========================================
@router.get("/latest")
def get_latest_event(
    runId: str = Query(..., description="Run 식별자"),
    eventType: Optional[str] = Query(None, description="이벤트 타입 필터 (선택)")
):
    """
    최신 이벤트 조회 (폴링용)

    Query Parameters:
    - runId: run_20250106_120000_xyz789 (필수)
    - eventType: LOOP_STARTED (선택, 특정 타입만 필터링)

    Response:
    ```json
    {
        "resultCode": "SUCCESS",
        "data": {
            "receivedAt": "2025-01-06T12:00:00Z",
            "event": {
                "eventType": "LOOP_PROGRESS",
                "runId": "run_xxx",
                "data": {...}
            },
            "_fileName": "20250106_120000_123456_LOOP_PROGRESS.json"
        }
    }
    ```
    """
    run_dir = _safe_join(EVENTS_ROOT, runId)

    if not run_dir.exists():
        # 폴링 용도에서는 404보다 SUCCESS + null이 편함
        return {
            "resultCode": "SUCCESS",
            "message": "Run not found",
            "data": None
        }

    files = _event_files(run_dir)

    if not files:
        return {
            "resultCode": "SUCCESS",
            "message": "No events",
            "data": None
        }

    # 이벤트 타입 필터링
    for p in files:
        if eventType is not None and f"_{eventType}.json" not in p.name:
            continue

        j = _read_json(p)
        if j is not None:
            j["_fileName"] = p.name
            return {
                "resultCode": "SUCCESS",
                "message": "ok",
                "data": j
            }

    return {
        "resultCode": "SUCCESS",
        "message": "No matching event",
        "data": None
    }
