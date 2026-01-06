# auto_labeling/v_1/api/routers/events.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body, Query

from auto_labeling.v_1.api.dto.common import ApiEnvelope
from auto_labeling.v_1.api.dto.event import EventIngestRequest, EventType

router = APIRouter(prefix="/events")

# ---------------------------------------------------------------------
# storage config
# ---------------------------------------------------------------------
# ✅ 하드코딩(/workspace) 제거: 현재 파일 기준으로 v_1 루트를 잡는다.
ROOT = Path(__file__).resolve().parents[2]  # .../auto_labeling/v_1

# ✅ 필요하면 env로 override 가능
# 예) V1_EVENTS_ROOT=/mnt/nas/V1_EVENTS
EVENTS_ROOT = Path(os.getenv("V1_EVENTS_ROOT", str(ROOT / "logs" / "events"))).resolve()
EVENTS_ROOT.mkdir(parents=True, exist_ok=True)


def _now_iso_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_ts() -> str:
    # 파일명 충돌 방지용 (micro 포함)
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _dump_event(event: EventIngestRequest) -> dict:
    # pydantic v1/v2 호환
    if hasattr(event, "model_dump"):
        return event.model_dump(by_alias=True, exclude_none=True)
    return event.dict(by_alias=True, exclude_none=True)


def _safe_join(base: Path, name: str) -> Path:
    """
    path traversal 방지용 safe join
    - runId에 ../ 등이 들어가도 EVENTS_ROOT 밖으로 못 나가게 차단
    """
    base_r = base.resolve()
    p = (base_r / name).resolve()
    if base_r == p:
        return p
    if base_r not in p.parents:
        raise HTTPException(status_code=400, detail="invalid runId")
    return p


def _event_files(run_dir: Path) -> List[Path]:
    """
    <run_dir> 아래 *.json 파일을 mtime 내림차순으로 정렬해 반환
    """
    files = [p for p in run_dir.glob("*.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------
# POST: ingest event (payload-only)
# ---------------------------------------------------------------------
@router.post(
    "",
    response_model=ApiEnvelope[None],
    summary="V1 이벤트 수신(외부/내부 콜백)",
    description=(
        "V1 Auto-Labeling 파이프라인에서 발생한 이벤트를 수신합니다.\n\n"
        "## 요청 포맷\n"
        "- 요청 바디는 **EventIngestRequest(payload-only)** 입니다.\n"
        "- 최소 필수:\n"
        "  - eventType\n"
        "  - runId\n\n"
        "## 저장\n"
        "- logs/events/<runId>/<ts>_<eventType>.json\n\n"
        "## 응답\n"
        "- ApiEnvelope로 ACK만 반환합니다."
    ),
)
def post_event(
    event: EventIngestRequest = Body(...),
):
    # validate (DTO에서 대부분 보장되지만 메시지 친절하게)
    # ✅ CamelModel이지만 내부 접근은 snake_case 우선
    if not getattr(event, "event_type", None):
        return ApiEnvelope(resultCode="FAIL", errorCode="EVT_4001", message="eventType is required", data=None)

    if not getattr(event, "run_id", None):
        return ApiEnvelope(resultCode="FAIL", errorCode="EVT_4002", message="runId is required", data=None)

    run_id = str(event.run_id).strip()
    event_type = str(event.event_type).strip()
    ts = _safe_ts()

    # ✅ POST에서도 safe_join 적용(보안 + 경로 꼬임 방지)
    out_dir = _safe_join(EVENTS_ROOT, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{ts}_{event_type}.json"
    out_file.write_text(
        json.dumps(
            {"receivedAt": _now_iso_z(), "event": _dump_event(event)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return ApiEnvelope(resultCode="SUCCESS", message="Event received", data=None)


# ---------------------------------------------------------------------
# GET 1) runs: runId 디렉토리 목록(최근순)
# ---------------------------------------------------------------------
@router.get(
    "/runs",
    response_model=ApiEnvelope[List[Dict[str, Any]]],
    summary="이벤트 run 목록 조회(최근순)",
    description="logs/events/ 아래 runId 디렉토리 목록을 최근 수정 시각 기준으로 반환합니다.",
)
def list_event_runs(
    limit: int = Query(50, ge=1, le=500),
):
    if not EVENTS_ROOT.exists():
        return ApiEnvelope(resultCode="SUCCESS", message="ok", data=[])

    runs = [p for p in EVENTS_ROOT.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    runs = runs[: int(limit)]

    items: List[Dict[str, Any]] = []
    for r in runs:
        items.append(
            {
                "runId": r.name,
                "runPath": str(r.resolve()),
                "updatedAt": datetime.fromtimestamp(r.stat().st_mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )

    return ApiEnvelope(resultCode="SUCCESS", message="ok", data=items)


# ---------------------------------------------------------------------
# GET 2) get: runId 이벤트 목록(pagination)
# ---------------------------------------------------------------------
@router.get(
    "/get",
    response_model=ApiEnvelope[Dict[str, Any]],
    summary="특정 run의 이벤트 목록 조회(pagination)",
)
def get_events(
    runId: str = Query(..., description="조회할 runId"),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5000),
):
    run_dir = _safe_join(EVENTS_ROOT, runId)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"run not found: {runId}")

    files = _event_files(run_dir)
    total = len(files)

    s = int(offset)
    e = min(s + int(limit), total)
    page = files[s:e]

    rows: List[dict] = []
    for p in page:
        j = _read_json(p)
        if j is not None:
            j["_fileName"] = p.name
            rows.append(j)

    data = {
        "runId": runId,
        "total": total,
        "offset": s,
        "limit": int(limit),
        "items": rows,
    }
    return ApiEnvelope(resultCode="SUCCESS", message="ok", data=data)


# ---------------------------------------------------------------------
# GET 3) latest: 특정 이벤트 타입 최신 1개
# ---------------------------------------------------------------------
@router.get(
    "/latest",
    response_model=ApiEnvelope[Optional[Dict[str, Any]]],
    summary="특정 run의 최신 이벤트 1개 조회(타입 필터 가능)",
)
def get_latest_event(
    runId: str = Query(..., description="조회할 runId"),
    eventType: Optional[EventType] = Query(None, description="필터할 이벤트 타입(선택)"),
):
    run_dir = _safe_join(EVENTS_ROOT, runId)
    if not run_dir.exists():
        # 폴링 용도에서는 404보다 SUCCESS + null이 편할 때가 많음
        return ApiEnvelope(resultCode="SUCCESS", message="run not found", data=None)

    files = _event_files(run_dir)
    if not files:
        return ApiEnvelope(resultCode="SUCCESS", message="no events", data=None)

    for p in files:
        if eventType is not None and f"_{eventType}.json" not in p.name:
            continue
        j = _read_json(p)
        if j is not None:
            j["_fileName"] = p.name
            return ApiEnvelope(resultCode="SUCCESS", message="ok", data=j)

    return ApiEnvelope(resultCode="SUCCESS", message="no matching event", data=None)
