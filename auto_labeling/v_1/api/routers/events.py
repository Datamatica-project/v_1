from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from fastapi import APIRouter, HTTPException, Body

from auto_labeling.v_1.api.dto.common import ApiEnvelope
from auto_labeling.v_1.api.dto.event import EventPayload

router = APIRouter(prefix="/events")

# ---------------------------------------------------------------------
# storage config
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("/workspace")
EVENTS_ROOT = PROJECT_ROOT / "auto_labeling" / "v_1" / "logs" / "events"


def _now_iso_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_ts() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _dump_event(event: EventPayload) -> dict:
    # pydantic v1/v2 호환
    if hasattr(event, "model_dump"):
        return event.model_dump(by_alias=True, exclude_none=True)
    return event.dict(by_alias=True, exclude_none=True)


@router.post(
    "",
    response_model=ApiEnvelope[None],
    summary="V1 이벤트 수신(외부/내부 콜백)",
    description=(
        "V1 Auto-Labeling 파이프라인에서 발생한 이벤트를 수신합니다.\n\n"
        "## 이 API의 의미\n"
        "- Worker/Orchestrator/Backend/FE 등 외부(또는 내부) 시스템이 loop 진행 상황을 추적하기 위한 **콜백/로그** API입니다.\n"
        "- 이 endpoint는 이벤트를 '수신'하고 ACK를 반환합니다. (저장/전달 정책은 프로젝트 구현에 따름)\n\n"
        "## 요청 포맷\n"
        "- 요청 바디는 `ApiEnvelope[EventPayload]` 입니다.\n"
        "- `data`(EventPayload)는 필수이며, 최소한 아래는 반드시 포함해야 합니다:\n"
        "  - data.eventType: 이벤트 타입\n"
        "  - data.runId: 실행(run) 식별자\n\n"
        "## eventType 사용 가이드\n"
        "- ROUND0_EXPORTED: round0 export 완료/실패 알림(주로 exportRelPath/manifestRelPath 포함)\n"
        "- ROUND_RESULT: 특정 round 완료 결과(주로 round, passCount/failCount/missCount 포함)\n"
        "- LOOP_FINISHED: 전체 loop 정상 종료\n"
        "- LOOP_FAILED: 전체 loop 실패 종료\n\n"
        "## 에러/정책\n"
        "- data가 없으면 400\n"
        "- eventType/runId가 없으면 ApiEnvelope(resultCode=FAIL, errorCode=...)로 응답\n"
    ),
)
def post_event(
    req: ApiEnvelope[EventPayload] = Body(...),
):
    # -----------------------------
    # validate request
    # -----------------------------
    if not req.data:
        raise HTTPException(status_code=400, detail="data is required")

    event = req.data

    if not getattr(event, "eventType", None):
        return ApiEnvelope(
            resultCode="FAIL",
            errorCode="EVT_4001",
            message="eventType is required",
            data=None,
        )

    if not getattr(event, "runId", None):
        return ApiEnvelope(
            resultCode="FAIL",
            errorCode="EVT_4002",
            message="runId is required",
            data=None,
        )

    # -----------------------------
    # side effect: persist event log
    # logs/events/<runId>/<timestamp>_<eventType>.json
    # -----------------------------
    run_id = event.runId
    event_type = event.eventType
    ts = _safe_ts()

    out_dir = EVENTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{ts}_{event_type}.json"
    out_file.write_text(
        json.dumps(
            {
                "receivedAt": _now_iso_z(),
                "event": _dump_event(event),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------
    # ack
    # -----------------------------
    return ApiEnvelope(
        resultCode="SUCCESS",
        message="Event received",
        data=None,
    )
