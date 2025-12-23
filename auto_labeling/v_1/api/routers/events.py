from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Body

from auto_labeling.v_1.api.dto.common import ApiEnvelope
from auto_labeling.v_1.api.dto.event import EventPayload

router = APIRouter(prefix="/events")


def _now_iso_z() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


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
    responses={
        200: {"description": "이벤트 수신 ACK(ApiEnvelope)"},
        400: {"description": "data 누락 등 요청 형식 오류"},
    },
)
def post_event(
    req: ApiEnvelope[EventPayload] = Body(
        ...,
        description=(
            "이벤트 수신 요청.\n\n"
            "### 구조\n"
            "{\n"
            '  "resultCode": "SUCCESS",   # (요청에서는 보통 의미 없음, 무시 가능)\n'
            '  "errorCode": null,\n'
            '  "message": null,\n'
            '  "data": {\n'
            '    "eventType": "ROUND_RESULT",\n'
            '    "runId": "run_001",\n'
            '    "jobId": "job_20251219_172000",\n'
            '    "status": "DONE",\n'
            '    "timestamp": "2025-12-19T07:09:32.872Z",\n'
            '    "round": 1,\n'
            '    "passCount": 1200,\n'
            '    "failCount": 340,\n'
            '    "missCount": 12,\n'
            '    "exportRelPath": "exports/run_001/round0",\n'
            '    "manifestRelPath": "exports/run_001/round0/manifest.json",\n'
            '    "extra": { "countUnit": "image" }\n'
            "  }\n"
            "}\n"
        ),
        examples=[
            {
                "resultCode": "SUCCESS",
                "errorCode": None,
                "message": None,
                "data": {
                    "eventType": "ROUND0_EXPORTED",
                    "runId": "run_001",
                    "jobId": "job_20251219_172000",
                    "status": "DONE",
                    "timestamp": "2025-12-19T07:09:32.872Z",
                    "round": 0,
                    "passCount": 1200,
                    "failCount": 340,
                    "missCount": 12,
                    "exportRelPath": "exports/run_001/round0",
                    "manifestRelPath": "exports/run_001/round0/manifest.json",
                    "extra": {"countUnit": "image"},
                },
            },
            {
                "resultCode": "SUCCESS",
                "errorCode": None,
                "message": None,
                "data": {
                    "eventType": "LOOP_FAILED",
                    "runId": "run_001",
                    "jobId": "job_20251219_172000",
                    "status": "FAILED",
                    "timestamp": "2025-12-19T07:20:11.000Z",
                    "message": "loop failed: out of disk",
                    "extra": {"node": "worker-0"},
                },
            },
        ],
    )
):
    # -----------------------------
    # validate request
    # -----------------------------
    if not req.data:
        # ApiEnvelope 형태라도 data 없으면 "요청 형식 오류"로 보는 게 명확함
        raise HTTPException(status_code=400, detail="data is required")

    event = req.data

    # eventType/runId가 없으면 envelope FAIL로 반환(스펙화)
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
    # side effect (현재는 print)
    # - 운영에서는 DB 저장, 큐 publish, 로그 적재 등을 수행할 수 있음
    # -----------------------------
    print("[EVENT]", event.model_dump())

    # -----------------------------
    # ack
    # -----------------------------
    return ApiEnvelope(
        resultCode="SUCCESS",
        message="Event received",
        data=None,
        # 필요하면 extra나 receivedAt 같은 걸 Envelope에 넣고 싶겠지만
        # 현재 ApiEnvelope 스키마가 없으니 message/data만 유지
    )
