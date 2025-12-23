from __future__ import annotations

from typing import Any, Dict, Optional, Literal
from pydantic import Field

from .base import CamelModel

EventType = Literal[
    "ROUND0_EXPORTED",
    "ROUND_RESULT",
    "LOOP_FINISHED",
    "LOOP_FAILED",
]


class EventPayload(CamelModel):
    eventType: EventType = Field(
        ...,
        description=(
            "이벤트 타입.\n"
            "- ROUND0_EXPORTED: round0 export 완료/실패 알림\n"
            "- ROUND_RESULT: 특정 round 결과 알림\n"
            "- LOOP_FINISHED: 전체 loop 정상 종료\n"
            "- LOOP_FAILED: 전체 loop 실패 종료"
        ),
        examples=["ROUND0_EXPORTED"],
    )
    runId: str = Field(
        ...,
        description="외부 추적용 run 식별자(보통 loop 실행 단위).",
        examples=["run_001"],
    )
    jobId: Optional[str] = Field(
        None,
        description="비동기 작업(job) 식별자. loop/run과 연계될 때 사용.",
        examples=["job_20251219_172000"],
    )
    status: str = Field(
        "DONE",
        description=(
            "이 이벤트가 보고하는 상태 문자열.\n"
            "권장 값: QUEUED | RUNNING | DONE | FAILED\n"
            "(시스템 내부 상태 체계에 맞게 확장 가능)"
        ),
        examples=["DONE"],
    )
    timestamp: Optional[str] = Field(
        None,
        description=(
            "이벤트 발생 시각(ISO8601).\n"
            "권장: KST(+09:00) 또는 UTC(Z)로 일관되게 사용.\n"
            "예: 2025-12-19T16:09:32.872+09:00"
        ),
        examples=["2025-12-19T16:09:32.872+09:00"],
    )
    message: Optional[str] = Field(
        None,
        description="사람이 읽는 메시지(성공/실패 요약, 경고/설명).",
        examples=["round0 export completed"],
    )

    round: Optional[int] = Field(
        None,
        description="round 기반 이벤트에서 round 번호. (ROUND_RESULT 등에서 주로 사용)",
        examples=[0, 1, 2],
    )

    passCount: Optional[int] = Field(
        None,
        description="PASS 개수(이미지 단위 또는 샘플 단위).",
        examples=[1200],
        ge=0,
    )
    failCount: Optional[int] = Field(
        None,
        description="FAIL 개수(이미지 단위 또는 샘플 단위).",
        examples=[340],
        ge=0,
    )
    missCount: Optional[int] = Field(
        None,
        description="MISS 개수(놓침/미분류/누락 케이스).",
        examples=[12],
        ge=0,
    )

    exportRelPath: Optional[str] = Field(
        None,
        description=(
            "export 결과 데이터가 저장된 위치를 가리키는 상대 경로.\n"
            "예: exports/<runId>/round0"
        ),
        examples=["exports/run_001/round0"],
    )
    manifestRelPath: Optional[str] = Field(
        None,
        description=(
            "이 export가 어떻게 만들어졌는지(조건/맥락)를 설명하는 manifest JSON 상대 경로.\n"
            "예: exports/<runId>/round0/manifest.json"
        ),
        examples=["exports/run_001/round0/manifest.json"],
    )


class EventRequest(CamelModel):
    eventType: EventType = Field(
        ...,
        description=(
            "이벤트 타입.\n"
            "- ROUND0_EXPORTED: round0 export 완료/실패\n"
            "- ROUND_RESULT: 특정 round 결과\n"
            "- LOOP_FINISHED: 전체 loop 정상 종료\n"
            "- LOOP_FAILED: 전체 loop 실패 종료"
        ),
        examples=["ROUND_RESULT"],
    )
    runId: str = Field(
        ...,
        description="외부 추적용 run 식별자(보통 loop 실행 단위).",
        examples=["run_001"],
    )

    jobId: Optional[str] = Field(
        None,
        description="비동기 작업(job) 식별자. loop/run과 연계될 때 사용.",
        examples=["job_20251219_172000"],
    )
    status: str = Field(
        "DONE",
        description="상태 문자열(권장: QUEUED | RUNNING | DONE | FAILED).",
        examples=["RUNNING"],
    )
    timestamp: Optional[str] = Field(
        None,
        description="이벤트 발생 시각(ISO8601). KST(+09:00) 또는 UTC(Z) 권장.",
        examples=["2025-12-19T16:10:01.100+09:00"],
    )
    message: Optional[str] = Field(
        None,
        description="사람이 읽는 메시지(성공/실패 요약, 경고/설명).",
        examples=["round 1 finished: fail_ratio improved"],
    )

    round: Optional[int] = Field(
        None,
        description="round 기반 이벤트에서 round 번호.",
        examples=[1],
    )

    passCount: Optional[int] = Field(
        None,
        description="PASS 개수(이미지/샘플 단위).",
        examples=[1500],
        ge=0,
    )
    failCount: Optional[int] = Field(
        None,
        description="FAIL 개수(이미지/샘플 단위).",
        examples=[220],
        ge=0,
    )
    missCount: Optional[int] = Field(
        None,
        description="MISS 개수(놓침/미분류/누락).",
        examples=[5],
        ge=0,
    )

    exportRelPath: Optional[str] = Field(
        None,
        description="export 결과 데이터 상대 경로(주로 ROUND0_EXPORTED에서 사용).",
        examples=["exports/run_001/round0"],
    )
    manifestRelPath: Optional[str] = Field(
        None,
        description="manifest JSON 상대 경로(주로 ROUND0_EXPORTED에서 사용).",
        examples=["exports/run_001/round0/manifest.json"],
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="확장 데이터(호환성 유지용). 스키마 밖 추가 정보는 여기에 넣는다.",
        examples=[{"teacherConfTh": 0.25, "gtAnchorRatio": 0.5}],
    )


class EventResponse(CamelModel):
    status: str = Field(
        "ACK",
        description="수신 ACK. 보통 'ACK' 고정(필요 시 'NACK' 확장 가능).",
        examples=["ACK"],
    )
    receivedAt: Optional[str] = Field(
        None,
        description="서버가 이벤트를 수신한 시각(ISO8601).",
        examples=["2025-12-19T16:10:02.000+09:00"],
    )
    runId: Optional[str] = Field(
        None,
        description="요청에 포함된 runId echo (추적용).",
        examples=["run_001"],
    )
    jobId: Optional[str] = Field(
        None,
        description="요청에 포함된 jobId echo (추적용).",
        examples=["job_20251219_172000"],
    )
