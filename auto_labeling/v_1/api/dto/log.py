from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal

from pydantic import Field

from .base import CamelModel


# ----------------------------------------
# 공통 타입 정의
# ----------------------------------------
LogLevel = Literal["INFO", "WARN", "ERROR", "DEBUG"]
LogScope = Literal["ingest", "loop", "round", "export", "system"]


# ========================================
# 로그 단건 아이템
# ========================================
class LogItem(CamelModel):
    """
    시스템 내부에서 발생한 단일 로그 항목
    """

    timestamp: datetime = Field(
        ...,
        description=(
            "로그가 기록된 시각.\n"
            "ISO8601 형식, KST(+09:00) 또는 UTC 기준"
        ),
        examples=["2025-12-19T16:20:01.123+09:00"],
    )

    level: LogLevel = Field(
        ...,
        description="로그 레벨 (INFO | WARN | ERROR | DEBUG)",
        examples=["INFO"],
    )

    scope: LogScope = Field(
        ...,
        description=(
            "로그가 발생한 기능 영역(scope).\n"
            "- ingest : 데이터 인입/등록\n"
            "- loop   : auto-labeling loop 전체\n"
            "- round  : 개별 round 처리\n"
            "- export : export 처리\n"
            "- system : 시스템/인프라 레벨"
        ),
        examples=["loop"],
    )

    ref_id: Optional[str] = Field(
        None,
        description=(
            "로그가 참조하는 식별자.\n"
            "예: runId, jobId, ingestId, roundId 등"
        ),
        examples=["run_001"],
    )

    message: str = Field(
        ...,
        description="로그 메시지 본문 (사람이 읽는 설명)",
        examples=["round 1 training started"],
    )

    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="추가 메타데이터 (파라미터, 통계, 에러 상세 등 자유 형식)",
        examples=[{"epoch": 1, "batch": 8}],
    )


# ========================================
# 로그 목록 응답
# ========================================
class LogListResponse(CamelModel):
    """
    로그 목록 조회 API 응답 DTO
    """

    items: List[LogItem] = Field(
        ...,
        description="로그 항목 리스트 (최신순 또는 시간순 정렬)"
    )

    total: int = Field(
        ...,
        description="조건에 맞는 전체 로그 개수",
        examples=[120],
        ge=0,
    )

    has_more: bool = Field(
        ...,
        description="다음 페이지 로그가 더 있는지 여부",
        examples=[False],
    )


# ========================================
# 진행률 조회 응답
# ========================================
class ProgressResponse(CamelModel):
    """
    특정 작업(run/round/export 등)의 현재 진행 상태를 나타내는 DTO
    """

    ref_id: str = Field(
        ...,
        description="진행 상황을 추적하는 기준 식별자 (runId, jobId 등)",
        examples=["job_20251219_172000"],
    )

    scope: LogScope = Field(
        ...,
        description="진행 상황이 속한 영역(scope)",
        examples=["round"],
    )

    stage: str = Field(
        ...,
        description=(
            "현재 처리 단계 이름.\n"
            "예: ingesting | training | predicting | exporting"
        ),
        examples=["training"],
    )

    progress: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "전체 대비 진행률 (0.0 ~ 1.0).\n"
            "0.5 = 50% 진행"
        ),
        examples=[0.42],
    )

    message: str = Field(
        ...,
        description="현재 단계에 대한 간단한 설명 메시지",
        examples=["epoch 2 / 5 진행 중"],
    )

    updated_at: datetime = Field(
        ...,
        description="진행 정보가 마지막으로 갱신된 시각 (ISO8601)",
        examples=["2025-12-19T16:22:10.000+09:00"],
    )


# ========================================
# 로그 생성 요청 (POST)
# ========================================
class LogCreateRequest(CamelModel):
    """
    외부 또는 내부 모듈에서 로그를 생성할 때 사용하는 요청 DTO
    """

    level: LogLevel = Field(
        ...,
        description="로그 레벨 (INFO | WARN | ERROR | DEBUG)",
        examples=["ERROR"],
    )

    scope: LogScope = Field(
        ...,
        description="로그가 속하는 기능 영역(scope)",
        examples=["export"],
    )

    ref_id: Optional[str] = Field(
        None,
        description="연관된 식별자 (runId, jobId, ingestId 등)",
        examples=["run_001"],
    )

    message: str = Field(
        ...,
        description="로그 메시지 본문",
        examples=["export failed: no disk space"],
    )

    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="추가 메타데이터 (에러 상세, 파라미터 등 자유 형식)",
        examples=[{"freeDiskMB": 120}],
    )
