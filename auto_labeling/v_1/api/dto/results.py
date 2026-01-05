# auto_labeling/v_1/api/dto/results.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import Field

from .base import CamelModel


# ----------------------------------------
# 공통 타입
# ----------------------------------------
RoundExportStatus = Literal["READY", "DONE", "FAILED", "UNKNOWN"]


# ========================================
# Run 목록 / RunInfo
# ========================================
class RunInfo(CamelModel):
    """
    프론트가 'run 단위'로 결과를 탐색할 때 필요한 최소 정보.
    - runId: 결과 묶음 키
    - status: DONE/FAILED/READY 등 (round0 export 기준 or loop 종료 기준)
    - createdAt: 생성 시각
    - exportRound0Ready: round0 export 결과를 조회할 수 있는지 여부
    """

    run_id: str = Field(
        ...,
        alias="runId",
        description="실행(run) 식별자. 결과 묶음 단위의 키.",
        examples=["run_001"],
    )

    status: RoundExportStatus = Field(
        "UNKNOWN",
        description="run의 대표 상태(보통 round0 export 또는 loop 종료 상태 기반).",
        examples=["DONE"],
    )

    created_at: Optional[datetime] = Field(
        None,
        alias="createdAt",
        description="run 생성 시각(ISO8601).",
        examples=["2025-12-19T16:09:32+09:00"],
    )

    updated_at: Optional[datetime] = Field(
        None,
        alias="updatedAt",
        description="run 갱신 시각(ISO8601).",
        examples=["2025-12-19T16:20:10+09:00"],
    )

    export_round0_ready: bool = Field(
        False,
        alias="exportRound0Ready",
        description="round0 export 결과가 준비되어 조회 가능한지 여부.",
        examples=[True],
    )

    # (선택) 연계 키
    job_id: Optional[str] = Field(
        None,
        alias="jobId",
        description="해당 run과 연계된 비동기 jobId(있을 때만).",
        examples=["job_20251219_172000"],
    )

    # (선택) 표시용 메모
    note: Optional[str] = Field(
        None,
        description="표시용 메모(오류 요약, 모드 등).",
        examples=["demo_always_train_gt"],
    )


class ListRunsResponse(CamelModel):
    items: List[RunInfo] = Field(
        default_factory=list,
        description="run 목록(최근순 권장).",
    )


# ========================================
# Round0 결과 조회 (프론트 친화)
# - export/round0/{runId}를 그대로 내리기보다,
#   프론트가 쓰기 쉬운 형태로 한 번 더 가공한 스펙
# ========================================
class Round0ResultResponse(CamelModel):
    run_id: str = Field(
        ...,
        alias="runId",
        description="조회 대상 runId",
        examples=["run_001"],
    )

    status: RoundExportStatus = Field(
        "UNKNOWN",
        description="round0 export 상태(READY | DONE | FAILED | UNKNOWN).",
        examples=["DONE"],
    )

    message: Optional[str] = Field(
        None,
        description="상태 요약 메시지(실패 사유, 완료 메시지 등).",
        examples=["round0 export completed"],
    )

    # 카운트
    pass_count: int = Field(
        0,
        alias="passCount",
        description="PASS 개수",
        ge=0,
        examples=[1200],
    )

    fail_count: int = Field(
        0,
        alias="failCount",
        description="FAIL 개수",
        ge=0,
        examples=[340],
    )

    miss_count: int = Field(
        0,
        alias="missCount",
        description="MISS 개수",
        ge=0,
        examples=[12],
    )

    # 경로(상대)
    export_rel_path: Optional[str] = Field(
        None,
        alias="exportRelPath",
        description="NAS/서버 기준 상대 경로 (예: exports/run_001/round0).",
        examples=["exports/run_001/round0"],
    )

    manifest_rel_path: Optional[str] = Field(
        None,
        alias="manifestRelPath",
        description="manifest JSON 상대 경로",
        examples=["exports/run_001/round0/manifest.json"],
    )

    # 경로(접근)
    download_base_url: Optional[str] = Field(
        None,
        alias="downloadBaseUrl",
        description=(
            "export 결과 접근 base URL 또는 UNC 경로.\n"
            "환경에 따라 http(s) URL 또는 \\\\DS... 형태가 될 수 있음."
        ),
        examples=[r"\\DS1821_1\V1_EXPORTS\exports\run_001\round0"],
    )

    manifest_url: Optional[str] = Field(
        None,
        alias="manifestUrl",
        description="manifest JSON 접근 URL/UNC 경로",
        examples=[r"\\DS1821_1\V1_EXPORTS\exports\run_001\round0\manifest.json"],
    )

    # (선택) 다운로드 zip URL을 API가 제공할 경우
    zip_url: Optional[str] = Field(
        None,
        alias="zipUrl",
        description=(
            "API 서버가 zip 생성/서빙을 제공하는 경우 zip 다운로드 URL.\n"
            "예: /api/v1/results/download?runId=run_001"
        ),
        examples=["/api/v1/results/download?runId=run_001"],
    )

    created_at: Optional[datetime] = Field(
        None,
        alias="createdAt",
        description="round0 결과 생성 시각",
    )

    updated_at: Optional[datetime] = Field(
        None,
        alias="updatedAt",
        description="round0 결과 갱신 시각",
    )

    # 확장
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="확장 메타(옵션/모드/환경 정보 등).",
        examples=[{"countUnit": "image"}],
    )


# ========================================
# (선택) Pagination 형태로 "results/get" 같은 API를 만들 때 사용
# ========================================
class ResultItem(CamelModel):
    """
    결과 리스트에서 한 row에 해당하는 아이템.
    - 결과를 '파일 단위'로 제공할 때 유용.
    - V1에서는 round0 export가 zip/manifest 중심이면 최소만 유지해도 됨.
    """

    name: str = Field(
        ...,
        description="결과 항목 이름(파일명/샘플 ID 등).",
        examples=["000123.jpg"],
    )

    which: Literal["pass", "fail", "miss", "unknown"] = Field(
        "unknown",
        description="PASS/FAIL/MISS 분류",
        examples=["pass"],
    )

    # (선택) 접근 경로들
    rel_path: Optional[str] = Field(
        None,
        alias="relPath",
        description="exportRelPath 기준 상대 경로",
        examples=["pass/images/000123.jpg"],
    )

    url: Optional[str] = Field(
        None,
        description="직접 접근 URL(서버가 서빙하는 경우).",
        examples=["/api/v1/results/file?runId=run_001&name=pass/images/000123.jpg"],
    )

    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="확장 메타(필요 시 bbox 요약/score 등).",
        examples=[{"nBoxes": 3}],
    )


class GetResultsResponse(CamelModel):
    """
    (선택) pagination 기반 결과 조회 응답.
    """
    run_id: str = Field(..., alias="runId")
    total: int = Field(..., description="전체 결과 개수", ge=0)
    offset: int = Field(..., description="pagination 시작 index", ge=0)
    limit: int = Field(..., description="pagination page size", ge=1)
    items: List[ResultItem] = Field(default_factory=list, description="결과 아이템 리스트")
