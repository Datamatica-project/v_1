from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, Literal

from pydantic import Field

from .base import CamelModel


# ----------------------------------------
# 공통 상태 타입
# ----------------------------------------
RoundExportStatus = Literal["READY", "DONE", "FAILED"]


# ========================================
# Round0 Export Notify (요청)
# ========================================
class Round0ExportNotifyRequest(CamelModel):
    """
    Round0 export 완료/실패 시 외부 시스템으로 전달되는 알림 요청 DTO
    """

    run_id: str = Field(
        ...,
        alias="runId",
        description="라운드0 export가 속한 실행(run) 식별자"
    )

    round: int = Field(
        0,
        description="라운드 번호 (round0 export이므로 기본값은 0)"
    )

    # 상태
    status: RoundExportStatus = Field(
        ...,
        description="Round0 export 처리 상태 (READY | DONE | FAILED)",
        examples=["DONE"]
    )

    message: Optional[str] = Field(
        None,
        description="상태에 대한 부가 설명 메시지 (실패 사유 또는 처리 요약)"
    )

    # 카운트
    pass_count: int = Field(
        default=0,
        alias="passCount",
        description="PASS로 분류된 이미지(또는 샘플) 개수",
        ge=0
    )

    fail_count: int = Field(
        default=0,
        alias="failCount",
        description="FAIL로 분류된 이미지(또는 샘플) 개수",
        ge=0
    )

    miss_count: int = Field(
        default=0,
        alias="missCount",
        description="MISS(누락/미검출) 이미지(또는 샘플) 개수",
        ge=0
    )

    # NAS 상대 경로 (shareRoot는 서버 고정)
    export_rel_path: str = Field(
        ...,
        alias="exportRelPath",
        description=(
            "Round0 export 결과 데이터가 저장된 NAS 상대 경로.\n"
            "실제 전체 경로는 서버에 고정된 shareRoot + exportRelPath 로 구성됨"
        )
    )

    manifest_rel_path: Optional[str] = Field(
        default=None,
        alias="manifestRelPath",
        description=(
            "Round0 export 결과를 생성한 조건과 맥락을 설명하는 "
            "manifest JSON 파일의 NAS 상대 경로"
        )
    )

    # 메타
    created_at: Optional[datetime] = Field(
        default=None,
        alias="createdAt",
        description=(
            "Round0 export 결과가 생성된 시각.\n"
            "ISO8601 형식, KST(+09:00) 또는 UTC 기준"
        )
    )

    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="확장 메타데이터 (실험 옵션, 내부 파라미터 등 자유 형식)"
    )


# ========================================
# Round0 Export Info (조회/응답)
# ========================================
class Round0ExportInfoResponse(CamelModel):
    """
    Round0 export 결과 조회 시 반환되는 응답 DTO
    """

    run_id: str = Field(
        ...,
        alias="runId",
        description="라운드0 export가 속한 실행(run) 식별자"
    )

    round: int = Field(
        ...,
        description="라운드 번호 (round0은 0)"
    )

    status: RoundExportStatus = Field(
        ...,
        description="Round0 export의 현재 또는 최종 상태 (READY | DONE | FAILED)"
    )

    message: Optional[str] = Field(
        None,
        description="상태에 대한 설명 메시지 (실패 사유 또는 처리 요약)"
    )

    pass_count: int = Field(
        ...,
        alias="passCount",
        description="PASS로 분류된 이미지(또는 샘플) 개수",
        ge=0
    )

    fail_count: int = Field(
        ...,
        alias="failCount",
        description="FAIL로 분류된 이미지(또는 샘플) 개수",
        ge=0
    )

    miss_count: int = Field(
        ...,
        alias="missCount",
        description="MISS(누락/미검출) 이미지(또는 샘플) 개수",
        ge=0
    )

    share_root: str = Field(
        ...,
        alias="shareRoot",
        description="NAS 또는 파일 서버의 기준 루트 경로 (서버 고정값)"
    )

    export_rel_path: str = Field(
        ...,
        alias="exportRelPath",
        description=(
            "Round0 export 결과 데이터가 저장된 NAS 상대 경로.\n"
            "전체 경로 = shareRoot + exportRelPath"
        )
    )

    manifest_rel_path: Optional[str] = Field(
        default=None,
        alias="manifestRelPath",
        description="manifest JSON 파일의 NAS 상대 경로"
    )

    download_base_url: Optional[str] = Field(
        default=None,
        alias="downloadBaseUrl",
        description=(
            "export 결과 파일을 다운로드할 때 사용하는 base URL.\n"
            "exportRelPath와 결합하여 실제 다운로드 URL을 구성"
        )
    )

    manifest_url: Optional[str] = Field(
        default=None,
        alias="manifestUrl",
        description="manifest JSON 파일의 직접 접근 URL"
    )

    created_at: Optional[datetime] = Field(
        default=None,
        alias="createdAt",
        description="Round0 export 결과 최초 생성 시각 (ISO8601)"
    )

    updated_at: Optional[datetime] = Field(
        default=None,
        alias="updatedAt",
        description="Round0 export 정보가 마지막으로 갱신된 시각 (ISO8601)"
    )

    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="확장 메타데이터 (추가 정보, 실험 옵션 등 자유 형식)"
    )
