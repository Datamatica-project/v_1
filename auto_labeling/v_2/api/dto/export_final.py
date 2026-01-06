# auto_labeling/v_1/api/dto/export_final.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, Literal

from pydantic import Field

from .base import CamelModel


FinalExportStatus = Literal["READY", "DONE", "FAILED"]


# ============================================================
# 1) 클라이언트(프론트) -> 백엔드: "final export 실행" 요청 DTO
# ============================================================
class ExportFinalRequest(CamelModel):
    student_weight: str = Field(
        ...,
        description="최종 export에 사용할 student 모델 가중치(.pt) 파일 경로 (로컬 경로)",
        alias="studentWeight",
    )

    export_root: Optional[str] = Field(
        None,
        description="export 결과가 저장될 출력 디렉토리 경로 (기본값: data/final_export)",
        alias="exportRoot",
    )

    export_conf: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="라벨 export 시 사용할 confidence threshold (0.0 ~ 1.0)",
        alias="exportConf",
    )

    pass_pool: Optional[str] = Field(
        None,
        description=(
            "PASS 이미지 풀을 고정해서 사용할 경우 지정하는 디렉토리 경로.\n"
            "지정하지 않으면 마지막 round 결과를 기준으로 자동 선택"
        ),
        alias="passPool",
    )

    merge_pass: bool = Field(
        True,
        description=(
            "round별로 생성된 pass_fail 결과를 pass 디렉토리에 합칠지 여부.\n"
            "True일 경우 모든 round의 PASS 결과를 병합하여 export"
        ),
        alias="mergePass",
    )

    round_root: Optional[str] = Field(
        None,
        description=(
            "특정 round 결과를 기준으로 export하고 싶을 때 사용하는 override 경로.\n"
            "예: data/round_r2"
        ),
        alias="roundRoot",
    )


class ExportFinalResponse(CamelModel):
    ok: bool = Field(
        True,
        description="export 성공 여부 (True: 성공, False: 실패)",
    )

    export_root: str = Field(
        ...,
        description="최종 export 결과가 저장된 디렉토리 경로",
        alias="exportRoot",
    )

    used_student_weight: str = Field(
        ...,
        description="실제 export에 사용된 student 모델 가중치(.pt) 경로",
        alias="usedStudentWeight",
    )

    detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="export 과정에 대한 상세 정보 또는 메타데이터",
    )


# ============================================================
# 2) (선택 A) 워커/파이프라인 -> 백엔드: "final export 완료 통지" DTO
#    - round0 방식 그대로 (registry + GET)로 가려면 이게 필요
# ============================================================
class ExportFinalNotifyRequest(CamelModel):
    run_id: str = Field(..., description="loop/export 실행을 식별하는 run id", alias="runId")

    status: FinalExportStatus = Field(..., description="final export 상태", alias="status")

    message: Optional[str] = Field(None, description="상태 메시지(에러 포함)", alias="message")

    # 결과 요약(선택)
    pass_count: Optional[int] = Field(0, ge=0, description="export된 pass 이미지 수", alias="passCount")
    pass_fail_count: Optional[int] = Field(0, ge=0, description="export된 pass_fail 이미지 수", alias="passFailCount")
    fail_fail_count: Optional[int] = Field(0, ge=0, description="export된 fail_fail 이미지 수", alias="failFailCount")
    miss_count: Optional[int] = Field(0, ge=0, description="export된 miss 이미지 수", alias="missCount")

    # 저장 위치(UNC/NAS 연동 고려)
    share_root: str = Field(
        "",
        description="NAS/UNC root (예: \\\\DS1821_1\\V1_EXPORTS). 없으면 빈 문자열",
        alias="shareRoot",
    )
    export_rel_path: str = Field(
        ...,
        description="share_root 기준 export 상대 경로 (예: run_001\\final_export)",
        alias="exportRelPath",
    )
    manifest_rel_path: Optional[str] = Field(
        None,
        description="(선택) manifest 상대 경로 (예: run_001\\final_export\\manifest.json)",
        alias="manifestRelPath",
    )

    # 시간: ms 없는 iso를 원하면, 생성하는 쪽에서 microsecond=0 유지 권장
    created_at: Optional[datetime] = Field(
        None,
        description="생성 시간(UTC 권장, ms 없이 내려가게 생성부에서 제어)",
        alias="createdAt",
    )

    extra: Optional[Dict[str, Any]] = Field(default_factory=dict, description="추가 메타", alias="extra")


class ExportFinalInfoResponse(CamelModel):
    run_id: str = Field(..., alias="runId")
    status: FinalExportStatus = Field(..., alias="status")
    message: Optional[str] = Field(None, alias="message")

    pass_count: int = Field(0, ge=0, alias="passCount")
    pass_fail_count: int = Field(0, ge=0, alias="passFailCount")
    fail_fail_count: int = Field(0, ge=0, alias="failFailCount")
    miss_count: int = Field(0, ge=0, alias="missCount")

    share_root: str = Field("", alias="shareRoot")
    export_rel_path: str = Field("", alias="exportRelPath")
    manifest_rel_path: Optional[str] = Field(None, alias="manifestRelPath")

    # 백엔드가 NAS URL 매핑을 제공하면 내려줄 수 있는 값들
    download_base_url: Optional[str] = Field(None, description="(선택) http download base url", alias="downloadBaseUrl")
    manifest_url: Optional[str] = Field(None, description="(선택) manifest url", alias="manifestUrl")

    created_at: Optional[datetime] = Field(None, alias="createdAt")
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")

    extra: Dict[str, Any] = Field(default_factory=dict, alias="extra")