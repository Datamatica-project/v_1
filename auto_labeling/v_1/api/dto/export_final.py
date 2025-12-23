from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class ExportFinalRequest(BaseModel):
    studentWeight: str = Field(
        ...,
        description="최종 export에 사용할 student 모델 가중치(.pt) 파일 경로 (로컬 경로)"
    )

    exportRoot: Optional[str] = Field(
        None,
        description="export 결과가 저장될 출력 디렉토리 경로 (기본값: data/final_export)"
    )

    exportConf: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="라벨 export 시 사용할 confidence threshold (0.0 ~ 1.0)"
    )

    passPool: Optional[str] = Field(
        None,
        description=(
            "PASS 이미지 풀을 고정해서 사용할 경우 지정하는 디렉토리 경로.\n"
            "지정하지 않으면 마지막 round 결과를 기준으로 자동 선택"
        )
    )

    mergePass: bool = Field(
        True,
        description=(
            "round별로 생성된 pass_fail 결과를 pass 디렉토리에 합칠지 여부.\n"
            "True일 경우 모든 round의 PASS 결과를 병합하여 export"
        )
    )

    roundRoot: Optional[str] = Field(
        None,
        description=(
            "특정 round 결과를 기준으로 export하고 싶을 때 사용하는 override 경로.\n"
            "예: data/round_r2"
        )
    )


class ExportFinalResponse(BaseModel):
    ok: bool = Field(
        True,
        description="export 성공 여부 (True: 성공, False: 실패)"
    )

    exportRoot: str = Field(
        ...,
        description="최종 export 결과가 저장된 디렉토리 경로"
    )

    usedStudentWeight: str = Field(
        ...,
        description="실제 export에 사용된 student 모델 가중치(.pt) 경로"
    )

    detail: Dict[str, Any] = Field(
        default_factory=dict,
        description="export 과정에 대한 상세 정보 또는 메타데이터"
    )
