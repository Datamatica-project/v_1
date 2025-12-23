from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import Field
from .base import CamelModel

class RunLoopRequest(CamelModel):
    cfg_path: Optional[str] = Field(
        None,
        description=(
            "Loop 실행에 사용할 설정 YAML 경로.\n"
            "미지정 시 서버 기본 loop 설정(cfg)을 사용."
        ),
        examples=["auto_labeling/v_1/configs/v1_loop.yaml"],
    )

    student_weight: Optional[str] = Field(
        None,
        description=(
            "초기 student 모델 가중치(.pt) 경로.\n"
            "미지정 시 registry에 등록된 최신 student weight 또는 기본값 사용."
        ),
        examples=["models/user_yolo/v1_000/best.pt"],
    )

    teacher_weight: Optional[str] = Field(
        None,
        description=(
            "Teacher 모델 가중치(.pt) 경로 override.\n"
            "미지정 시 teacher_model.yaml에 정의된 기본 teacher 사용."
        ),
        examples=["models/teacher/weights/yolov11x_teacher.pt"],
    )

    export_root: Optional[str] = Field(
        None,
        description=(
            "Loop 결과(export)가 저장될 출력 루트 경로.\n"
            "미지정 시 서버 기본 export 경로 사용."
        ),
        examples=["data/exports/run_001"],
    )

    base_model: str = Field(
        "",
        alias="baseModel",
        description=(
            "GT 기반 초기 학습에 사용할 base 모델 가중치(.pt) 경로.\n"
            "빈 문자열 또는 미지정 시 서버 기본 base model 사용."
        ),
        examples=["models/pretrained/yolov11x.pt"],
    )

    gt_epochs: int = Field(
        30,
        alias="gtEpochs",
        description=(
            "GT 데이터로 student를 초기 학습할 때 사용할 epoch 수.\n"
            "미지정 시 기본값 30 사용."
        ),
        ge=1,
        examples=[30],
    )

    gt_imgsz: int = Field(
        640,
        alias="gtImgsz",
        description=(
            "GT 학습 시 입력 이미지 크기(imgsz).\n"
            "미지정 시 기본값 640 사용."
        ),
        ge=128,
        examples=[640],
    )

    gt_batch: int = Field(
        8,
        alias="gtBatch",
        description=(
            "GT 학습 시 batch size.\n"
            "미지정 시 기본값 8 사용."
        ),
        ge=1,
        examples=[8],
    )

class RunLoopResponse(CamelModel):
    job_id: str = Field(
        ...,
        description="비동기로 실행된 loop 작업의 job 식별자",
        examples=["job_20251219_172000"],
    )

    status: str = Field(
        ...,
        description=(
            "job의 초기 상태.\n"
            "일반적으로 QUEUED 또는 RUNNING."
        ),
        examples=["QUEUED"],
    )

class JobStatusResponse(CamelModel):
    """
    Loop 작업 상태 조회 응답 DTO
    """

    job_id: str = Field(
        ...,
        description="조회 대상 loop 작업의 job 식별자",
        examples=["job_20251219_172000"],
    )

    status: str = Field(
        ...,
        description=(
            "현재 job 상태.\n"
            "QUEUED | RUNNING | DONE | FAILED | NOT_FOUND"
        ),
        examples=["RUNNING"],
    )

    stats: Dict[str, Any] = Field(
        ...,
        description=(
            "job 실행 중/완료 시 수집된 상태 정보 및 통계.\n"
            "진행률, round 정보, 에러 메시지 등이 포함될 수 있음."
        ),
        examples=[{"round": 1, "progress": 0.42}],
    )
