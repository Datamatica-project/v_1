"""
모델별 GT/Unlabeled 데이터 업로드 및 등록 DTO (앙상블 방식)

각 모델(YOLO, Model2, Model3)별로 독립적인 GT 및 Unlabeled 데이터 관리
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import Field
from .base import CamelModel


# ========================================
# GT 업로드 및 등록
# ========================================
class GTUploadRequest(CamelModel):
    """
    GT ZIP 업로드 요청 (multipart/form-data)

    파일은 FastAPI UploadFile로 별도 처리
    이 DTO는 추가 메타데이터용

    Example:
        {
            "modelName": "yolo",
            "sourceName": "client_A",
            "datasetName": "202501_batch1"
        }
    """
    model_name: Literal["yolo", "model2", "model3"] = Field(
        ...,
        alias="modelName",
        description="모델 이름 (yolo, model2, model3)"
    )

    source_name: Optional[str] = Field(
        default=None,
        alias="sourceName",
        description="GT 제공처 (선택)"
    )

    dataset_name: Optional[str] = Field(
        default=None,
        alias="datasetName",
        description="데이터셋 이름 (선택)"
    )


class GTUploadResponse(CamelModel):
    """
    GT ZIP 업로드 응답

    Example:
        {
            "ingestId": "gt_yolo_20250106_120000_abc123",
            "modelName": "yolo",
            "status": "UPLOADED",
            "extractedPath": "/workspace/.../raw_ingest/gt_yolo_xxx/extracted",
            "next": "/api/v2/yolo/gt/register?ingestId=gt_yolo_20250106_120000_abc123"
        }
    """
    ingest_id: str = Field(
        ...,
        alias="ingestId",
        description="Ingest 식별자"
    )

    model_name: str = Field(
        ...,
        alias="modelName",
        description="모델 이름"
    )

    status: str = Field(
        ...,
        description="업로드 상태 (UPLOADED, FAILED)"
    )

    extracted_path: Optional[str] = Field(
        default=None,
        alias="extractedPath",
        description="압축 해제된 경로"
    )

    next: Optional[str] = Field(
        default=None,
        description="다음 단계 API 엔드포인트"
    )

    error: Optional[str] = Field(
        default=None,
        description="에러 메시지 (실패 시)"
    )


class GTRegisterRequest(CamelModel):
    """
    GT 등록 요청 (표준화 + 현재 GT 갱신)

    Example:
        {
            "ingestId": "gt_yolo_20250106_120000_abc123",
            "copyMode": "symlink",
            "strict": false
        }
    """
    ingest_id: str = Field(
        ...,
        alias="ingestId",
        description="Ingest 식별자"
    )

    copy_mode: Literal["symlink", "copy"] = Field(
        default="symlink",
        alias="copyMode",
        description="복사 모드 (symlink | copy)"
    )

    strict: bool = Field(
        default=False,
        description="엄격 모드 (라벨 검증 실패 시 중단)"
    )


class GTRegisterResponse(CamelModel):
    """
    GT 등록 응답

    Example:
        {
            "ingestId": "gt_yolo_20250106_120000_abc123",
            "modelName": "yolo",
            "status": "DONE",
            "registeredPath": ".../data/gt_data/yolo/GT_gt_yolo_xxx",
            "currentGTPath": ".../data/gt_data/yolo/GT.file",
            "summary": {
                "ok": 1500,
                "skip": 0,
                "error": 0
            }
        }
    """
    ingest_id: str = Field(
        ...,
        alias="ingestId",
        description="Ingest 식별자"
    )

    model_name: str = Field(
        ...,
        alias="modelName",
        description="모델 이름"
    )

    status: str = Field(
        ...,
        description="등록 상태 (DONE, FAILED)"
    )

    registered_path: Optional[str] = Field(
        default=None,
        alias="registeredPath",
        description="등록된 GT 버전 경로"
    )

    current_gt_path: Optional[str] = Field(
        default=None,
        alias="currentGTPath",
        description="현재 활성 GT 경로 (symlink)"
    )

    summary: Optional[Dict[str, int]] = Field(
        default=None,
        description="처리 요약 (ok, skip, error)"
    )

    error: Optional[str] = Field(
        default=None,
        description="에러 메시지 (실패 시)"
    )


# ========================================
# Unlabeled 업로드
# ========================================
class UnlabeledUploadRequest(CamelModel):
    """
    Unlabeled 이미지 ZIP 업로드 요청

    Example:
        {
            "modelName": "yolo",
            "datasetName": "202501_unlabeled_batch1"
        }
    """
    model_name: Literal["yolo", "model2", "model3"] = Field(
        ...,
        alias="modelName",
        description="모델 이름 (yolo, model2, model3)"
    )

    dataset_name: Optional[str] = Field(
        default=None,
        alias="datasetName",
        description="데이터셋 이름 (선택)"
    )


class UnlabeledUploadResponse(CamelModel):
    """
    Unlabeled 업로드 응답

    Example:
        {
            "ingestId": "unlabeled_yolo_20250106_120000_xyz789",
            "modelName": "yolo",
            "status": "DONE",
            "addedImages": 500,
            "unlabeledDir": ".../data/unlabeled/yolo/images"
        }
    """
    ingest_id: str = Field(
        ...,
        alias="ingestId",
        description="Ingest 식별자"
    )

    model_name: str = Field(
        ...,
        alias="modelName",
        description="모델 이름"
    )

    status: str = Field(
        ...,
        description="업로드 상태 (DONE, FAILED)"
    )

    added_images: int = Field(
        0,
        alias="addedImages",
        ge=0,
        description="추가된 이미지 수"
    )

    unlabeled_dir: Optional[str] = Field(
        default=None,
        alias="unlabeledDir",
        description="Unlabeled 이미지 디렉토리"
    )

    error: Optional[str] = Field(
        default=None,
        description="에러 메시지 (실패 시)"
    )


# ========================================
# GT/Unlabeled 목록 조회
# ========================================
class GTVersionInfo(CamelModel):
    """
    GT 버전 정보

    Example:
        {
            "versionId": "GT_gt_yolo_20250106_120000_abc123",
            "modelName": "yolo",
            "isCurrent": true,
            "createdAt": "2025-01-06T12:00:00+09:00",
            "imageCount": 1500,
            "sourceName": "client_A",
            "datasetName": "202501_batch1"
        }
    """
    version_id: str = Field(
        ...,
        alias="versionId",
        description="GT 버전 ID"
    )

    model_name: str = Field(
        ...,
        alias="modelName",
        description="모델 이름"
    )

    is_current: bool = Field(
        False,
        alias="isCurrent",
        description="현재 활성 GT 여부"
    )

    created_at: Optional[datetime] = Field(
        None,
        alias="createdAt",
        description="생성 시각"
    )

    image_count: int = Field(
        0,
        alias="imageCount",
        ge=0,
        description="이미지 수"
    )

    source_name: Optional[str] = Field(
        default=None,
        alias="sourceName",
        description="GT 제공처"
    )

    dataset_name: Optional[str] = Field(
        default=None,
        alias="datasetName",
        description="데이터셋 이름"
    )


class ListGTVersionsResponse(CamelModel):
    """
    GT 버전 목록 조회 응답

    Example:
        {
            "modelName": "yolo",
            "versions": [
                {
                    "versionId": "GT_gt_yolo_20250106_120000_abc123",
                    "modelName": "yolo",
                    "isCurrent": true,
                    "createdAt": "2025-01-06T12:00:00+09:00",
                    "imageCount": 1500
                }
            ]
        }
    """
    model_name: str = Field(
        ...,
        alias="modelName",
        description="모델 이름"
    )

    versions: list[GTVersionInfo] = Field(
        default_factory=list,
        description="GT 버전 목록"
    )


class UnlabeledInfoResponse(CamelModel):
    """
    Unlabeled 이미지 정보 조회 응답

    Example:
        {
            "modelName": "yolo",
            "imageCount": 500,
            "unlabeledDir": ".../data/unlabeled/yolo/images"
        }
    """
    model_name: str = Field(
        ...,
        alias="modelName",
        description="모델 이름"
    )

    image_count: int = Field(
        0,
        alias="imageCount",
        ge=0,
        description="이미지 수"
    )

    unlabeled_dir: Optional[str] = Field(
        default=None,
        alias="unlabeledDir",
        description="Unlabeled 디렉토리"
    )
