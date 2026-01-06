"""
모델별 GT/Unlabeled 데이터 업로드 및 관리 Router (앙상블 방식)

각 모델(YOLO, Model2, Model3)별로 독립적인 엔드포인트 제공
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from tempfile import NamedTemporaryFile

from auto_labeling.v_2.api.dto import (
    GTUploadRequest,
    GTUploadResponse,
    GTRegisterRequest,
    GTRegisterResponse,
    UnlabeledUploadRequest,
    UnlabeledUploadResponse,
    ListGTVersionsResponse,
    GTVersionInfo,
    UnlabeledInfoResponse,
)
from auto_labeling.v_2.services.data_manager import GTManager, UnlabeledManager


router = APIRouter(prefix="/api/v2", tags=["Data Ingest (v2 Ensemble)"])


# ========================================
# YOLO GT/Unlabeled API
# ========================================
@router.post("/yolo/gt/upload", response_model=GTUploadResponse)
async def upload_yolo_gt(
    file: UploadFile = File(..., description="GT ZIP file (images/, labels/)"),
    source_name: str = Form(None, description="GT 제공처"),
    dataset_name: str = Form(None, description="데이터셋 이름")
):
    """
    YOLO 모델용 GT ZIP 업로드

    ZIP 구조:
    ```
    gt.zip
    ├── images/
    │   ├── img001.jpg
    │   └── ...
    └── labels/
        ├── img001.txt
        └── ...
    ```
    """
    manager = GTManager("yolo")

    # 임시 파일로 저장
    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = manager.upload_gt_zip(
            zip_path=tmp_path,
            source_name=source_name,
            dataset_name=dataset_name
        )

        if result["status"] == "FAILED":
            raise HTTPException(status_code=400, detail=result["error"])

        return GTUploadResponse(
            ingest_id=result["ingest_id"],
            model_name="yolo",
            status=result["status"],
            extracted_path=result["extracted_path"],
            next=f"/api/v2/yolo/gt/register?ingestId={result['ingest_id']}"
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/yolo/gt/register", response_model=GTRegisterResponse)
async def register_yolo_gt(
    ingest_id: str = Query(..., alias="ingestId", description="Ingest 식별자"),
    copy_mode: Literal["symlink", "copy"] = Query("symlink", alias="copyMode"),
    strict: bool = Query(False, description="엄격 모드 (라벨 검증 실패 시 중단)")
):
    """
    YOLO GT 등록 (표준화 + 현재 GT 갱신)

    - 라벨 검증
    - GT 버전 디렉토리 생성
    - 현재 GT 심볼릭 링크 갱신
    """
    manager = GTManager("yolo")

    result = manager.register_gt(
        ingest_id=ingest_id,
        copy_mode=copy_mode,
        strict=strict
    )

    if result["status"] == "FAILED":
        raise HTTPException(status_code=400, detail=result["error"])

    return GTRegisterResponse(
        ingest_id=result["ingest_id"],
        model_name="yolo",
        status=result["status"],
        registered_path=result["registered_path"],
        current_gt_path=result["current_gt_path"],
        summary=result["summary"]
    )


@router.get("/yolo/gt/versions", response_model=ListGTVersionsResponse)
async def list_yolo_gt_versions():
    """YOLO GT 버전 목록 조회"""
    manager = GTManager("yolo")
    versions = manager.list_gt_versions()

    return ListGTVersionsResponse(
        model_name="yolo",
        versions=[
            GTVersionInfo(
                version_id=v["version_id"],
                model_name="yolo",
                is_current=v["is_current"],
                created_at=v["created_at"],
                image_count=v["image_count"]
            )
            for v in versions
        ]
    )


@router.post("/yolo/unlabeled/upload", response_model=UnlabeledUploadResponse)
async def upload_yolo_unlabeled(
    file: UploadFile = File(..., description="Unlabeled 이미지 ZIP"),
    dataset_name: str = Form(None, description="데이터셋 이름")
):
    """
    YOLO 모델용 Unlabeled 이미지 업로드

    ZIP에서 모든 이미지 파일을 추출하여 data/unlabeled/yolo/images/에 추가
    """
    manager = UnlabeledManager("yolo")

    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = manager.upload_unlabeled_zip(
            zip_path=tmp_path,
            dataset_name=dataset_name
        )

        if result["status"] == "FAILED":
            raise HTTPException(status_code=400, detail=result["error"])

        return UnlabeledUploadResponse(
            ingest_id=result["ingest_id"],
            model_name="yolo",
            status=result["status"],
            added_images=result["added_images"],
            unlabeled_dir=result["unlabeled_dir"]
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/yolo/unlabeled/info", response_model=UnlabeledInfoResponse)
async def get_yolo_unlabeled_info():
    """YOLO Unlabeled 이미지 정보 조회"""
    manager = UnlabeledManager("yolo")
    info = manager.get_info()

    return UnlabeledInfoResponse(
        model_name="yolo",
        image_count=info["image_count"],
        unlabeled_dir=info["unlabeled_dir"]
    )


# ========================================
# Model2 GT/Unlabeled API
# ========================================
@router.post("/model2/gt/upload", response_model=GTUploadResponse)
async def upload_model2_gt(
    file: UploadFile = File(...),
    source_name: str = Form(None),
    dataset_name: str = Form(None)
):
    """Model2용 GT ZIP 업로드"""
    manager = GTManager("model2")

    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = manager.upload_gt_zip(tmp_path, source_name, dataset_name)

        if result["status"] == "FAILED":
            raise HTTPException(status_code=400, detail=result["error"])

        return GTUploadResponse(
            ingest_id=result["ingest_id"],
            model_name="model2",
            status=result["status"],
            extracted_path=result["extracted_path"],
            next=f"/api/v2/model2/gt/register?ingestId={result['ingest_id']}"
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/model2/gt/register", response_model=GTRegisterResponse)
async def register_model2_gt(
    ingest_id: str = Query(..., alias="ingestId"),
    copy_mode: Literal["symlink", "copy"] = Query("symlink", alias="copyMode"),
    strict: bool = Query(False)
):
    """Model2 GT 등록"""
    manager = GTManager("model2")
    result = manager.register_gt(ingest_id, copy_mode, strict)

    if result["status"] == "FAILED":
        raise HTTPException(status_code=400, detail=result["error"])

    return GTRegisterResponse(
        ingest_id=result["ingest_id"],
        model_name="model2",
        status=result["status"],
        registered_path=result["registered_path"],
        current_gt_path=result["current_gt_path"],
        summary=result["summary"]
    )


@router.get("/model2/gt/versions", response_model=ListGTVersionsResponse)
async def list_model2_gt_versions():
    """Model2 GT 버전 목록 조회"""
    manager = GTManager("model2")
    versions = manager.list_gt_versions()

    return ListGTVersionsResponse(
        model_name="model2",
        versions=[
            GTVersionInfo(
                version_id=v["version_id"],
                model_name="model2",
                is_current=v["is_current"],
                created_at=v["created_at"],
                image_count=v["image_count"]
            )
            for v in versions
        ]
    )


@router.post("/model2/unlabeled/upload", response_model=UnlabeledUploadResponse)
async def upload_model2_unlabeled(
    file: UploadFile = File(...),
    dataset_name: str = Form(None)
):
    """Model2용 Unlabeled 이미지 업로드"""
    manager = UnlabeledManager("model2")

    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = manager.upload_unlabeled_zip(tmp_path, dataset_name)

        if result["status"] == "FAILED":
            raise HTTPException(status_code=400, detail=result["error"])

        return UnlabeledUploadResponse(
            ingest_id=result["ingest_id"],
            model_name="model2",
            status=result["status"],
            added_images=result["added_images"],
            unlabeled_dir=result["unlabeled_dir"]
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/model2/unlabeled/info", response_model=UnlabeledInfoResponse)
async def get_model2_unlabeled_info():
    """Model2 Unlabeled 이미지 정보 조회"""
    manager = UnlabeledManager("model2")
    info = manager.get_info()

    return UnlabeledInfoResponse(
        model_name="model2",
        image_count=info["image_count"],
        unlabeled_dir=info["unlabeled_dir"]
    )


# ========================================
# Model3 GT/Unlabeled API
# ========================================
@router.post("/model3/gt/upload", response_model=GTUploadResponse)
async def upload_model3_gt(
    file: UploadFile = File(...),
    source_name: str = Form(None),
    dataset_name: str = Form(None)
):
    """Model3용 GT ZIP 업로드"""
    manager = GTManager("model3")

    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = manager.upload_gt_zip(tmp_path, source_name, dataset_name)

        if result["status"] == "FAILED":
            raise HTTPException(status_code=400, detail=result["error"])

        return GTUploadResponse(
            ingest_id=result["ingest_id"],
            model_name="model3",
            status=result["status"],
            extracted_path=result["extracted_path"],
            next=f"/api/v2/model3/gt/register?ingestId={result['ingest_id']}"
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/model3/gt/register", response_model=GTRegisterResponse)
async def register_model3_gt(
    ingest_id: str = Query(..., alias="ingestId"),
    copy_mode: Literal["symlink", "copy"] = Query("symlink", alias="copyMode"),
    strict: bool = Query(False)
):
    """Model3 GT 등록"""
    manager = GTManager("model3")
    result = manager.register_gt(ingest_id, copy_mode, strict)

    if result["status"] == "FAILED":
        raise HTTPException(status_code=400, detail=result["error"])

    return GTRegisterResponse(
        ingest_id=result["ingest_id"],
        model_name="model3",
        status=result["status"],
        registered_path=result["registered_path"],
        current_gt_path=result["current_gt_path"],
        summary=result["summary"]
    )


@router.get("/model3/gt/versions", response_model=ListGTVersionsResponse)
async def list_model3_gt_versions():
    """Model3 GT 버전 목록 조회"""
    manager = GTManager("model3")
    versions = manager.list_gt_versions()

    return ListGTVersionsResponse(
        model_name="model3",
        versions=[
            GTVersionInfo(
                version_id=v["version_id"],
                model_name="model3",
                is_current=v["is_current"],
                created_at=v["created_at"],
                image_count=v["image_count"]
            )
            for v in versions
        ]
    )


@router.post("/model3/unlabeled/upload", response_model=UnlabeledUploadResponse)
async def upload_model3_unlabeled(
    file: UploadFile = File(...),
    dataset_name: str = Form(None)
):
    """Model3용 Unlabeled 이미지 업로드"""
    manager = UnlabeledManager("model3")

    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = manager.upload_unlabeled_zip(tmp_path, dataset_name)

        if result["status"] == "FAILED":
            raise HTTPException(status_code=400, detail=result["error"])

        return UnlabeledUploadResponse(
            ingest_id=result["ingest_id"],
            model_name="model3",
            status=result["status"],
            added_images=result["added_images"],
            unlabeled_dir=result["unlabeled_dir"]
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/model3/unlabeled/info", response_model=UnlabeledInfoResponse)
async def get_model3_unlabeled_info():
    """Model3 Unlabeled 이미지 정보 조회"""
    manager = UnlabeledManager("model3")
    info = manager.get_info()

    return UnlabeledInfoResponse(
        model_name="model3",
        image_count=info["image_count"],
        unlabeled_dir=info["unlabeled_dir"]
    )
