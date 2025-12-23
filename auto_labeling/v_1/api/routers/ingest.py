# auto_labeling/v_1/api/routers/ingest.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Path as PathParam
from pathlib import Path
import json
import zipfile
import time
import shutil
import uuid
from typing import Dict, Any

from auto_labeling.v_1.src.gt_register import register_gt_yolo

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]  # .../v_1
DATA = ROOT / "data"
RAW = DATA / "raw_ingest"               # ingest 임시 저장소 (zip/extracted/status.json)
GT_VERS = DATA / "GT_versions"          # 등록된 GT 버전 보관 (GT_<ingest_id>)
GT_CUR = DATA / "GT"                    # 현재 GT 심볼릭 링크(또는 디렉토리)
UNLAB = DATA / "unlabeled" / "images"   # unlabeled 이미지 풀


def _new_id(prefix: str) -> str:
    """ingestId 생성: {prefix}_YYYYmmdd_HHMMSS_<rand>"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{suf}"


def _save_status(d: Path, payload: dict) -> None:
    """RAW/<ingest_id>/status.json 저장"""
    d.mkdir(parents=True, exist_ok=True)
    p = d / "status.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@router.post(
    "/gt/ingests/upload",
    summary="GT ZIP 업로드",
    description=(
        "사람/외부 시스템이 생성한 GT 라벨 ZIP 파일을 업로드합니다.\n\n"
        "### ZIP 구조 규칙(필수)\n"
        "- images/ : 이미지 파일들\n"
        "- labels/ : YOLO 라벨(.txt) 파일들\n"
        "- 라벨 포맷: `class cx cy w h` (0~1 정규화)\n\n"
        "업로드 후에는 반드시 **`POST /api/v1/gt/ingests/{ingestId}/register`** 를 호출하여\n"
        "서버의 GT 저장소(GT_versions)로 등록해야 합니다."
    ),
    responses={
        200: {"description": "업로드 성공 (ingestId 및 extracted 경로 반환)"},
        400: {"description": "업로드/압축해제 실패 또는 요청 오류"},
        500: {"description": "서버 내부 오류"},
    },
)
async def upload_gt_zip(
    file: UploadFile = File(
        ...,
        description=(
            "GT ZIP 파일(binary).\n"
            "반드시 images/ 와 labels/ 디렉토리를 포함해야 합니다."
        ),
    ),
    source_name: str = Form(
        "",
        description=(
            "GT 라벨 제공처(선택).\n"
            "예: vendorA, internal, human"
        ),
    ),
    dataset_name: str = Form(
        "",
        description=(
            "데이터셋 이름/버전(선택).\n"
            "예: TS2025_GT_v1"
        ),
    ),
):
    ingest_id = _new_id("gt")
    ingest_dir = RAW / ingest_id
    ingest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ingest_dir / "payload.zip"
    zip_path.write_bytes(await file.read())

    extracted = ingest_dir / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extracted)

    status: Dict[str, Any] = {
        "ingest_id": ingest_id,
        "status": "UPLOADED",
        "source_name": source_name,
        "dataset_name": dataset_name,
        "raw_zip": str(zip_path),
        "extracted": str(extracted),
        "next": f"/api/v1/gt/ingests/{ingest_id}/register",
        "note": "이 단계는 업로드/압축해제만 수행합니다. 반드시 register를 호출하세요.",
    }
    _save_status(ingest_dir, status)
    return status


@router.post(
    "/gt/ingests/{ingest_id}/register",
    summary="GT 등록(표준화 + 현재 GT 링크 갱신)",
    description=(
        "업로드된 GT ZIP(extracted)을 검증/정리하여 GT_versions 아래에 버전으로 등록합니다.\n\n"
        "### 동작 요약\n"
        "1) extracted/images, extracted/labels 존재 여부 확인\n"
        "2) `GT_versions/GT_<ingestId>`로 복사 또는 심볼릭 링크 생성\n"
        "3) `data/GT` 심볼릭 링크를 최신 버전으로 갱신\n\n"
        "### copyMode 정책\n"
        "- symlink: 원본을 보존하고 링크만 생성(빠름, 원본 경로가 유지되어야 함)\n"
        "- copy: 실제 파일 복사(안전, 느림)\n\n"
        "### strict 정책\n"
        "- false: 일부 라벨/이미지 누락이 있어도 가능한 범위에서 처리\n"
        "- true: 형식/매칭 문제가 있으면 실패 처리"
    ),
    responses={
        200: {"description": "등록 성공 (registered 경로, current_GT 링크, summary 반환)"},
        404: {"description": "ingestId가 없거나 extracted가 없음"},
        400: {"description": "images/labels 구조 불일치 또는 파라미터 오류"},
    },
)
def register_gt(
    ingest_id: str = PathParam(
        ...,
        description="upload 단계에서 발급된 GT ingestId",
        examples=["gt_20251219_172000_ab12cd34"],
    ),
    copy_mode: str = Query(
        "symlink",
        description=(
            "파일 처리 방식.\n"
            "- symlink: 심볼릭 링크 생성(빠름)\n"
            "- copy: 실제 파일 복사(안전)\n"
        ),
        examples=["symlink"],
    ),
    strict: bool = Query(
        False,
        description=(
            "엄격 모드 여부.\n"
            "- true: 누락/불일치가 있으면 실패\n"
            "- false: 가능한 범위에서 처리"
        ),
        examples=[False],
    ),
):
    ingest_dir = RAW / ingest_id
    extracted = ingest_dir / "extracted"
    if not extracted.exists():
        raise HTTPException(404, "ingest not found or not extracted")

    if not (extracted / "images").exists() or not (extracted / "labels").exists():
        raise HTTPException(400, "zip must contain images/ and labels/")

    dst_root = GT_VERS / f"GT_{ingest_id}"
    dst_root.mkdir(parents=True, exist_ok=True)

    summary = register_gt_yolo(
        src_root=extracted,
        dst_root=dst_root,
        copy_mode=copy_mode,
        strict=strict,
    )

    # ✅ 현재 GT 링크 갱신 (data/GT -> GT_versions/GT_<ingest_id>)
    GT_CUR.parent.mkdir(parents=True, exist_ok=True)
    if GT_CUR.exists() or GT_CUR.is_symlink():
        if GT_CUR.is_symlink() or GT_CUR.is_file():
            GT_CUR.unlink()
        else:
            shutil.rmtree(GT_CUR, ignore_errors=True)

    GT_CUR.symlink_to(dst_root, target_is_directory=True)

    status: Dict[str, Any] = {
        "ingest_id": ingest_id,
        "status": "DONE",
        "registered": str(dst_root),
        "current_GT": str(GT_CUR),
        "summary": summary,
        "note": "data/GT는 최신 등록 버전을 가리키는 심볼릭 링크입니다.",
    }
    _save_status(ingest_dir, status)
    return status


@router.post(
    "/unlabeled/ingests/upload",
    summary="Unlabeled ZIP 업로드(이미지 풀에 추가)",
    description=(
        "라벨이 없는 이미지 ZIP 파일을 업로드하고, 추출된 이미지들을 `data/unlabeled/images`에 추가합니다.\n\n"
        "### ZIP 구조 규칙\n"
        "- 구조는 자유(하위 폴더 포함 가능)\n"
        "- 서버는 추출된 파일 중 이미지 확장자만 탐색하여 복사합니다.\n"
        "  (jpg/jpeg/png/bmp/webp)\n\n"
        "### 주의\n"
        "- 현재 구현은 파일명을 그대로 사용하여 복사합니다.\n"
        "  동일한 파일명이 있으면 덮어쓸 수 있으므로, 운영에서는 파일명 충돌 정책(리네임/해시)이 필요할 수 있습니다."
    ),
    responses={
        200: {"description": "업로드 성공 (추가된 이미지 수 반환)"},
        400: {"description": "업로드/압축해제 실패 또는 요청 오류"},
    },
)
async def upload_unlabeled_zip(
    file: UploadFile = File(
        ...,
        description="Unlabeled 이미지 ZIP 파일(binary). 하위 폴더 포함 가능.",
    ),
    dataset_name: str = Form(
        "",
        description="데이터셋 이름/버전(선택). 예: TS2025_unlabeled_v1",
    ),
):
    ingest_id = _new_id("unlabeled")
    ingest_dir = RAW / ingest_id
    ingest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = ingest_dir / "payload.zip"
    zip_path.write_bytes(await file.read())

    extracted = ingest_dir / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extracted)

    UNLAB.mkdir(parents=True, exist_ok=True)

    moved = 0
    for p in extracted.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            shutil.copy2(p, UNLAB / p.name)
            moved += 1

    status: Dict[str, Any] = {
        "ingest_id": ingest_id,
        "status": "DONE",
        "dataset_name": dataset_name,
        "added_images": moved,
        "unlabeled_dir": str(UNLAB),
        "note": "현재는 파일명 그대로 복사합니다. 동일 파일명 충돌 정책이 필요할 수 있습니다.",
    }
    _save_status(ingest_dir, status)
    return status