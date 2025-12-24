#각 함수마다 무엇을 하는 친구들인지 간단하게 적어두었습니다.

from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Path as PathParam
from pathlib import Path
import json
import zipfile
import time
import shutil
import uuid
from typing import Dict, Any

from auto_labeling.demo.src.gt_register import register_gt_yolo

router = APIRouter()
# data 루트 및 하위 디렉토리 정의
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw_ingest"
GT_VERS = DATA / "GT_versions"
GT_CUR = DATA / "GT"
UNLAB = DATA / "unlabeled" / "images"

#ingest_id 생성
def _new_id(prefix: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{suf}"

#ingest 단계별 상태/메타데이터 기록용
def _save_status(d: Path, payload: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "status.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

#GT 업로드
@router.post("/gt/ingests/upload")
async def upload_gt_zip(
    file: UploadFile = File(...),
    source_name: str = Form(""),
    dataset_name: str = Form(""),
):
    """
    GT ZIP 파일 업로드 + 압축 해제

    - 입력:
      - images/ + labels/ 를 포함한 ZIP
    - 수행:
      - raw_ingest/<ingest_id>/payload.zip 저장
      - raw_ingest/<ingest_id>/extracted/ 압축 해제
      - status.json 기록
    - 주의:
      - 이 단계에서는 GT 등록을 하지 않음
      - 반드시 /register API를 추가 호출해야 함
    """
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

#gt 등록 (표준 형식으로 변환)
@router.post("/gt/ingests/{ingest_id}/register")
def register_gt(
    ingest_id: str = PathParam(...),
    copy_mode: str = Query("symlink"),
    strict: bool = Query(False),
):
    """
    업로드된 GT를 정식 GT 버전으로 등록

    - 입력:
      - raw_ingest/<ingest_id>/extracted
    - 수행:
      1) images/labels 구조 검증
      2) YOLO 라벨 검증 (class id / 정규화 범위)
      3) GT_versions/GT_<ingest_id> 생성
      4) data/GT 심볼릭 링크 갱신 -> 매번 달라지는 gt 경로 추적
      5) data.yaml 생성 (YOLO 학습용)
    """
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


@router.post("/unlabeled/ingests/upload")
async def upload_unlabeled_zip(
    file: UploadFile = File(...),
    dataset_name: str = Form(""),
):
    """
    라벨 없는 이미지 ZIP 업로드

    - 입력:
      - 이미지 파일들만 포함된 ZIP (하위 폴더 허용)
    - 수행:
      1) ZIP 저장 + 압축 해제
      2) 이미지 파일만 필터링
      3) data/unlabeled/images 로 복사
    - 주의:
      - 현재는 파일명 그대로 복사
      - 파일명 충돌 시 덮어쓸 수 있음
    """
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
