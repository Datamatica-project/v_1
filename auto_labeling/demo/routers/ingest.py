# demo/routers/ingest.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Path as PathParam
from pathlib import Path
import json
import zipfile
import time
import shutil
import uuid
from typing import Dict, Any, Literal

from auto_labeling.demo.src.gt_register import register_gt_yolo

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw_ingest"
GT_VERS = DATA / "GT_versions"
GT_CUR = DATA / "GT"
UNLAB = DATA / "unlabeled" / "images"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
OnConflict = Literal["skip", "overwrite", "rename"]


def _new_id(prefix: str) -> str:
    """prefix + timestamp + random suffix로 ingest_id 생성"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{suf}"


def _save_status(d: Path, payload: dict) -> None:
    """ingest_dir/status.json 저장 (프론트/디버깅 용)"""
    d.mkdir(parents=True, exist_ok=True)
    p = d / "status.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_dir(d: Path) -> None:
    """디렉토리 내부 파일/폴더 전부 삭제(디렉토리는 유지)"""
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def _copy_image_with_policy(src: Path, dst_dir: Path, *, on_conflict: OnConflict) -> tuple[bool, str]:
    """
    이미지 1장을 dst_dir로 복사하는데, 동일 파일명 충돌 정책 적용
    return: (copied?, final_filename)
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if not dst.exists():
        shutil.copy2(src, dst)
        return True, dst.name

    if on_conflict == "skip":
        return False, dst.name

    if on_conflict == "overwrite":
        shutil.copy2(src, dst)
        return True, dst.name

    # rename
    stem, suf = src.stem, src.suffix
    k = 1
    while True:
        cand = dst_dir / f"{stem}__{k}{suf}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return True, cand.name
        k += 1


@router.post("/gt/ingests/upload")
async def upload_gt_zip(
    file: UploadFile = File(...),
    source_name: str = Form("", alias="sourceName"),
    dataset_name: str = Form("", alias="datasetName"),
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
      4) data/GT 심볼릭 링크 갱신 -> 최신 GT 버전 추적
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
    dataset_name: str = Form("", alias="datasetName"),
    clear_before: bool = Form(True, alias="clearBefore"),          # ✅ demo: 기본 True
    on_conflict: OnConflict = Form("skip", alias="onConflict"),     # ✅ clear_before=False일 때 유용
):
    """
    Unlabeled ZIP 업로드 + 압축 해제 + unlabeled/images 반영

    - demo 기본 정책:
      - clearBefore=True → 업로드마다 UNLAB(=data/unlabeled/images) 초기화 후 이번 업로드만 유지
    - 충돌 정책(onConflict):
      - skip: 동일 파일명은 건너뜀
      - overwrite: 덮어씀
      - rename: __1, __2 ... 붙여서 저장
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

    # ✅ 핵심: demo에서는 “이번 업로드만 대상으로” 쓰기 위해 기본 clear
    if bool(clear_before):
        _clear_dir(UNLAB)

    moved = 0
    skipped = 0
    renamed = 0
    overwrote = 0

    for p in extracted.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue

        before_exists = (UNLAB / p.name).exists()
        ok, final_name = _copy_image_with_policy(p, UNLAB, on_conflict=on_conflict)

        if ok:
            moved += 1
            if before_exists and on_conflict == "overwrite":
                overwrote += 1
            if on_conflict == "rename" and final_name != p.name:
                renamed += 1
        else:
            skipped += 1

    status: Dict[str, Any] = {
        "ingest_id": ingest_id,
        "status": "DONE",
        "dataset_name": dataset_name,
        "clear_before": bool(clear_before),
        "on_conflict": on_conflict,
        "added_images": moved,
        "skipped_images": skipped,
        "renamed_images": renamed,
        "overwrote_images": overwrote,
        "unlabeled_dir": str(UNLAB),
        "note": "demo 기본값은 clearBefore=True로 unlabeled를 매 업로드마다 초기화합니다.",
    }
    _save_status(ingest_dir, status)
    return status
