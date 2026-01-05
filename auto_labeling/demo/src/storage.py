# demo/src/storage.py
from __future__ import annotations

from dataclasses import dataclass
from fastapi import UploadFile, HTTPException
from pathlib import Path
import hashlib
import os
import shutil
import zipfile
from typing import Dict, List, Tuple, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha1_file(p: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


async def save_upload_file(upload: UploadFile, dst_path: Path, *, chunk_size: int = 1024 * 1024) -> int:
    """
    UploadFile을 chunk로 저장 (대용량 zip 대비)
    return: total bytes written
    """
    ensure_dir(dst_path.parent)

    total = 0
    with dst_path.open("wb") as out:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    return total


def safe_extract_zip(zip_path: Path, dst_dir: Path) -> int:
    """
    zip-slip 방지 압축해제.
    return: extracted file count
    """
    ensure_dir(dst_dir)
    extracted = 0

    base = dst_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            # 디렉토리는 건너뜀
            if info.is_dir():
                continue

            # zip 내부 경로를 안전하게 정규화
            name = info.filename.replace("\\", "/")

            # 절대경로 / 상위 경로 탈출 차단
            if name.startswith("/") or name.startswith("../") or "/../" in name:
                raise HTTPException(400, f"unsafe zip entry: {info.filename}")

            out_path = (dst_dir / name).resolve()
            if not str(out_path).startswith(str(base)):
                raise HTTPException(400, f"zip-slip detected: {info.filename}")

            ensure_dir(out_path.parent)

            with z.open(info, "r") as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted += 1

    return extracted


@dataclass
class IngestResult:
    moved: int
    skipped: int
    collisions_avoided: int
    items: List[Dict[str, str]]  # {"src": "...", "dst": "..."}


def ingest_images_to_dir(
    extracted_root: Path,
    target_dir: Path,
    *,
    rename_by_hash: bool = True,
    keep_original_name_if_unique: bool = False,
) -> IngestResult:
    """
    extracted_root 아래에서 이미지 파일을 찾아 target_dir로 copy2.

    rename_by_hash=True:
      dst_name = <sha1>_<orig_name>
    keep_original_name_if_unique=True:
      target_dir에 동일 이름이 없으면 orig_name을 그대로 사용(충돌 시 hash로 회피)
    """
    ensure_dir(target_dir)

    moved = 0
    skipped = 0
    collisions_avoided = 0
    items: List[Dict[str, str]] = []

    for p in extracted_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            orig = p.name
            dst_name = orig

            if rename_by_hash:
                h = sha1_file(p)
                dst_name = f"{h}_{orig}"
            elif keep_original_name_if_unique:
                if (target_dir / orig).exists():
                    h = sha1_file(p)
                    dst_name = f"{h}_{orig}"
                    collisions_avoided += 1
            else:
                # 그대로 복사(충돌 시 덮어씀) → 데모에서는 비추
                pass

            dst = target_dir / dst_name
            if dst.exists():
                # 같은 해시/같은 이름이 이미 있으면 스킵
                skipped += 1
                continue

            shutil.copy2(p, dst)
            moved += 1
            items.append({"src": str(p), "dst": str(dst)})
        except Exception:
            skipped += 1
            continue

    return IngestResult(
        moved=moved,
        skipped=skipped,
        collisions_avoided=collisions_avoided,
        items=items,
    )
