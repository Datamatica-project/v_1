"""
모델별 GT/Unlabeled 데이터 관리 서비스 (앙상블 방식)

각 모델(YOLO, Model2, Model3)별로 독립적인 데이터 디렉토리 관리
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
import os
import shutil
import zipfile
from datetime import datetime
import yaml

from auto_labeling.v_1.scripts.logger import log


# ========================================
# 설정
# ========================================
V2_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = V2_ROOT.parent / "data"

# 모델별 디렉토리
GT_DATA_ROOT = DATA_ROOT / "gt_data"
UNLABELED_ROOT = DATA_ROOT / "unlabeled"
RAW_INGEST_ROOT = DATA_ROOT / "raw_ingest"

# 이미지 확장자
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 모델 이름 타입
ModelName = Literal["yolo", "model2", "model3"]


# ========================================
# 유틸리티 함수
# ========================================
def _generate_ingest_id(prefix: str, model_name: str) -> str:
    """Ingest ID 생성: {prefix}_{model_name}_{timestamp}_{random}"""
    import random
    import string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{model_name}_{timestamp}_{rand_suffix}"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """파일 복사 또는 심볼릭 링크 생성"""
    mode = (mode or "copy").lower().strip()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode == "symlink":
        os.symlink(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def _check_yolo_label(label_path: Path, num_classes: int) -> tuple[bool, str]:
    """YOLO 라벨 형식 검증 (class cx cy w h)"""
    try:
        txt = label_path.read_text(encoding="utf-8").strip()
        if not txt:
            return True, "ok(empty)"
        lines = txt.splitlines()
    except Exception as e:
        return False, f"read_error: {e}"

    for ln, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 5:
            return False, f"line {ln}: invalid format ({len(parts)} fields)"

        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            return False, f"line {ln}: invalid class id"

        if cls_id < 0 or cls_id >= num_classes:
            return False, f"line {ln}: class id out of range ({cls_id})"

        try:
            vals = list(map(float, parts[1:]))
        except ValueError:
            return False, f"line {ln}: invalid float values"

        for v in vals:
            if v < 0.0 or v > 1.0:
                return False, f"line {ln}: value out of [0,1] ({v})"

    return True, "ok"


# ========================================
# GT 관리
# ========================================
class GTManager:
    """GT 데이터 관리"""

    def __init__(self, model_name: ModelName):
        self.model_name = model_name
        self.model_gt_root = GT_DATA_ROOT / model_name
        self.model_gt_root.mkdir(parents=True, exist_ok=True)

        # 클래스 정보 로드 (v_1 설정 참조)
        self.class_yaml = V2_ROOT.parent / "v_1" / "configs" / "classes.yaml"
        self.class_names = self._load_class_names()
        self.num_classes = len(self.class_names)

    def _load_class_names(self) -> dict[int, str]:
        """classes.yaml에서 클래스 정보 로드"""
        if not self.class_yaml.exists():
            log(f"[GT_MANAGER] classes.yaml not found: {self.class_yaml}, using default")
            return {0: "object"}  # 기본값

        try:
            with open(self.class_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            names = data.get("names", {0: "object"})
            return {int(k): str(v) for k, v in names.items()}
        except Exception as e:
            log(f"[GT_MANAGER] Error loading classes.yaml: {e}, using default")
            return {0: "object"}

    def upload_gt_zip(
        self,
        zip_path: Path,
        source_name: Optional[str] = None,
        dataset_name: Optional[str] = None
    ) -> dict:
        """
        GT ZIP 업로드 및 압축 해제

        Returns:
            {
                "ingest_id": str,
                "status": str,
                "extracted_path": str,
                "error": Optional[str]
            }
        """
        ingest_id = _generate_ingest_id("gt", self.model_name)
        ingest_dir = RAW_INGEST_ROOT / ingest_id
        extracted_dir = ingest_dir / "extracted"

        try:
            ingest_dir.mkdir(parents=True, exist_ok=True)

            # ZIP 압축 해제
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extracted_dir)

            log(f"[GT_MANAGER] Extracted GT ZIP: {extracted_dir}")

            # images/ 및 labels/ 존재 확인
            img_dir = extracted_dir / "images"
            lbl_dir = extracted_dir / "labels"

            if not img_dir.exists() or not lbl_dir.exists():
                return {
                    "ingest_id": ingest_id,
                    "status": "FAILED",
                    "extracted_path": None,
                    "error": "ZIP must contain images/ and labels/ directories"
                }

            # 메타데이터 저장
            meta = {
                "ingest_id": ingest_id,
                "model_name": self.model_name,
                "source_name": source_name,
                "dataset_name": dataset_name,
                "uploaded_at": datetime.now().isoformat(),
            }

            meta_path = ingest_dir / "meta.yaml"
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.dump(meta, f)

            return {
                "ingest_id": ingest_id,
                "status": "UPLOADED",
                "extracted_path": str(extracted_dir),
                "error": None
            }

        except Exception as e:
            log(f"[GT_MANAGER] Upload failed: {e}")
            return {
                "ingest_id": ingest_id,
                "status": "FAILED",
                "extracted_path": None,
                "error": str(e)
            }

    def register_gt(
        self,
        ingest_id: str,
        copy_mode: Literal["symlink", "copy"] = "symlink",
        strict: bool = False
    ) -> dict:
        """
        GT 등록 (표준화 + 현재 GT 갱신)

        Returns:
            {
                "ingest_id": str,
                "status": str,
                "registered_path": str,
                "current_gt_path": str,
                "summary": {"ok": int, "skip": int, "error": int},
                "error": Optional[str]
            }
        """
        ingest_dir = RAW_INGEST_ROOT / ingest_id
        extracted_dir = ingest_dir / "extracted"

        if not extracted_dir.exists():
            return {
                "ingest_id": ingest_id,
                "status": "FAILED",
                "registered_path": None,
                "current_gt_path": None,
                "summary": None,
                "error": f"Ingest directory not found: {extracted_dir}"
            }

        try:
            # GT 버전 디렉토리 생성
            gt_version_dir = self.model_gt_root / f"GT_{ingest_id}"
            gt_version_dir.mkdir(parents=True, exist_ok=True)

            src_img_dir = extracted_dir / "images"
            src_lbl_dir = extracted_dir / "labels"
            dst_img_dir = gt_version_dir / "images"
            dst_lbl_dir = gt_version_dir / "labels"

            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            # 이미지 파일 수집
            imgs = []
            for ext in IMAGE_EXTS:
                imgs.extend(sorted(src_img_dir.glob(f"*{ext}")))

            n_ok = 0
            n_skip = 0
            n_err = 0

            log(f"[GT_MANAGER] Registering GT: {ingest_id}")
            log(f"[GT_MANAGER] Images: {len(imgs)}, Classes: {self.num_classes}")

            for img_path in imgs:
                lbl_path = src_lbl_dir / f"{img_path.stem}.txt"

                if not lbl_path.exists():
                    n_skip += 1
                    log(f"[GT_MANAGER] SKIP(no label): {img_path.name}")
                    continue

                # 라벨 검증
                valid, msg = _check_yolo_label(lbl_path, self.num_classes)

                if not valid:
                    n_err += 1
                    log(f"[GT_MANAGER] ERROR: {img_path.name} - {msg}")
                    if strict:
                        raise ValueError(f"Label validation failed: {img_path.name} - {msg}")
                    continue

                # 파일 복사/링크
                dst_img = dst_img_dir / img_path.name
                dst_lbl = dst_lbl_dir / f"{img_path.stem}.txt"

                _link_or_copy(img_path, dst_img, copy_mode)
                _link_or_copy(lbl_path, dst_lbl, copy_mode)

                n_ok += 1

            log(f"[GT_MANAGER] Summary: ok={n_ok}, skip={n_skip}, error={n_err}")

            # data.yaml 생성
            data_yaml_path = gt_version_dir / "data.yaml"
            data_yaml_content = {
                "path": str(gt_version_dir.absolute()),
                "train": "images",
                "val": "images",
                "nc": self.num_classes,
                "names": self.class_names
            }

            with open(data_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(data_yaml_content, f)

            # 현재 GT 심볼릭 링크 갱신
            current_gt_link = self.model_gt_root / "GT.file"
            if current_gt_link.exists() or current_gt_link.is_symlink():
                current_gt_link.unlink()

            os.symlink(str(gt_version_dir), str(current_gt_link))

            log(f"[GT_MANAGER] Current GT updated: {current_gt_link} -> {gt_version_dir}")

            return {
                "ingest_id": ingest_id,
                "status": "DONE",
                "registered_path": str(gt_version_dir),
                "current_gt_path": str(current_gt_link),
                "summary": {"ok": n_ok, "skip": n_skip, "error": n_err},
                "error": None
            }

        except Exception as e:
            log(f"[GT_MANAGER] Registration failed: {e}")
            return {
                "ingest_id": ingest_id,
                "status": "FAILED",
                "registered_path": None,
                "current_gt_path": None,
                "summary": None,
                "error": str(e)
            }

    def list_gt_versions(self) -> list[dict]:
        """
        GT 버전 목록 조회

        Returns:
            [
                {
                    "version_id": str,
                    "is_current": bool,
                    "created_at": datetime,
                    "image_count": int,
                }
            ]
        """
        versions = []
        current_gt_link = self.model_gt_root / "GT.file"
        current_target = None

        if current_gt_link.is_symlink():
            current_target = current_gt_link.resolve()

        for gt_dir in sorted(self.model_gt_root.glob("GT_*"), reverse=True):
            if not gt_dir.is_dir():
                continue

            img_dir = gt_dir / "images"
            image_count = 0

            if img_dir.exists():
                for ext in IMAGE_EXTS:
                    image_count += len(list(img_dir.glob(f"*{ext}")))

            versions.append({
                "version_id": gt_dir.name,
                "is_current": (current_target == gt_dir.resolve()),
                "created_at": datetime.fromtimestamp(gt_dir.stat().st_mtime),
                "image_count": image_count,
            })

        return versions


# ========================================
# Unlabeled 관리
# ========================================
class UnlabeledManager:
    """Unlabeled 이미지 관리"""

    def __init__(self, model_name: ModelName):
        self.model_name = model_name
        self.unlabeled_dir = UNLABELED_ROOT / model_name / "images"
        self.unlabeled_dir.mkdir(parents=True, exist_ok=True)

    def upload_unlabeled_zip(
        self,
        zip_path: Path,
        dataset_name: Optional[str] = None
    ) -> dict:
        """
        Unlabeled ZIP 업로드 및 이미지 추가

        Returns:
            {
                "ingest_id": str,
                "status": str,
                "added_images": int,
                "unlabeled_dir": str,
                "error": Optional[str]
            }
        """
        ingest_id = _generate_ingest_id("unlabeled", self.model_name)

        try:
            temp_dir = RAW_INGEST_ROOT / ingest_id / "extracted"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # ZIP 압축 해제
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)

            # 모든 이미지 파일 수집 (재귀적으로)
            images = []
            for ext in IMAGE_EXTS:
                images.extend(temp_dir.rglob(f"*{ext}"))

            log(f"[UNLABELED_MANAGER] Found {len(images)} images in ZIP")

            # 이미지 복사
            added_count = 0
            for img_path in images:
                dst_path = self.unlabeled_dir / img_path.name

                # 중복 파일명 처리
                if dst_path.exists():
                    stem = img_path.stem
                    suffix = img_path.suffix
                    counter = 1
                    while dst_path.exists():
                        dst_path = self.unlabeled_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                shutil.copy2(img_path, dst_path)
                added_count += 1

            log(f"[UNLABELED_MANAGER] Added {added_count} images to {self.unlabeled_dir}")

            # 임시 디렉토리 삭제
            shutil.rmtree(RAW_INGEST_ROOT / ingest_id, ignore_errors=True)

            return {
                "ingest_id": ingest_id,
                "status": "DONE",
                "added_images": added_count,
                "unlabeled_dir": str(self.unlabeled_dir),
                "error": None
            }

        except Exception as e:
            log(f"[UNLABELED_MANAGER] Upload failed: {e}")
            return {
                "ingest_id": ingest_id,
                "status": "FAILED",
                "added_images": 0,
                "unlabeled_dir": None,
                "error": str(e)
            }

    def get_info(self) -> dict:
        """
        Unlabeled 이미지 정보 조회

        Returns:
            {
                "image_count": int,
                "unlabeled_dir": str
            }
        """
        image_count = 0

        if self.unlabeled_dir.exists():
            for ext in IMAGE_EXTS:
                image_count += len(list(self.unlabeled_dir.glob(f"*{ext}")))

        return {
            "image_count": image_count,
            "unlabeled_dir": str(self.unlabeled_dir)
        }
