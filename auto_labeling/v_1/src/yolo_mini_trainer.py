# auto_labeling/v_1/src/yolo_mini_trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import shutil
import yaml
from datetime import datetime
from typing import Optional, Iterable

from ultralytics import YOLO
from auto_labeling.v_1.scripts.logger import log, log_json

ROOT = Path(__file__).resolve().parents[1]
CLASS_YAML = ROOT / "configs" / "classes.yaml"  # 공통 클래스 정의 경로


# ------------------------------------------------------------
# 1) classes.yaml 로더
# ------------------------------------------------------------
def load_class_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"[MiniTrainer] classes.yaml not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names")
    if names is None:
        raise KeyError(f"[MiniTrainer] 'names' key not in {yaml_path}")

    fixed: dict[int, str] = {}
    for k, v in names.items():
        fixed[int(k)] = str(v)

    return fixed


# ------------------------------------------------------------
# 2) student weight 네이밍 규칙
# ------------------------------------------------------------
def _build_student_round_weight_name(
    base_weights: Path,
    round_index: int,
    sampler: str,
    anchor_ratio: int,
    date_str: Optional[str] = None,
) -> str:
    """
    예:
      base_weights = yolov11x_student_h10_init_20251210.pt
      -> yolov11x_student_r2_online_fail_gt4_20251212.pt
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    stem = base_weights.stem

    if "_student" in stem:
        model_name = stem.split("_student")[0]
    else:
        model_name = stem

    # 파일명이 숫자로 시작하면 model 접두어 붙이기
    if model_name and model_name[0].isdigit():
        model_name = f"model{model_name}"

    sampler = sampler.lower().strip().replace(" ", "_")

    filename = f"{model_name}_student_r{round_index}_{sampler}_gt{anchor_ratio}_{date_str}.pt"
    return filename


def _iter_images(img_dir: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs: list[Path] = []
    for e in exts:
        imgs.extend(sorted(img_dir.glob(f"*{e}")))
    return imgs


# ------------------------------------------------------------
# 3) label_dir -> (img_dir.parent/labels) 로 정리 (YOLO 표준)
#    - label_dir이 이미 target이면 "자기복사" 방지
# ------------------------------------------------------------
def _prepare_labels_for_yolo(
    img_dir: Path,
    label_dir: Path,
) -> Path:
    """
    img_dir   : round 이미지 디렉토리 (예: data/round_r1/images)
    label_dir : pseudo-label txt 들이 있는 디렉토리

    반환: YOLO data.yaml 에서 사용할 root (img_dir의 부모)
         즉, root/images 와 root/labels 구조
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"[MiniTrainer] img_dir not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"[MiniTrainer] label_dir not found: {label_dir}")

    data_root = img_dir.parent
    yolo_lbl_dir = data_root / "labels"
    yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    # label_dir이 이미 yolo_lbl_dir이면 복사 생략(자기복사 방지)
    try:
        same_dir = label_dir.resolve() == yolo_lbl_dir.resolve()
    except Exception:
        same_dir = str(label_dir) == str(yolo_lbl_dir)

    n = 0
    if same_dir:
        log(f"[MiniTrainer] label_dir == target labels dir ({yolo_lbl_dir}) → 복사 생략")
        return data_root

    for src_lbl in sorted(label_dir.glob("*.txt")):
        dst_lbl = yolo_lbl_dir / src_lbl.name
        # 혹시라도 동일 파일이면 스킵
        try:
            if src_lbl.resolve() == dst_lbl.resolve():
                continue
        except Exception:
            pass
        shutil.copy2(src_lbl, dst_lbl)
        n += 1

    log(f"[MiniTrainer] labels 정리: {label_dir} → {yolo_lbl_dir} (copied={n})")
    return data_root


# ------------------------------------------------------------
# 4) Teacher pseudo-label 기반 mini train (Micro-FT)
#    - lr0, freeze 옵션 반영
#    - augmentation 최소화 옵션 제공(기본 off 권장)
# ------------------------------------------------------------
def train_on_teacher_pseudo(
    base_weights: Path,
    img_dir: Path,
    label_dir: Path,
    out_weights: Optional[Path] = None,
    *,
    round_index: int = 1,
    sampler: str = "online_fail",
    anchor_ratio: int = 10,
    epochs: int = 2,
    imgsz: int = 1280,
    device: str = "0",
    lr0: float = 5e-5,
    freeze_backbone: bool = True,
    freeze_layers: int = 10,
    batch: int | None = None,
    workers: int | None = None,
    # 마이크로 FT 안정화 옵션(기본은 최소 증강)
    disable_mosaic: bool = True,
    disable_mixup: bool = True,
    close_mosaic: int = 0,
    seed: int = 0,
) -> Path:
    """
    base_weights : 현재 student weight (이걸 기준으로 미세 학습)
    img_dir      : round 이미지 디렉토리 (…/round_rX/images)
    label_dir    : round 라벨 디렉토리 (teacher_pseudo + GT anchor)
    out_weights  : 새로 저장할 student weight 경로 (None이면 규칙대로 생성)

    freeze_backbone=True 인 경우:
      - Ultralytics train(freeze=freeze_layers) 로 일부 레이어 고정
    """
    base_weights = Path(base_weights)
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    if not base_weights.exists():
        raise FileNotFoundError(f"[MiniTrainer] base_weights not found: {base_weights}")

    # out_weights 자동 네이밍
    if out_weights is None:
        filename = _build_student_round_weight_name(
            base_weights=base_weights,
            round_index=round_index,
            sampler=sampler,
            anchor_ratio=anchor_ratio,
        )
        out_dir = base_weights.parent
        out_weights = out_dir / filename

    out_weights = Path(out_weights)
    out_weights.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 sanity
    imgs = _iter_images(img_dir)
    if not imgs:
        raise RuntimeError(f"[MiniTrainer] img_dir has no images: {img_dir}")

    # labels 존재 체크
    lbls = sorted(label_dir.glob("*.txt"))
    if not lbls:
        raise RuntimeError(f"[MiniTrainer] label_dir has no labels: {label_dir}")

    log(
        f"[MiniTrainer] 시작 – base={base_weights.name}, out={out_weights.name}, "
        f"round={round_index}, sampler={sampler}, gt={anchor_ratio}, "
        f"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, "
        f"freeze_backbone={freeze_backbone}, freeze_layers={freeze_layers}, "
        f"batch={batch}, workers={workers}"
    )

    # data.yaml
    data_root = _prepare_labels_for_yolo(img_dir=img_dir, label_dir=label_dir)

    data_yaml = out_weights.parent / f"data_teacher_pseudo_r{round_index}.yaml"

    try:
        rel_train = img_dir.relative_to(data_root)
        rel_train_str = str(rel_train)
    except Exception:
        rel_train_str = str(img_dir)

    names = load_class_names(CLASS_YAML)

    cfg = {
        "path": str(data_root),
        "train": rel_train_str,
        "val": rel_train_str,
        "names": names,
    }

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # 모델 로드
    model = YOLO(str(base_weights))

    log_json({
        "event": "student_train_start",
        "base_weight": base_weights.name,
        "out_weight": out_weights.name,
        "round": round_index,
        "sampler": sampler,
        "anchor_ratio": anchor_ratio,
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "lr0": lr0,
        "freeze_backbone": freeze_backbone,
        "freeze_layers": freeze_layers,
        "batch": batch,
        "workers": workers,
        "disable_mosaic": disable_mosaic,
        "disable_mixup": disable_mixup,
        "seed": seed,
    })

    # train 인자 구성
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        device=device,
        project=str(out_weights.parent),
        name=f"teacher_fail_ft_r{round_index}",
        save=True,
        exist_ok=True,
        lr0=float(lr0),
        seed=int(seed),
        close_mosaic=int(close_mosaic),
    )

    if batch is not None:
        train_kwargs["batch"] = int(batch)
    if workers is not None:
        train_kwargs["workers"] = int(workers)

    # 마이크로 FT: 증강 최소화(권장)
    if disable_mosaic:
        train_kwargs["mosaic"] = 0.0
    if disable_mixup:
        train_kwargs["mixup"] = 0.0

    # freeze
    if freeze_backbone:
        train_kwargs["freeze"] = int(freeze_layers)

    # 학습 실행
    results = model.train(**train_kwargs)

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"[MiniTrainer] best.pt not found at {best}")

    shutil.copy2(best, out_weights)

    log(f"[MiniTrainer] 완료 – new student weights: {out_weights}")
    log_json({
        "event": "student_train_end",
        "out_weight": out_weights.name,
        "round": round_index,
        "sampler": sampler,
        "anchor_ratio": anchor_ratio,
        "best_path": str(best),
    })

    return out_weights

if __name__ == "__main__":
    base = ROOT / "models" / "user_yolo" / "weights" / "yolov11x_student_h10_init_20251210.pt"
    imgs = ROOT / "data" / "round_r1" / "images"
    labels = ROOT / "data" / "round_r1" / "labels"

    train_on_teacher_pseudo(
        base_weights=base,
        img_dir=imgs,
        label_dir=labels,
        out_weights=None,
        round_index=1,
        sampler="online_fail",
        anchor_ratio=10,
        epochs=2,
        imgsz=1536,
        device="0",
        lr0=5e-5,
        freeze_backbone=True,
        freeze_layers=10,
        batch=4,
        workers=4,
        disable_mosaic=True,
        disable_mixup=True,
        seed=0,
    )
