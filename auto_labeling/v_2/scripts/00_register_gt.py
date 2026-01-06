# auto_labeling/v_1/scripts/00_register_gt.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import shutil
import argparse

import yaml

from auto_labeling.v_1.scripts.logger import log, log_json

ROOT = Path(__file__).resolve().parents[1]
CLASS_YAML = ROOT / "configs" / "classes.yaml"


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------
def _iter_images(d: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs: list[Path] = []
    for e in exts:
        imgs.extend(sorted(d.glob(f"*{e}")))
    return imgs


def _load_class_names() -> dict[int, str]:
    if not CLASS_YAML.exists():
        raise FileNotFoundError(f"[GT_REGISTER] classes.yaml not found: {CLASS_YAML}")

    with open(CLASS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names")
    if names is None:
        raise KeyError("[GT_REGISTER] 'names' key not in classes.yaml")

    fixed: dict[int, str] = {}
    for k, v in names.items():
        fixed[int(k)] = str(v)

    return fixed


def _check_yolo_label(
    label_path: Path,
    *,
    num_classes: int,
) -> tuple[bool, str]:
    """
    YOLO txt sanity check
    """
    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        return False, f"read_error: {e}"

    for ln, line in enumerate(lines, 1):
        parts = line.strip().split()
        if len(parts) != 5:
            return False, f"line {ln}: invalid format ({len(parts)} fields)"

        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= num_classes:
            return False, f"line {ln}: class id out of range ({cls_id})"

        vals = list(map(float, parts[1:]))
        for v in vals:
            if v < 0.0 or v > 1.0:
                return False, f"line {ln}: value out of [0,1] ({v})"

    return True, "ok"


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Register GT dataset (YOLO)")

    ap.add_argument(
        "--src",
        type=str,
        required=True,
        help="source GT root (must contain images/ and labels/)",
    )
    ap.add_argument(
        "--dst",
        type=str,
        default=str(ROOT / "data" / "GT"),
        help="destination GT root (default: data/GT)",
    )
    ap.add_argument(
        "--copy_mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="copy or symlink images/labels",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="strict mode: stop on first error",
    )

    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    src_img = src_root / "images"
    src_lbl = src_root / "labels"

    if not src_img.exists() or not src_lbl.exists():
        raise FileNotFoundError("[GT_REGISTER] src must contain images/ and labels/")

    dst_img = dst_root / "images"
    dst_lbl = dst_root / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names()
    num_classes = len(class_names)

    imgs = _iter_images(src_img)

    n_ok = 0
    n_skip = 0
    n_err = 0

    log(f"[GT_REGISTER] src={src_root}")
    log(f"[GT_REGISTER] dst={dst_root}")
    log(f"[GT_REGISTER] images={len(imgs)}, classes={num_classes}")

    for img_path in imgs:
        lbl_path = src_lbl / f"{img_path.stem}.txt"

        if not lbl_path.exists():
            n_skip += 1
            log(f"[GT_REGISTER] SKIP(no label): {img_path.name}")
            continue

        ok, reason = _check_yolo_label(lbl_path, num_classes=num_classes)
        if not ok:
            n_err += 1
            log(f"[GT_REGISTER] ERROR {img_path.name}: {reason}")
            log_json({
                "event": "gt_register_error",
                "image": img_path.name,
                "reason": reason,
            })
            if args.strict:
                raise RuntimeError(f"[GT_REGISTER] strict mode stop: {img_path.name}")
            continue

        # copy or symlink
        if args.copy_mode == "symlink":
            if not (dst_img / img_path.name).exists():
                (dst_img / img_path.name).symlink_to(img_path)
            if not (dst_lbl / lbl_path.name).exists():
                (dst_lbl / lbl_path.name).symlink_to(lbl_path)
        else:
            shutil.copy2(img_path, dst_img / img_path.name)
            shutil.copy2(lbl_path, dst_lbl / lbl_path.name)

        n_ok += 1

    # data.yaml 생성
    data_yaml = dst_root / "data.yaml"
    data_cfg = {
        "path": str(dst_root),
        "train": "images",
        "val": "images",
        "names": class_names,
    }

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)

    log(f"[GT_REGISTER] DONE: ok={n_ok}, skip={n_skip}, error={n_err}")
    log(f"[GT_REGISTER] data.yaml created: {data_yaml}")

    log_json({
        "event": "gt_register_summary",
        "src": str(src_root),
        "dst": str(dst_root),
        "ok": n_ok,
        "skip": n_skip,
        "error": n_err,
        "num_classes": num_classes,
    })


if __name__ == "__main__":
    main()
