from __future__ import annotations

from pathlib import Path
import os
import shutil
import yaml

from auto_labeling.v_1.scripts.logger import log, log_json

ROOT = Path(__file__).resolve().parents[1]
CLASS_YAML = ROOT / "configs" / "classes.yaml"


def _load_class_names() -> dict[int, str]:
    if not CLASS_YAML.exists():
        raise FileNotFoundError(f"[GT_REGISTER] classes.yaml not found: {CLASS_YAML}")
    with open(CLASS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names")
    if names is None:
        raise KeyError("[GT_REGISTER] 'names' key not in classes.yaml")
    return {int(k): str(v) for k, v in names.items()}


def _check_yolo_label(label_path: Path, *, num_classes: int) -> tuple[bool, str]:
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
        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= num_classes:
            return False, f"line {ln}: class id out of range ({cls_id})"
        vals = list(map(float, parts[1:]))
        for v in vals:
            if v < 0.0 or v > 1.0:
                return False, f"line {ln}: value out of [0,1] ({v})"
    return True, "ok"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    mode = (mode or "copy").lower().strip()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def register_gt_yolo(
    src_root: Path,
    dst_root: Path,
    *,
    copy_mode: str = "copy",   # copy | symlink
    strict: bool = False,
) -> dict:
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    src_img = src_root / "images"
    src_lbl = src_root / "labels"
    if not src_img.exists() or not src_lbl.exists():
        raise FileNotFoundError("[GT_REGISTER] src_root must contain images/ and labels/")

    dst_img = dst_root / "images"
    dst_lbl = dst_root / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names()
    num_classes = len(class_names)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs = []
    for e in exts:
        imgs += sorted(src_img.glob(f"*{e}"))

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
            log_json({"event": "gt_register_error", "image": img_path.name, "reason": reason})
            if strict:
                raise RuntimeError(f"[GT_REGISTER] strict mode stop: {img_path.name}")
            continue

        _link_or_copy(img_path, dst_img / img_path.name, copy_mode)
        _link_or_copy(lbl_path, dst_lbl / lbl_path.name, copy_mode)
        n_ok += 1

    # data.yaml 생성
    data_yaml = dst_root / "data.yaml"
    cfg = {"path": str(dst_root), "train": "images", "val": "images", "names": class_names}
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    summary = {"ok": n_ok, "skip": n_skip, "error": n_err, "dst": str(dst_root)}
    log(f"[GT_REGISTER] DONE: {summary}")
    log_json({"event": "gt_register_summary", **summary})
    return summary
