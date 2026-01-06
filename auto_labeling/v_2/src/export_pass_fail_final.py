# auto_labeling/v_1/src/export_pass_fail_final.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO
from auto_labeling.v_1.scripts.logger import log


def _iter_images(dirp: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs: list[Path] = []
    for e in exts:
        imgs.extend(sorted(dirp.glob(f"*{e}")))
    return imgs


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_bytes(src.read_bytes())


def _write_yolo_label(model: YOLO, img_path: Path, out_lbl_path: Path, *, device: str, conf_th: float) -> None:
    res = model.predict(
        source=str(img_path),
        conf=float(conf_th),
        verbose=False,
        device=str(device),
    )[0]

    boxes = res.boxes
    lines: list[str] = []
    if boxes is not None and len(boxes) > 0:
        for cls_id, xywhn in zip(boxes.cls.cpu().tolist(), boxes.xywhn.cpu().tolist()):
            cx, cy, w, h = map(float, xywhn)
            lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.write_text("\n".join(lines), encoding="utf-8")  # 빈 파일 허용


def export_pass_fail_final(
    *,
    student_weights: Path,
    out_root: Path,
    # ✅ PASS source들을 여러 개 받을 수 있게
    pass_img_dirs: List[Path],
    # ✅ FAIL-only 결과들
    pass_fail_img_dir: Optional[Path] = None,
    fail_fail_img_dir: Optional[Path] = None,
    miss_img_dir: Optional[Path] = None,
    # 옵션
    device: str = "0",
    conf_th: float = 0.3,
    merge_pass_into_pass_dir: bool = True,
) -> None:


    if not student_weights.exists():
        raise FileNotFoundError(f"[FINAL_EXPORT] student weight not found: {student_weights}")

    out_root.mkdir(parents=True, exist_ok=True)

    out_pass_img = out_root / "pass" / "images"
    out_pass_lbl = out_root / "pass" / "labels"

    out_pass_fail_img = out_root / "pass_fail" / "images"
    out_pass_fail_lbl = out_root / "pass_fail" / "labels"

    out_fail_fail_img = out_root / "fail_fail" / "images"
    out_miss_img = out_root / "miss" / "images"

    for d in [out_pass_img, out_pass_lbl, out_pass_fail_img, out_pass_fail_lbl, out_fail_fail_img, out_miss_img]:
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(student_weights))

    total_pass = 0
    for d in pass_img_dirs:
        if d is None:
            continue
        d = Path(d)
        if not d.exists():
            log(f"[FINAL_EXPORT] skip pass_img_dir(not found): {d}")
            continue
        imgs = _iter_images(d)
        log(f"[FINAL_EXPORT] PASS src={d} images={len(imgs)}")
        for img_path in imgs:
            _copy_file(img_path, out_pass_img / img_path.name)
            _write_yolo_label(model, img_path, out_pass_lbl / f"{img_path.stem}.txt", device=device, conf_th=conf_th)
        total_pass += len(imgs)

    if pass_fail_img_dir is not None:
        d = Path(pass_fail_img_dir)
        if d.exists():
            imgs = _iter_images(d)
            log(f"[FINAL_EXPORT] PASS_FAIL images={len(imgs)} from {d}")
            for img_path in imgs:
                _copy_file(img_path, out_pass_fail_img / img_path.name)
                _write_yolo_label(model, img_path, out_pass_fail_lbl / f"{img_path.stem}.txt", device=device, conf_th=conf_th)

                if merge_pass_into_pass_dir:
                    _copy_file(img_path, out_pass_img / img_path.name)
                    _write_yolo_label(model, img_path, out_pass_lbl / f"{img_path.stem}.txt", device=device, conf_th=conf_th)
        else:
            log(f"[FINAL_EXPORT] skip pass_fail_img_dir(not found): {d}")

    if fail_fail_img_dir is not None:
        d = Path(fail_fail_img_dir)
        if d.exists():
            imgs = _iter_images(d)
            log(f"[FINAL_EXPORT] FAIL_FAIL images={len(imgs)} from {d}")
            for img_path in imgs:
                _copy_file(img_path, out_fail_fail_img / img_path.name)
        else:
            log(f"[FINAL_EXPORT] skip fail_fail_img_dir(not found): {d}")

    if miss_img_dir is not None:
        d = Path(miss_img_dir)
        if d.exists():
            imgs = _iter_images(d)
            log(f"[FINAL_EXPORT] MISS images={len(imgs)} from {d}")
            for img_path in imgs:
                _copy_file(img_path, out_miss_img / img_path.name)
        else:
            log(f"[FINAL_EXPORT] skip miss_img_dir(not found): {d}")

    log("[FINAL_EXPORT] export 완료 (pass / pass_fail / fail_fail / miss)")
