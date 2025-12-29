# test.py
# 현재 문제점
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2

from augmentation import AugConfig, build_train_augment, apply_augment

YoloBBox = Tuple[float, float, float, float]


def read_yolo_txt(label_path: Path) -> tuple[List[int], List[YoloBBox]]:
    class_labels: List[int] = []
    bboxes: List[YoloBBox] = []
    if not label_path.exists():
        return class_labels, bboxes

    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return class_labels, bboxes

    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        class_labels.append(cls)
        bboxes.append((xc, yc, w, h))
    return class_labels, bboxes


def write_yolo_txt(label_path: Path, class_labels: List[int], bboxes: List[YoloBBox]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, (xc, yc, w, h) in zip(class_labels, bboxes):
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_boxes(image_bgr, class_labels: List[int], bboxes: List[YoloBBox]):
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    for cls, (xc, yc, bw, bh) in zip(class_labels, bboxes):
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, str(cls), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--label", required=True, help="path to yolo txt label")
    ap.add_argument("--outdir", default="aug_out", help="output directory root")
    ap.add_argument("--n", type=int, default=5, help="how many augmented samples to generate")
    ap.add_argument("--save_viz", action="store_true", help="save visualization images with boxes")
    args = ap.parse_args()

    img_path = Path(args.image)
    lbl_path = Path(args.label)

    # ✅ 원하는 출력 구조
    out_root = Path(args.outdir)
    out_img_dir = out_root / "images"
    out_lbl_dir = out_root / "labels"
    out_viz_dir = out_root / "result"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    out_viz_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    class_labels, bboxes = read_yolo_txt(lbl_path)

    cfg = AugConfig()
    aug = build_train_augment(cfg)

    stem = img_path.stem
    suffix = img_path.suffix.lower()

    for i in range(args.n):
        # ✅ cycle 기반: aug_id로 4개 증강을 순환 적용
        aug_img, aug_bboxes, aug_classes = apply_augment(
            aug,
            image,
            bboxes,
            class_labels,
            cfg=cfg,
            aug_id=i,
        )

        out_img_path = out_img_dir / f"{stem}_aug{i:02d}{suffix}"
        out_lbl_path = out_lbl_dir / f"{stem}_aug{i:02d}.txt"

        cv2.imwrite(str(out_img_path), aug_img)
        write_yolo_txt(out_lbl_path, list(aug_classes), list(aug_bboxes))

        if args.save_viz:
            viz = draw_boxes(aug_img, list(aug_classes), list(aug_bboxes))
            viz_path = out_viz_dir / f"{stem}_aug{i:02d}_viz{suffix}"
            cv2.imwrite(str(viz_path), viz)

    print(f"[OK] wrote {args.n} samples to: {out_root.resolve()}")
    print(f" - images: {out_img_dir.resolve()}")
    print(f" - labels: {out_lbl_dir.resolve()}")
    if args.save_viz:
        print(f" - result: {out_viz_dir.resolve()}")


if __name__ == "__main__":
    main()
