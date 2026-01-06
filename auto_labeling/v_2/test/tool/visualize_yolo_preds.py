# auto_labeling/v_1/test/tool/visualize_yolo_preds.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse

import cv2


# 우리가 쓰는 클래스 id -> 이름 매핑
CLASS_NAMES = {
    0: "car",
    1: "traffic_sign",
    2: "truck",
    3: "person",
    4: "special_vehicle",
    5: "bicycle",
    6: "motorcycle",
    7: "bus",
    8: "pedestrian",
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="YOLO txt(out_labels) + images를 이용해서 시각화 이미지 생성 (디버그 포함)"
    )
    ap.add_argument(
        "--img_dir",
        type=str,
        default="auto_labeling/v_1/test/images",
        help="원본 이미지 폴더",
    )
    ap.add_argument(
        "--label_dir",
        type=str,
        default="auto_labeling/v_1/test/out_labels",
        help="YOLO txt 라벨 폴더",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="auto_labeling/v_1/test/vis",
        help="박스 그린 결과 이미지가 저장될 폴더",
    )
    return ap.parse_args()


def load_yolo_labels(label_path: Path):
    """
    YOLO txt 파일을 읽어서 [ (cls_id, cx,cy,w,h), ... ] 리스트로 반환
    """
    if not label_path.exists():
        print(f"  - [INFO] 라벨 파일 없음: {label_path.name}")
        return []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        print(f"  - [INFO] 라벨 파일은 있는데 내용이 비어 있음: {label_path.name}")
        return []

    lines = text.splitlines()
    boxes = []
    for line in lines:
        if not line.strip():
            continue

        parts = line.split()
        # cls cx cy w h  or  cls cx cy w h conf
        if len(parts) < 5:
            print(f"  - [WARN] 형식 이상한 라벨 라인: {line}")
            continue

        try:
            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            print(f"  - [WARN] 숫자 파싱 실패 라벨 라인: {line}")
            continue

        boxes.append((cls_id, cx, cy, w, h))

    print(f"  - [INFO] 라벨 로드 완료: {label_path.name}, 박스 {len(boxes)}개")
    return boxes


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=3):
    """
    img: BGR 이미지 (numpy)
    boxes: [(cls_id, cx,cy,w,h), ...]
    """
    h, w = img.shape[:2]

    for cls_id, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        cls_name = CLASS_NAMES.get(cls_id, str(cls_id))
        label = f"{cls_name}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        ty1 = max(y1 - th - baseline, 0)
        ty2 = y1
        tx1 = x1
        tx2 = x1 + tw

        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return img


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)

    print(f"[CFG] img_dir   = {img_dir}")
    print(f"[CFG] label_dir = {label_dir}")
    print(f"[CFG] out_dir   = {out_dir}")

    if not img_dir.exists():
        print(f"[ERROR] 이미지 폴더 없음: {img_dir}")
        return
    if not label_dir.exists():
        print(f"[WARN] 라벨 폴더 없음: {label_dir} (그래도 이미지만 복사해서 저장)")
        # 그래도 out_dir만 만들고 진행
    out_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(
        [p for p in img_dir.rglob("*") if p.suffix.lower() in img_exts]
    )

    print(f"[INFO] 시각화 대상 이미지 수: {len(img_paths)}")

    for i, img_path in enumerate(img_paths, start=1):
        print(f"\n[VIS] ({i}/{len(img_paths)}) 이미지: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  - [WARN] 이미지 로드 실패: {img_path}")
            continue

        label_path = label_dir / f"{img_path.stem}.txt"
        print(f"  - [DEBUG] 라벨 경로: {label_path}")
        boxes = load_yolo_labels(label_path)

        if boxes:
            img_drawn = draw_boxes(img, boxes)
        else:
            img_drawn = img.copy()
            cv2.putText(
                img_drawn,
                "NO BOX",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img_drawn)
        print(f"  - [INFO] 저장: {out_path}")

    print(f"\n[DONE] 시각화 완료. 결과 폴더: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
