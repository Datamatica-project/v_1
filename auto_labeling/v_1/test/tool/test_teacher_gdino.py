# auto_labeling/v_1/test/tool/test_teacher_gdino.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse

from auto_labeling.v_1.src.teacher_runner_DINO import (
    load_teacher_config,
    TeacherRunner,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="테스트용: GroundingDINO Teacher로 이미지 몇 장만 추론해보기"
    )
    ap.add_argument(
        "--teacher_cfg",
        type=str,
        default="auto_labeling/v_1/configs/teacher_model_DINOl.yaml",
        help="teacher_model_DINOl.yaml 경로",
    )
    ap.add_argument(
        "--img_dir",
        type=str,
        default="auto_labeling/v_1/test/images",
        help="테스트용 이미지 폴더",
    )
    ap.add_argument(
        "--num_images",
        type=int,
        default=3,
        help="최대 몇 장까지 테스트할지",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="auto_labeling/v_1/test/out_labels",
        help="YOLO txt 라벨이 저장될 폴더",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    teacher_cfg_path = Path(args.teacher_cfg)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)

    if not img_dir.exists():
        print(f"[ERROR] 이미지 폴더가 없습니다: {img_dir}")
        print("  → 몇 장 넣고 다시 실행해줘.")
        return

    # 1) Teacher 설정 로드
    t_cfg = load_teacher_config(teacher_cfg_path)

    # 2) 출력 경로를 test용 out_dir로 오버라이드
    t_cfg.output_label_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) TeacherRunner 생성
    runner = TeacherRunner(t_cfg)

    # 4) 이미지 몇 장만 골라서 테스트
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(
        [p for p in img_dir.rglob("*") if p.suffix.lower() in img_exts]
    )

    if not img_paths:
        print(f"[ERROR] {img_dir} 안에서 이미지 파일을 찾지 못했습니다.")
        return

    img_paths = img_paths[: args.num_images]
    print(f"[INFO] 테스트 대상 이미지 수: {len(img_paths)}")

    for i, img_path in enumerate(img_paths, start=1):
        print(f"\n[TEST] ({i}/{len(img_paths)}) {img_path.name} 추론 중...")
        preds = runner.infer_image(img_path)

        print(f"  - 감지된 박스 수: {len(preds)}")
        for j, (cls_id, cx, cy, bw, bh) in enumerate(preds[:5], start=1):
            print(f"    [{j}] cls={cls_id}, cx={cx:.3f}, cy={cy:.3f}, w={bw:.3f}, h={bh:.3f}")

        runner.save_yolo_label(img_path, preds)

    print(f"\n[DONE] YOLO txt 라벨이 여기 저장됨: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
