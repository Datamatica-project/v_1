# auto_labeling/v1/test/test_vis_debug.py
# -*- coding: utf-8 -*-

from pathlib import Path

from auto_labeling.v_1.src.teacher_runner import TeacherRunner


def main():
    ROOT = Path(__file__).resolve().parents[1]  # v1/ 디렉토리
    cfg_path = ROOT / "configs" / "teacher_model.yaml"

    img_dir = ROOT / "test" / "images"
    out_label_dir = ROOT / "test" / "out_labels"
    out_vis_dir = ROOT / "test" / "debug_vis"

    runner = TeacherRunner(cfg_path)
    runner.run_dir(img_dir, out_label_dir, out_vis_dir)

    print("[DONE] test_vis_debug finished")


if __name__ == "__main__":
    main()
