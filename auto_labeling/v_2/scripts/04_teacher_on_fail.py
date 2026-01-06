# auto_labeling/v_1/scripts/04_teacher_on_fail.py
from __future__ import annotations

from pathlib import Path
import argparse

from auto_labeling.v_1.scripts.logger import log
from auto_labeling.v_1.src.teacher_runner import run_teacher_on_fail

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description="Run Teacher on FAIL-only dirs")

    ap.add_argument(
        "--round",
        type=int,
        default=-1,
        help="round index (e.g., 1). -1 = latest round",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="fail_sample",
        choices=["fail_sample", "fail_fail"],
        help="which fail dir to use inside round_rX",
    )
    ap.add_argument("--teacher", type=str, default="", help="teacher weights override (.pt)")
    ap.add_argument("--device", type=str, default="", help="device override (e.g., 0, cpu)")
    ap.add_argument("--conf", type=float, default=-1.0, help="conf_th override")
    ap.add_argument("--max_samples", type=int, default=-1, help="max_fail_samples override")
    ap.add_argument(
        "--out_labels",
        type=str,
        default="",
        help="output labels dir override (default: round_rX/teacher_pseudo/labels)",
    )

    args = ap.parse_args()

    # resolve round
    data_root = ROOT / "data"
    if args.round >= 0:
        round_root = data_root / f"round_r{args.round}"
        if not round_root.exists():
            raise FileNotFoundError(f"[04_TEACHER] round dir not found: {round_root}")
    else:
        # latest round_rX
        rounds = []
        for p in data_root.glob("round_r*"):
            if p.is_dir():
                try:
                    r = int(p.name.replace("round_r", ""))
                    rounds.append((r, p))
                except Exception:
                    pass
        if not rounds:
            raise FileNotFoundError("[04_TEACHER] no round_r* found under data/")
        rounds.sort(key=lambda x: x[0])
        round_root = rounds[-1][1]

    # pick fail_dir
    fail_dir = round_root / args.mode / "images"
    if not fail_dir.exists():
        raise FileNotFoundError(f"[04_TEACHER] fail_dir not found: {fail_dir}")

    # output labels
    if args.out_labels.strip():
        out_labels = Path(args.out_labels.strip())
    else:
        out_labels = round_root / "teacher_pseudo" / "labels"

    teacher_w = Path(args.teacher.strip()) if args.teacher.strip() else None
    device = args.device.strip() if args.device.strip() else None
    conf_th = None if args.conf < 0 else float(args.conf)
    max_samples = None if args.max_samples < 0 else int(args.max_samples)

    log(f"[04_TEACHER] round_root = {round_root}")
    log(f"[04_TEACHER] fail_dir   = {fail_dir}")
    log(f"[04_TEACHER] out_labels = {out_labels}")

    run_teacher_on_fail(
        fail_img_dir=fail_dir,
        out_label_dir=out_labels,
        teacher_weights=teacher_w,
        device=device,
        max_fail_samples=max_samples,
        conf_th=conf_th,
    )


if __name__ == "__main__":
    main()
