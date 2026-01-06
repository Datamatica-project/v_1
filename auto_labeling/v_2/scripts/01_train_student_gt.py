# auto_labeling/v_1/scripts/01_train_student_gt.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import yaml

from ultralytics import YOLO
from auto_labeling.v_1.scripts.logger import log, log_json

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------
def _build_student_init_name(
    base_model: Path,
    *,
    tag: str = "h10",
) -> str:

    date_str = datetime.now().strftime("%Y%m%d")
    stem = base_model.stem
    if stem.endswith(".pt"):
        stem = stem.replace(".pt", "")
    return f"{stem}_student_{tag}_init_{date_str}.pt"


def _check_gt_ready(gt_root: Path) -> None:
    if not gt_root.exists():
        raise FileNotFoundError(f"[GT_TRAIN] GT root not found: {gt_root}")

    for sub in ["images", "labels", "data.yaml"]:
        if not (gt_root / sub).exists():
            raise FileNotFoundError(f"[GT_TRAIN] missing {sub} in {gt_root}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train initial Student on GT only")

    ap.add_argument(
        "--gt",
        type=str,
        default=str(ROOT / "data" / "GT"),
        help="GT root (images/ labels/ data.yaml)",
    )
    ap.add_argument(
        "--base",
        type=str,
        default=str(ROOT / "models" / "pretrained" / "weights" / "best.pt"),
        help="base pretrained YOLO weight",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", type=str, default="h10", help="student tag (e.g. h10)")
    ap.add_argument("--project", type=str, default=str(ROOT / "models" / "user_yolo"))
    ap.add_argument("--name", type=str, default="student_gt_init")

    args = ap.parse_args()

    gt_root = Path(args.gt)
    base_model = Path(args.base)
    project = Path(args.project)

    _check_gt_ready(gt_root)

    if not base_model.exists():
        raise FileNotFoundError(f"[GT_TRAIN] base model not found: {base_model}")

    out_dir = project / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("[GT_TRAIN] ===== Student GT Training START =====")
    log(f"[GT_TRAIN] GT        = {gt_root}")
    log(f"[GT_TRAIN] base     = {base_model.name}")
    log(f"[GT_TRAIN] epochs   = {args.epochs}")
    log(f"[GT_TRAIN] imgsz    = {args.imgsz}")
    log(f"[GT_TRAIN] batch    = {args.batch}")
    log(f"[GT_TRAIN] device   = {args.device}")

    model = YOLO(str(base_model))

    results = model.train(
        data=str(gt_root / "data.yaml"),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        project=str(out_dir),
        name=str(args.name),
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError("[GT_TRAIN] best.pt not found after training")

    weights_dir = project / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    final_name = _build_student_init_name(base_model, tag=args.tag)
    final_path = weights_dir / final_name
    final_path.write_bytes(best.read_bytes())

    log(f"[GT_TRAIN] FINAL student weight saved: {final_path}")

    log_json({
        "event": "student_gt_init",
        "gt_root": str(gt_root),
        "base_model": base_model.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "out_weight": final_name,
    })

    log("[GT_TRAIN] ===== DONE =====")


if __name__ == "__main__":
    main()
