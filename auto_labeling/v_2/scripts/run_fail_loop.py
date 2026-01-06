# auto_labeling/v_1/scripts/run_fail_loop.py
from __future__ import annotations

from pathlib import Path
import argparse

from auto_labeling.v_1.scripts.logger import log
from auto_labeling.v_1.src.loop_controller import load_loop_cfg, run_loop, ROOT


def _iter_images(d: Path) -> int:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    n = 0
    for e in exts:
        n += len(list(d.glob(f"*{e}")))
    return n


def main():
    ap = argparse.ArgumentParser(description="V1 FAIL-only loop runner")

    ap.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "v1_loop_real.yaml"))
    ap.add_argument("--student", type=str, default="", help="student weight (.pt), empty=default")

    ap.add_argument(
        "--fail_dir",
        type=str,
        default=str(ROOT / "data" / "fail" / "images"),
        help="initial fail pool images dir (FAIL-only loop input)",
    )

    ap.add_argument("--copy_mode", type=str, default="", help="copy/symlink/hardlink (override loop_cfg.copy_mode)")
    ap.add_argument("--max_samples", type=int, default=-1, help="max samples per split (override loop_cfg.max_samples)")
    ap.add_argument("--save_labels", action="store_true", help="copy labels too (requires GT labels)")

    args = ap.parse_args()

    loop_cfg = load_loop_cfg(Path(args.cfg))

    # overrides
    if args.copy_mode.strip():
        loop_cfg["copy_mode"] = args.copy_mode.strip()
    if args.max_samples >= 0:
        loop_cfg["max_samples"] = int(args.max_samples)
    if args.save_labels:
        loop_cfg["save_labels"] = True

    student_w = Path(args.student) if args.student else None
    fail_dir = Path(args.fail_dir)

    if not fail_dir.exists():
        raise FileNotFoundError(f"[RUN] fail_dir not found: {fail_dir}")

    n_fail_imgs = _iter_images(fail_dir)
    log(f"[RUN] fail_dir = {fail_dir} (images={n_fail_imgs})")
    if n_fail_imgs == 0:
        log("[RUN] fail_dir has 0 images → nothing to do")
        return

    final_w, history = run_loop(
        loop_cfg=loop_cfg,
        student_w=student_w,
        initial_fail_img_dir=fail_dir,
    )

    log(f"[RUN] FINAL student weight = {final_w}")
    log(f"[RUN] rounds = {len(history)}")

    if history:
        last_r = history[-1]["round"]
        log(f"[RUN] last round dir = {ROOT / 'data' / f'round_r{last_r}'}")
        log(f"[RUN] last fail_fail dir = {ROOT / 'data' / f'round_r{last_r}' / 'fail_fail' / 'images'}")
        log(f"[RUN] last pass_fail dir = {ROOT / 'data' / f'round_r{last_r}' / 'pass_fail' / 'images'}")
        log(f"[RUN] last miss dir      = {ROOT / 'data' / f'round_r{last_r}' / 'miss' / 'images'}")


if __name__ == "__main__":
    main()
