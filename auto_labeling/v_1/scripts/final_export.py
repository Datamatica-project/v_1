# auto_labeling/v_1/scripts/final_export.py
from __future__ import annotations

from pathlib import Path
import argparse

from auto_labeling.v_1.scripts.logger import log
from auto_labeling.v_1.src.loop_controller import ROOT, load_loop_cfg, run_loop
from auto_labeling.v_1.src.export_pass_fail_final import export_pass_fail_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "v1_loop.yaml"))
    ap.add_argument("--student", type=str, default="", help="initial student weight (.pt), empty = default")

    ap.add_argument("--export_root", type=str, default=str(ROOT / "data" / "final_export"))
    ap.add_argument("--export_conf", type=float, default=0.3)

    # (선택) 고정 PASS pool을 최종 pass에 포함하고 싶으면
    ap.add_argument("--pass_pool", type=str, default="", help="optional PASS pool images dir")

    # pass_fail을 pass로도 합칠지
    ap.add_argument("--merge_pass", action="store_true", help="merge pass_fail into pass/")

    args = ap.parse_args()

    loop_cfg = load_loop_cfg(Path(args.cfg))
    student_w = Path(args.student) if args.student else None

    # 1) loop 수행 (teacher는 configs/teacher_model.yaml)
    final_w, history = run_loop(loop_cfg=loop_cfg, student_w=student_w)

    # 2) last round
    if history:
        last_round = int(history[-1]["round"])
    else:
        last_round = int(loop_cfg.get("max_rounds", 1))
        log(f"[FINAL_EXPORT] WARNING: history empty → fallback last_round={last_round}")

    round_root = ROOT / "data" / f"round_r{last_round}"
    pass_fail_img_dir = round_root / "pass_fail" / "images"
    fail_fail_img_dir = round_root / "fail_fail" / "images"
    miss_img_dir = round_root / "miss" / "images"

    pass_img_dirs = []
    if args.pass_pool.strip():
        pass_img_dirs.append(Path(args.pass_pool.strip()))

    export_pass_fail_final(
        student_weights=final_w,
        out_root=Path(args.export_root),
        pass_img_dirs=pass_img_dirs,
        pass_fail_img_dir=pass_fail_img_dir,
        fail_fail_img_dir=fail_fail_img_dir,
        miss_img_dir=miss_img_dir,
        device=str(loop_cfg.get("device", "0")),
        conf_th=float(args.export_conf),
        merge_pass_into_pass_dir=bool(args.merge_pass),
    )

    log("[FINAL_EXPORT] DONE")


if __name__ == "__main__":
    main()
