from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime

from ultralytics import YOLO

from auto_labeling.v_1.scripts.logger import log, log_json
from auto_labeling.v_1.src.loop_controller import ROOT, load_loop_cfg, run_loop
from auto_labeling.v_1.src.export_pass_fail_final import export_pass_fail_final

def train_student_on_gt(
    gt_root: Path,
    out_dir: Path,
    *,
    base_model: Path,
    epochs: int,
    imgsz: int,
    batch: int | None,
    device: str,
) -> Path:
    data_yaml = gt_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"[TRAIN_GT] data.yaml not found: {data_yaml}")

    out_dir.mkdir(parents=True, exist_ok=True)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"student_gt_init_{tag}"

    log(f"[TRAIN_GT] base_model={base_model}")
    log(f"[TRAIN_GT] data={data_yaml}")
    log(f"[TRAIN_GT] epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}")

    model = YOLO(str(base_model))
    save_project = Path(out_dir) / "runs"
    save_project.mkdir(parents=True, exist_ok=True)

    model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=(int(batch) if batch is not None else None),
        device=device,
        name=run_name,
        project=str(save_project),
        verbose=True,
    )

    cand = sorted(save_project.rglob("weights/best.pt"))
    if not cand:
        raise FileNotFoundError(f"[TRAIN_GT] best.pt not found under {save_project}/**/weights/best.pt")

    best_pt = cand[-1]  # 가장 최근
    final_name = out_dir / f"student_h10_init_{tag}.pt"
    final_name.write_bytes(best_pt.read_bytes())

    latest_name = out_dir / "student_h10_init_latest.pt"
    latest_name.write_bytes(final_name.read_bytes())

    log(f"[TRAIN_GT] best_pt={best_pt}")
    log(f"[TRAIN_GT] saved={final_name}")
    log(f"[TRAIN_GT] latest_saved={latest_name}")

    log_json(
        {
            "event": "student_gt_init_done",
            "gt_root": str(gt_root),
            "base_model": str(base_model),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "best_pt": str(best_pt),
            "saved_weight": str(final_name),
            "latest_weight": str(latest_name),
        }
    )

    return final_name



def main():
    parser = argparse.ArgumentParser(description="V1 End-to-End Auto Labeling Test Runner (GT + UNLABELS)")

    parser.add_argument("--cfg", type=str, default=str(ROOT / "configs" / "v1_loop.yaml"))
    parser.add_argument("--gt_dir", type=str, default=str(ROOT / "data" / "gt"))
    parser.add_argument("--unlabels_dir", type=str, default=str(ROOT / "data" / "unlabels" / "images"))

    parser.add_argument("--student", type=str, default="", help="initial student weight (.pt)")
    parser.add_argument("--train_gt", action="store_true", help="train student from GT(10%) before loop")
    parser.add_argument("--base_model", type=str, default="", help="pretrained base model (.pt), used when --train_gt")

    parser.add_argument("--gt_epochs", type=int, default=10)
    parser.add_argument("--gt_imgsz", type=int, default=640)
    parser.add_argument("--gt_batch", type=int, default=8)

    parser.add_argument("--export_root", type=str, default=str(ROOT / "data" / "final_set"))
    parser.add_argument("--export_conf", type=float, default=0.3)
    parser.add_argument("--merge_pass", action="store_true", help="merge pass_fail into pass/ for final training set")

    parser.add_argument("--pass_pool", type=str, default=str(ROOT / "data" / "pass" / "images"))

    args = parser.parse_args()

    loop_cfg = load_loop_cfg(Path(args.cfg))
    device = str(loop_cfg.get("device", "0"))

    gt_root = Path(args.gt_dir)
    unlabels_img_dir = Path(args.unlabels_dir)

    if not unlabels_img_dir.exists():
        raise FileNotFoundError(f"[TEST_MAIN] unlabels images dir not found: {unlabels_img_dir}")

    student_w: Path | None = Path(args.student) if args.student else None

    if args.train_gt:
        base_model = Path(args.base_model) if args.base_model else None
        if base_model is None or not base_model.exists():
            raise FileNotFoundError("[TEST_MAIN] --train_gt requires --base_model <pretrained.pt> (existing file)")

        out_dir = ROOT / "models" / "user_yolo" / "weights"
        student_w = train_student_on_gt(
            gt_root=gt_root,
            out_dir=out_dir,
            base_model=base_model,
            epochs=int(args.gt_epochs),
            imgsz=int(args.gt_imgsz),
            batch=int(args.gt_batch),
            device=device,
        )

    if student_w is None:
        raise RuntimeError("[TEST_MAIN] Provide --student OR use --train_gt (no default student weight).")

    log("[TEST_MAIN] ===== V1 E2E START (GT + UNLABELS) =====")
    log(f"[TEST_MAIN] gt_dir={gt_root}")
    log(f"[TEST_MAIN] unlabels_dir={unlabels_img_dir}")
    log(f"[TEST_MAIN] initial_student={student_w}")

    # ---------------------------------------
    # 1) Loop 실행 (unlabels 전체를 초기 풀로)
    # ---------------------------------------
    final_student_w, history = run_loop(
        loop_cfg=loop_cfg,
        student_w=student_w,
        initial_fail_img_dir=unlabels_img_dir,
    )

    log("[TEST_MAIN] LOOP DONE")
    log(f"[TEST_MAIN] FINAL student weight = {final_student_w}")
    log(f"[TEST_MAIN] TOTAL rounds = {len(history)}")

    if not history:
        log("[TEST_MAIN] WARNING: history empty → export skip")
        log("[TEST_MAIN] ===== END =====")
        return

    # ---------------------------------------
    # 2) 라운드별 export (PASS / PASS_FAIL / MISS)
    # ---------------------------------------
    export_root = Path(args.export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    pass_pool_dir = Path(args.pass_pool) if args.pass_pool else None
    pass_img_dirs = []
    if pass_pool_dir and pass_pool_dir.exists():
        pass_img_dirs.append(pass_pool_dir)

    for rs in history:
        r = int(rs.get("round", 0))
        round_root = ROOT / "data" / f"round_r{r}"

        pass_fail_img_dir = round_root / "pass_fail" / "images"
        fail_fail_img_dir = round_root / "fail_fail" / "images"
        miss_img_dir = round_root / "miss" / "images"

        # ✅ loop_controller에서 new_weight를 "full path"로 기록하도록 바꿀 것(아래 참고)
        w = (rs.get("new_weight", "") or "").strip()
        student_w_round = Path(w) if w else Path(final_student_w)

        out_round = export_root / f"round_r{r}"
        log(f"[TEST_MAIN] EXPORT round={r} -> {out_round}")
        log(f"[TEST_MAIN]   using student weight: {student_w_round}")

        export_pass_fail_final(
            student_weights=student_w_round,
            out_root=out_round,
            pass_img_dirs=pass_img_dirs,
            pass_fail_img_dir=pass_fail_img_dir,
            fail_fail_img_dir=fail_fail_img_dir,
            miss_img_dir=miss_img_dir,
            device=device,
            conf_th=float(args.export_conf),
            merge_pass_into_pass_dir=bool(args.merge_pass),
        )

        log_json(
            {
                "event": "round_export_done",
                "round": r,
                "out_round": str(out_round),
                "student_weight": str(student_w_round),
            }
        )

    log("[TEST_MAIN] ===== E2E DONE =====")


if __name__ == "__main__":
    main()
