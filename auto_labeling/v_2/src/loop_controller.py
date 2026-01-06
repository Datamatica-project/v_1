from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
import shutil
from typing import Optional, Any, List, Dict

import yaml

from auto_labeling.v_1.scripts.logger import log, log_json
from auto_labeling.v_1.src.pass_fail_filter import split_pass_fail
from auto_labeling.v_1.src.teacher_runner import run_teacher_on_fail
from auto_labeling.v_1.src.yolo_mini_trainer import train_on_teacher_pseudo

ROOT = Path(__file__).resolve().parents[1]

def _iter_images(d: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out: list[Path] = []
    for e in exts:
        out.extend(sorted(d.glob(f"*{e}")))
    return out


def _clear_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for p in d.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def _link_or_copy(src: Path, dst: Path, mode: str = "copy") -> None:
    mode = (mode or "copy").lower().strip()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    if mode == "symlink":
        os.symlink(str(src), str(dst))
    elif mode == "hardlink":
        os.link(str(src), str(dst))
    else:
        dst.write_bytes(src.read_bytes())


def _copy_files(src_paths: List[Path], dst_dir: Path, *, copy_mode: str = "copy") -> int:
    """
    용량 줄이려면 symlink/hardlink도 쓰도록 통일.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_paths:
        if not p.exists():
            continue
        _link_or_copy(p, dst_dir / p.name, mode=copy_mode)
        n += 1
    return n


def _student_make_yolo_labels(
    *,
    weights: Path,
    img_dir: Path,
    out_label_dir: Path,
    device: str = "0",
    imgsz: int = 640,
    conf: float = 0.01,
    max_det: int = 300,
) -> int:

    from ultralytics import YOLO

    out_label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    results = model.predict(
        source=str(img_dir),
        conf=float(conf),
        imgsz=int(imgsz),
        device=device,
        max_det=int(max_det),
        verbose=False,
        save=False,
    )

    n = 0
    for r in results:
        img_path = Path(getattr(r, "path", ""))
        if not img_path.name:
            continue

        lbl_path = out_label_dir / f"{img_path.stem}.txt"

        if r.boxes is None or len(r.boxes) == 0:
            lbl_path.write_text("", encoding="utf-8")
            n += 1
            continue

        cls = r.boxes.cls.detach().cpu().numpy().astype(int)
        xywhn = r.boxes.xywhn.detach().cpu().numpy()

        lines: list[str] = []
        for c, (x, y, w, h) in zip(cls, xywhn):
            lines.append(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n += 1

    return n

def load_loop_cfg(cfg_path: Optional[Path] = None) -> dict:
    if cfg_path is None:
        cfg_path = ROOT / "configs" / "v1_loop_real.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("loop", data)



@dataclass
class RoundStats:
    round: int
    n_total: int
    n_pass: int
    n_fail_student: int
    fail_ratio: float
    fail_sample_n: int
    teacher_success: int
    teacher_miss: int
    train_imgs: int
    new_weight: Optional[str] = None


# -----------------------------
# helpers (metric stop)
# -----------------------------
def _metric_direction(metric_name: str) -> str:
    m = (metric_name or "").strip().lower()
    if m in {"fail_ratio", "n_fail_student", "fail_count", "n_fail"}:
        return "lower"
    return "higher"


def _get_metric(rs: RoundStats, metric_name: str) -> Optional[float]:
    key = (metric_name or "").strip()
    if not key:
        return None
    v = getattr(rs, key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


# -----------------------------
# core steps
# -----------------------------
def _prepare_round_dirs(round_root: Path) -> dict[str, Path]:
    """
    ✅ r0: pass/images + pass/labels
    ✅ r>=1: pass_fail/images + pass_fail/labels
    """
    dirs = {
        # r0 진짜 PASS
        "pass_img": round_root / "pass" / "images",
        "pass_lbl": round_root / "pass" / "labels",

        # r>=1 FAIL→PASS 승급분
        "pass_fail_img": round_root / "pass_fail" / "images",
        "pass_fail_lbl": round_root / "pass_fail" / "labels",

        # FAIL 잔존분(다음 라운드 입력)
        "fail_fail_img": round_root / "fail_fail" / "images",

        "fail_sample_img": round_root / "fail_sample" / "images",
        "teacher_lbl": round_root / "teacher_pseudo" / "labels",
        "miss_img": round_root / "miss" / "images",
        "train_img": round_root / "train" / "images",
        "train_lbl": round_root / "train" / "labels",
    }
    for d in dirs.values():
        _clear_dir(d)
    return dirs


def _teacher_infer_with_miss(
    *,
    fail_sample_imgs: List[Path],
    fail_sample_dir: Path,
    out_teacher_lbl_dir: Path,
    teacher_conf_th: float,
) -> tuple[List[Path], List[Path]]:
    """
    teacher_weights/device는 teacher_model.yaml에서 처리.
    loop에서는 teacher_conf_th만 override한다.
    """
    used_imgs = run_teacher_on_fail(
        fail_img_dir=fail_sample_dir,
        out_label_dir=out_teacher_lbl_dir,
        teacher_weights=None,      # teacher_model.yaml(weights_uri)로 해결
        device=None,               # teacher_model.yaml(device)로 해결
        max_fail_samples=None,     # 이미 sample_dir에 들어있음
        conf_th=float(teacher_conf_th),
    )

    used_stems = {p.stem for p in used_imgs}
    miss_imgs = [p for p in fail_sample_imgs if p.stem not in used_stems]
    return used_imgs, miss_imgs


def _build_trainset(
    *,
    teacher_success_imgs: List[Path],
    teacher_lbl_dir: Path,
    train_img_dir: Path,
    train_lbl_dir: Path,
    use_gt_anchor: bool,
    gt_anchor_ratio: float,
    gt_img_dir: Path,
    gt_lbl_dir: Path,
    copy_mode: str = "copy",
) -> int:
    # (A) teacher-success
    for img_p in teacher_success_imgs:
        lbl_p = teacher_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        _link_or_copy(img_p, train_img_dir / img_p.name, mode=copy_mode)
        _link_or_copy(lbl_p, train_lbl_dir / lbl_p.name, mode=copy_mode)

    # (B) GT anchor
    if use_gt_anchor:
        gt_pool = _iter_images(gt_img_dir)
        if gt_pool:
            n_anchor = max(1, int(len(gt_pool) * float(gt_anchor_ratio)))
            pick = random.sample(gt_pool, min(n_anchor, len(gt_pool)))
            added = 0
            for img in pick:
                lbl = gt_lbl_dir / f"{img.stem}.txt"
                if not lbl.exists():
                    continue
                _link_or_copy(img, train_img_dir / img.name, mode=copy_mode)
                _link_or_copy(lbl, train_lbl_dir / lbl.name, mode=copy_mode)
                added += 1
            log(f"[ROUND] GT anchor added = {added} (ratio={gt_anchor_ratio})")

    return len(_iter_images(train_img_dir))


def run_one_round(
    *,
    round_idx: int,
    student_w: Path,
    loop_cfg: dict,
    prev_fail_img_dir: Path,
) -> tuple[Optional[Path], RoundStats]:
    device = str(loop_cfg.get("device", "0"))

    # Student PASS/FAIL
    conf_th = float(loop_cfg.get("conf_pass_threshold", 0.7))
    allow_empty = bool(loop_cfg.get("allow_empty_boxes_pass", True))

    # Teacher (override conf만 loop에서)
    teacher_conf_th = float(loop_cfg.get("teacher_conf_th", 0.5))
    use_teacher = bool(loop_cfg.get("use_teacher", True))

    fs = loop_cfg.get("fail_sampler", {}) or {}
    fail_sample_max = int(fs.get("max_fail_per_round", 500))

    use_gt_anchor = bool(loop_cfg.get("use_gt_anchor", True))
    gt_anchor_ratio = float(loop_cfg.get("gt_anchor_ratio", 0.5))
    copy_mode = str(loop_cfg.get("copy_mode", "copy"))

    # train opts
    round_epochs = int(loop_cfg.get("round_epochs", 2))
    round_lr = float(loop_cfg.get("round_lr", 5e-5))
    freeze_backbone = bool(loop_cfg.get("freeze_backbone", True))
    freeze_layers = int(loop_cfg.get("freeze_layers", 10))
    batch = loop_cfg.get("round_batch", None)
    workers = loop_cfg.get("round_workers", None)
    imgsz = int(loop_cfg.get("imgsz", 640))

    # PASS 라벨 생성 옵션
    save_pass_labels = bool(loop_cfg.get("save_pass_labels", True))
    pass_label_conf = float(loop_cfg.get("pass_label_conf", float(loop_cfg.get("nms_conf_floor", 0.01))))

    # GT anchor용
    gt_img_dir = ROOT / "data" / "GT" / "images"
    gt_lbl_dir = ROOT / "data" / "GT" / "labels"

    prev_fail_img_dir = Path(prev_fail_img_dir)
    if not prev_fail_img_dir.exists():
        raise FileNotFoundError(f"[ROUND {round_idx}] prev_fail_img_dir not found: {prev_fail_img_dir}")

    round_root = ROOT / "data" / f"round_r{round_idx}"
    dirs = _prepare_round_dirs(round_root)

    # ✅ r0는 pass/, r>=1은 pass_fail/ 로 분리
    if round_idx == 0:
        pass_img_dir = dirs["pass_img"]
        pass_lbl_dir = dirs["pass_lbl"]
        pass_tag = "PASS"
    else:
        pass_img_dir = dirs["pass_fail_img"]
        pass_lbl_dir = dirs["pass_fail_lbl"]
        pass_tag = "PASS_FAIL"

    log(f"\n===== ROUND {round_idx} START (FAIL-only) =====")
    log(f"[ROUND {round_idx}] student = {student_w.name}")
    log(f"[ROUND {round_idx}] prev_fail = {prev_fail_img_dir}")
    log(f"[ROUND {round_idx}] copy_mode = {copy_mode}")

    # PASS/FAIL on prev_fail
    stats_raw = split_pass_fail(
        weights=student_w,
        src_img_dir=prev_fail_img_dir,
        pass_img_dir=pass_img_dir,
        fail_img_dir=dirs["fail_fail_img"],
        conf_th=conf_th,
        allow_empty_boxes_pass=allow_empty,
        device=device,
        copy_mode=copy_mode,
        pass_fail_policy=dict(loop_cfg.get("pass_fail_policy", {}) or {}),
        max_samples=loop_cfg.get("max_samples", None),

        save_labels=False,
        src_label_dir=None,
        pass_label_dir=None,
        fail_label_dir=None,

        nms_conf_floor=float(loop_cfg.get("nms_conf_floor", 0.01)),
        max_det=int(loop_cfg.get("max_det", 300)),
    )

    n_pass = int(stats_raw["pass"])
    n_fail = int(stats_raw["fail"])
    n_total = int(stats_raw["used_imgs"])
    fail_ratio = n_fail / max(1, n_total)

    log(f"[ROUND {round_idx}] {pass_tag}={n_pass}, FAIL_FAIL={n_fail}, FAIL_ratio={fail_ratio:.4f}")

    # ✅ PASS 라벨 생성(학생 pseudo-label)
    if save_pass_labels and n_pass > 0:
        made = _student_make_yolo_labels(
            weights=student_w,
            img_dir=pass_img_dir,
            out_label_dir=pass_lbl_dir,
            device=device,
            imgsz=imgsz,
            conf=pass_label_conf,
            max_det=int(loop_cfg.get("max_det", 300)),
        )
        log(f"[ROUND {round_idx}] {pass_tag} labels made = {made} -> {pass_lbl_dir}")

    if n_fail == 0:
        rs = RoundStats(
            round=round_idx,
            n_total=n_total,
            n_pass=n_pass,
            n_fail_student=0,
            fail_ratio=0.0,
            fail_sample_n=0,
            teacher_success=0,
            teacher_miss=0,
            train_imgs=0,
            new_weight=None,
        )
        log(f"[ROUND {round_idx}] FAIL_FAIL=0 → 종료(수렴)")
        log_json({"event": "round_summary", **rs.__dict__})
        return None, rs

    # FAIL sample
    fail_pool = _iter_images(dirs["fail_fail_img"])
    if not fail_pool:
        rs = RoundStats(
            round=round_idx,
            n_total=n_total,
            n_pass=n_pass,
            n_fail_student=n_fail,
            fail_ratio=fail_ratio,
            fail_sample_n=0,
            teacher_success=0,
            teacher_miss=0,
            train_imgs=0,
            new_weight=None,
        )
        log(f"[ROUND {round_idx}] fail_fail 이미지 0장 → 종료")
        log_json({"event": "round_summary", **rs.__dict__})
        return None, rs

    fail_sample_n = min(fail_sample_max, len(fail_pool))
    fail_sample = random.sample(fail_pool, fail_sample_n)
    _copy_files(fail_sample, dirs["fail_sample_img"], copy_mode=copy_mode)
    log(f"[ROUND {round_idx}] FAIL sample = {fail_sample_n}")

    # Teacher
    if not use_teacher:
        rs = RoundStats(
            round=round_idx,
            n_total=n_total,
            n_pass=n_pass,
            n_fail_student=n_fail,
            fail_ratio=fail_ratio,
            fail_sample_n=fail_sample_n,
            teacher_success=0,
            teacher_miss=fail_sample_n,
            train_imgs=0,
            new_weight=None,
        )
        log(f"[ROUND {round_idx}] Teacher disabled → 종료")
        log_json({"event": "round_summary", **rs.__dict__})
        return None, rs

    teacher_success_imgs, teacher_miss_imgs = _teacher_infer_with_miss(
        fail_sample_imgs=fail_sample,
        fail_sample_dir=dirs["fail_sample_img"],
        out_teacher_lbl_dir=dirs["teacher_lbl"],
        teacher_conf_th=teacher_conf_th,
    )

    _copy_files(teacher_miss_imgs, dirs["miss_img"], copy_mode=copy_mode)
    log(f"[ROUND {round_idx}] Teacher SUCCESS = {len(teacher_success_imgs)}")
    log(f"[ROUND {round_idx}] Teacher MISS    = {len(teacher_miss_imgs)}")

    if not teacher_success_imgs:
        rs = RoundStats(
            round=round_idx,
            n_total=n_total,
            n_pass=n_pass,
            n_fail_student=n_fail,
            fail_ratio=fail_ratio,
            fail_sample_n=fail_sample_n,
            teacher_success=0,
            teacher_miss=len(teacher_miss_imgs),
            train_imgs=0,
            new_weight=None,
        )
        log(f"[ROUND {round_idx}] Teacher SUCCESS=0 → 종료")
        log_json({"event": "round_summary", **rs.__dict__})
        return None, rs

    # trainset
    train_imgs = _build_trainset(
        teacher_success_imgs=teacher_success_imgs,
        teacher_lbl_dir=dirs["teacher_lbl"],
        train_img_dir=dirs["train_img"],
        train_lbl_dir=dirs["train_lbl"],
        use_gt_anchor=use_gt_anchor,
        gt_anchor_ratio=gt_anchor_ratio,
        gt_img_dir=gt_img_dir,
        gt_lbl_dir=gt_lbl_dir,
        copy_mode=copy_mode,
    )

    if train_imgs == 0:
        rs = RoundStats(
            round=round_idx,
            n_total=n_total,
            n_pass=n_pass,
            n_fail_student=n_fail,
            fail_ratio=fail_ratio,
            fail_sample_n=fail_sample_n,
            teacher_success=len(teacher_success_imgs),
            teacher_miss=len(teacher_miss_imgs),
            train_imgs=0,
            new_weight=None,
        )
        log(f"[ROUND {round_idx}] train imgs=0 → 종료")
        log_json({"event": "round_summary", **rs.__dict__})
        return None, rs

    # Micro-FT
    new_w = train_on_teacher_pseudo(
        base_weights=student_w,
        img_dir=dirs["train_img"],
        label_dir=dirs["train_lbl"],
        out_weights=None,
        round_index=round_idx,
        sampler="fail_only",
        anchor_ratio=int(gt_anchor_ratio * 10),
        epochs=round_epochs,
        imgsz=imgsz,
        device=device,
        lr0=round_lr,
        freeze_backbone=freeze_backbone,
        freeze_layers=freeze_layers,
        batch=batch if batch is None else int(batch),
        workers=workers if workers is None else int(workers),
    )

    rs = RoundStats(
        round=round_idx,
        n_total=n_total,
        n_pass=n_pass,
        n_fail_student=n_fail,
        fail_ratio=fail_ratio,
        fail_sample_n=fail_sample_n,
        teacher_success=len(teacher_success_imgs),
        teacher_miss=len(teacher_miss_imgs),
        train_imgs=train_imgs,
        new_weight=str(new_w),
    )

    log(f"[ROUND {round_idx}] NEW student = {new_w.name}")
    log(f"===== ROUND {round_idx} END =====\n")
    log_json({"event": "round_summary", **rs.__dict__})
    return new_w, rs


def run_loop(
    *,
    loop_cfg: dict,
    student_w: Optional[Path] = None,
    initial_fail_img_dir: Optional[Path] = None,
) -> tuple[Path, list[Dict[str, Any]]]:
    if student_w is None:
        raise ValueError(
            "[run_loop] student_w is None. "
            "Provide --student or train student from GT before loop."
        )
    if initial_fail_img_dir is None:
        initial_fail_img_dir = ROOT / "data" / "fail" / "images"
    initial_fail_img_dir = Path(initial_fail_img_dir)

    if not initial_fail_img_dir.exists():
        raise FileNotFoundError(f"[LOOP] initial_fail_img_dir not found: {initial_fail_img_dir}")

    max_rounds = int(loop_cfg.get("max_rounds", 2))

    enable_fail_ratio_stop = bool(loop_cfg.get("enable_fail_ratio_stop", True))
    min_fail_ratio = float(loop_cfg.get("min_fail_ratio", 0.01))

    enable_fail_count_stop = bool(loop_cfg.get("enable_fail_count_stop", True))
    min_fail_count = int(loop_cfg.get("min_fail_count", 300))

    enable_metric_stop = bool(loop_cfg.get("enable_metric_stop", False))
    metric_name = str(loop_cfg.get("metric_name", "fail_ratio"))
    min_improvement = float(loop_cfg.get("min_improvement", 0.0))
    patience_rounds = int(loop_cfg.get("patience_rounds", 2))

    best_metric: Optional[float] = None
    no_improve = 0
    direction = _metric_direction(metric_name)

    history: list[Dict[str, Any]] = []

    prev_fail_img_dir = initial_fail_img_dir
    log(f"[LOOP] start(FAIL-only): student={student_w.name}, max_rounds={max_rounds} (r0..r{max_rounds-1})")
    log("[LOOP] teacher weights: from configs/teacher_model.yaml (weights_uri -> cache)")
    log(f"[LOOP] initial_fail_img_dir = {prev_fail_img_dir}")

    # ✅ r0부터 시작
    for r in range(0, max_rounds):
        new_w, rs = run_one_round(
            round_idx=r,
            student_w=student_w,
            loop_cfg=loop_cfg,
            prev_fail_img_dir=prev_fail_img_dir,
        )
        history.append(rs.__dict__)

        if enable_fail_ratio_stop and rs.fail_ratio <= min_fail_ratio:
            log(f"[LOOP] STOP: fail_ratio {rs.fail_ratio:.4f} <= {min_fail_ratio:.4f}")
            break

        if enable_fail_count_stop and rs.n_fail_student < min_fail_count:
            log(f"[LOOP] STOP: fail_count {rs.n_fail_student} < {min_fail_count}")
            break

        if enable_metric_stop:
            cur = _get_metric(rs, metric_name)
            if cur is None:
                log(f"[LOOP] metric_stop enabled but metric '{metric_name}' not found → ignore")
            else:
                if best_metric is None:
                    best_metric = cur
                    no_improve = 0
                    log(f"[LOOP] metric init: {metric_name}={best_metric:.6f} (dir={direction})")
                else:
                    improvement = (best_metric - cur) if direction == "lower" else (cur - best_metric)

                    if improvement >= min_improvement:
                        best_metric = cur
                        no_improve = 0
                        log(f"[LOOP] metric improved: {metric_name}={cur:.6f} (Δ={improvement:.6f} >= {min_improvement})")
                    else:
                        no_improve += 1
                        log(
                            f"[LOOP] metric no-improve: {metric_name}={cur:.6f} (Δ={improvement:.6f} < {min_improvement}) "
                            f"patience={no_improve}/{patience_rounds}"
                        )

                    if no_improve >= patience_rounds:
                        log(
                            f"[LOOP] STOP(metric): no improvement for {patience_rounds} rounds "
                            f"(metric={metric_name}, min_improvement={min_improvement})"
                        )
                        break

        if new_w is None:
            log("[LOOP] STOP: new weight is None")
            break

        next_fail_dir = ROOT / "data" / f"round_r{r}" / "fail_fail" / "images"
        if not next_fail_dir.exists():
            raise FileNotFoundError(f"[LOOP] next fail_fail dir not found: {next_fail_dir}")

        prev_fail_img_dir = next_fail_dir
        student_w = new_w

    log(f"[LOOP] done: final_student={student_w.name}")
    return student_w, history
