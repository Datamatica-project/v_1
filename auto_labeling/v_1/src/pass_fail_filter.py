from __future__ import annotations

from pathlib import Path
import os
import random
import math
from typing import Optional, Literal, Dict, Any

from ultralytics import YOLO

from auto_labeling.v_1.scripts.logger import log, log_json

ROOT = Path(__file__).resolve().parents[1]

PolicyMode = Literal["min", "ratio", "drop_lowest"]


def _iter_images(d: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out: list[Path] = []
    for e in exts:
        out.extend(sorted(d.glob(f"*{e}")))
    return out


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


def _decide_pass(
    confs: list[float],
    *,
    conf_th: float,
    policy: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    """
    return: (is_pass, debug_dict)
    """
    if not confs:
        return False, {"n_boxes": 0}

    th = float(conf_th)

    mode = str(policy.get("mode", "ratio")).strip().lower()
    if mode == "min":
        is_pass = (min(confs) >= th)
        return is_pass, {"mode": "min", "min_conf": float(min(confs)), "n_boxes": len(confs)}

    if mode == "ratio":
        # low-conf 비율이 max_low_conf_ratio 이하이면 PASS
        max_low = float(policy.get("max_low_conf_ratio", 0.25))
        max_low = max(0.0, min(1.0, max_low))
        low = sum(1 for c in confs if float(c) < th)
        low_ratio = low / max(1, len(confs))
        is_pass = (low_ratio <= max_low)
        return is_pass, {
            "mode": "ratio",
            "n_boxes": len(confs),
            "low": int(low),
            "low_ratio": float(low_ratio),
            "max_low_conf_ratio": float(max_low),
            "min_conf": float(min(confs)),
        }

    if mode == "drop_lowest":
        # 하위 drop_lowest_ratio 만큼 버리고 남은 것들의 min>=th면 PASS
        drop_r = float(policy.get("drop_lowest_ratio", 0.20))
        drop_r = max(0.0, min(0.99, drop_r))
        n = len(confs)
        k = int(math.floor(n * drop_r))
        k = max(0, min(n - 1, k))  # 최소 1개는 남겨야 함
        s = sorted(float(c) for c in confs)
        kept = s[k:]
        is_pass = (min(kept) >= th)
        return is_pass, {
            "mode": "drop_lowest",
            "n_boxes": n,
            "drop_lowest_ratio": float(drop_r),
            "dropped_k": int(k),
            "min_conf_kept": float(min(kept)),
            "min_conf_all": float(min(confs)),
        }

    # unknown → 안전하게 min 정책으로 fallback
    is_pass = (min(confs) >= th)
    return is_pass, {"mode": "min(fallback)", "min_conf": float(min(confs)), "n_boxes": len(confs)}


def split_pass_fail(
    weights: Path,
    src_img_dir: Path,
    pass_img_dir: Path,
    fail_img_dir: Path,
    conf_th: float = 0.7,
    allow_empty_boxes_pass: bool = True,
    *,
    device: str | int | None = "0",
    nms_conf_floor: float = 0.01,
    max_det: int = 300,
    max_samples: int | None = None,
    copy_mode: str = "copy",
    save_labels: bool = False,
    src_label_dir: Path | None = None,
    pass_label_dir: Path | None = None,
    fail_label_dir: Path | None = None,
    # ✅ NEW: YAML pass_fail_policy 그대로 받음
    pass_fail_policy: Optional[Dict[str, Any]] = None,
):
    pass_img_dir.mkdir(parents=True, exist_ok=True)
    fail_img_dir.mkdir(parents=True, exist_ok=True)

    if save_labels:
        if src_label_dir is None:
            raise ValueError("[PASS/FAIL] save_labels=True 인데 src_label_dir=None 입니다.")
        if pass_label_dir is None or fail_label_dir is None:
            raise ValueError("[PASS/FAIL] save_labels=True 인데 pass_label_dir/fail_label_dir가 없습니다.")
        pass_label_dir.mkdir(parents=True, exist_ok=True)
        fail_label_dir.mkdir(parents=True, exist_ok=True)

    policy = pass_fail_policy or {"mode": "ratio", "max_low_conf_ratio": 0.25, "drop_lowest_ratio": 0.20}
    log(
        f"[PASS/FAIL] 시작 – weight={Path(weights).name}, "
        f"conf_th={conf_th}, allow_empty={allow_empty_boxes_pass}, "
        f"policy={policy}, nms_conf_floor={nms_conf_floor}, max_det={max_det}, "
        f"max_samples={max_samples}, copy_mode={copy_mode}, save_labels={save_labels}"
    )

    model = YOLO(str(weights))

    all_img_paths = _iter_images(Path(src_img_dir))
    n_total = len(all_img_paths)

    if max_samples is not None and max_samples < n_total:
        img_paths = random.sample(all_img_paths, max_samples)
    else:
        img_paths = all_img_paths

    n_used = len(img_paths)
    n_pass = 0
    n_fail = 0
    n_empty = 0

    for i, img_path in enumerate(img_paths, 1):
        results = model.predict(
            source=str(img_path),
            conf=float(nms_conf_floor),
            max_det=int(max_det),
            verbose=False,
            device=device,
        )

        boxes = results[0].boxes if results else None

        if boxes is None or len(boxes) == 0:
            n_empty += 1
            is_pass = bool(allow_empty_boxes_pass)
            debug = {"mode": "empty", "n_boxes": 0}
        else:
            confs = [float(x) for x in boxes.conf.cpu().tolist()]
            is_pass, debug = _decide_pass(confs, conf_th=float(conf_th), policy=policy)

        dst_img = (pass_img_dir if is_pass else fail_img_dir) / img_path.name
        _link_or_copy(img_path, dst_img, mode=copy_mode)

        if is_pass:
            n_pass += 1
        else:
            n_fail += 1

        if save_labels and src_label_dir is not None:
            src_lbl = Path(src_label_dir) / f"{img_path.stem}.txt"
            if src_lbl.exists():
                if is_pass:
                    _link_or_copy(src_lbl, Path(pass_label_dir) / src_lbl.name, mode=copy_mode)
                else:
                    _link_or_copy(src_lbl, Path(fail_label_dir) / src_lbl.name, mode=copy_mode)

        if i % 50 == 0 or i == n_used:
            log(
                f"[PASS/FAIL] 진행 {i}/{n_used} ({i / n_used * 100:.1f}%) "
                f"PASS={n_pass}, FAIL={n_fail}, EMPTY={n_empty}"
            )

        # (가벼운 per-image debug 로그)
        log_json(
            {
                "tag": "pass_fail_item",
                "image": img_path.name,
                "is_pass": bool(is_pass),
                "conf_th": float(conf_th),
                "allow_empty": bool(allow_empty_boxes_pass),
                "policy": policy,
                **debug,
            }
        )

    stats = {
        "total_imgs": n_total,
        "used_imgs": n_used,
        "pass": n_pass,
        "fail": n_fail,
        "empty_boxes": n_empty,
        "conf_th": float(conf_th),
        "allow_empty_boxes_pass": bool(allow_empty_boxes_pass),
        "pass_fail_policy": policy,
        "nms_conf_floor": float(nms_conf_floor),
        "max_det": int(max_det),
        "max_samples": None if max_samples is None else int(max_samples),
        "copy_mode": str(copy_mode),
        "save_labels": bool(save_labels),
    }

    log(f"[PASS/FAIL] 완료 – 전체={n_total}, 사용={n_used}, PASS={n_pass}, FAIL={n_fail}, EMPTY={n_empty}")
    log_json({"tag": "pass_fail_split", **stats})
    return stats
