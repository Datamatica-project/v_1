# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import random
from typing import List, Optional, Any

import argparse
import yaml
import cv2
from ultralytics import YOLO

from auto_labeling.v_1.scripts.logger import log, log_json
try:
    from auto_labeling.v_1.storage.fetch import fetch_s3_to_cache
except Exception:
    fetch_s3_to_cache = None


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "teacher_model.yaml"
CLASS_YAML = ROOT / "configs" / "classes.yaml"


def load_teacher_cfg() -> dict:
    if not CFG_PATH.exists():
        raise FileNotFoundError(f"[TeacherRunner] config not found: {CFG_PATH}")

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "teacher" not in data:
        raise KeyError(f"[TeacherRunner] 'teacher' key not in {CFG_PATH}")

    return data["teacher"]


def load_class_names(yaml_path: Path) -> dict[int, str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"[TeacherRunner] classes.yaml not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names")
    if names is None:
        raise KeyError(f"[TeacherRunner] 'names' key not in {yaml_path}")

    fixed: dict[int, str] = {}
    for k, v in names.items():
        fixed[int(k)] = str(v)
    return fixed


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return ROOT / p


def _norm_name(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def _iter_images(dirp: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out: list[Path] = []
    for e in exts:
        out.extend(sorted(dirp.glob(f"*{e}")))
    return out


def _latest_round_root(data_root: Path) -> Optional[Path]:
    """data/round_r1, data/round_r2 ... 중 가장 큰 r을 찾아 반환"""
    if not data_root.exists():
        return None
    rounds = []
    for p in data_root.glob("round_r*"):
        if p.is_dir():
            try:
                r = int(p.name.replace("round_r", ""))
                rounds.append((r, p))
            except Exception:
                continue
    if not rounds:
        return None
    rounds.sort(key=lambda x: x[0])
    return rounds[-1][1]


def _parse_classes_field(v: Any) -> Optional[list[int]]:
    """
    teacher_model.yaml의 classes:
      - null -> None
      - [0,1,2] -> 그대로
      - ["0","1"] -> int 변환
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        out: list[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                raise ValueError(f"[TeacherRunner] classes must be list[int], got bad element: {x!r}")
        return out
    raise ValueError(f"[TeacherRunner] classes must be list[int] or null, got: {type(v)}")


# main
def run_teacher_on_fail(
    fail_img_dir: Path,
    out_label_dir: Path,
    teacher_weights: Optional[Path] = None,  # CLI override (로컬 경로)
    device: Optional[str] = None,
    max_fail_samples: Optional[int] = None,
    conf_th: Optional[float] = None,
) -> List[Path]:
    cfg = load_teacher_cfg()

    # ---------------------------------------
    # 1) weights resolve (MinIO/S3 uri -> cache) / device / params
    # ---------------------------------------
    if teacher_weights is None:
        weights_uri = str(cfg.get("weights_uri", "")).strip()

        if weights_uri:
            if fetch_s3_to_cache is None:
                # storage 모듈이 없는 환경: uri fetch 불가 → 로컬 weights로 fallback
                log(
                    "[TeacherRunner] weights_uri is set but storage.fetch is unavailable. "
                    "Falling back to local cfg['weights']."
                )
                teacher_weights = _resolve_path(
                    cfg.get("weights", "models/teacher/weights/yolov11x_teacher_v1_20251211.pt")
                )
            else:
                cache_dir = _resolve_path(cfg.get("weights_cache_dir", "models/teacher/weights_cache"))
                teacher_weights = fetch_s3_to_cache(weights_uri, cache_dir=cache_dir)
        else:
            teacher_weights = _resolve_path(
                cfg.get("weights", "models/teacher/weights/yolov11x_teacher_v1_20251211.pt"))
    else:
        teacher_weights = Path(teacher_weights)

    if device is None:
        device = str(cfg.get("device", "0"))
    device = device.strip()

    if max_fail_samples is None:
        max_fail_samples = int(cfg.get("max_fail_samples", 1000))

    # conf key 호환: conf_th 우선, 없으면 conf
    if conf_th is None:
        if "conf_th" in cfg:
            conf_th = float(cfg.get("conf_th", 0.3))
        else:
            conf_th = float(cfg.get("conf", 0.3))

    imgsz = int(cfg.get("imgsz", 1280))
    iou = float(cfg.get("iou", 0.7))
    max_det = int(cfg.get("max_det", 300))
    agnostic_nms = bool(cfg.get("agnostic_nms", False))
    classes = _parse_classes_field(cfg.get("classes", None))

    save_conf = bool(cfg.get("save_conf", False))
    enable_vis = bool(cfg.get("enable_vis", False))
    vis_dir = _resolve_path(cfg.get("vis_dir", "logs/teacher_vis"))
    if enable_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # gate
    min_boxes_for_success = int(cfg.get("min_boxes_for_success", 1))
    min_conf_for_success = float(cfg.get("min_conf_for_success", 0.0))  # 0이면 비활성
    min_box_area_norm = float(cfg.get("min_box_area_norm", 0.0))
    max_boxes_per_image = int(cfg.get("max_boxes_per_image", 0))

    # ---------------------------------------
    # 2) classes.yaml 로드 (keep_classes name 지원)
    # ---------------------------------------
    try:
        class_names = load_class_names(CLASS_YAML)
        log(f"[TeacherRunner] classes.yaml 로드 완료: {len(class_names)} classes")
        name2id = {_norm_name(v): k for k, v in class_names.items()}
    except Exception as e:
        class_names = None
        name2id = {}
        log(f"[TeacherRunner] WARNING: classes.yaml 로드 실패 → id만 사용 ({e})")

    # keep_classes 처리 (name or id)
    keep_cfg = cfg.get("keep_classes", None)
    allowed_ids: Optional[set[int]] = None

    if keep_cfg is not None:
        allowed_ids = set()
        if len(keep_cfg) > 0 and isinstance(keep_cfg[0], str):
            # 이름 기반
            if not name2id:
                log("[TeacherRunner] WARNING: keep_classes가 name 리스트인데 classes.yaml 없음 → 필터 무시")
                allowed_ids = None
            else:
                for name in keep_cfg:
                    nid = name2id.get(_norm_name(name))
                    if nid is None:
                        log(f"[TeacherRunner] WARNING: keep_classes name '{name}' not in classes.yaml")
                    else:
                        allowed_ids.add(int(nid))
        else:
            # id 기반
            for cid in keep_cfg:
                allowed_ids.add(int(cid))

    # ---------------------------------------
    # 3) logs
    # ---------------------------------------
    log(f"[TeacherRunner] fail_img_dir : {fail_img_dir}")
    log(f"[TeacherRunner] out_label_dir: {out_label_dir}")
    log(f"[TeacherRunner] weights     : {teacher_weights}")
    log(f"[TeacherRunner] device      : {device}")
    log(f"[TeacherRunner] imgsz       : {imgsz}")
    log(f"[TeacherRunner] conf_th     : {conf_th}")
    log(f"[TeacherRunner] iou         : {iou}")
    log(f"[TeacherRunner] max_det     : {max_det}")
    log(f"[TeacherRunner] agnostic_nms: {agnostic_nms}")
    log(f"[TeacherRunner] classes     : {classes}")
    log(f"[TeacherRunner] max_samples : {max_fail_samples}")
    log(f"[TeacherRunner] save_conf   : {save_conf}")
    log(f"[TeacherRunner] enable_vis  : {enable_vis} ({vis_dir})")
    log(f"[TeacherRunner] gate(min_boxes)        : {min_boxes_for_success}")
    log(f"[TeacherRunner] gate(min_conf)         : {min_conf_for_success}")
    log(f"[TeacherRunner] gate(min_area_norm)    : {min_box_area_norm}")
    log(f"[TeacherRunner] gate(max_boxes_image)  : {max_boxes_per_image}")

    if allowed_ids is not None:
        if class_names:
            keep_names = [class_names.get(i, f"id_{i}") for i in sorted(allowed_ids)]
            log(f"[TeacherRunner] keep_classes (ids): {sorted(allowed_ids)} → {keep_names}")
        else:
            log(f"[TeacherRunner] keep_classes (ids): {sorted(allowed_ids)}")

    if not Path(teacher_weights).exists():
        raise FileNotFoundError(f"[TeacherRunner] teacher weights not found: {teacher_weights}")

    out_label_dir.mkdir(parents=True, exist_ok=True)

    img_paths = _iter_images(Path(fail_img_dir))
    if not img_paths:
        log("[TeacherRunner] FAIL 이미지가 없습니다.")
        return []

    if max_fail_samples and len(img_paths) > max_fail_samples:
        img_paths = random.sample(img_paths, max_fail_samples)

    log(f"[TeacherRunner] 대상 FAIL 이미지 수: {len(img_paths)}")

    model = YOLO(str(teacher_weights))

    used: List[Path] = []

    zero_box_cnt = 0
    filtered_all_cnt = 0
    few_box_cnt = 0
    low_conf_cnt = 0
    small_area_cnt = 0
    too_many_box_cnt = 0
    success_cnt = 0

    for i, img_path in enumerate(img_paths):
        log(f"[TeacherRunner] [{i+1}/{len(img_paths)}] {img_path.name}")

        pred_kwargs = dict(
            source=str(img_path),
            save=False,
            verbose=False,
            conf=float(conf_th),
            iou=float(iou),
            imgsz=int(imgsz),
            max_det=int(max_det),
            agnostic_nms=bool(agnostic_nms),
            device=device,
        )
        if classes is not None:
            pred_kwargs["classes"] = classes

        results = model.predict(**pred_kwargs)[0]

        if enable_vis:
            try:
                vis = results.plot()  # BGR numpy
                out_vis = vis_dir / f"{img_path.stem}.jpg"
                cv2.imwrite(str(out_vis), vis)
            except Exception as e:
                log(f"[TeacherRunner] vis save failed: {img_path.name} ({e})")

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            zero_box_cnt += 1
            continue

        cls_list = boxes.cls.cpu().tolist()
        xywhn_list = boxes.xywhn.cpu().tolist()
        conf_list = boxes.conf.cpu().tolist()

        lines: list[str] = []
        kept_classes: list[int] = []
        kept_confs: list[float] = []

        for cls_id, xywhn, score in zip(cls_list, xywhn_list, conf_list):
            cid = int(cls_id)

            if allowed_ids is not None and cid not in allowed_ids:
                continue

            cx, cy, w, h = map(float, xywhn)

            if min_box_area_norm > 0.0 and (w * h) < min_box_area_norm:
                small_area_cnt += 1
                continue

            if save_conf:
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {float(score):.6f}")
            else:
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            kept_classes.append(cid)
            kept_confs.append(float(score))

        if not lines:
            filtered_all_cnt += 1
            continue

        if max_boxes_per_image > 0 and len(lines) > max_boxes_per_image:
            too_many_box_cnt += 1
            log_json({
                "event": "teacher_fail_too_many_boxes",
                "image": img_path.name,
                "boxes": len(lines),
                "max_boxes_per_image": max_boxes_per_image,
            })
            continue

        if len(lines) < min_boxes_for_success:
            few_box_cnt += 1
            log_json({
                "event": "teacher_fail_too_few_boxes",
                "image": img_path.name,
                "boxes": len(lines),
                "min_boxes_for_success": min_boxes_for_success,
            })
            continue

        if min_conf_for_success > 0.0:
            if min(kept_confs) < min_conf_for_success:
                low_conf_cnt += 1
                log_json({
                    "event": "teacher_fail_low_conf",
                    "image": img_path.name,
                    "min_conf": float(min(kept_confs)),
                    "min_conf_for_success": float(min_conf_for_success),
                })
                continue

        label_path = out_label_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        used.append(img_path)
        success_cnt += 1

        class_info = None
        if class_names:
            class_info = [class_names.get(c, f"id_{c}") for c in kept_classes]

        log_json({
            "event": "teacher_pseudo",
            "image": img_path.name,
            "boxes": len(lines),
            "conf_th": float(conf_th),
            "min_conf_in_image": float(min(kept_confs)) if kept_confs else None,
            "classes": kept_classes,
            "class_names": class_info,
            "save_conf": bool(save_conf),
            "imgsz": int(imgsz),
            "iou": float(iou),
            "max_det": int(max_det),
            "agnostic_nms": bool(agnostic_nms),
        })

    log(f"[TeacherRunner] pseudo-label SUCCESS 이미지 수: {success_cnt}")
    log(f"[TeacherRunner] zero_box(완전 실패) 이미지 수: {zero_box_cnt}")
    log(f"[TeacherRunner] filtered_all(필터 후 0박스) 이미지 수: {filtered_all_cnt}")
    log(f"[TeacherRunner] too_few_boxes(<{min_boxes_for_success}) 이미지 수: {few_box_cnt}")
    if min_conf_for_success > 0:
        log(f"[TeacherRunner] low_conf(<{min_conf_for_success}) 이미지 수: {low_conf_cnt}")
    if min_box_area_norm > 0:
        log(f"[TeacherRunner] small_area(<{min_box_area_norm}) skip 수: {small_area_cnt}")
    if max_boxes_per_image > 0:
        log(f"[TeacherRunner] too_many_boxes(>{max_boxes_per_image}) 이미지 수: {too_many_box_cnt}")
    log(f"[TeacherRunner] 최종 used 이미지 수: {len(used)}")

    log_json({
        "event": "teacher_summary",
        "total": len(img_paths),
        "success": success_cnt,
        "zero_box": zero_box_cnt,
        "filtered_all": filtered_all_cnt,
        "few_box": few_box_cnt,
        "low_conf": low_conf_cnt,
        "small_area_skips": small_area_cnt,
        "too_many_box": too_many_box_cnt,
        "conf_th": float(conf_th),
        "imgsz": int(imgsz),
        "iou": float(iou),
        "max_det": int(max_det),
        "agnostic_nms": bool(agnostic_nms),
        "classes": classes,
        "min_boxes_for_success": min_boxes_for_success,
        "min_conf_for_success": float(min_conf_for_success),
        "min_box_area_norm": float(min_box_area_norm),
        "max_boxes_per_image": max_boxes_per_image,
        "save_conf": bool(save_conf),
        "enable_vis": bool(enable_vis),
    })

    return used


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TeacherRunner (FAIL-only friendly)")
    ap.add_argument("--fail_dir", type=str, default="", help="fail images dir (default: latest round fail_fail/images)")
    ap.add_argument("--out_labels", type=str, default="", help="output labels dir (default: next to fail_dir)")
    ap.add_argument("--teacher", type=str, default="", help="teacher weights path override (local .pt)")
    ap.add_argument("--device", type=str, default="", help="device override (e.g., 0, cuda:0, cpu)")
    ap.add_argument("--max_samples", type=int, default=-1, help="max fail samples override")
    ap.add_argument("--conf", type=float, default=-1.0, help="conf_th override")

    args = ap.parse_args()

    data_root = ROOT / "data"
    latest_round = _latest_round_root(data_root)

    if args.fail_dir.strip():
        fail_dir = Path(args.fail_dir.strip())
    else:
        if latest_round is None:
            fail_dir = data_root / "fail" / "images"  # fallback
        else:
            fail_dir = latest_round / "fail_fail" / "images"

    if args.out_labels.strip():
        out_labels = Path(args.out_labels.strip())
    else:
        if "round_r" in str(fail_dir):
            out_labels = fail_dir.parents[2] / "teacher_pseudo" / "labels"
        else:
            out_labels = data_root / "teacher_pseudo" / "labels"

    teacher_w = Path(args.teacher.strip()) if args.teacher.strip() else None
    device = args.device.strip() if args.device.strip() else None
    max_samples = None if args.max_samples < 0 else int(args.max_samples)
    conf_th = None if args.conf < 0 else float(args.conf)

    run_teacher_on_fail(
        fail_img_dir=fail_dir,
        out_label_dir=out_labels,
        teacher_weights=teacher_w,
        device=device,
        max_fail_samples=max_samples,
        conf_th=conf_th,
    )
