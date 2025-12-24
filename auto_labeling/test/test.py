# test.py
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2

from yolov11x import YoloV11xModel, PredictConfig


def iter_images(img_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out: List[Path] = []
    for e in exts:
        out.extend(sorted(img_dir.glob(f"*{e}")))
    return out


def link_or_copy(src: Path, dst: Path, mode: str = "copy") -> None:
    mode = (mode or "copy").lower().strip()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(str(src), str(dst))
    elif mode == "hardlink":
        os.link(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def decide_pass(
    confs: List[float],
    *,
    conf_threshold: float,
    allow_empty_boxes_pass: bool,
) -> Tuple[bool, Dict[str, Any]]:
    if not confs:
        return bool(allow_empty_boxes_pass), {"n_boxes": 0, "mode": "empty"}

    m = float(min(confs))
    is_pass = m >= float(conf_threshold)
    return is_pass, {"n_boxes": len(confs), "mode": "min", "min_conf": m}


def split_pass_fail(
    *,
    weights: Path,
    src_img_dir: Path,
    out_dir: Path,
    conf_threshold: float = 0.7,
    allow_empty_boxes_pass: bool = True,
    device: str = "0",
    imgsz: int = 1280,
    copy_mode: str = "copy",
    export_pass_labels: bool = False,
    export_label_conf: float = 0.3,
    save_pass_viz: bool = True,
    save_fail_images: bool = False,
    save_fail_list: bool = True,
) -> Dict[str, Any]:
    src_img_dir = Path(src_img_dir)
    out_dir = Path(out_dir)

    pass_img_dir = out_dir / "pass" / "images"
    pass_lbl_dir = out_dir / "pass" / "labels"
    pass_viz_dir = out_dir / "pass" / "viz"
    fail_img_dir = out_dir / "fail" / "images"

    pass_img_dir.mkdir(parents=True, exist_ok=True)
    if export_pass_labels:
        pass_lbl_dir.mkdir(parents=True, exist_ok=True)
    if save_pass_viz:
        pass_viz_dir.mkdir(parents=True, exist_ok=True)
    if save_fail_images:
        fail_img_dir.mkdir(parents=True, exist_ok=True)

    model = YoloV11xModel(
        weights=weights,
        cfg=PredictConfig(device=device, imgsz=imgsz),
    )

    imgs = iter_images(src_img_dir)
    n_total = len(imgs)
    n_pass = 0
    n_fail = 0
    n_empty = 0

    debug_items: List[Dict[str, Any]] = []
    fail_list: List[str] = []

    for img_path in imgs:
        confs = model.predict_confs(img_path)

        if not confs:
            n_empty += 1

        is_pass, dbg = decide_pass(
            confs,
            conf_threshold=conf_threshold,
            allow_empty_boxes_pass=allow_empty_boxes_pass,
        )

        if is_pass:
            n_pass += 1

            link_or_copy(img_path, pass_img_dir / img_path.name, mode=copy_mode)

            if export_pass_labels:
                lines = model.predict_yolo_labels(img_path, conf_th=float(export_label_conf))
                (pass_lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            if save_pass_viz:
                viz = model.predict_viz_pass(img_path, conf_th=float(conf_threshold))
                cv2.imwrite(str(pass_viz_dir / img_path.name), viz)

        else:
            n_fail += 1
            fail_list.append(img_path.name)
            if save_fail_images:
                link_or_copy(img_path, fail_img_dir / img_path.name, mode=copy_mode)

        debug_items.append(
            {
                "image": img_path.name,
                "isPass": bool(is_pass),
                "confThreshold": float(conf_threshold),
                "allowEmptyBoxesPass": bool(allow_empty_boxes_pass),
                **dbg,
            }
        )

    summary = {
        "srcImgDir": str(src_img_dir),
        "outDir": str(out_dir),
        "weights": str(weights),
        "confThreshold": float(conf_threshold),
        "allowEmptyBoxesPass": bool(allow_empty_boxes_pass),
        "total": int(n_total),
        "pass": int(n_pass),
        "fail": int(n_fail),
        "emptyBoxes": int(n_empty),
        "exportPassLabels": bool(export_pass_labels),
        "exportLabelConf": float(export_label_conf),
        "savePassViz": bool(save_pass_viz),
        "saveFailImages": bool(save_fail_images),
        "saveFailList": bool(save_fail_list),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "items.json").write_text(json.dumps(debug_items, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_fail_list:
        (out_dir / "fail_list.json").write_text(
            json.dumps({"fail": fail_list}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def main():
    ap = argparse.ArgumentParser(description="Demo PASS splitter (FAIL no-save by default)")
    ap.add_argument("--weights", type=str, required=True, help="trained .pt")
    ap.add_argument("--src", type=str, required=True, help="images dir")
    ap.add_argument("--out", type=str, required=True, help="output dir")
    ap.add_argument("--conf", type=float, default=0.7, help="pass threshold (min conf)")
    ap.add_argument("--allow_empty", action="store_true", help="if no boxes => PASS")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--copy_mode", type=str, default="copy", choices=["copy", "symlink", "hardlink"])

    ap.add_argument("--export_pass_labels", action="store_true", help="write YOLO labels for PASS images")
    ap.add_argument("--export_label_conf", type=float, default=0.3, help="conf for exported labels")

    ap.add_argument("--save_pass_viz", action="store_true", help="save PASS viz images")
    ap.add_argument("--save_fail_images", action="store_true", help="(optional) also save FAIL images")
    ap.add_argument("--no_fail_list", action="store_true", help="do NOT write fail_list.json")

    args = ap.parse_args()

    summary = split_pass_fail(
        weights=Path(args.weights),
        src_img_dir=Path(args.src),
        out_dir=Path(args.out),
        conf_threshold=float(args.conf),
        allow_empty_boxes_pass=bool(args.allow_empty),
        device=str(args.device),
        imgsz=int(args.imgsz),
        copy_mode=str(args.copy_mode),
        export_pass_labels=bool(args.export_pass_labels),
        export_label_conf=float(args.export_label_conf),
        save_pass_viz=bool(args.save_pass_viz),
        save_fail_images=bool(args.save_fail_images),
        save_fail_list=(not bool(args.no_fail_list)),
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
