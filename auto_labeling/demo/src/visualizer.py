# demo/src/visualizer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _try_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _try_import_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        return Image, ImageDraw, ImageFont
    except Exception:
        return None, None, None


@dataclass
class DrawConfig:
    show_label: bool = True
    show_conf: bool = True
    line_thickness: int = 2
    font_scale: float = 0.5  # cv2 기준
    font_thickness: int = 1


def _label_text(box: Dict[str, Any], class_names: Optional[Dict[int, str]], cfg: DrawConfig) -> str:
    cid = int(box.get("classId", -1))
    conf = box.get("conf", None)
    name = class_names.get(cid, str(cid)) if class_names else str(cid)

    if cfg.show_conf and conf is not None:
        return f"{name} {float(conf):.2f}"
    return f"{name}"


def draw_xyxy_boxes_on_image(
    img_path: Path,
    boxes: List[Dict[str, Any]],
    *,
    class_names: Optional[Dict[int, str]] = None,
    cfg: Optional[DrawConfig] = None,
) -> "object":
    """
    boxes: [{classId:int, conf:float, xyxy:[x1,y1,x2,y2]}]
    return: numpy array(BGR) if cv2 available else PIL Image
    """
    cfg = cfg or DrawConfig()
    cv2 = _try_import_cv2()

    if cv2 is not None:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"failed to read image: {img_path}")

        h, w = img.shape[:2]
        for b in boxes:
            x1, y1, x2, y2 = b.get("xyxy", [0, 0, 0, 0])
            x1 = int(_clamp(float(x1), 0, w - 1))
            x2 = int(_clamp(float(x2), 0, w - 1))
            y1 = int(_clamp(float(y1), 0, h - 1))
            y2 = int(_clamp(float(y2), 0, h - 1))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), int(cfg.line_thickness))

            if cfg.show_label:
                text = _label_text(b, class_names, cfg)
                cv2.putText(
                    img,
                    text,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    float(cfg.font_scale),
                    (0, 0, 0),
                    int(cfg.font_thickness) + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    text,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    float(cfg.font_scale),
                    (255, 255, 255),
                    int(cfg.font_thickness),
                    cv2.LINE_AA,
                )
        return img

    # fallback PIL
    Image, ImageDraw, ImageFont = _try_import_pil()
    if Image is None:
        raise RuntimeError("Neither cv2 nor PIL is available for visualization")

    im = Image.open(str(img_path)).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size

    for b in boxes:
        x1, y1, x2, y2 = b.get("xyxy", [0, 0, 0, 0])
        x1 = int(_clamp(float(x1), 0, w - 1))
        x2 = int(_clamp(float(x2), 0, w - 1))
        y1 = int(_clamp(float(y1), 0, h - 1))
        y2 = int(_clamp(float(y2), 0, h - 1))

        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=int(cfg.line_thickness))
        if cfg.show_label:
            text = _label_text(b, class_names, cfg)
            draw.text((x1, max(0, y1 - 12)), text, fill=(255, 255, 255))

    return im


def save_preview(
    img_path: Path,
    boxes: List[Dict[str, Any]],
    out_path: Path,
    *,
    class_names: Optional[Dict[int, str]] = None,
    cfg: Optional[DrawConfig] = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = draw_xyxy_boxes_on_image(img_path, boxes, class_names=class_names, cfg=cfg)

    cv2 = _try_import_cv2()
    if cv2 is not None and hasattr(rendered, "shape"):
        cv2.imwrite(str(out_path), rendered)
        return out_path

    # PIL Image
    rendered.save(str(out_path))
    return out_path


def build_gallery_html(
    previews_dir: Path,
    out_html: Path,
    *,
    title: str = "Demo Predictions Gallery",
    max_items: int = 500,
) -> Path:
    """
    previews_dir의 이미지들을 간단한 html로 보여줌.
    브라우저로 열면 QA/데모에 매우 좋음.
    """
    previews = []
    if previews_dir.exists():
        for p in previews_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                previews.append(p)
    previews.sort(key=lambda p: p.name)
    previews = previews[: int(max_items)]

    out_html.parent.mkdir(parents=True, exist_ok=True)

    # 상대 경로로 참조 (html과 previews가 같은 run_dir 아래라고 가정)
    lines = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'/>")
    lines.append(f"<title>{title}</title>")
    lines.append(
        "<style>"
        "body{font-family:Arial,sans-serif;margin:20px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;}"
        ".card{border:1px solid #ddd;border-radius:10px;padding:8px;}"
        "img{width:100%;height:auto;border-radius:8px;}"
        ".name{font-size:12px;color:#555;word-break:break-all;margin-top:6px;}"
        "</style></head><body>"
    )
    lines.append(f"<h2>{title}</h2>")
    lines.append(f"<p>count: {len(previews)}</p>")
    lines.append("<div class='grid'>")
    for p in previews:
        rel = p.relative_to(out_html.parent).as_posix()
        lines.append("<div class='card'>")
        lines.append(f"<img src='{rel}'/>")
        lines.append(f"<div class='name'>{p.name}</div>")
        lines.append("</div>")
    lines.append("</div></body></html>")

    out_html.write_text("\n".join(lines), encoding="utf-8")
    return out_html
