from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional
import uuid
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from augmentation import AugConfig, build_train_augment, apply_augment

YoloBBox = Tuple[float, float, float, float]

app = FastAPI(title="Augmentation API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 개발용: 전부 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
AUG_COUNT = 4
INCLUDE_ORIGINAL = True
SAVE_VIZ = True
TAGS = ["flip", "scale", "blur", "color"]

class BBoxItem(BaseModel):
    class_id: int = Field(ge=0)
    bbox: Tuple[float, float, float, float]  # (cx, cy, w, h), normalized yolo

class AugmentMeta(BaseModel):
    image: Optional[str] = None
    bboxes: List[BBoxItem]
    bbox_format: Literal["yolo"] = "yolo"

def validate_yolo_norm(bboxes: List[YoloBBox]) -> None:
    for (xc, yc, w, h) in bboxes:
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            raise ValueError(f"bbox not normalized in [0,1]: {(xc, yc, w, h)}")
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"bbox w/h must be > 0: {(xc, yc, w, h)}")
# -----------------------
# utils
# -----------------------
def parse_yolo_txt_bytes(txt_bytes: bytes) -> tuple[List[int], List[YoloBBox]]:
    class_labels: List[int] = []
    bboxes: List[YoloBBox] = []

    txt = txt_bytes.decode("utf-8", errors="ignore").strip()
    if not txt:
        return class_labels, bboxes

    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        class_labels.append(cls)
        bboxes.append((xc, yc, w, h))
    return class_labels, bboxes


def yolo_txt_bytes(class_labels: List[int], bboxes: List[YoloBBox]) -> bytes:
    lines = []
    for cls, (xc, yc, w, h) in zip(class_labels, bboxes):
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return (("\n".join(lines) + ("\n" if lines else ""))).encode("utf-8")


def draw_boxes(image_bgr: np.ndarray, class_labels: List[int], bboxes: List[YoloBBox]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    for cls, (xc, yc, bw, bh) in zip(class_labels, bboxes):
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, str(cls), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def imdecode_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def imencode_like_ext(image_bgr: np.ndarray, ext: str) -> bytes:
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ValueError(f"Failed to encode image as {ext}")
    return buf.tobytes()


def guess_ext(filename: str | None) -> str:
    name = Path(filename or "image.jpg")
    ext = name.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    if ext == ".jpeg":
        ext = ".jpg"
    return ext


def ext_to_mime(ext: str) -> str:
    e = ext.lower()
    if e == ".png":
        return "image/png"
    if e == ".webp":
        return "image/webp"
    return "image/jpeg"


def make_multipart_mixed(parts: List[tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = f"----aug-{uuid.uuid4().hex}"
    chunks: List[bytes] = []

    for name, ctype, payload in parts:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n'.encode())
        chunks.append(f"Content-Type: {ctype}\r\n\r\n".encode())
        chunks.append(payload)
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    return body, f"multipart/mixed; boundary={boundary}"


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/augment")
async def augment(
    image: UploadFile = File(..., description="Input image file (jpg/png/webp)"),
    meta: UploadFile = File(..., description="JSON metadata (bboxes)"),
):
    try:
        img_bytes = await image.read()
        meta_bytes = await meta.read()

        img = imdecode_upload(img_bytes)

        req = AugmentMeta.model_validate_json(meta_bytes)
        if req.bbox_format != "yolo":
            raise ValueError("only bbox_format='yolo' is supported")

        class_labels: List[int] = [it.class_id for it in req.bboxes]
        bboxes: List[YoloBBox] = [tuple(it.bbox) for it in req.bboxes]
        validate_yolo_norm(bboxes)

        in_name = Path(image.filename or (req.image or "image.jpg"))
        stem = in_name.stem
        ext = guess_ext(image.filename or req.image)
        mime = ext_to_mime(ext)

        cfg = AugConfig()
        aug = build_train_augment(cfg)

        parts: List[tuple[str, str, bytes]] = []
        total = (1 if INCLUDE_ORIGINAL else 0) + AUG_COUNT

        for idx in range(total):
            if INCLUDE_ORIGINAL and idx == 0:
                out_img = img
                out_cls = class_labels
                out_bb = bboxes
                tag = "orig"
            else:
                aug_id = idx - (1 if INCLUDE_ORIGINAL else 0)
                out_img, out_bb, out_cls = apply_augment(
                    aug, img, bboxes, class_labels, cfg=cfg, aug_id=aug_id
                )
                tag = TAGS[aug_id % 4]

            img_name = f"{stem}_aug{idx:02d}_{tag}{ext}"
            lbl_name = f"{stem}_aug{idx:02d}_{tag}.txt"

            parts.append((f"image_{idx:02d}", mime, imencode_like_ext(out_img, ext)))
            parts.append((f"label_{idx:02d}", "text/plain; charset=utf-8",
                          yolo_txt_bytes(list(out_cls), list(out_bb))))

            if SAVE_VIZ:
                viz = draw_boxes(out_img, list(out_cls), list(out_bb))
                parts.append((f"viz_{idx:02d}", mime, imencode_like_ext(viz, ext)))

            parts.append((f"image_name_{idx:02d}", "text/plain; charset=utf-8", img_name.encode("utf-8")))
            parts.append((f"label_name_{idx:02d}", "text/plain; charset=utf-8", lbl_name.encode("utf-8")))

        body, content_type = make_multipart_mixed(parts)

        return Response(
            content=body,
            media_type=content_type,
            headers={
                "X-Aug-Total": str(total),
                "X-Aug-Has-Viz": "1" if SAVE_VIZ else "0",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"augment failed: {e}")

