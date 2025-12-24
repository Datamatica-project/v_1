from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

ImagePath = Union[str, Path]


@dataclass
class PredictConfig:
    device: str = "0"
    imgsz: int = 1280
    nms_conf_floor: float = 0.01
    iou: float = 0.7
    max_det: int = 300


class YoloV11xModel:
    def __init__(self, weights: ImagePath, cfg: Optional[PredictConfig] = None):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"weights not found: {self.weights}")
        self.cfg = cfg or PredictConfig()
        self.model = YOLO(str(self.weights))

    def predict_confs(self, image: ImagePath) -> List[float]:
        r = self.model.predict(
            source=str(image),
            conf=float(self.cfg.nms_conf_floor),
            iou=float(self.cfg.iou),
            imgsz=int(self.cfg.imgsz),
            max_det=int(self.cfg.max_det),
            device=str(self.cfg.device),
            verbose=False,
            save=False,
        )
        if not r:
            return []
        boxes = r[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        return [float(x) for x in boxes.conf.detach().cpu().tolist()]

    def predict_yolo_labels(
        self,
        image: ImagePath,
        *,
        conf_th: float,
    ) -> List[str]:
        r = self.model.predict(
            source=str(image),
            conf=float(conf_th),
            iou=float(self.cfg.iou),
            imgsz=int(self.cfg.imgsz),
            max_det=int(self.cfg.max_det),
            device=str(self.cfg.device),
            verbose=False,
            save=False,
        )
        if not r:
            return []

        boxes = r[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        cls_list = boxes.cls.detach().cpu().tolist()
        xywhn_list = boxes.xywhn.detach().cpu().tolist()

        lines: List[str] = []
        for cid, xywhn in zip(cls_list, xywhn_list):
            cx, cy, w, h = map(float, xywhn)
            lines.append(f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        return lines

    def predict_viz_pass(
        self,
        image: ImagePath,
        *,
        conf_th: float,
        draw_conf: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_badge: bool = True,
    ) -> np.ndarray:
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"image not found: {image}")

        r = self.model.predict(
            source=str(image),
            conf=float(self.cfg.nms_conf_floor),
            iou=float(self.cfg.iou),
            imgsz=int(self.cfg.imgsz),
            max_det=int(self.cfg.max_det),
            device=str(self.cfg.device),
            verbose=False,
            save=False,
        )
        if not r:
            return img

        boxes = r[0].boxes
        if boxes is None or len(boxes) == 0:
            return img

        h, w = img.shape[:2]

        xywhn = boxes.xywhn.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy().astype(int)

        kept = 0
        for (cx, cy, bw, bh), conf, cid in zip(xywhn, confs, clss):
            if float(conf) < float(conf_th):
                continue

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            if draw_conf:
                label = f"{cid}:{float(conf):.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            kept += 1

        if draw_badge:
            badge = f"PASS (th={float(conf_th):.2f}, boxes={kept})"
            cv2.putText(
                img,
                badge,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

        return img
