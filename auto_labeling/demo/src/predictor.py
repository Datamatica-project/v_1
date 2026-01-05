# demo/src/predictor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


@dataclass
class PredictConfig:
    weights_path: Optional[str] = None
    device: str = "0"
    imgsz: int = 640
    conf_th: float = 0.25
    iou_th: float = 0.7


class Predictor:
    def __init__(self, *, cfg: PredictConfig):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. `pip install ultralytics`")

        self.cfg = cfg
        if not cfg.weights_path:
            raise RuntimeError("weightsPath is required for demo unless you set a default in Predictor")

        self.model = YOLO(str(cfg.weights_path))

    def predict_one(self, img_path: Path, *, save_preview_dir: Optional[Path] = None) -> Dict[str, Any]:
        res = self.model.predict(
            source=str(img_path),
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf_th),
            iou=float(self.cfg.iou_th),
            device=str(self.cfg.device),
            verbose=False,
        )

        r0 = res[0]
        boxes = []
        if getattr(r0, "boxes", None) is not None and r0.boxes is not None:
            # xyxy + conf + cls
            xyxy = r0.boxes.xyxy.cpu().numpy()
            conf = r0.boxes.conf.cpu().numpy()
            cls = r0.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                boxes.append(
                    {
                        "classId": int(cls[i]),
                        "conf": float(conf[i]),
                        "xyxy": [x1, y1, x2, y2],
                    }
                )

        preview_rel = None
        if save_preview_dir is not None:
            save_preview_dir.mkdir(parents=True, exist_ok=True)
            im = r0.plot()
            out_path = save_preview_dir / (img_path.stem + "_pred.jpg")
            try:
                import cv2  # type: ignore

                cv2.imwrite(str(out_path), im)
            except Exception:
                from PIL import Image

                Image.fromarray(im[..., ::-1]).save(out_path)  # BGR->RGB
            preview_rel = str(out_path)

        return {
            "imagePath": str(img_path),
            "boxes": boxes,
            "previewPath": preview_rel,
        }
