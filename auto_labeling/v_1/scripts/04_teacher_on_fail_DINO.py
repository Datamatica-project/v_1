from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import yaml
import torch
# GroundingDINO 관련 import (설치 필요)
from groundingdino.util.inference import load_model, predict


@dataclass
class TeacherConfig:
    model_config: Path
    model_weights: Path
    device: str
    box_threshold: float
    text_threshold: float
    class_prompt: str
    class_map: Dict[str, int]
    output_label_dir: Path
    save_debug_vis: bool = False
    debug_vis_dir: Path | None = None


def load_teacher_config(yaml_path: Path) -> TeacherConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["teacher"]

    return TeacherConfig(
        model_config=Path(cfg["model_config"]),
        model_weights=Path(cfg["model_weights"]),
        device=cfg.get("device", "cuda:0"),
        box_threshold=float(cfg.get("box_threshold", 0.1)),
        text_threshold=float(cfg.get("text_threshold", 0.1)),
        class_prompt=cfg["class_prompt"],
        class_map=cfg["class_map"],
        output_label_dir=Path(cfg["output_label_dir"]),
        save_debug_vis=bool(cfg.get("save_debug_vis", False)),
        debug_vis_dir=Path(cfg["debug_vis_dir"]) if cfg.get("debug_vis_dir") else None,
    )


class TeacherRunner:
    def __init__(self, config: TeacherConfig):
        self.cfg = config

        print(f"[TeacherRunner] Loading GroundingDINO from:")
        print(f"  cfg    = {self.cfg.model_config}")
        print(f"  weight = {self.cfg.model_weights}")

        self.model = load_model(
            str(self.cfg.model_config),
            str(self.cfg.model_weights),
            device=self.cfg.device,
        )
        self.cfg.output_label_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_debug_vis and self.cfg.debug_vis_dir is not None:
            self.cfg.debug_vis_dir.mkdir(parents=True, exist_ok=True)

    # phrase → class_id 매핑
    def _phrase_to_class_id(self, phrase: str) -> int | None:
        key = phrase.lower().strip()
        return self.cfg.class_map.get(key, None)

    def infer_image(self, img_path: Path) -> List[Tuple[int, float, float, float, float]]:
        """
        GroundingDINO 추론 후 YOLO 포맷 (cx, cy, w, h, 0~1) 으로 변환.

        return: list of (class_id, x_center, y_center, w, h)  [모두 0~1]
        """
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            return []

        img_h, img_w = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        boxes, logits, phrases = predict(
            model=self.model,
            image=img_tensor,
            caption=self.cfg.class_prompt,
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
            device=self.cfg.device,
        )

        results: List[Tuple[int, float, float, float, float]] = []

        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            cls_id = self._phrase_to_class_id(phrase)
            if cls_id is None:
                print(f"[DEBUG] Skip phrase not in class_map: {phrase}")
                continue

            # GroundingDINO boxes 형식: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.tolist()

            # 혹시 픽셀 단위로 오는 버전을 대비한 방어 코드
            max_val = max(x1, y1, x2, y2)
            if max_val > 1.5:  # 0~1 범위를 확실히 벗어나면 픽셀이라고 가정
                x1 /= float(img_w)
                x2 /= float(img_w)
                y1 /= float(img_h)
                y2 /= float(img_h)

            # 이제 x1~x2, y1~y2 는 0~1 범위의 xyxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = (x2 - x1)
            bh = (y2 - y1)

            # clamp to [0,1]
            cx = float(max(0.0, min(1.0, cx)))
            cy = float(max(0.0, min(1.0, cy)))
            bw = float(max(0.0, min(1.0, bw)))
            bh = float(max(0.0, min(1.0, bh)))

            if bw <= 0.0 or bh <= 0.0:
                print(f"[DEBUG] Skip zero-size box at index {i}: {box.tolist()}")
                continue

            results.append((cls_id, cx, cy, bw, bh))

        print(f"[TeacherRunner] {img_path.name}: kept {len(results)} boxes")
        return results

    def save_yolo_label(self, img_path: Path, preds: List[Tuple[int, float, float, float, float]]):
        """
        preds: list of (class_id, cx, cy, w, h)  모두 0~1 기준
        """
        if not preds:
            print(f"[TeacherRunner] No boxes for {img_path.name}, skip txt")
            return

        txt_path = self.cfg.output_label_dir / f"{img_path.stem}.txt"
        lines = []
        for cls_id, cx, cy, bw, bh in preds:
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("".join(lines), encoding="utf-8")
        print(f"[TeacherRunner] Saved YOLO label: {txt_path}")
