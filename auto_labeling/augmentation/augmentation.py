# augmentation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import albumentations as A

YoloBBox = Tuple[float, float, float, float]


@dataclass
class AugConfig:

    p_fliplr: float = 1.0

    scale_limit: float = 0.2

    blur_limit: Tuple[int, int] = (3, 7)
    sigma_limit: Tuple[float, float] = (0.6, 1.6)

    brightness: float = 0.45
    contrast: float = 0.45
    saturation: float = 0.45
    hue: float = 0.1

    min_visibility: float = 0.0
    clip_bboxes: bool = True


def _bbox_params(cfg: AugConfig) -> A.BboxParams:
    return A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=float(cfg.min_visibility),
        clip=bool(cfg.clip_bboxes),
    )


def _t_flip(cfg: AugConfig) -> A.BasicTransform:
    return A.HorizontalFlip(p=cfg.p_fliplr)


def _t_scale(cfg: AugConfig) -> A.BasicTransform:
    lo = 1.0 - cfg.scale_limit
    hi = 1.0 + cfg.scale_limit
    return A.Affine(scale=(lo, hi), fit_output=True, p=1.0)


def _t_blur(cfg: AugConfig) -> A.BasicTransform:
    return A.GaussianBlur(
        blur_limit=cfg.blur_limit,
        sigma_limit=cfg.sigma_limit,
        p=1.0,
    )


def _t_color(cfg: AugConfig) -> A.BasicTransform:
    return A.ColorJitter(
        brightness=cfg.brightness,
        contrast=cfg.contrast,
        saturation=cfg.saturation,
        hue=cfg.hue,
        p=1.0,
    )


def build_train_augment(cfg: AugConfig) -> None:
    return None


def apply_augment(
    _aug_unused,
    image_bgr,
    bboxes: List[YoloBBox],
    class_labels: List[int],
    *,
    cfg: AugConfig,
    aug_id: int,
):
    transforms = [
        _t_flip(cfg),
        _t_scale(cfg),
        _t_blur(cfg),
        _t_color(cfg),
    ]
    k = int(aug_id) % 4

    pipe = A.Compose(
        [transforms[k]],
        bbox_params=_bbox_params(cfg),
    )

    out = pipe(image=image_bgr, bboxes=bboxes, class_labels=class_labels)
    return out["image"], out["bboxes"], out["class_labels"]
