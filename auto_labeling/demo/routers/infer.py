# demo/routers/infer.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
import json
import uuid
import random

import yaml
import cv2
from ultralytics import YOLO

from auto_labeling.demo.routers.ingest import DATA

router = APIRouter(prefix="/api/demo", tags=["demo-infer"])

PROJECT_ROOT = DATA.parent
MODEL_YAML = PROJECT_ROOT / "demo" / "configs" / "model.yaml"


def _now_iso_no_ms() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _load_model_yaml() -> dict:
    if not MODEL_YAML.exists():
        raise HTTPException(500, f"model.yaml not found: {MODEL_YAML}")
    try:
        return yaml.safe_load(MODEL_YAML.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise HTTPException(500, f"failed to load model.yaml: {e}")


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (PROJECT_ROOT / pp).resolve()


def _get_default_output_base(cfg: dict) -> Path:
    out = cfg.get("output") or {}
    base = out.get("base_dir") or out.get("baseDir") or "demo/data/demo_runs"
    return _resolve_path(str(base))


def _iter_images(img_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out: List[Path] = []
    for e in exts:
        out.extend(sorted(img_dir.glob(f"*{e}")))
    return out


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# Swagger "string" 등 더미값을 무시하기 위한 유틸
_MEANINGLESS_STR = {"", "string", "String", "STRING", "null", "None"}


def _norm_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s in _MEANINGLESS_STR:
        return None
    return s


def _merge_defaults(req: "DemoRunRequest", cfg: dict) -> "DemoRunRequest":
    """
    요청이 비어있거나 의미없는 값이면 model.yaml 기본값으로 채움
    """
    weights_cfg = cfg.get("weights") or {}
    device_cfg = cfg.get("device") or {}
    predict_cfg = cfg.get("predict") or {}
    pf_cfg = cfg.get("pass_fail") or {}

    if req.batch_size is None:
        req.batch_size = int(predict_cfg.get("batch", 32))
    if _norm_str(req.result_abs_path) is None:
        req.result_abs_path = str(_get_default_output_base(cfg))
    if _norm_str(req.image_dir) is None:
        req.image_dir = str(DATA / "unlabeled" / "images")

    # weights: 없으면 weights.path 사용
    if _norm_str(req.weights) is None:
        req.weights = str(weights_cfg.get("path") or "")

    # device: 없으면 device.default
    if _norm_str(req.device) is None:
        req.device = str(device_cfg.get("default") or "0")

    # predict
    if req.imgsz is None:
        req.imgsz = int(predict_cfg.get("imgsz", 640))
    if req.iou is None:
        req.iou = float(predict_cfg.get("iou", 0.7))

    # pass/fail
    if req.conf_pass_threshold is None:
        req.conf_pass_threshold = float(pf_cfg.get("pass_conf", 0.7))
    if req.allow_empty_boxes_pass is None:
        req.allow_empty_boxes_pass = bool(pf_cfg.get("empty_is_pass", True))

    # policy 기본값 보정
    if not isinstance(req.pass_fail_policy, dict) or not req.pass_fail_policy:
        req.pass_fail_policy = {"mode": "min"}
    else:
        # Swagger 기본 additionalProp1 같은 케이스 방어
        mode = str(req.pass_fail_policy.get("mode", "min")).strip().lower()
        if mode in _MEANINGLESS_STR:
            req.pass_fail_policy["mode"] = "min"

    return req


PolicyMode = Literal["min", "ratio", "drop_lowest"]


def _decide_pass(
    confs: List[float],
    *,
    conf_pass_threshold: float,
    policy: Dict[str, Any],
    allow_empty_boxes_pass: bool,
) -> tuple[bool, Dict[str, Any]]:
    if not confs:
        return bool(allow_empty_boxes_pass), {"mode": "empty", "n_boxes": 0}

    th = float(conf_pass_threshold)
    mode = str(policy.get("mode", "min")).strip().lower()

    if mode == "min":
        m = float(min(confs))
        return (m >= th), {"mode": "min", "min_conf": m, "n_boxes": len(confs)}

    if mode == "ratio":
        max_low = float(policy.get("max_low_conf_ratio", 0.25))
        max_low = max(0.0, min(1.0, max_low))
        low = sum(1 for c in confs if float(c) < th)
        low_ratio = low / max(1, len(confs))
        return (low_ratio <= max_low), {
            "mode": "ratio",
            "n_boxes": len(confs),
            "low": int(low),
            "low_ratio": float(low_ratio),
            "max_low_conf_ratio": float(max_low),
            "min_conf": float(min(confs)),
        }

    if mode == "drop_lowest":
        drop_r = float(policy.get("drop_lowest_ratio", 0.2))
        drop_r = max(0.0, min(0.99, drop_r))
        n = len(confs)
        k = int(n * drop_r)
        k = max(0, min(n - 1, k))
        s = sorted(float(c) for c in confs)
        kept = s[k:]
        return (float(min(kept)) >= th), {
            "mode": "drop_lowest",
            "n_boxes": n,
            "drop_lowest_ratio": float(drop_r),
            "dropped_k": int(k),
            "min_conf_kept": float(min(kept)),
            "min_conf_all": float(min(confs)),
        }

    m = float(min(confs))
    return (m >= th), {"mode": "min(fallback)", "min_conf": m, "n_boxes": len(confs)}

# schemas (external = camelCase)
class DemoRunRequest(BaseModel):
    #  Optional로 변경 (빈 바디 허용 핵심)
    result_abs_path: Optional[str] = Field(
        default=None,
        alias="resultAbsPath",
        description="결과 저장 base 디렉토리(절대경로). 비어있으면 model.yaml output.base_dir 사용.",
    )

    image_dir: Optional[str] = Field(
        default=None,
        alias="imageDir",
        description="추론 대상 이미지 디렉토리. 비어있으면 DATA/unlabeled/images 사용.",
    )

    weights: Optional[str] = Field(default=None, description="YOLO weights 경로. 없으면 model.yaml weights.path 사용.")
    device: Optional[str] = Field(default=None, description="device (예: 0, cuda:0, cpu)")
    batch_size: Optional[int] = Field(
        default=None,
        alias="batchSize",
        ge=1,
        le=512,
        description="YOLO predict batch size. None이면 model.yaml predict.batch 또는 기본값 사용.",
    )
    imgsz: Optional[int] = Field(default=None, description="inference imgsz")
    iou: Optional[float] = Field(default=None, description="NMS IoU")

    max_det: int = Field(default=300, alias="maxDet", description="max detections per image")
    nms_conf_floor: float = Field(default=0.05, alias="nmsConfFloor", description="predict conf floor")

    conf_pass_threshold: Optional[float] = Field(default=None, alias="confPassThreshold", description="PASS 판정 threshold")
    allow_empty_boxes_pass: Optional[bool] = Field(default=None, alias="allowEmptyBoxesPass", description="0 box일 때 PASS 여부")

    pass_fail_policy: Dict[str, Any] = Field(
        default_factory=dict,
        alias="passFailPolicy",
        description="PASS/FAIL 정책: min | ratio | drop_lowest",
    )

    save_result_images: bool = Field(default=True, alias="saveResultImages", description="박스 시각화 이미지 저장 여부")
    save_fail_images: bool = Field(default=True, alias="saveFailImages", description="FAIL도 시각화 저장할지")
    max_save_images: int = Field(default=-1, alias="maxSaveImages", description="시각화 이미지 저장 최대 개수(-1=무제한)")
    sample_seed: int = Field(default=0, alias="sampleSeed", description="샘플링 시드")

    run_id: Optional[str] = Field(default=None, alias="runId", description="run ID(옵션). 없으면 자동 생성")

    class Config:
        populate_by_name = True


class DemoRunResponse(BaseModel):
    run_id: str = Field(..., alias="runId")
    run_path: str = Field(..., alias="runPath")
    status: str
    created_at: str = Field(..., alias="createdAt")

    class Config:
        populate_by_name = True


def _gen_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:8]
    return f"run_{ts}_{suf}"


@router.post(
    "/run",
    response_model=DemoRunResponse,
    summary="Demo inference 실행 (PASS/FAIL + 결과 이미지 저장 + preds.jsonl 생성)",
)
def run_demo(body: Optional[DemoRunRequest] = Body(default=None)) -> DemoRunResponse:
    # 빈 바디도 허용: None이면 default request 생성 후 merge
    cfg = _load_model_yaml()
    req = body or DemoRunRequest()
    req = _merge_defaults(req, cfg)

    # ---- output base ----
    out_base = _resolve_path(str(req.result_abs_path))
    if not out_base.is_absolute():
        raise HTTPException(400, "resultAbsPath must be absolute path")
    out_base.mkdir(parents=True, exist_ok=True)

    run_id = _norm_str(req.run_id) or _gen_run_id()
    run_dir = out_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "status.json"
    created_at = _now_iso_no_ms()

    _write_json(
        status_path,
        {
            "runId": run_id,
            "status": "RUNNING",
            "createdAt": created_at,
            "startedAt": created_at,
            "resultAbsPath": str(out_base),
            "runPath": str(run_dir),
        },
    )

    weights = _norm_str(req.weights)
    if not weights:
        raise HTTPException(400, "weights not provided and model.yaml has no weights.path")
    weights_path = _resolve_path(weights)
    if not weights_path.exists():
        raise HTTPException(400, f"weights not found: {weights_path}")

    img_dir = _resolve_path(str(req.image_dir))
    if not img_dir.exists():
        raise HTTPException(400, f"imageDir not found: {img_dir}")

    img_paths = _iter_images(img_dir)
    if not img_paths:
        _write_json(
            status_path,
            {
                "runId": run_id,
                "status": "DONE",
                "createdAt": created_at,
                "finishedAt": _now_iso_no_ms(),
                "total": 0,
                "pass": 0,
                "fail": 0,
                "message": "no images",
                "resultAbsPath": str(out_base),
                "runPath": str(run_dir),
            },
        )
        return DemoRunResponse(runId=run_id, runPath=str(run_dir), status="DONE", createdAt=created_at)
    result_dir = run_dir / "result"
    pass_dir = result_dir / "pass"
    fail_dir = result_dir / "fail"
    if req.save_result_images:
        pass_dir.mkdir(parents=True, exist_ok=True)
        if req.save_fail_images:
            fail_dir.mkdir(parents=True, exist_ok=True)

    preds_jsonl = run_dir / "preds.jsonl"
    save_indices: Optional[set[int]] = None
    if req.save_result_images and req.max_save_images is not None and int(req.max_save_images) >= 0:
        k = int(req.max_save_images)
        if k == 0:
            save_indices = set()
        elif k < len(img_paths):
            random.seed(int(req.sample_seed))
            save_indices = set(random.sample(range(len(img_paths)), k))
    model = YOLO(str(weights_path))

    n_pass = 0
    n_fail = 0
    n_error = 0

    pred_kwargs = dict(
        conf=float(req.nms_conf_floor),
        iou=float(req.iou),
        imgsz=int(req.imgsz),
        max_det=int(req.max_det),
        device=str(req.device),
        verbose=False,
        save=False,
    )

    for idx, img_path in enumerate(img_paths):
        try:
            r0 = model.predict(source=str(img_path), **pred_kwargs)[0]
            boxes = r0.boxes

            out_boxes: List[Dict[str, Any]] = []
            confs: List[float] = []

            if boxes is not None and len(boxes) > 0:
                cls_list = boxes.cls.detach().cpu().tolist()
                conf_list = boxes.conf.detach().cpu().tolist()
                xyxy_list = boxes.xyxy.detach().cpu().tolist()

                for c, s, xyxy in zip(cls_list, conf_list, xyxy_list):
                    confs.append(float(s))
                    x1, y1, x2, y2 = map(float, xyxy)
                    out_boxes.append({"classId": int(c), "conf": float(s), "xyxy": [x1, y1, x2, y2]})

            passed, debug = _decide_pass(
                confs,
                conf_pass_threshold=float(req.conf_pass_threshold),
                policy=dict(req.pass_fail_policy or {"mode": "min"}),
                allow_empty_boxes_pass=bool(req.allow_empty_boxes_pass),
            )

            if passed:
                n_pass += 1
            else:
                n_fail += 1

            result_path: Optional[str] = None
            preview_path: Optional[str] = None

            should_save = bool(req.save_result_images)
            if save_indices is not None:
                should_save = should_save and (idx in save_indices)

            if should_save:
                out_dir = pass_dir if passed else (fail_dir if req.save_fail_images else None)
                if out_dir is not None:
                    vis = r0.plot()  # BGR
                    out_img = out_dir / f"{img_path.stem}.jpg"
                    cv2.imwrite(str(out_img), vis)
                    result_path = str(out_img)
                    preview_path = result_path

            _append_jsonl(
                preds_jsonl,
                {
                    "type": "pred",
                    "imagePath": str(img_path),
                    "boxes": out_boxes,
                    "pass": bool(passed),
                    "resultPath": result_path,
                    "previewPath": preview_path,
                    "debug": debug,
                },
            )

        except Exception as e:
            n_error += 1
            _append_jsonl(preds_jsonl, {"type": "error", "imagePath": str(img_path), "error": str(e)})

    finished_at = _now_iso_no_ms()
    _write_json(
        status_path,
        {
            "runId": run_id,
            "status": "DONE" if n_error == 0 else "DONE_WITH_ERRORS",
            "createdAt": created_at,
            "startedAt": created_at,
            "finishedAt": finished_at,
            "resultAbsPath": str(out_base),
            "runPath": str(run_dir),
            "weights": str(weights_path),
            "imageDir": str(img_dir),
            "total": len(img_paths),
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
            "saveResultImages": bool(req.save_result_images),
            "saveFailImages": bool(req.save_fail_images),
            "maxSaveImages": int(req.max_save_images),
            "config": {
                "device": str(req.device),
                "imgsz": int(req.imgsz),
                "iou": float(req.iou),
                "maxDet": int(req.max_det),
                "nmsConfFloor": float(req.nms_conf_floor),
                "confPassThreshold": float(req.conf_pass_threshold),
                "allowEmptyBoxesPass": bool(req.allow_empty_boxes_pass),
                "passFailPolicy": dict(req.pass_fail_policy or {"mode": "min"}),
            },
        },
    )

    return DemoRunResponse(
        runId=run_id,
        runPath=str(run_dir),
        status="DONE" if n_error == 0 else "DONE_WITH_ERRORS",
        createdAt=created_at,
    )
