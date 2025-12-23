# auto_labeling/v_1/worker/routers/export_round0_worker.py
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Body
from pydantic import BaseModel, Field
from pathlib import Path
import os
import json
import shutil
from typing import Dict, Any, Optional

import requests

from auto_labeling.v_1.scripts.logger import log_json


router = APIRouter(prefix="/api/v1")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
V1_ROOT = PROJECT_ROOT / "auto_labeling" / "v_1"

# PASS source (누적 PASS)
PASS_IMG_DIR = Path(os.getenv("PASS_IMG_DIR", str(V1_ROOT / "data" / "pass" / "images")))
PASS_LBL_DIR = Path(os.getenv("PASS_LBL_DIR", str(V1_ROOT / "data" / "pass" / "labels")))

# NAS root (컨테이너 내부 마운트 경로)
NAS_ROOT = Path(os.getenv("NAS_ROOT", "/mnt/nas"))
NAS_PASS_ROOT = Path(os.getenv("NAS_PASS_ROOT", str(NAS_ROOT / "v1" / "pass")))

API_BASE = os.getenv("V1_API_BASE", "http://api:8010").rstrip("/")
API_V1_PREFIX = os.getenv("V1_API_PREFIX", "/v1").rstrip("/")
API_NOTIFY_PATH = f"{API_V1_PREFIX}/export/round0/notify"

# jobs
JOB_DIR = Path(os.getenv("WORKER_JOB_DIR", str(V1_ROOT / "logs" / "jobs")))
JOB_DIR.mkdir(parents=True, exist_ok=True)


class ExportRound0Request(BaseModel):
    # 외부 통신은 camelCase
    run_id: str = Field(..., alias="runId")
    async_notify: bool = Field(True, alias="asyncNotify")


def _now_iso_z() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _job_path(run_id: str) -> Path:
    # list_jobs가 job_*.json 을 보니까 여기도 job_ prefix로 저장
    return JOB_DIR / f"job_{run_id}.json"


def _write_job(run_id: str, obj: Dict[str, Any]) -> None:
    _job_path(run_id).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_images(d: Path) -> int:
    if not d.exists():
        return 0
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _sync_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def _notify_round0_to_api(*, run_id: str, pass_count: int, export_rel_path: str) -> None:
    payload = {
        "runId": run_id,
        "round": 0,
        "status": "DONE",
        "message": "PASS exported to NAS",
        "passCount": int(pass_count),
        "failCount": 0,
        "missCount": 0,
        "exportRelPath": export_rel_path,  # ex) "v1/pass/<runId>"
        "manifestRelPath": None,
        "extra": {},
    }

    url = f"{API_BASE}{API_NOTIFY_PATH}"
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        log_json(
            {
                "time": _now_iso_z(),
                "level": "WARN",
                "scope": "export",
                "refId": run_id,
                "message": "round0_notify_failed",
                "data": {"error": str(e), "url": url, "payload": payload},
            }
        )


def _export_pass_to_nas(run_id: str) -> Dict[str, Any]:
    if not PASS_IMG_DIR.exists():
        raise FileNotFoundError(f"[EXPORT_ROUND0] pass images not found: {PASS_IMG_DIR}")
    if not PASS_LBL_DIR.exists():
        raise FileNotFoundError(f"[EXPORT_ROUND0] pass labels not found: {PASS_LBL_DIR}")

    out_root = NAS_PASS_ROOT / run_id
    img_dst = out_root / "images"
    lbl_dst = out_root / "labels"

    _sync_dir(PASS_IMG_DIR, img_dst)
    _sync_dir(PASS_LBL_DIR, lbl_dst)

    pass_count = _count_images(img_dst)

    meta = {
        "runId": run_id,
        "status": "PASS_SAVED",
        "savedAt": _now_iso_z(),
        "nasPassPath": str(out_root),
        "imagesDir": str(img_dst),
        "labelsDir": str(lbl_dst),
        "passCount": int(pass_count),
        "exportRelPath": f"v1/pass/{run_id}",
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


@router.post("/api/export/round0")
def export_round0(
    bg: BackgroundTasks,
    req: ExportRound0Request = Body(...),
):
    run_id = req.run_id
    async_notify = bool(req.async_notify)

    # job 파일 기록 (/logs/jobs 용)
    try:
        _write_job(
            run_id,
            {
                "jobId": f"job_{run_id}",
                "type": "exportRound0",
                "status": "RUNNING",
                "updatedAt": _now_iso_z(),
            },
        )
    except Exception:
        pass

    try:
        meta = _export_pass_to_nas(run_id)

        # job 업데이트
        try:
            _write_job(
                run_id,
                {
                    "jobId": f"job_{run_id}",
                    "type": "exportRound0",
                    "status": "DONE",
                    "updatedAt": _now_iso_z(),
                    "result": {
                        "exportRelPath": meta.get("exportRelPath"),
                        "passCount": meta.get("passCount"),
                    },
                },
            )
        except Exception:
            pass

        # ✅ API notify (옵션: background)
        if async_notify:
            bg.add_task(
                _notify_round0_to_api,
                run_id=run_id,
                pass_count=int(meta.get("passCount", 0)),
                export_rel_path=str(meta.get("exportRelPath", f"v1/pass/{run_id}")),
            )
        else:
            _notify_round0_to_api(
                run_id=run_id,
                pass_count=int(meta.get("passCount", 0)),
                export_rel_path=str(meta.get("exportRelPath", f"v1/pass/{run_id}")),
            )

        return meta

    except Exception as e:
        try:
            _write_job(
                run_id,
                {
                    "jobId": f"job_{run_id}",
                    "type": "exportRound0",
                    "status": "FAILED",
                    "updatedAt": _now_iso_z(),
                    "error": str(e),
                },
            )
        except Exception:
            pass
        return {"runId": run_id, "status": "FAILED", "error": str(e)}
