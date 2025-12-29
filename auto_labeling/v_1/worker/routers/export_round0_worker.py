from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import os
import json
import shutil
from typing import Dict, Any, Tuple

import requests

from auto_labeling.v_1.scripts.logger import log_json


# NOTE:
# - 실제 최종 경로는 worker app 쪽 include_router(prefix=...)에 의해 결정됨
# - 이 파일은 /export 아래만 책임진다.
router = APIRouter(prefix="/export")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
V1_ROOT = PROJECT_ROOT / "auto_labeling" / "v_1"

# 누적 PASS(기본)
PASS_IMG_DIR = Path(os.getenv("PASS_IMG_DIR", str(V1_ROOT / "data" / "pass" / "images")))
PASS_LBL_DIR = Path(os.getenv("PASS_LBL_DIR", str(V1_ROOT / "data" / "pass" / "labels")))

# round0 PASS fallback (누적 PASS가 없을 때 자동으로 round_r0/pass 사용)
ROUND0_PASS_IMG_DIR = Path(os.getenv("ROUND0_PASS_IMG_DIR", str(V1_ROOT / "data" / "round_r0" / "pass" / "images")))
ROUND0_PASS_LBL_DIR = Path(os.getenv("ROUND0_PASS_LBL_DIR", str(V1_ROOT / "data" / "round_r0" / "pass" / "labels")))

# NAS root (컨테이너 내부 마운트 경로)
NAS_ROOT = Path(os.getenv("NAS_ROOT", "/mnt/nas"))
NAS_PASS_ROOT = Path(os.getenv("NAS_PASS_ROOT", str(NAS_ROOT / "v1" / "pass")))

# ✅ 백엔드(= api 컨테이너) notify endpoint
# - prefix 꼬임 방지: base + notify_path로 단순화
API_BASE_URL = os.getenv("API_BASE_URL", "http://v1-api:8010").rstrip("/")
API_NOTIFY_PATH = os.getenv("API_NOTIFY_PATH", "/api/v1/export/round0/notify").strip()
if not API_NOTIFY_PATH.startswith("/"):
    API_NOTIFY_PATH = "/" + API_NOTIFY_PATH

# ✅ 백엔드에 넘길 shareRoot (UNC/URL 매핑의 기준값)
# - api 쪽 NAS_URL_MAP key로 쓰고 싶으면, "공유루트" 값을 정확히 넣는다.
# - 예: "\\\\DS1821_1\\home"  또는  "smb://DS1821_1/home"  또는  "/mnt/nas"
SHARE_ROOT = os.getenv("V1_EXPORT_SHARE_ROOT", str(NAS_ROOT)).strip()

# jobs
JOB_DIR = Path(os.getenv("WORKER_JOB_DIR", str(V1_ROOT / "logs" / "jobs")))
JOB_DIR.mkdir(parents=True, exist_ok=True)


class ExportRound0Request(BaseModel):
    # 외부 통신 camelCase
    run_id: str = Field(..., alias="runId")
    async_notify: bool = Field(True, alias="asyncNotify")

    # pydantic v1 호환
    class Config:
        allow_population_by_field_name = True


def _now_iso_z() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _job_path(run_id: str) -> Path:
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


def _resolve_pass_dirs() -> Tuple[Path, Path, str]:
    """
    export0가 참조할 PASS 원천 디렉토리 결정.
    1) 누적 PASS(data/pass) 우선
    2) 없으면 round_r0/pass fallback
    """
    if PASS_IMG_DIR.exists() and PASS_LBL_DIR.exists():
        return PASS_IMG_DIR, PASS_LBL_DIR, "pass"
    if ROUND0_PASS_IMG_DIR.exists() and ROUND0_PASS_LBL_DIR.exists():
        return ROUND0_PASS_IMG_DIR, ROUND0_PASS_LBL_DIR, "round_r0_pass"
    # 둘 다 없으면 누적 PASS 기준으로 에러를 내되, 메시지를 명확히
    return PASS_IMG_DIR, PASS_LBL_DIR, "missing"


def _notify_round0_to_api(*, run_id: str, pass_count: int, export_rel_path: str) -> None:
    payload = {
        "runId": run_id,
        "round": 0,
        "status": "DONE",
        "message": "PASS exported to NAS",
        "passCount": int(pass_count),
        "failCount": 0,
        "missCount": 0,
        "shareRoot": str(SHARE_ROOT),     # ✅ 추가(중요)
        "exportRelPath": export_rel_path, # ex) "v1/pass/<runId>"
        "manifestRelPath": None,
        "extra": {},
    }

    url = f"{API_BASE_URL}{API_NOTIFY_PATH}"
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
    src_img_dir, src_lbl_dir, src_kind = _resolve_pass_dirs()

    if not src_img_dir.exists():
        raise FileNotFoundError(
            f"[EXPORT_ROUND0] pass images not found. tried={src_kind} path={src_img_dir}"
        )
    if not src_lbl_dir.exists():
        raise FileNotFoundError(
            f"[EXPORT_ROUND0] pass labels not found. tried={src_kind} path={src_lbl_dir}"
        )

    out_root = NAS_PASS_ROOT / run_id
    img_dst = out_root / "images"
    lbl_dst = out_root / "labels"

    _sync_dir(src_img_dir, img_dst)
    _sync_dir(src_lbl_dir, lbl_dst)

    pass_count = _count_images(img_dst)

    meta = {
        "runId": run_id,
        "status": "PASS_SAVED",
        "savedAt": _now_iso_z(),
        "sourceKind": src_kind,             # ✅ 어디서 가져왔는지 기록
        "sourceImagesDir": str(src_img_dir),
        "sourceLabelsDir": str(src_lbl_dir),
        "nasPassPath": str(out_root),
        "imagesDir": str(img_dst),
        "labelsDir": str(lbl_dst),
        "passCount": int(pass_count),
        "shareRoot": str(SHARE_ROOT),
        "exportRelPath": f"v1/pass/{run_id}",
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


@router.post("/round0")
def export_round0(bg: BackgroundTasks, req: ExportRound0Request = Body(...)):
    run_id = req.run_id
    async_notify = bool(req.async_notify)

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
                        "shareRoot": meta.get("shareRoot"),
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

        # ✅ 실패는 HTTP 에러로 주는 게 프론트/백엔드에서 다루기 편함
        raise HTTPException(
            status_code=500,
            detail={"runId": run_id, "status": "FAILED", "error": str(e)},
        )
