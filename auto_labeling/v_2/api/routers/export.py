from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Body, Query, Path as PathParam

from auto_labeling.v_1.api.dto.export_final import (
    ExportFinalNotifyRequest,
    ExportFinalInfoResponse,
)

# -----------------------------------------------------------------------------
# Router (⚠️ prefix는 server.py에서만 /api/v1 붙인다)
# -----------------------------------------------------------------------------
router = APIRouter(prefix="/export", tags=["export"])

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../auto_labeling/v_1
EXPORTS_DIR = ROOT / "logs" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# export 결과를 서버 로컬에서 접근하기 위한 루트 (preview/index 생성용)
V1_EXPORT_LOCAL_ROOT = os.getenv("V1_EXPORT_LOCAL_ROOT", "").strip() or None

# NAS 등 외부 served base url (선택)
NAS_SERVED_BASE_URL = os.getenv("V1_EXPORT_NAS_SERVED_BASE_URL", "").strip() or None


# -----------------------------------------------------------------------------
# time / json helpers
# -----------------------------------------------------------------------------
def _utc_now_iso_z() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _registry_path(run_id: str) -> Path:
    return EXPORTS_DIR / f"final_{run_id}.json"


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, payload: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert_registry(run_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    p = _registry_path(run_id)
    now = _utc_now_iso_z()

    if p.exists():
        cur = _read_json(p)
        created_at = cur.get("createdAt") or now
    else:
        cur = {}
        created_at = now

    merged = dict(cur)
    merged.update(patch)
    merged["runId"] = run_id
    merged["createdAt"] = created_at
    merged["updatedAt"] = now

    _write_json(p, merged)
    return merged


def _registry_to_response(payload: Dict[str, Any]) -> ExportFinalInfoResponse:
    return ExportFinalInfoResponse(
        run_id=payload.get("runId", ""),
        status=payload.get("status", "READY"),
        message=payload.get("message"),
        pass_count=int(payload.get("passCount", 0) or 0),
        pass_fail_count=int(payload.get("passFailCount", 0) or 0),
        fail_fail_count=int(payload.get("failFailCount", 0) or 0),
        miss_count=int(payload.get("missCount", 0) or 0),
        share_root=payload.get("shareRoot", ""),
        export_rel_path=payload.get("exportRelPath", ""),
        manifest_rel_path=payload.get("manifestRelPath"),
        download_base_url=payload.get("downloadBaseUrl"),
        manifest_url=payload.get("manifestUrl"),
        created_at=payload.get("createdAt"),
        updated_at=payload.get("updatedAt"),
        extra=payload.get("extra") or {},
    )


def _resolve_export_root(export_rel_path: str) -> Optional[Path]:
    if not V1_EXPORT_LOCAL_ROOT:
        return None
    return (Path(V1_EXPORT_LOCAL_ROOT) / export_rel_path).resolve()


def _to_served_url(rel_path: str) -> Optional[str]:
    if not NAS_SERVED_BASE_URL:
        return None
    return NAS_SERVED_BASE_URL.rstrip("/") + "/" + rel_path.replace("\\", "/").lstrip("/")


# =============================================================================
# 1) Worker → API : notify (RUNNING / DONE / FAILED)
# =============================================================================
@router.post(
    "/final/notify",
    response_model=ExportFinalInfoResponse,
    summary="Final export registry upsert (worker notify)",
)
def notify_export_final(
    req: ExportFinalNotifyRequest = Body(...),
) -> ExportFinalInfoResponse:
    run_id = req.run_id.strip()
    if not run_id:
        raise HTTPException(400, "runId is required")

    merged = _upsert_registry(
        run_id,
        {
            "status": req.status,
            "message": req.message,
            "passCount": req.pass_count or 0,
            "passFailCount": req.pass_fail_count or 0,
            "failFailCount": req.fail_fail_count or 0,
            "missCount": req.miss_count or 0,
            "shareRoot": req.share_root or "",
            "exportRelPath": req.export_rel_path,
            "manifestRelPath": req.manifest_rel_path,
            "extra": req.extra or {},
        },
    )

    # served url 보강(선택)
    if merged.get("exportRelPath"):
        if NAS_SERVED_BASE_URL:
            merged["downloadBaseUrl"] = _to_served_url(merged["exportRelPath"])
        if merged.get("manifestRelPath") and NAS_SERVED_BASE_URL:
            merged["manifestUrl"] = _to_served_url(merged["manifestRelPath"])

        _write_json(_registry_path(run_id), merged)

    return _registry_to_response(merged)


# =============================================================================
# 2) Frontend / External : GET final export info
# =============================================================================
@router.get(
    "/final/{run_id}",
    response_model=ExportFinalInfoResponse,
    summary="Get final export registry",
)
def get_export_final(
    run_id: str = PathParam(...),
    buildResult: bool = Query(False, description="true면 preview/index 빌드(옵션)"),
):
    run_id = run_id.strip()
    if not run_id:
        raise HTTPException(400, "run_id is required")

    p = _registry_path(run_id)
    if not p.exists():
        raise HTTPException(404, f"final export registry not found: {run_id}")

    payload = _read_json(p)

    # (선택) result/preview/index 빌드
    if buildResult:
        export_rel = payload.get("exportRelPath")
        if not export_rel:
            raise HTTPException(500, "exportRelPath missing in registry")

        export_root = _resolve_export_root(export_rel)
        if export_root is None or not export_root.exists():
            raise HTTPException(501, "V1_EXPORT_LOCAL_ROOT not configured or path missing")

        # ⚠️ 여기서 preview/index 빌드 함수 호출 가능
        # build_final_preview(export_root, run_id)

    return _registry_to_response(payload)
