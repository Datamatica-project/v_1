# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Body, Path as PathParam

from auto_labeling.v_1.api.dto.export_round0 import (
    Round0ExportNotifyRequest,
    Round0ExportInfoResponse,
)

router = APIRouter(prefix="/export")

ROOT = Path(__file__).resolve().parents[2]  # .../auto_labeling/v_1
EXPORT_REG_DIR = ROOT / "logs" / "exports"
EXPORT_REG_DIR.mkdir(parents=True, exist_ok=True)

# ✅ NAS share root (서버 고정 정책)
NAS_SHARE_ROOT = os.getenv("V1_EXPORT_NAS_ROOT", r"\\DS1821_1\V1_EXPORTS")

# ✅ NAS를 HTTP로 서빙할 경우(옵션): served base url을 매핑해서 manifestUrl/downloadBaseUrl을 생성
NAS_SERVED_BASE_URL = os.getenv("V1_EXPORT_NAS_SERVED_BASE_URL", "").strip() or None
NAS_URL_MAP: Dict[str, str] = (
    {NAS_SHARE_ROOT: NAS_SERVED_BASE_URL} if NAS_SERVED_BASE_URL else {}
)


def _utc_now() -> datetime:
    """UTC now"""
    return datetime.now(timezone.utc)


def _run_file(run_id: str) -> Path:
    """run별 round0 export 레지스트리 파일 경로"""
    return EXPORT_REG_DIR / f"round0_{run_id}.json"


def _safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    """레지스트리 JSON 안전 로드 (없거나 파싱 실패 시 None)"""
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    """레지스트리 JSON 저장"""
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _join_unc(share_root: str, rel_path: str) -> str:
    """
    UNC 경로 조합.
    - share_root: 서버 고정 NAS 루트 (예: \\DS1821_1\\V1_EXPORTS)
    - rel_path: export 결과 상대경로 (예: exports/run_001/round0)
    """
    if not share_root:
        return rel_path
    s = share_root.rstrip("\\/")
    r = rel_path.lstrip("\\/")
    return s + "\\" + r


def _to_served_url(share_root: str, rel_path: str) -> Optional[str]:
    """
    NAS를 HTTP로 서빙하는 환경에서 다운로드 URL을 생성.
    - V1_EXPORT_NAS_SERVED_BASE_URL이 설정되어 있으면 share_root -> base_url 매핑으로 URL 생성
    """
    base = NAS_URL_MAP.get(share_root)
    if not base:
        return None
    return base.rstrip("/") + "/" + rel_path.replace("\\", "/").lstrip("/")


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    """ISO8601 string -> datetime (Z 포함 가능)"""
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


@router.post(
    "/round0/notify",
    response_model=Round0ExportInfoResponse,
    summary="Round0 export 결과 알림(등록)",
    description=(
        "Round0 export가 완료(또는 실패)되었음을 서버에 알리고, 서버는 이를 레지스트리 파일로 저장합니다.\n\n"
        "### 이 API의 의미\n"
        "- round0 export 결과(카운트/경로/manifest 경로)를 **runId 기준으로 등록**합니다.\n"
        "- 등록된 결과는 `GET /api/v1/export/round0/{runId}` 로 조회할 수 있습니다.\n\n"
        "### 경로 정책(중요)\n"
        "- exportRelPath / manifestRelPath는 **NAS share root 기준 상대 경로**입니다.\n"
        "- 서버는 NAS 루트(`V1_EXPORT_NAS_ROOT`)를 고정값으로 사용하며,\n"
        "  응답에서 `shareRoot` + `exportRelPath`로 실제 경로(UNC 또는 served URL)를 구성합니다.\n\n"
        "### downloadBaseUrl / manifestUrl 생성 정책\n"
        "- `V1_EXPORT_NAS_SERVED_BASE_URL`이 설정되어 있으면 HTTP URL을 생성\n"
        "- 없으면 UNC(\\\\DS...\\...) 경로를 반환"
    ),
    responses={
        200: {"description": "등록 성공 및 등록된 정보 반환"},
        400: {"description": "필수 필드 누락(runId/exportRelPath 등)"},
    },
)
def notify_round0_export(
    req: Round0ExportNotifyRequest = Body(
        ...,
        description=(
            "Round0 export 알림 요청 본문.\n\n"
            "필수:\n"
            "- runId (string)\n"
            "- status (string: READY|DONE|FAILED)\n"
            "- exportRelPath (string)\n\n"
            "선택:\n"
            "- manifestRelPath (string|null)\n"
            "- passCount/failCount/missCount (int)\n"
            "- message (string|null)\n"
            "- createdAt (datetime|null)\n"
            "- extra (object|null)\n"
        ),
        examples=[
            {
                "runId": "run_001",
                "round": 0,
                "status": "DONE",
                "message": "round0 export completed",
                "passCount": 1200,
                "failCount": 340,
                "missCount": 12,
                "exportRelPath": "exports/run_001/round0",
                "manifestRelPath": "exports/run_001/round0/manifest.json",
                "createdAt": "2025-12-19T07:09:32.872Z",
                "extra": {"countUnit": "image"},
            }
        ],
    )
):
    now = _utc_now()
    share_root = NAS_SHARE_ROOT

    run_id = req.run_id.strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="runId is required")

    export_rel_path = req.export_rel_path.strip()
    if not export_rel_path:
        raise HTTPException(status_code=400, detail="exportRelPath is required")

    created_at = (req.created_at or now).isoformat()
    updated_at = now.isoformat()

    record: Dict[str, Any] = {
        # ✅ 내부 저장은 snake_case 유지 (외부 응답/요청은 camelCase DTO로 처리)
        "run_id": run_id,
        "round": int(req.round or 0),
        "status": req.status,
        "message": req.message,

        "pass_count": int(req.pass_count or 0),
        "fail_count": int(req.fail_count or 0),
        "miss_count": int(req.miss_count or 0),

        # ✅ NAS 정책 고정
        "share_root": share_root,
        "export_rel_path": export_rel_path,
        "manifest_rel_path": req.manifest_rel_path,

        "created_at": created_at,
        "updated_at": updated_at,
        "extra": req.extra or {},
    }

    _write_json(_run_file(run_id), record)

    download_base_url = _to_served_url(share_root, export_rel_path) or _join_unc(share_root, export_rel_path)
    manifest_url = None
    if req.manifest_rel_path:
        manifest_url = _to_served_url(share_root, req.manifest_rel_path) or _join_unc(share_root, req.manifest_rel_path)

    return Round0ExportInfoResponse(
        run_id=run_id,
        round=record["round"],
        status=req.status,
        message=req.message,

        pass_count=record["pass_count"],
        fail_count=record["fail_count"],
        miss_count=record["miss_count"],

        share_root=share_root,
        export_rel_path=export_rel_path,
        manifest_rel_path=req.manifest_rel_path,

        download_base_url=download_base_url,
        manifest_url=manifest_url,

        created_at=_parse_dt(created_at),
        updated_at=_parse_dt(updated_at),
        extra=record["extra"],
    )


@router.get(
    "/round0/{run_id}",
    response_model=Round0ExportInfoResponse,
    summary="Round0 export 결과 조회",
    description=(
        "notify로 등록된 round0 export 정보를 runId로 조회합니다.\n\n"
        "### 동작\n"
        "- logs/exports/round0_{runId}.json 레지스트리를 읽어 응답합니다.\n"
        "- NAS 루트는 서버 고정값(`V1_EXPORT_NAS_ROOT`)을 기준으로 downloadBaseUrl/manifestUrl을 계산합니다."
    ),
    responses={
        200: {"description": "조회 성공"},
        404: {"description": "해당 runId의 round0 export 정보가 없음"},
    },
)
def get_round0_export(
    run_id: str = PathParam(
        ...,
        description="조회할 runId (notify에서 등록한 runId)",
        examples=["run_001"],
    )
):
    record = _safe_read_json(_run_file(run_id))
    if not record:
        raise HTTPException(status_code=404, detail="round0 export not found")

    # ✅ NAS 고정 정책: 저장된 share_root가 달라도 응답은 고정값 기준으로 계산
    share_root = NAS_SHARE_ROOT

    export_rel_path = str(record.get("export_rel_path", "") or "")
    manifest_rel_path = record.get("manifest_rel_path")

    download_base_url = _to_served_url(share_root, export_rel_path) or _join_unc(share_root, export_rel_path)
    manifest_url = None
    if manifest_rel_path:
        manifest_url = _to_served_url(share_root, str(manifest_rel_path)) or _join_unc(share_root, str(manifest_rel_path))

    return Round0ExportInfoResponse(
        run_id=str(record.get("run_id", run_id)),
        round=int(record.get("round", 0)),
        status=str(record.get("status", "UNKNOWN")),
        message=record.get("message"),

        pass_count=int(record.get("pass_count", 0)),
        fail_count=int(record.get("fail_count", 0)),
        miss_count=int(record.get("miss_count", 0)),

        share_root=share_root,
        export_rel_path=export_rel_path,
        manifest_rel_path=manifest_rel_path,

        download_base_url=download_base_url,
        manifest_url=manifest_url,

        created_at=_parse_dt(record.get("created_at")),
        updated_at=_parse_dt(record.get("updated_at")),
        extra=record.get("extra") or {},
    )
