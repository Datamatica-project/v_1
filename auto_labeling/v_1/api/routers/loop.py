# auto_labeling/v_1/api/routers/loop.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Body, Path as PathParam, Query
from pathlib import Path
import os
import time
import requests
from typing import Any, Dict, Optional

from auto_labeling.v_1.api.dto.loop import RunLoopRequest, RunLoopResponse, JobStatusResponse

router = APIRouter(prefix="/loop")

WORKER_BASE_URL = os.getenv("WORKER_BASE_URL", "http://v1-worker:8011").rstrip("/")
DEFAULT_TIMEOUT = (3.0, 300.0)

WORKER_LOOP_PREFIX = "/api/v1/loop"
API_BASE_URL = os.getenv("API_BASE_URL", "http://v1-api:8010").rstrip("/")

ROOT = Path(__file__).resolve().parents[2]  # .../auto_labeling/v_1
PREVIEW_REG_DIR = ROOT / "logs" / "previews"
PREVIEW_REG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# errors (worker proxy vs api-internal)
# ---------------------------------------------------------------------
def _raise_502(prefix: str, *, url: str, err: Exception, resp: Optional[requests.Response] = None) -> None:
    detail: Dict[str, Any] = {"message": f"{prefix}: {err}", "url": url}
    if resp is not None:
        detail["status_code"] = resp.status_code
        detail["body_preview"] = (resp.text or "")[:1000]
    raise HTTPException(status_code=502, detail=detail)


def _post_json_worker(url: str, payload: dict, *, params: dict | None = None) -> dict:
    try:
        r = requests.post(url, json=payload, params=params, timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        _raise_502("WORKER POST failed", url=url, err=e)

    if not r.ok:
        _raise_502("WORKER POST bad status", url=url, err=RuntimeError("bad status"), resp=r)

    try:
        return r.json()
    except Exception as e:
        _raise_502("WORKER POST invalid json", url=url, err=e, resp=r)


def _get_json_worker(url: str) -> dict:
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        _raise_502("WORKER GET failed", url=url, err=e)

    if not r.ok:
        _raise_502("WORKER GET bad status", url=url, err=RuntimeError("bad status"), resp=r)

    try:
        return r.json()
    except Exception as e:
        _raise_502("WORKER GET invalid json", url=url, err=e, resp=r)


def _post_json_api_internal(url: str, payload: dict, *, params: dict | None = None) -> tuple[bool, dict]:
    try:
        r = requests.post(url, json=payload, params=params, timeout=DEFAULT_TIMEOUT)
    except Exception as e:
        return False, {"message": f"API_INTERNAL POST failed: {e}", "url": url}

    if not r.ok:
        return False, {
            "message": "API_INTERNAL POST bad status",
            "url": url,
            "status_code": r.status_code,
            "body_preview": (r.text or "")[:1000],
        }

    try:
        return True, r.json()
    except Exception as e:
        return False, {
            "message": f"API_INTERNAL POST invalid json: {e}",
            "url": url,
            "status_code": r.status_code,
            "body_preview": (r.text or "")[:1000],
        }


# ---------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------
def _normalize_keys(d: dict) -> dict:
    if "job_id" in d and "jobId" not in d:
        d["jobId"] = d["job_id"]
    if "jobId" in d and "job_id" not in d:
        d["job_id"] = d["jobId"]

    if "run_id" in d and "runId" not in d:
        d["runId"] = d["run_id"]
    if "runId" in d and "run_id" not in d:
        d["run_id"] = d["runId"]

    return d


def _now_iso_z_no_ms() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slim_stats(stats: dict) -> dict:
    """
    ✅ 프론트로 내려보낼 stats에서 초장문 로그를 제거한다.
    """
    if not isinstance(stats, dict):
        return {}

    s = dict(stats)

    for k in ("stdout", "stderr"):
        v = s.get(k)
        if isinstance(v, dict):
            v = dict(v)
            v.pop("head", None)
            v.pop("tail", None)
            if not v:
                s.pop(k, None)
            else:
                s[k] = v
        else:
            if k in s:
                s.pop(k, None)

    for k in ("head", "tail", "log", "logs", "trainer_args", "engine_args"):
        if k in s and isinstance(s.get(k), (str, dict, list)):
            s.pop(k, None)

    tp = s.get("tail_preview")
    if isinstance(tp, str) and len(tp) > 4000:
        s["tail_preview"] = tp[-4000:]

    return s


def _get_round0_status(stats: dict) -> str:
    """
    ✅ worker가 내려주는 공식 round0 상태.
    - 우선 camelCase: round0Status
    - 없으면 snake_case: round0_status
    """
    if not isinstance(stats, dict):
        return ""
    v = stats.get("round0Status")
    if v is None:
        v = stats.get("round0_status")
    return str(v or "").upper().strip()


def _ensure_final_stub(run_id: str, *, reason: str = "loop.status(round0Status=DONE)") -> dict:
    """
    ✅ round0 DONE 직후:
    - final registry가 없을 수 있으니 READY 상태로 먼저 upsert해서 404를 방지한다.
    - (주의) export/final/notify 쪽에서 DONE/FAILED를 READY로 덮어쓰지 않도록 downgrade 방지 권장
    """
    url = f"{API_BASE_URL}/api/v1/export/final/notify"
    payload = {
        "runId": run_id,
        "status": "READY",
        "message": "round0 done, waiting final export",
        "exportRelPath": f"exports/{run_id}/final",
        "manifestRelPath": None,
        "passCount": 0,
        "passFailCount": 0,
        "failFailCount": 0,
        "missCount": 0,
        "extra": {"stubCreatedBy": reason},
    }

    ok, resp_or_err = _post_json_api_internal(url, payload)

    return {
        "ok": bool(ok),
        "builtAt": _now_iso_z_no_ms(),
        "request": {"method": "POST", "url": url, "body": payload},
        "response": resp_or_err if ok else None,
        "error": None if ok else resp_or_err,
    }


# ---------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------
@router.post("/run", response_model=RunLoopResponse, summary="Loop 실행(Worker 프록시)")
def run_loop(req: RunLoopRequest = Body(default_factory=RunLoopRequest)):
    url = f"{WORKER_BASE_URL}{WORKER_LOOP_PREFIX}/run"
    payload = req.model_dump(by_alias=True, exclude_none=True)

    data = _post_json_worker(url, payload)
    data = _normalize_keys(data)
    return RunLoopResponse(**data)


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Loop 상태 조회 (결과 생성은 /api/v1/results/* 가 runId 기반 멱등 처리)",
)
def get_status(
    job_id: str = PathParam(..., description="조회할 jobId (/loop/run 응답에서 반환)"),
    ensureFinalStub: bool = Query(
        True,
        description="round0Status=DONE일 때 final export registry READY stub을 upsert해서 404를 방지할지 여부",
    ),
):
    # 1) worker status proxy
    url = f"{WORKER_BASE_URL}{WORKER_LOOP_PREFIX}/status/{job_id}"
    data = _get_json_worker(url)
    data = _normalize_keys(data)

    status = (data.get("status") or "").upper().strip()

    raw_stats = data.get("stats") or {}
    stats = _slim_stats(raw_stats)

    # 2) runId 없으면 여기서 끝
    run_id = (data.get("runId") or "").strip()
    if not run_id:
        stats["finalExportStubEnsured"] = {
            "ok": False,
            "builtAt": _now_iso_z_no_ms(),
            "error": {"message": "skip final stub: worker did not return runId"},
        }
        stats["results"] = {
            "note": "Results are built by /api/v1/results/* endpoints (idempotent).",
            "suggested": {
                "round0BuildPreviewSet": "/api/v1/results/round0/buildPreviewSet (POST, body: {runId})",
                "round0Preview": "/api/v1/results/round0/preview?runId=<runId>",
                "finalBuildPreviewSet": "/api/v1/results/final/buildPreviewSet (POST, body: {runId})",
                "finalPreview": "/api/v1/results/final/preview?runId=<runId>",
            },
        }
        data["stats"] = stats
        return JobStatusResponse(**data)

    # 3) ✅ round0는 worker의 "공식 필드"를 기준으로 판단
    round0_status = _get_round0_status(raw_stats)
    stats["round0Status"] = round0_status or stats.get("round0Status") or stats.get("round0_status") or ""

    # round0 부가 필드: worker가 이미 내려줬으면 그대로 노출
    # (worker에서 camelCase mirror까지 만든 상태라면 raw_stats/ stats 어디든 있을 수 있음)
    for k in (
        "round0ExportRelPath",
        "round0Counts",
        "round0Notify",
        "round0UpdatedAt",
        "round0Note",
        "round0_export_rel_path",
        "round0_counts",
        "round0_notify",
        "round0_updated_at",
        "round0_note",
    ):
        if k in raw_stats and k not in stats:
            stats[k] = raw_stats.get(k)

    # 4) ✅ final stub upsert는 "round0Status == DONE" 일 때만
    if ensureFinalStub:
        if round0_status == "DONE":
            stats["finalExportStubEnsured"] = _ensure_final_stub(run_id)
        else:
            stats["finalExportStubEnsured"] = {
                "ok": False,
                "builtAt": _now_iso_z_no_ms(),
                "error": {
                    "message": "skip final stub: round0Status is not DONE",
                    "round0Status": round0_status,
                    "loopStatus": status,
                },
            }

    # ✅ loop/status는 결과 생성(POST)을 절대 하지 않는다.
    stats["results"] = {
        "note": "Results are built by /api/v1/results/* endpoints (idempotent).",
        "runId": run_id,
        "suggested": {
            "round0BuildPreviewSet": "/api/v1/results/round0/buildPreviewSet (POST, body: {runId})",
            "round0Preview": f"/api/v1/results/round0/preview?runId={run_id}",
            "finalBuildPreviewSet": "/api/v1/results/final/buildPreviewSet (POST, body: {runId})",
            "finalPreview": f"/api/v1/results/final/preview?runId={run_id}",
        },
    }

    data["stats"] = stats
    return JobStatusResponse(**data)
