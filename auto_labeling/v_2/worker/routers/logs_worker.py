from __future__ import annotations

from fastapi import APIRouter, Query
from pathlib import Path
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from auto_labeling.v_1.api.dto.log import LogListResponse
from auto_labeling.v_1.scripts.logger import load_logs

router = APIRouter()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
V1_ROOT = PROJECT_ROOT / "auto_labeling" / "v_1"
JOB_DIR = Path(os.getenv("WORKER_JOB_DIR", str(V1_ROOT / "logs" / "jobs")))
JOB_DIR.mkdir(parents=True, exist_ok=True)

def _read_json(p: Path) -> Dict[str, Any]:
    d = json.loads(p.read_text(encoding="utf-8"))
    if "job_id" in d and "jobId" not in d:
        d["jobId"] = d["job_id"]
    if "updated_at" in d and "updatedAt" not in d:
        d["updatedAt"] = d["updated_at"]
    return d

def _to_iso_timestamp(v: Any) -> str:
    if v is None:
        return datetime.now(tz=timezone.utc).isoformat()

    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(float(v), tz=timezone.utc).isoformat()

    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()

    if isinstance(v, str):
        s = v.strip()
        if " " in s and "T" not in s:
            s = s.replace(" ", "T")
        return s

    return datetime.now(tz=timezone.utc).isoformat()


def _normalize_log_item(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    if "timestamp" not in out and "time" in out:
        out["timestamp"] = out.pop("time")

    out["timestamp"] = _to_iso_timestamp(out.get("timestamp"))

    if "ref_id" in out and "refId" not in out:
        out["refId"] = out["ref_id"]

    return out

@router.get("/api/logs", response_model=LogListResponse)
def get_logs(
    scope: Optional[str] = Query(None, description="ingest | loop | export | system"),
    ref_id: Optional[str] = Query(None, alias="refId"),
    level: Optional[str] = Query(None, description="INFO | WARN | ERROR"),
    limit: int = Query(100, ge=1, le=1000),
    since: Optional[datetime] = Query(None, description="이 timestamp 이후 로그만 조회 (cursor)"),
):
    raw_items = load_logs(
        scope=scope,
        ref_id=ref_id,
        level=level,
        limit=limit,
        since=since,
    )

    items: List[Dict[str, Any]] = []
    for x in raw_items or []:
        try:
            if isinstance(x, dict):
                items.append(_normalize_log_item(x))
            else:
                items.append(_normalize_log_item(dict(x)))  # type: ignore[arg-type]
        except Exception:
            items.append({"timestamp": datetime.now(tz=timezone.utc).isoformat(), "message": str(x)})

    return LogListResponse(
        items=items,
        total=len(items),
        has_more=len(items) == limit,
    )


@router.get("/api/logs/jobs")
def list_jobs(limit: int = 50) -> Dict[str, Any]:
    files = sorted(JOB_DIR.glob("job_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    files = files[: max(1, min(limit, 500))]

    items: List[Dict[str, Any]] = []
    for p in files:
        try:
            d = _read_json(p)
            items.append(
                {
                    "jobId": d.get("jobId", p.stem),
                    "status": d.get("status", "UNKNOWN"),
                    "updatedAt": d.get("updatedAt", d.get("updated_at", None)),
                    "path": str(p),
                }
            )
        except Exception:
            items.append({"jobId": p.stem, "status": "BROKEN", "path": str(p)})

    return {"count": len(items), "items": items}


@router.get("/api/logs/jobs/{job_id}")
def read_job(job_id: str) -> Dict[str, Any]:
    # list_jobs가 job_*.json 을 보니까, read도 둘 다 허용해주는 게 편함
    candidates = [
        JOB_DIR / f"{job_id}.json",
        JOB_DIR / f"job_{job_id}.json",
    ]
    p = next((c for c in candidates if c.exists()), candidates[0])

    if not p.exists():
        return {"jobId": job_id, "status": "NOT_FOUND"}

    try:
        return _read_json(p)
    except Exception as e:
        return {"jobId": job_id, "status": "BROKEN", "error": str(e), "path": str(p)}
