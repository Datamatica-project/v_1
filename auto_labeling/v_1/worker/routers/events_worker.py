from __future__ import annotations

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
import os
import requests
from typing import Any, Dict, Optional, Literal

router = APIRouter(prefix="/api/v1")

BACKEND_NOTIFY_URL = os.getenv("BACKEND_NOTIFY_URL", "").strip()
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "").strip()

DEFAULT_TIMEOUT = (3, 15)
EventType = Literal[
    "ROUND0_EXPORTED",   # round0 PASS export 완료 (NAS 경로 포함)
    "ROUND_RESULT",      # round별 pass/fail/miss 요약
    "LOOP_FINISHED",     # loop 종료 최종 요약
    "LOOP_FAILED",       # loop 실패 요약
]


class EventPayload(BaseModel):
    event_type: EventType = Field(..., alias="eventType")
    run_id: str = Field(..., alias="runId")

    round: Optional[int] = Field(default=None, alias="round")

    status: str = Field(default="DONE", alias="status")

    pass_count: Optional[int] = Field(default=None, alias="passCount")
    fail_count: Optional[int] = Field(default=None, alias="failCount")
    miss_count: Optional[int] = Field(default=None, alias="missCount")

    export_rel_path: Optional[str] = Field(default=None, alias="exportRelPath")
    manifest_rel_path: Optional[str] = Field(default=None, alias="manifestRelPath")

    message: Optional[str] = Field(default=None, alias="message")
    extra: Dict[str, Any] = Field(default_factory=dict, alias="extra")

    job_id: Optional[str] = Field(default=None, alias="jobId")
    timestamp: Optional[str] = Field(default=None, alias="timestamp")


def _post_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not BACKEND_NOTIFY_URL:
        return {"status": "FAILED", "error": "BACKEND_NOTIFY_URL is empty"}

    headers = {"Content-Type": "application/json"}
    if WORKER_TOKEN:
        headers["X-Worker-Token"] = WORKER_TOKEN

    try:
        r = requests.post(
            BACKEND_NOTIFY_URL,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        return {
            "status": "OK" if r.ok else "FAILED",
            "httpStatus": r.status_code,
            "response": (r.text or "")[:1000],
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e), "url": BACKEND_NOTIFY_URL}


@router.post("/api/events")
def emit_event(req: EventPayload = Body(...)) -> Dict[str, Any]:

    payload = req.model_dump(by_alias=True, exclude_none=True)
    return _post_event(payload)
