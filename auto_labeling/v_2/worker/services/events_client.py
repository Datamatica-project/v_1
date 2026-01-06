# auto_labeling/v_1/worker/services/events_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
import os
import time

import requests


EventType = Literal[
    "ROUND0_EXPORTED",   # round0 PASS export 완료 (NAS 경로 포함)
    "ROUND_RESULT",      # round별 pass/fail/miss 요약
    "LOOP_FINISHED",     # loop 종료 최종 요약
    "LOOP_FAILED",       # loop 실패 요약
]


def _now_iso_z() -> str:
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _coalesce(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


@dataclass(frozen=True)
class EventsClientConfig:
    """
    Worker -> Backend 이벤트 notify 클라이언트.

    환경변수 우선순위:
      1) BACKEND_EVENTS_URL           (권장: http://api:8010/v1/events)
      2) BACKEND_NOTIFY_URL           (이전 호환)
      3) V1_API_BASE + V1_API_PREFIX + "/events"
         (예: http://api:8010 + /v1 + /events => http://api:8010/v1/events)

    인증:
      - WORKER_TOKEN 또는 EVENTS_WORKER_TOKEN 이 있으면 X-Worker-Token 헤더로 전송
    """

    backend_events_url: str
    worker_token: str = ""
    timeout_connect_s: float = 3.0
    timeout_read_s: float = 15.0
    retries: int = 2
    backoff_s: float = 0.6  # 재시도 간격(지수 백오프 기본값)

    @staticmethod
    def from_env() -> "EventsClientConfig":
        api_base = os.getenv("V1_API_BASE", "http://api:8010").rstrip("/")
        api_prefix = os.getenv("V1_API_PREFIX", "/v1").rstrip("/")

        url = _coalesce(
            os.getenv("BACKEND_EVENTS_URL"),
            os.getenv("BACKEND_NOTIFY_URL"),  # legacy
        )
        if not url:
            url = f"{api_base}{api_prefix}/events"

        token = _coalesce(os.getenv("WORKER_TOKEN"), os.getenv("EVENTS_WORKER_TOKEN")) or ""
        timeout_connect = float(os.getenv("EVENTS_TIMEOUT_CONNECT", "3"))
        timeout_read = float(os.getenv("EVENTS_TIMEOUT_READ", "15"))
        retries = int(os.getenv("EVENTS_RETRIES", "2"))
        backoff = float(os.getenv("EVENTS_BACKOFF_S", "0.6"))

        return EventsClientConfig(
            backend_events_url=url,
            worker_token=token,
            timeout_connect_s=timeout_connect,
            timeout_read_s=timeout_read,
            retries=retries,
            backoff_s=backoff,
        )


class EventsClient:
    def __init__(self, cfg: Optional[EventsClientConfig] = None) -> None:
        self.cfg = cfg or EventsClientConfig.from_env()

    # ---------------------------
    # Public API (외부 통신 camelCase)
    # ---------------------------
    def emit_event(
        self,
        *,
        event_type: EventType,
        run_id: str,
        round: Optional[int] = None,
        status: str = "DONE",
        pass_count: Optional[int] = None,
        fail_count: Optional[int] = None,
        miss_count: Optional[int] = None,
        share_root: Optional[str] = None,
        export_rel_path: Optional[str] = None,
        manifest_rel_path: Optional[str] = None,
        message: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        백엔드 /v1/events 로 이벤트 POST

        반환 형태:
          {
            "status": "OK" | "FAILED",
            "httpStatus": int | None,
            "response": str | None,
            "error": str | None,
            "url": str
          }
        """
        payload: Dict[str, Any] = {
            "eventType": event_type,
            "runId": run_id,
            "status": status,
            "timestamp": timestamp or _now_iso_z(),
        }

        if round is not None:
            payload["round"] = int(round)

        if pass_count is not None:
            payload["passCount"] = int(pass_count)
        if fail_count is not None:
            payload["failCount"] = int(fail_count)
        if miss_count is not None:
            payload["missCount"] = int(miss_count)

        # round0 export/manifest용
        if share_root is not None:
            payload["shareRoot"] = str(share_root)
        if export_rel_path is not None:
            payload["exportRelPath"] = str(export_rel_path)
        if manifest_rel_path is not None:
            payload["manifestRelPath"] = str(manifest_rel_path)

        if message is not None:
            payload["message"] = str(message)
        if job_id is not None:
            payload["jobId"] = str(job_id)
        payload["extra"] = extra or {}

        return self._post(payload)

    # ---------------------------
    # Convenience helpers
    # ---------------------------
    def round0_exported(
        self,
        *,
        run_id: str,
        pass_count: int,
        share_root: str,
        export_rel_path: str,
        message: str = "PASS exported",
        extra: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.emit_event(
            event_type="ROUND0_EXPORTED",
            run_id=run_id,
            round=0,
            status="DONE",
            pass_count=pass_count,
            fail_count=0,
            miss_count=0,
            share_root=share_root,
            export_rel_path=export_rel_path,
            message=message,
            extra=extra,
            job_id=job_id,
        )

    def round_result(
        self,
        *,
        run_id: str,
        round: int,
        pass_count: int,
        fail_count: int,
        miss_count: int = 0,
        message: str = "round summary",
        extra: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.emit_event(
            event_type="ROUND_RESULT",
            run_id=run_id,
            round=round,
            status="DONE",
            pass_count=pass_count,
            fail_count=fail_count,
            miss_count=miss_count,
            message=message,
            extra=extra,
            job_id=job_id,
        )

    def loop_finished(
        self,
        *,
        run_id: str,
        pass_count: Optional[int] = None,
        fail_count: Optional[int] = None,
        miss_count: Optional[int] = None,
        message: str = "loop finished",
        extra: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.emit_event(
            event_type="LOOP_FINISHED",
            run_id=run_id,
            status="DONE",
            pass_count=pass_count,
            fail_count=fail_count,
            miss_count=miss_count,
            message=message,
            extra=extra,
            job_id=job_id,
        )

    def loop_failed(
        self,
        *,
        run_id: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.emit_event(
            event_type="LOOP_FAILED",
            run_id=run_id,
            status="FAILED",
            message=message,
            extra=extra,
            job_id=job_id,
        )

    # ---------------------------
    # Internal (snake_case)
    # ---------------------------
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.cfg.backend_events_url.strip()
        if not url:
            return {"status": "FAILED", "httpStatus": None, "response": None, "error": "backend_events_url is empty", "url": ""}

        headers = {"Content-Type": "application/json"}
        if self.cfg.worker_token:
            headers["X-Worker-Token"] = self.cfg.worker_token

        timeout = (self.cfg.timeout_connect_s, self.cfg.timeout_read_s)

        last_err: Optional[str] = None
        last_status: Optional[int] = None
        last_resp: Optional[str] = None

        attempts = max(1, int(self.cfg.retries) + 1)
        for i in range(attempts):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=timeout)
                last_status = r.status_code
                last_resp = (r.text or "")[:1000]

                if r.ok:
                    return {"status": "OK", "httpStatus": last_status, "response": last_resp, "error": None, "url": url}

                # 4xx는 재시도해도 의미 없는 경우가 많아서 즉시 반환
                if 400 <= r.status_code < 500:
                    return {"status": "FAILED", "httpStatus": last_status, "response": last_resp, "error": "client_error", "url": url}

                last_err = f"server_error_http_{r.status_code}"
            except Exception as e:
                last_err = str(e)

            # retry
            if i < attempts - 1:
                time.sleep(self.cfg.backoff_s * (2**i))

        return {"status": "FAILED", "httpStatus": last_status, "response": last_resp, "error": last_err or "unknown_error", "url": url}
