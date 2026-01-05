# auto_labeling/v_1/worker/export.py
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException
from pathlib import Path
import os
import time
import requests
from typing import List, Optional, Dict, Any

from auto_labeling.v_1.src.export_pass_fail_final import export_pass_fail_final

router = APIRouter(prefix="/api/v1", tags=["worker-export"])

# -----------------------------------------------------------------------------
# Paths
# - 파일 위치 기준으로 v_1 루트를 고정해서, 컨테이너 mount/PROJECT_ROOT에 덜 흔들리게 함
# - 필요 시 PROJECT_ROOT/V1_ROOT env로 override 가능
# -----------------------------------------------------------------------------
DEFAULT_V1_ROOT = Path(__file__).resolve().parents[1]  # .../auto_labeling/v_1
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(DEFAULT_V1_ROOT.parents[1])))  # best-effort
V1_ROOT = Path(os.getenv("V1_ROOT", str(DEFAULT_V1_ROOT)))

API_BASE_URL = os.getenv("API_BASE_URL", "http://v1-api:8010").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "10.0"))

# (선택) NAS share root를 registry에 같이 남기고 싶으면 설정
V1_EXPORT_SHARE_ROOT = os.getenv("V1_EXPORT_SHARE_ROOT", "").strip() or None


def _now_iso_no_ms_local() -> str:
    # worker 이벤트용 timestamp(로컬) - ms 제거
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=API_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"ok": True}


def _emit_event(
    run_id: str,
    job_id: str,
    event_type: str,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "resultCode": "SUCCESS",
        "message": "event",
        "data": {
            "eventType": event_type,
            "runId": run_id,
            "jobId": job_id,
            "status": status,
            "timestamp": _now_iso_no_ms_local(),
            "extra": extra or {},
        },
    }
    return _post_json(f"{API_BASE_URL}/api/v1/events", payload)


def _notify_final_export(
    *,
    run_id: str,
    status: str,
    message: str,
    export_rel_path: str,
    manifest_rel_path: Optional[str],
    counts: Optional[Dict[str, int]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    API 서버의 레지스트리 upsert endpoint 호출.
    - createdAt/updatedAt은 API 서버가 통일 포맷(UTC Z, ms 제거)으로 부여하므로 worker에서 보내지 않음.
    """
    payload: Dict[str, Any] = {
        "runId": run_id,
        "status": status,
        "message": message,
        "exportRelPath": export_rel_path,
        "manifestRelPath": manifest_rel_path,
        "extra": extra or {},
    }
    if V1_EXPORT_SHARE_ROOT:
        payload["shareRoot"] = V1_EXPORT_SHARE_ROOT

    c = counts or {}
    payload["passCount"] = int(c.get("passCount", 0) or 0)
    payload["passFailCount"] = int(c.get("passFailCount", 0) or 0)
    payload["failFailCount"] = int(c.get("failFailCount", 0) or 0)
    payload["missCount"] = int(c.get("missCount", 0) or 0)

    return _post_json(f"{API_BASE_URL}/api/v1/export/final/notify", payload)


@router.post("/export/final", summary="최종 Export 수행(Worker)")
def export_final(
    run_id: str = Body(..., embed=True),
    job_id: str = Body("", embed=True),
    export_root: str = Body("", embed=True),
    conf_th: float = Body(0.3, embed=True),
    device: str = Body("0", embed=True),
):
    """
    - 실제 export 결과는 out_root 아래에 생성한다.
    - export_root 미지정 시: {V1_ROOT}/exports/{run_id}/final 로 통일
      (API notify의 exportRelPath와 1:1로 맞추기 위함)
    - 시작 즉시 status=RUNNING notify로 "스텁 레지스트리"를 선생성한다(404 방지)
    - 완료/실패 시 DONE/FAILED notify로 갱신한다.
    """
    run_id = (run_id or "").strip()
    if not run_id:
        raise HTTPException(400, "run_id is required")

    # 1) out_root 결정 (exports로 통일)
    if export_root:
        out_root = Path(export_root)
        out_root = out_root if out_root.is_absolute() else (V1_ROOT / out_root)
    else:
        out_root = V1_ROOT / "exports" / run_id / "final"
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        rel_from_v1 = out_root.relative_to(V1_ROOT).as_posix()
        export_rel_path = rel_from_v1
    except Exception:
        export_rel_path = f"exports/{run_id}/final"
    manifest_rel_path: Optional[str] = f"{export_rel_path}/manifest.json"

    w = os.getenv("FINAL_EXPORT_STUDENT_WEIGHTS", "").strip()
    student_weights = Path(w) if w else (V1_ROOT / "models" / "user_yolo" / "weights" / "student_h10_init_latest.pt")
    if not student_weights.exists():
        try:
            _notify_final_export(
                run_id=run_id,
                status="FAILED",
                message=f"student weights not found: {student_weights}",
                export_rel_path=export_rel_path,
                manifest_rel_path=None,
                counts=None,
                extra={"outRoot": str(out_root), "studentWeights": str(student_weights), "v1Root": str(V1_ROOT)},
            )
        except Exception:
            pass
        _emit_event(
            run_id=run_id,
            job_id=job_id or "",
            event_type="FINAL_EXPORTED",
            status="FAILED",
            extra={"reason": "student weights missing", "studentWeights": str(student_weights)},
        )
        raise HTTPException(500, f"student weights not found: {student_weights}")

    try:
        _notify_final_export(
            run_id=run_id,
            status="RUNNING",
            message="final export started",
            export_rel_path=export_rel_path,
            manifest_rel_path=manifest_rel_path,
            counts={"passCount": 0, "passFailCount": 0, "failFailCount": 0, "missCount": 0},
            extra={
                "outRoot": str(out_root),
                "studentWeights": str(student_weights),
                "v1Root": str(V1_ROOT),
                "device": device,
                "confTh": float(conf_th),
                "jobId": job_id or "",
            },
        )
    except Exception:
        # 레지스트리 기록 실패가 export 자체를 막지는 않도록
        pass

    _emit_event(
        run_id=run_id,
        job_id=job_id or "",
        event_type="FINAL_EXPORTING",
        status="RUNNING",
        extra={"exportRelPath": export_rel_path},
    )

    # 6) PASS/FAIL/MISS source dir (파이프라인에 맞춰 조절)
    pass_img_dirs: List[Path] = [
        V1_ROOT / "data" / "pass" / "images",
        V1_ROOT / "data" / "round_r0" / "pass" / "images",
    ]
    pass_fail_img_dir = V1_ROOT / "data" / "pass_fail" / "images"
    fail_fail_img_dir = V1_ROOT / "data" / "fail_fail" / "images"
    miss_img_dir = V1_ROOT / "data" / "miss" / "images"

    started = time.time()

    # 7) 실제 export 수행 (+ DONE/FAILED notify)
    try:
        export_pass_fail_final(
            student_weights=student_weights,
            out_root=out_root,
            pass_img_dirs=pass_img_dirs,
            pass_fail_img_dir=pass_fail_img_dir,
            fail_fail_img_dir=fail_fail_img_dir,
            miss_img_dir=miss_img_dir,
            device=device,
            conf_th=conf_th,
        )

        # DONE 직전에 manifest 존재 여부 재계산
        manifest_path = out_root / "manifest.json"
        manifest_rel_path_done = f"{export_rel_path}/manifest.json" if manifest_path.exists() else None

        counts = {"passCount": 0, "passFailCount": 0, "failFailCount": 0, "missCount": 0}

        notify_resp = _notify_final_export(
            run_id=run_id,
            status="DONE",
            message="final export completed",
            export_rel_path=export_rel_path,
            manifest_rel_path=manifest_rel_path_done,
            counts=counts,
            extra={
                "outRoot": str(out_root),
                "studentWeights": str(student_weights),
                "v1Root": str(V1_ROOT),
                "durationSec": round(time.time() - started, 3),
            },
        )

        _emit_event(
            run_id=run_id,
            job_id=job_id or "",
            event_type="FINAL_EXPORTED",
            status="DONE",
            extra={"exportRelPath": export_rel_path, "manifestRelPath": manifest_rel_path_done},
        )

        return {
            "runId": run_id,
            "jobId": job_id,
            "outRoot": str(out_root),
            "exportRelPath": export_rel_path,
            "manifestRelPath": manifest_rel_path_done,
            "notify": notify_resp,
        }

    except Exception as e:
        # FAILED 직전에 manifest 존재 여부 재계산
        manifest_path = out_root / "manifest.json"
        manifest_rel_path_fail = f"{export_rel_path}/manifest.json" if manifest_path.exists() else None

        try:
            _notify_final_export(
                run_id=run_id,
                status="FAILED",
                message=str(e),
                export_rel_path=export_rel_path,
                manifest_rel_path=manifest_rel_path_fail,
                counts=None,
                extra={
                    "outRoot": str(out_root),
                    "studentWeights": str(student_weights),
                    "v1Root": str(V1_ROOT),
                    "durationSec": round(time.time() - started, 3),
                },
            )
        except Exception:
            pass

        _emit_event(
            run_id=run_id,
            job_id=job_id or "",
            event_type="FINAL_EXPORTED",
            status="FAILED",
            extra={"exportRelPath": export_rel_path, "error": str(e)},
        )
        raise
