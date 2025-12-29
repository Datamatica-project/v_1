from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Body
from pathlib import Path
import os
import time
import json
import subprocess
from typing import Dict, Any, Optional

import requests  # ✅ 추가

from auto_labeling.v_1.api.dto.loop import RunLoopRequest, RunLoopResponse, JobStatusResponse

router = APIRouter(prefix="/api/v1/loop")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
V1_ROOT = PROJECT_ROOT / "auto_labeling" / "v_1"

DEFAULT_CFG_PATH = str(V1_ROOT / "configs" / "v1_loop.yaml")
DEFAULT_BASE_MODEL = str(V1_ROOT / "models" / "pretrained" / "weights" / "best.pt")

LATEST_STUDENT = V1_ROOT / "models" / "user_yolo" / "weights" / "student_h10_init_latest.pt"
SCRIPT_PATH = V1_ROOT / "scripts" / "test_main.py"

DEFAULT_GT_EPOCHS = int(os.getenv("DEFAULT_GT_EPOCHS", "30"))
DEFAULT_GT_IMGSZ = int(os.getenv("DEFAULT_GT_IMGSZ", "640"))
DEFAULT_GT_BATCH = int(os.getenv("DEFAULT_GT_BATCH", "8"))

DEMO_ALWAYS_TRAIN_GT = os.getenv("DEMO_ALWAYS_TRAIN_GT", "true").strip().lower() in ("1", "true", "yes", "y")

DEFAULT_JOB_DIR = V1_ROOT / "logs" / "jobs"
JOB_DIR = Path(os.getenv("WORKER_JOB_DIR", str(DEFAULT_JOB_DIR)))
JOB_DIR.mkdir(parents=True, exist_ok=True)

# ✅ API 주소 (worker -> api 콜)
API_BASE_URL = os.getenv("API_BASE_URL", "http://v1-api:8010").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "10.0"))

# ✅ 자동 후처리 on/off
AUTO_EXPORT_ROUND0 = os.getenv("AUTO_EXPORT_ROUND0", "true").strip().lower() in ("1", "true", "yes", "y")
AUTO_EMIT_EVENTS = os.getenv("AUTO_EMIT_EVENTS", "true").strip().lower() in ("1", "true", "yes", "y")


def _new_job_id() -> str:
    return time.strftime("job_%Y%m%d_%H%M%S") + f"_{os.getpid()}"


def _job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _write_job(job_id: str, payload: Dict[str, Any]) -> None:
    payload["job_id"] = job_id
    payload["updated_at"] = time.time()
    _job_path(job_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_job(job_id: str) -> Dict[str, Any]:
    p = _job_path(job_id)
    if not p.exists():
        return {"job_id": job_id, "status": "NOT_FOUND"}
    return json.loads(p.read_text(encoding="utf-8"))


def _norm_path(p: Optional[str]) -> Optional[Path]:
    s = (p or "").strip()
    if not s:
        return None
    pp = Path(s)
    if not pp.is_absolute():
        pp = PROJECT_ROOT / pp
    return pp


def _post_json(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, timeout=API_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"ok": True}


def _notify_round0_export(*, run_id: str, job_id: str, extra: dict | None = None) -> dict:
    """
    ✅ worker가 export0 완료를 api에 알림
    - 현재는 'notify'만 보내는 최소 버전
    - 실제 zip/manifest 생성은 export_round0_worker 쪽에서 하거나,
      추후 서비스로 분리해서 여기서 실행하도록 확장 가능
    """
    payload = {
        "eventType": "ROUND0_EXPORTED",
        "runId": run_id,
        "jobId": job_id,
        "status": "DONE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "message": "round0 export completed (auto)",
        "extra": extra or {},
    }

    # 1) export notify (API 저장용)
    notify_url = f"{API_BASE_URL}/api/v1/export/round0/notify"
    notify_resp = _post_json(notify_url, payload)

    return {"notify": notify_resp}


def _emit_event(*, event_type: str, run_id: str, job_id: str, status: str, extra: dict | None = None) -> dict:
    payload = {
        "eventType": event_type,
        "runId": run_id,
        "jobId": job_id,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "extra": extra or {},
    }
    url = f"{API_BASE_URL}/api/v1/events"
    return _post_json(url, payload)


def _run_loop_subprocess(job_id: str, req: RunLoopRequest) -> None:
    _write_job(job_id, {"status": "RUNNING"})

    if not SCRIPT_PATH.exists():
        _write_job(job_id, {"status": "FAILED", "error": f"[WORKER] test_main.py not found: {SCRIPT_PATH}"})
        return

    cmd: list[str] = ["python", str(SCRIPT_PATH)]

    cfg = (getattr(req, "cfg_path", "") or "").strip()
    cfg_path = cfg if cfg else DEFAULT_CFG_PATH
    cmd += ["--cfg", cfg_path]

    teacher_path = _norm_path(getattr(req, "teacher_weight", None))

    export_root = (getattr(req, "export_root", "") or "").strip()
    if export_root:
        cmd += ["--export_root", export_root]

    used_mode = "demo_always_train_gt" if DEMO_ALWAYS_TRAIN_GT else "student_policy"

    if DEMO_ALWAYS_TRAIN_GT:
        cmd += ["--train_gt"]

        base_model_path = _norm_path(getattr(req, "base_model", None)) or Path(DEFAULT_BASE_MODEL)
        if not base_model_path.is_absolute():
            base_model_path = PROJECT_ROOT / base_model_path

        if not base_model_path.exists():
            _write_job(job_id, {
                "status": "FAILED",
                "error": f"[WORKER] baseModel not found: {base_model_path}",
                "cmd": cmd,
                "defaults": {"defaultCfgPath": DEFAULT_CFG_PATH, "defaultBaseModel": DEFAULT_BASE_MODEL},
            })
            return

        cmd += ["--base_model", str(base_model_path)]

        gt_epochs = int(getattr(req, "gt_epochs", 0) or DEFAULT_GT_EPOCHS)
        gt_imgsz = int(getattr(req, "gt_imgsz", 0) or DEFAULT_GT_IMGSZ)
        gt_batch = int(getattr(req, "gt_batch", 0) or DEFAULT_GT_BATCH)

        cmd += ["--gt_epochs", str(gt_epochs), "--gt_imgsz", str(gt_imgsz), "--gt_batch", str(gt_batch)]

        gt_dir = os.getenv("GT_DIR", str(V1_ROOT / "data" / "GT"))
        unlabels_dir = os.getenv("UNLABELED_DIR", str(V1_ROOT / "data" / "unlabeled" / "images"))
        cmd += ["--gt_dir", gt_dir, "--unlabels_dir", unlabels_dir]

        used_student = ""
    else:
        # (기존 로직 유지)
        student_path = _norm_path(getattr(req, "student_weight", None))
        if student_path and student_path.exists():
            cmd += ["--student", str(student_path)]
            used_student = str(student_path)
            used_mode = "student_explicit"
        elif LATEST_STUDENT.exists():
            cmd += ["--student", str(LATEST_STUDENT)]
            used_student = str(LATEST_STUDENT)
            used_mode = "student_latest"
        else:
            cmd += ["--train_gt"]
            used_mode = "train_gt_first"
            base_model_path = _norm_path(getattr(req, "base_model", None)) or Path(DEFAULT_BASE_MODEL)
            if not base_model_path.is_absolute():
                base_model_path = PROJECT_ROOT / base_model_path
            if not base_model_path.exists():
                _write_job(job_id, {
                    "status": "FAILED",
                    "error": f"[WORKER] baseModel not found: {base_model_path}",
                    "cmd": cmd,
                    "defaults": {"defaultCfgPath": DEFAULT_CFG_PATH, "defaultBaseModel": DEFAULT_BASE_MODEL},
                })
                return
            cmd += ["--base_model", str(base_model_path)]
            gt_epochs = int(getattr(req, "gt_epochs", 0) or DEFAULT_GT_EPOCHS)
            gt_imgsz = int(getattr(req, "gt_imgsz", 0) or DEFAULT_GT_IMGSZ)
            gt_batch = int(getattr(req, "gt_batch", 0) or DEFAULT_GT_BATCH)
            cmd += ["--gt_epochs", str(gt_epochs), "--gt_imgsz", str(gt_imgsz), "--gt_batch", str(gt_batch)]
            used_student = ""

    _write_job(job_id, {
        "status": "RUNNING",
        "mode": used_mode,
        "cmd": cmd,
        "resolved": {
            "cfg": cfg_path,
            "student": used_student,
            "teacher": str(teacher_path) if teacher_path else "",
            "export_root": export_root,
            "latest_path": str(LATEST_STUDENT),
            "default_cfg": DEFAULT_CFG_PATH,
            "default_base_model": DEFAULT_BASE_MODEL,
            "demo_always_train_gt": DEMO_ALWAYS_TRAIN_GT,
            "api_base_url": API_BASE_URL,
            "auto_export_round0": AUTO_EXPORT_ROUND0,
            "auto_emit_events": AUTO_EMIT_EVENTS,
        },
    })

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )

        # ✅ 여기서 "자동 export0 + event" 실행
        post_actions: dict[str, Any] = {}
        post_errors: list[str] = []

        if AUTO_EXPORT_ROUND0:
            try:
                post_actions["export_round0"] = _notify_round0_export(
                    run_id=job_id,  # run_id를 job_id로 쓰는 현재 규칙 그대로
                    job_id=job_id,
                    extra={
                        "passDir": str(V1_ROOT / "data" / "pass"),
                        "failDir": str(V1_ROOT / "data" / "fail"),
                        "exportRoot": export_root,
                    },
                )
            except Exception as e:
                post_errors.append(f"export_round0_failed: {e}")

        if AUTO_EMIT_EVENTS:
            try:
                post_actions["event_loop_finished"] = _emit_event(
                    event_type="LOOP_FINISHED",
                    run_id=job_id,
                    job_id=job_id,
                    status="DONE",
                    extra={"mode": used_mode, "postErrors": post_errors},
                )
            except Exception as e:
                post_errors.append(f"emit_event_failed: {e}")

        _write_job(job_id, {
            "status": "DONE",
            "stdout": (proc.stdout or "")[-5000:],
            "stderr": (proc.stderr or "")[-5000:],
            "cmd": cmd,
            "latest_exists": LATEST_STUDENT.exists(),
            "post_actions": post_actions,
            "post_errors": post_errors,
        })

    except subprocess.CalledProcessError as e:
        # ✅ 실패 이벤트(옵션)
        if AUTO_EMIT_EVENTS:
            try:
                _emit_event(
                    event_type="LOOP_FAILED",
                    run_id=job_id,
                    job_id=job_id,
                    status="FAILED",
                    extra={"error": str(e)},
                )
            except Exception:
                pass

        _write_job(job_id, {
            "status": "FAILED",
            "error": str(e),
            "stdout": (e.stdout or "")[-5000:],
            "stderr": (e.stderr or "")[-5000:],
            "cmd": cmd,
            "latest_exists": LATEST_STUDENT.exists(),
        })
    except Exception as e:
        if AUTO_EMIT_EVENTS:
            try:
                _emit_event(
                    event_type="LOOP_FAILED",
                    run_id=job_id,
                    job_id=job_id,
                    status="FAILED",
                    extra={"error": str(e)},
                )
            except Exception:
                pass

        _write_job(job_id, {
            "status": "FAILED",
            "error": str(e),
            "cmd": cmd,
            "latest_exists": LATEST_STUDENT.exists(),
        })


@router.post("/run", response_model=RunLoopResponse)
def run_loop(
    bg: BackgroundTasks,
    req: RunLoopRequest = Body(default_factory=RunLoopRequest),
):
    job_id = _new_job_id()
    _write_job(job_id, {"status": "QUEUED"})
    bg.add_task(_run_loop_subprocess, job_id, req)
    return RunLoopResponse(job_id=job_id, status="QUEUED", jobId=job_id)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    d = _read_job(job_id)
    return JobStatusResponse(job_id=job_id, status=d.get("status", "UNKNOWN"), stats=d, jobId=job_id)
