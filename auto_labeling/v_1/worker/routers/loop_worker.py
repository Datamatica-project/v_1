from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Body
from pathlib import Path
import os
import time
import json
import subprocess
from typing import Dict, Any, Optional

from auto_labeling.v_1.api.dto.loop import RunLoopRequest, RunLoopResponse, JobStatusResponse

# ✅ Worker: /v1/loop 하위로 고정 (게이트웨이 호출 규약과 동일)
router = APIRouter(prefix="/api/v1/loop")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
V1_ROOT = PROJECT_ROOT / "auto_labeling" / "v_1"

DEFAULT_CFG_PATH = str(V1_ROOT / "configs" / "v1_loop.yaml")
DEFAULT_BASE_MODEL = str(V1_ROOT / "models" / "pretrained" / "yolov8n.pt")

LATEST_STUDENT = V1_ROOT / "models" / "user_yolo" / "weights" / "student_h10_init_latest.pt"
SCRIPT_PATH = V1_ROOT / "scripts" / "test_main.py"

DEFAULT_GT_EPOCHS = int(os.getenv("DEFAULT_GT_EPOCHS", "30"))
DEFAULT_GT_IMGSZ = int(os.getenv("DEFAULT_GT_IMGSZ", "640"))
DEFAULT_GT_BATCH = int(os.getenv("DEFAULT_GT_BATCH", "8"))

DEMO_ALWAYS_TRAIN_GT = os.getenv("DEMO_ALWAYS_TRAIN_GT", "true").strip().lower() in ("1", "true", "yes", "y")

DEFAULT_JOB_DIR = V1_ROOT / "logs" / "jobs"
JOB_DIR = Path(os.getenv("WORKER_JOB_DIR", str(DEFAULT_JOB_DIR)))
JOB_DIR.mkdir(parents=True, exist_ok=True)


def _new_job_id() -> str:
    # 기존 규칙 유지 (job_ prefix 포함)
    return time.strftime("job_%Y%m%d_%H%M%S") + f"_{os.getpid()}"


def _job_path(job_id: str) -> Path:
    # job_*.json 형태로 저장
    return JOB_DIR / f"{job_id}.json"


def _write_job(job_id: str, payload: Dict[str, Any]) -> None:
    payload["job_id"] = job_id
    payload["updated_at"] = time.time()
    _job_path(job_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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


def _run_loop_subprocess(job_id: str, req: RunLoopRequest) -> None:
    _write_job(job_id, {"status": "RUNNING"})

    if not SCRIPT_PATH.exists():
        _write_job(job_id, {
            "status": "FAILED",
            "error": f"[WORKER] test_main.py not found: {SCRIPT_PATH}",
        })
        return

    cmd: list[str] = ["python", str(SCRIPT_PATH)]

    cfg = (getattr(req, "cfg_path", "") or "").strip()
    cfg_path = cfg if cfg else DEFAULT_CFG_PATH
    cmd += ["--cfg", cfg_path]

    # teacher는 현재 test_main.py에서 안 쓰는 것으로 보이므로 resolved에만 남김
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
                "defaults": {
                    "defaultCfgPath": DEFAULT_CFG_PATH,
                    "defaultBaseModel": DEFAULT_BASE_MODEL,
                },
            })
            return

        cmd += ["--base_model", str(base_model_path)]

        gt_epochs = int(getattr(req, "gt_epochs", 0) or DEFAULT_GT_EPOCHS)
        gt_imgsz = int(getattr(req, "gt_imgsz", 0) or DEFAULT_GT_IMGSZ)
        gt_batch = int(getattr(req, "gt_batch", 0) or DEFAULT_GT_BATCH)

        cmd += ["--gt_epochs", str(gt_epochs)]
        cmd += ["--gt_imgsz", str(gt_imgsz)]
        cmd += ["--gt_batch", str(gt_batch)]

        gt_dir = os.getenv("GT_DIR", str(V1_ROOT / "data" / "GT"))
        unlabels_dir = os.getenv("UNLABELED_DIR", str(V1_ROOT / "data" / "unlabeled" / "images"))
        cmd += ["--gt_dir", gt_dir]
        cmd += ["--unlabels_dir", unlabels_dir]

        used_student = ""
    else:
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
                    "defaults": {
                        "defaultCfgPath": DEFAULT_CFG_PATH,
                        "defaultBaseModel": DEFAULT_BASE_MODEL,
                    },
                })
                return

            cmd += ["--base_model", str(base_model_path)]

            gt_epochs = int(getattr(req, "gt_epochs", 0) or DEFAULT_GT_EPOCHS)
            gt_imgsz = int(getattr(req, "gt_imgsz", 0) or DEFAULT_GT_IMGSZ)
            gt_batch = int(getattr(req, "gt_batch", 0) or DEFAULT_GT_BATCH)

            cmd += ["--gt_epochs", str(gt_epochs)]
            cmd += ["--gt_imgsz", str(gt_imgsz)]
            cmd += ["--gt_batch", str(gt_batch)]

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
        _write_job(job_id, {
            "status": "DONE",
            "stdout": (proc.stdout or "")[-5000:],
            "stderr": (proc.stderr or "")[-5000:],
            "cmd": cmd,
            "latest_exists": LATEST_STUDENT.exists(),
        })
    except subprocess.CalledProcessError as e:
        _write_job(job_id, {
            "status": "FAILED",
            "error": str(e),
            "stdout": (e.stdout or "")[-5000:],
            "stderr": (e.stderr or "")[-5000:],
            "cmd": cmd,
            "latest_exists": LATEST_STUDENT.exists(),
        })
    except Exception as e:
        _write_job(job_id, {
            "status": "FAILED",
            "error": str(e),
            "cmd": cmd,
            "latest_exists": LATEST_STUDENT.exists(),
        })


@router.post("/api/run", response_model=RunLoopResponse)
def run_loop(
    bg: BackgroundTasks,
    req: RunLoopRequest = Body(default_factory=RunLoopRequest),  # {} 허용
):
    job_id = _new_job_id()
    _write_job(job_id, {"status": "QUEUED"})
    bg.add_task(_run_loop_subprocess, job_id, req)

    # DTO 호환을 위해 snake/camel 둘 다 만족시키는 형태로 반환
    return RunLoopResponse(job_id=job_id, status="QUEUED", jobId=job_id)


# ✅ Gateway가 호출하는 worker endpoint: GET /v1/loop/status/{job_id}
@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    d = _read_job(job_id)
    return JobStatusResponse(job_id=job_id, status=d.get("status", "UNKNOWN"), stats=d, jobId=job_id)
