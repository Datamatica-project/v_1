# auto_labeling/v_1/worker/routers/loop_worker.py
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Body
from pathlib import Path
import os
import time
import json
import subprocess
from typing import Dict, Any, Optional, Tuple, List

import requests

from auto_labeling.v_1.api.dto.loop import RunLoopRequest, RunLoopResponse, JobStatusResponse

router = APIRouter(prefix="/api/v1/loop")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace"))
V1_ROOT = PROJECT_ROOT / "auto_labeling" / "v_1"

DEFAULT_CFG_PATH = str(V1_ROOT / "configs" / "v1_loop_real.yaml")
DEFAULT_BASE_MODEL = str(V1_ROOT / "models" / "pretrained" / "weights" / "best.pt")

LATEST_STUDENT = V1_ROOT / "models" / "user_yolo" / "weights" / "student_h10_init_latest.pt"
SCRIPT_PATH = V1_ROOT / "scripts" / "test_main.py"

DEFAULT_GT_EPOCHS = int(os.getenv("DEFAULT_GT_EPOCHS", "1"))
DEFAULT_GT_IMGSZ = int(os.getenv("DEFAULT_GT_IMGSZ", "640"))
DEFAULT_GT_BATCH = int(os.getenv("DEFAULT_GT_BATCH", "8"))

DEMO_ALWAYS_TRAIN_GT = os.getenv("DEMO_ALWAYS_TRAIN_GT", "true").strip().lower() in ("1", "true", "yes", "y")

DEFAULT_JOB_DIR = V1_ROOT / "logs" / "jobs"
JOB_DIR = Path(os.getenv("WORKER_JOB_DIR", str(DEFAULT_JOB_DIR)))
JOB_DIR.mkdir(parents=True, exist_ok=True)

API_BASE_URL = os.getenv("API_BASE_URL", "http://v1-api:8010").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "10.0"))

AUTO_EXPORT_ROUND0 = os.getenv("AUTO_EXPORT_ROUND0", "true").strip().lower() in ("1", "true", "yes", "y")
AUTO_EMIT_EVENTS = os.getenv("AUTO_EMIT_EVENTS", "true").strip().lower() in ("1", "true", "yes", "y")

# test_main.py가 <export_root>/round_r0 로 export 한다는 가정
DEFAULT_EXPORT_ROOT = os.getenv("DEFAULT_EXPORT_ROOT", "auto_labeling/v_1/data").strip()

# stdout/stderr 저장 전략
STD_TAIL = int(os.getenv("WORKER_STD_TAIL", "5000"))
STD_HEAD = int(os.getenv("WORKER_STD_HEAD", "2000"))


# -----------------------------
# time helpers
# -----------------------------
def _now_iso_z_no_ms() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -----------------------------
# job helpers
# -----------------------------
def _new_job_id() -> str:
    return time.strftime("job_%Y%m%d_%H%M%S") + f"_{os.getpid()}"


def _new_run_id() -> str:
    ts = time.strftime("run_%Y%m%d_%H%M%S")
    suf = f"{os.getpid()}_{int(time.time() * 1000) % 100000:05d}"
    return f"{ts}_{suf}"


def _job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _read_job(job_id: str) -> Dict[str, Any]:
    p = _job_path(job_id)
    if not p.exists():
        return {"job_id": job_id, "status": "NOT_FOUND"}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_job(job_id: str, payload: Dict[str, Any]) -> None:
    """
    ✅ 중요: job 파일을 '덮어쓰기'가 아니라 '병합 업데이트'로 처리.
    - 여러 단계에서 _write_job을 여러 번 호출해도 이전 필드가 날아가지 않게 함.
    """
    cur = _read_job(job_id)
    if not isinstance(cur, dict):
        cur = {"job_id": job_id}

    merged = dict(cur)
    merged.update(payload)
    merged["job_id"] = job_id
    merged["updated_at"] = time.time()

    _job_path(job_id).write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_path(p: Optional[str]) -> Optional[Path]:
    s = (p or "").strip()
    if not s:
        return None
    pp = Path(s)
    if not pp.is_absolute():
        pp = PROJECT_ROOT / pp
    return pp


def _clip_head_tail(s: str, *, head: int = STD_HEAD, tail: int = STD_TAIL) -> Dict[str, str]:
    s = s or ""
    return {"head": s[:head], "tail": s[-tail:] if tail > 0 else ""}


# -----------------------------
# http helpers
# -----------------------------
def _post_json(url: str, payload: dict) -> dict:
    """
    422 등 validation 에러의 'detail'을 jobs에 남기기 위해
    raise_for_status() 대신 response.text를 포함해 예외를 던진다.
    """
    r = requests.post(url, json=payload, timeout=API_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {url} failed: {r.status_code} {r.text}")
    try:
        return r.json()
    except Exception:
        return {"ok": True}


# -----------------------------
# export path helpers
# -----------------------------
def _count_images(img_dir: Path) -> int:
    if not img_dir.exists():
        return 0
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    n = 0
    for e in exts:
        n += len(list(img_dir.glob(f"*{e}")))
    return n


def _sanitize_rel_path(p: str) -> str:
    s = (p or "").strip().replace("\\", "/")
    return s.rstrip("/").rstrip("\\")


def _build_round0_export_paths(export_root: str, run_id: str) -> Tuple[str, str, Path]:
    """
    ✅ A안 정책: export_root 기본은 data로 통일.
    - test_main.py는 out_round = export_root / "round_r0" 로 export 한다는 가정.

    round0 폴더:
      <export_root>/round_r0
    """
    export_root_s = _sanitize_rel_path(export_root)
    if not export_root_s:
        export_root_s = _sanitize_rel_path(DEFAULT_EXPORT_ROOT)

    export_rel_path = f"{export_root_s}/round_r0"
    export_abs_root = _norm_path(export_root_s) or (PROJECT_ROOT / export_root_s)
    export_abs_round0 = (export_abs_root / "round_r0").resolve()
    return export_root_s, export_rel_path, export_abs_round0


# -----------------------------
# notify / events
# -----------------------------
def _notify_round0_export(
    *,
    run_id: str,
    status: str,
    export_rel_path: str,
    pass_count: int,
    fail_count: int,
    miss_count: int,
    message: str | None = None,
    manifest_rel_path: str | None = None,
    extra: dict | None = None,
) -> Tuple[dict, dict, str]:
    """
    /api/v1/export/round0/notify 로 전송.
    반환: (response_json, request_payload, notify_url)
    """
    payload = {
        "runId": run_id,
        "round": 0,
        "status": status,  # "DONE" | "FAILED" | "READY"
        "message": message,
        "passCount": int(pass_count),
        "failCount": int(fail_count),
        "missCount": int(miss_count),
        "exportRelPath": export_rel_path,
        "manifestRelPath": manifest_rel_path,
        "createdAt": _now_iso_z_no_ms(),
        "extra": extra or {"countUnit": "image"},
    }
    notify_url = f"{API_BASE_URL}/api/v1/export/round0/notify"
    resp = _post_json(notify_url, payload)
    return resp, payload, notify_url


def _emit_event(
    *,
    event_type: str,
    run_id: str,
    job_id: str,
    status: str,
    extra: dict | None = None,
) -> Tuple[dict, dict, str]:
    """
    /api/v1/events 로 전송.
    반환: (response_json, request_payload, url)
    """
    payload = {
        "eventType": event_type,  # "LOOP_DONE" | "LOOP_FAILED"
        "runId": run_id,
        "jobId": job_id,
        "message": status,
        "payload": {
            "status": status,
            "timestamp": _now_iso_z_no_ms(),
            "extra": extra or {},
        },
    }
    url = f"{API_BASE_URL}/api/v1/events"
    resp = _post_json(url, payload)
    return resp, payload, url


# -----------------------------
# round0 status helpers
# -----------------------------
def _set_round0_status(
    job_id: str,
    *,
    round0_status: str,
    export_rel_path: str | None = None,
    counts: dict | None = None,
    notify: dict | None = None,
    note: str | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "round0_status": round0_status,
        "round0_updated_at": _now_iso_z_no_ms(),
    }
    if export_rel_path is not None:
        payload["round0_export_rel_path"] = export_rel_path
    if counts is not None:
        payload["round0_counts"] = counts
    if notify is not None:
        payload["round0_notify"] = notify
    if note is not None:
        payload["round0_note"] = note
    _write_job(job_id, payload)


# -----------------------------
# main worker routine
# -----------------------------
def _run_loop_subprocess(job_id: str, run_id: str, req: RunLoopRequest) -> None:
    # ✅ round0 필드 기본값을 명시적으로 세팅
    _write_job(
        job_id,
        {
            "status": "RUNNING",
            "run_id": run_id,
            "round0_status": "RUNNING",
            "round0_updated_at": _now_iso_z_no_ms(),
        },
    )

    if not SCRIPT_PATH.exists():
        _set_round0_status(job_id, round0_status="FAILED", note="test_main.py missing")
        _write_job(
            job_id,
            {"status": "FAILED", "run_id": run_id, "error": f"[WORKER] test_main.py not found: {SCRIPT_PATH}"},
        )
        return

    cmd: List[str] = ["python", str(SCRIPT_PATH)]

    cfg = (getattr(req, "cfg_path", "") or "").strip()
    cfg_path = cfg if cfg else DEFAULT_CFG_PATH
    cmd += ["--cfg", cfg_path]

    teacher_weight = (getattr(req, "teacher_weight", "") or "").strip()
    teacher_path = _norm_path(teacher_weight) if teacher_weight else None

    export_root = (getattr(req, "export_root", "") or "").strip()
    if not export_root:
        export_root = DEFAULT_EXPORT_ROOT
    cmd += ["--export_root", export_root]

    used_mode = "demo_always_train_gt" if DEMO_ALWAYS_TRAIN_GT else "student_policy"

    # -----------------------------
    # build cmd: train_gt or student policy
    # -----------------------------
    if DEMO_ALWAYS_TRAIN_GT:
        cmd += ["--train_gt"]

        base_model_path = _norm_path(getattr(req, "base_model", None)) or Path(DEFAULT_BASE_MODEL)
        if not base_model_path.is_absolute():
            base_model_path = PROJECT_ROOT / base_model_path

        if not base_model_path.exists():
            _set_round0_status(job_id, round0_status="FAILED", note="baseModel not found")
            _write_job(
                job_id,
                {
                    "status": "FAILED",
                    "run_id": run_id,
                    "error": f"[WORKER] baseModel not found: {base_model_path}",
                    "cmd": cmd,
                    "defaults": {"defaultCfgPath": DEFAULT_CFG_PATH, "defaultBaseModel": DEFAULT_BASE_MODEL},
                },
            )
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
                _set_round0_status(job_id, round0_status="FAILED", note="baseModel not found")
                _write_job(
                    job_id,
                    {
                        "status": "FAILED",
                        "run_id": run_id,
                        "error": f"[WORKER] baseModel not found: {base_model_path}",
                        "cmd": cmd,
                        "defaults": {"defaultCfgPath": DEFAULT_CFG_PATH, "defaultBaseModel": DEFAULT_BASE_MODEL},
                    },
                )
                return

            cmd += ["--base_model", str(base_model_path)]
            gt_epochs = int(getattr(req, "gt_epochs", 0) or DEFAULT_GT_EPOCHS)
            gt_imgsz = int(getattr(req, "gt_imgsz", 0) or DEFAULT_GT_IMGSZ)
            gt_batch = int(getattr(req, "gt_batch", 0) or DEFAULT_GT_BATCH)
            cmd += ["--gt_epochs", str(gt_epochs), "--gt_imgsz", str(gt_imgsz), "--gt_batch", str(gt_batch)]
            used_student = ""

    # 기록
    _write_job(
        job_id,
        {
            "status": "RUNNING",
            "run_id": run_id,
            "mode": used_mode,
            "cmd": cmd,
            "resolved": {
                "cfg": cfg_path,
                "student": used_student,
                "teacher": str(teacher_path) if teacher_path else "",
                "export_root": export_root,
                "default_export_root": DEFAULT_EXPORT_ROOT,
                "latest_path": str(LATEST_STUDENT),
                "default_cfg": DEFAULT_CFG_PATH,
                "default_base_model": DEFAULT_BASE_MODEL,
                "demo_always_train_gt": DEMO_ALWAYS_TRAIN_GT,
                "api_base_url": API_BASE_URL,
                "auto_export_round0": AUTO_EXPORT_ROUND0,
                "auto_emit_events": AUTO_EMIT_EVENTS,
            },
        },
    )

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )

        post_actions: Dict[str, Any] = {}
        post_errors: List[str] = []

        # round0 export notify
        if AUTO_EXPORT_ROUND0:
            try:
                _export_root_s, export_rel_path, export_abs_round0 = _build_round0_export_paths(export_root, run_id)

                post_actions["round0_paths"] = {
                    "exportRoot": export_root,
                    "exportRelPath": export_rel_path,
                    "exportAbsRound0": str(export_abs_round0),
                    "exists": export_abs_round0.exists(),
                }

                if not export_abs_round0.exists():
                    extra = {
                        "exportRoot": export_root,
                        "exportAbsRound0": str(export_abs_round0),
                        "reason": "export_abs_round0_not_found",
                    }
                    resp, payload, url = _notify_round0_export(
                        run_id=run_id,
                        status="FAILED",
                        export_rel_path=export_rel_path,
                        pass_count=0,
                        fail_count=0,
                        miss_count=0,
                        message="round0 export folder missing",
                        manifest_rel_path=None,
                        extra=extra,
                    )
                    post_actions["export_round0_notify"] = resp
                    post_actions["export_round0_notify_request"] = payload
                    post_actions["export_round0_notify_url"] = url

                    _set_round0_status(
                        job_id,
                        round0_status="FAILED",
                        export_rel_path=export_rel_path,
                        counts={"pass": 0, "fail": 0, "miss": 0},
                        notify={"response": resp, "request": payload, "url": url},
                        note="round0 export folder missing",
                    )
                else:
                    pass_cnt = _count_images(export_abs_round0 / "pass" / "images")
                    fail_cnt = _count_images(export_abs_round0 / "fail" / "images")
                    miss_cnt = _count_images(export_abs_round0 / "miss" / "images")

                    resp, payload, url = _notify_round0_export(
                        run_id=run_id,
                        status="DONE",
                        export_rel_path=export_rel_path,
                        pass_count=pass_cnt,
                        fail_count=fail_cnt,
                        miss_count=miss_cnt,
                        message="round0 export completed (auto, data-root)",
                        manifest_rel_path=None,
                        extra={
                            "exportRoot": export_root,
                            "exportAbsRound0": str(export_abs_round0),
                            "passDir": str(export_abs_round0 / "pass"),
                            "failDir": str(export_abs_round0 / "fail"),
                            "missDir": str(export_abs_round0 / "miss"),
                        },
                    )
                    post_actions["export_round0_notify"] = resp
                    post_actions["export_round0_notify_request"] = payload
                    post_actions["export_round0_notify_url"] = url

                    _set_round0_status(
                        job_id,
                        round0_status="DONE",
                        export_rel_path=export_rel_path,
                        counts={"pass": int(pass_cnt), "fail": int(fail_cnt), "miss": int(miss_cnt)},
                        notify={"response": resp, "request": payload, "url": url},
                        note="round0 export notify done",
                    )

            except Exception as e:
                post_errors.append(f"export_round0_notify_failed: {e}")
                cur = _read_job(job_id)
                if str(cur.get("round0_status") or "").upper() != "DONE":
                    _set_round0_status(job_id, round0_status="FAILED", note=f"export_round0_notify_failed: {e}")

        # events
        if AUTO_EMIT_EVENTS:
            try:
                resp, payload, url = _emit_event(
                    event_type="LOOP_DONE",
                    run_id=run_id,
                    job_id=job_id,
                    status="DONE",
                    extra={"mode": used_mode, "postErrors": post_errors},
                )
                post_actions["event_loop_done"] = resp
                post_actions["event_loop_done_request"] = payload
                post_actions["event_loop_done_url"] = url
            except Exception as e:
                post_errors.append(f"emit_event_failed: {e}")

        _write_job(
            job_id,
            {
                "status": "DONE",
                "run_id": run_id,
                "stdout": _clip_head_tail(proc.stdout or ""),
                "stderr": _clip_head_tail(proc.stderr or ""),
                "cmd": cmd,
                "latest_exists": LATEST_STUDENT.exists(),
                "post_actions": post_actions,
                "post_errors": post_errors,
            },
        )

    except subprocess.CalledProcessError as e:
        post_errors: List[str] = []
        post_actions: Dict[str, Any] = {}

        cur = _read_job(job_id)
        if str(cur.get("round0_status") or "").upper() != "DONE":
            _set_round0_status(job_id, round0_status="FAILED", note=f"subprocess_failed: {e}")

        if AUTO_EMIT_EVENTS:
            try:
                resp, payload, url = _emit_event(
                    event_type="LOOP_FAILED",
                    run_id=run_id,
                    job_id=job_id,
                    status="FAILED",
                    extra={"error": str(e)},
                )
                post_actions["event_loop_failed"] = resp
                post_actions["event_loop_failed_request"] = payload
                post_actions["event_loop_failed_url"] = url
            except Exception as ee:
                post_errors.append(f"emit_event_failed: {ee}")

        _write_job(
            job_id,
            {
                "status": "FAILED",
                "run_id": run_id,
                "error": str(e),
                "stdout": _clip_head_tail(e.stdout or ""),
                "stderr": _clip_head_tail(e.stderr or ""),
                "cmd": cmd,
                "latest_exists": LATEST_STUDENT.exists(),
                "post_actions": post_actions,
                "post_errors": post_errors,
            },
        )

    except Exception as e:
        post_errors: List[str] = []
        post_actions: Dict[str, Any] = {}

        cur = _read_job(job_id)
        if str(cur.get("round0_status") or "").upper() != "DONE":
            _set_round0_status(job_id, round0_status="FAILED", note=f"exception: {e}")

        if AUTO_EMIT_EVENTS:
            try:
                resp, payload, url = _emit_event(
                    event_type="LOOP_FAILED",
                    run_id=run_id,
                    job_id=job_id,
                    status="FAILED",
                    extra={"error": str(e)},
                )
                post_actions["event_loop_failed"] = resp
                post_actions["event_loop_failed_request"] = payload
                post_actions["event_loop_failed_url"] = url
            except Exception as ee:
                post_errors.append(f"emit_event_failed: {ee}")

        _write_job(
            job_id,
            {
                "status": "FAILED",
                "run_id": run_id,
                "error": str(e),
                "cmd": cmd,
                "latest_exists": LATEST_STUDENT.exists(),
                "post_actions": post_actions,
                "post_errors": post_errors,
            },
        )


# -----------------------------
# routes
# -----------------------------
@router.post("/run", response_model=RunLoopResponse)
def run_loop(
    bg: BackgroundTasks,
    req: RunLoopRequest = Body(default_factory=RunLoopRequest),
):
    job_id = _new_job_id()
    run_id = _new_run_id()

    # ✅ QUEUED 시점에도 round0 필드 생성
    _write_job(
        job_id,
        {
            "status": "QUEUED",
            "run_id": run_id,
            "round0_status": "QUEUED",
            "round0_updated_at": _now_iso_z_no_ms(),
        },
    )
    bg.add_task(_run_loop_subprocess, job_id, run_id, req)

    return RunLoopResponse(job_id=job_id, run_id=run_id, status="QUEUED", jobId=job_id, runId=run_id)


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str):
    d = _read_job(job_id)
    run_id = d.get("run_id") or job_id

    # ✅ 프론트/백엔드가 camelCase로 바로 읽을 수 있게 mirror 필드 추가
    stats = dict(d)

    # 기본 키들 (snake -> camel)
    if "round0_status" in stats and "round0Status" not in stats:
        stats["round0Status"] = stats.get("round0_status")
    if "round0_export_rel_path" in stats and "round0ExportRelPath" not in stats:
        stats["round0ExportRelPath"] = stats.get("round0_export_rel_path")
    if "round0_counts" in stats and "round0Counts" not in stats:
        stats["round0Counts"] = stats.get("round0_counts")
    if "round0_notify" in stats and "round0Notify" not in stats:
        stats["round0Notify"] = stats.get("round0_notify")
    if "round0_updated_at" in stats and "round0UpdatedAt" not in stats:
        stats["round0UpdatedAt"] = stats.get("round0_updated_at")

    return JobStatusResponse(
        job_id=job_id,
        run_id=run_id,
        status=d.get("status", "UNKNOWN"),
        stats=stats,
        jobId=job_id,
        runId=run_id,
    )
