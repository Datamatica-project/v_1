from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from fastapi import APIRouter, HTTPException, Body

from auto_labeling.v_1.api.dto.export_final import ExportFinalRequest, ExportFinalResponse
from auto_labeling.v_1.src.export_pass_fail_final import export_pass_fail_final

# (있으면 쓰고, 없어도 동작하도록 안전하게)
try:
    from auto_labeling.v_1.scripts.logger import log_json  # type: ignore
except Exception:  # pragma: no cover
    def log_json(event: Dict[str, Any]) -> None:  # fallback
        print(json.dumps(event, ensure_ascii=False))


# ✅ API 서버 라우터는 /export 하위만 담당.
# server.py에서 include_router(..., prefix="/v1") 하면 최종: /api/v1/export/final
router = APIRouter(prefix="/export")

ROOT = Path(__file__).resolve().parents[2]  # .../v_1
JOB_DIR = ROOT / "logs" / "jobs"
JOB_DIR.mkdir(parents=True, exist_ok=True)


def _utc_now_iso_z() -> str:
    """
    현재 시각을 UTC ISO8601(Z) 문자열로 반환.
    - 예: 2025-12-19T07:09:32.872Z
    """
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _write_job(job_id: str, payload: Dict[str, Any]) -> None:
    """
    job 상태를 logs/jobs/{jobId}.json 으로 저장.
    - 프론트/외부 시스템은 jobId를 이용해 상태 추적 가능
    - 저장 실패는 본 작업 실패로 간주하지 않음(로깅 목적이므로)
    """
    p = JOB_DIR / f"{job_id}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@router.post(
    "/final",
    response_model=ExportFinalResponse,
    summary="최종 Export 수행",
    description=(
        "루프가 생성한 PASS/FAIL/MISS 풀을 기반으로 **최종 라벨 export**를 수행합니다.\n\n"
        "## 이 API가 하는 일(의미)\n"
        "- Student 모델로 최종 추론/라벨 내보내기를 수행하여, 학습/납품/검수용 결과를 생성합니다.\n"
        "- export 결과는 `exportRoot` 아래에 디렉토리 구조로 저장됩니다.\n"
        "- 작업 상태는 `logs/jobs/{jobId}.json`에 기록되어 외부에서 추적할 수 있습니다.\n\n"
        "## 입력(요청) 필드 의미\n"
        "- `studentWeight` (필수): export에 사용할 Student 모델 가중치(.pt) 파일 경로\n"
        "- `exportRoot` (선택): 결과를 저장할 출력 루트 경로\n"
        "  - 미지정 시 기본값: `data/final_export`\n"
        "- `exportConf` (선택): export 시 사용할 confidence threshold (0~1)\n"
        "  - 낮을수록 더 많은 bbox가 포함되고, 높을수록 보수적으로 저장됨\n"
        "- `passPool` (선택): PASS 풀을 특정 디렉토리로 고정하고 싶을 때 지정\n"
        "  - 지정 시 이 디렉토리의 이미지들을 PASS 입력으로 사용\n"
        "- `mergePass` (선택, 기본 True): 라운드별 PASS 계열을 하나로 합쳐 export할지 여부\n"
        "  - True: pass_fail 등 PASS로 취급 가능한 풀을 PASS로 병합\n"
        "  - False: PASS/기타 그룹을 구분해서 export\n"
        "- `roundRoot` (선택): 특정 라운드 결과 디렉토리를 명시적으로 지정\n"
        "  - 예: `data/round_r2`\n\n"
        "## roundRoot 자동 선택 정책\n"
        "- `roundRoot`를 주지 않으면 `data/round_r*` 중 가장 큰 r(최신 라운드)을 자동 선택합니다.\n"
        "- 라운드 디렉토리가 하나도 없으면:\n"
        "  - pass_fail/fail_fail/miss 입력이 없을 수 있으며, PASS(pool)만으로 export될 수 있습니다.\n\n"
        "## 출력(결과) 의미\n"
        "- export 결과는 `exportRoot` 아래에 저장됩니다.\n"
        "- 본 API는 성공 시 ExportFinalResponse로 `exportRoot`, `usedStudentWeight`를 반환합니다.\n"
        "- 추가 추적을 위해 response.detail에 `jobId`를 포함합니다.\n\n"
        "## 실패/예외\n"
        "- `studentWeight` 경로가 없으면 400 반환\n"
        "- export 과정 내부 오류는 500으로 전파될 수 있음(현재 구현은 raise 그대로)\n"
    ),
    responses={
        200: {"description": "export 수행 완료"},
        400: {"description": "요청 값 오류(예: studentWeight 경로 없음)"},
        500: {"description": "export 중 내부 오류"},
    },
)
def export_final(
    req: ExportFinalRequest = Body(
        ...,
        description=(
            "최종 export 실행 요청 본문(JSON).\n\n"
            "### 필수\n"
            "- studentWeight (string): 모델 가중치(.pt) 파일 경로\n\n"
            "### 선택\n"
            "- exportRoot (string|null): 결과 출력 루트\n"
            "- exportConf (number): confidence threshold (0~1)\n"
            "- passPool (string|null): PASS 이미지 디렉토리(고정 풀)\n"
            "- mergePass (boolean): PASS 병합 여부\n"
            "- roundRoot (string|null): 사용할 round 결과 디렉토리 override\n\n"
            "### 주의\n"
            "- passPool/roundRoot는 로컬 파일시스템 경로이며, 서버/컨테이너 내부 경로 기준입니다.\n"
            "- exportConf는 높을수록 저장되는 bbox가 줄어드는 경향이 있습니다.\n"
        ),
        examples=[
            {
                "studentWeight": "models/user_yolo/v1_001/best.pt",
                "exportRoot": "data/final_export/run_001",
                "exportConf": 0.3,
                "passPool": None,
                "mergePass": True,
                "roundRoot": "data/round_r2",
            }
        ],
    )
):
    # -----------------------------
    # job init
    # -----------------------------
    job_id = f"job_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}_export_final"
    created_at = _utc_now_iso_z()

    # job 시작 상태 기록(실패해도 export 계속 진행)
    try:
        _write_job(
            job_id,
            {
                "jobId": job_id,
                "type": "exportFinal",
                "status": "RUNNING",
                "createdAt": created_at,
                "updatedAt": created_at,
                "request": {
                    "studentWeight": req.studentWeight,
                    "exportRoot": req.exportRoot,
                    "roundRoot": req.roundRoot,
                    "passPool": req.passPool,
                    "exportConf": req.exportConf,
                    "mergePass": req.mergePass,
                },
            },
        )
    except Exception:
        pass

    # -----------------------------
    # validate studentWeight
    # -----------------------------
    student_weights = Path(req.studentWeight)
    if not student_weights.exists():
        now = _utc_now_iso_z()
        try:
            _write_job(
                job_id,
                {
                    "jobId": job_id,
                    "type": "exportFinal",
                    "status": "FAILED",
                    "createdAt": created_at,
                    "updatedAt": now,
                    "error": f"studentWeight not found: {student_weights}",
                },
            )
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"studentWeight not found: {student_weights}")

    # -----------------------------
    # resolve export root
    # -----------------------------
    out_root = Path(req.exportRoot) if req.exportRoot else (ROOT / "data" / "final_export")
    out_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # resolve last round root
    # -----------------------------
    round_root: Optional[Path]
    if req.roundRoot:
        round_root = Path(req.roundRoot)
    else:
        data_root = ROOT / "data"
        rounds: list[tuple[int, Path]] = []
        for p in data_root.glob("round_r*"):
            if p.is_dir():
                try:
                    r = int(p.name.replace("round_r", ""))
                    rounds.append((r, p))
                except Exception:
                    continue
        rounds.sort(key=lambda x: x[0])
        round_root = rounds[-1][1] if rounds else None

    # -----------------------------
    # candidate pools from round_root
    # -----------------------------
    pass_fail_img_dir = None
    fail_fail_img_dir = None
    miss_img_dir = None

    if round_root is not None:
        pass_fail_img_dir = round_root / "pass_fail" / "images"
        fail_fail_img_dir = round_root / "fail_fail" / "images"
        miss_img_dir = round_root / "miss" / "images"

    # PASS pool override
    pass_img_dirs: List[Path] = []
    if req.passPool:
        pass_img_dirs.append(Path(req.passPool))

    # -----------------------------
    # run export
    # -----------------------------
    started_at = _utc_now_iso_z()
    log_json(
        {
            "timestamp": started_at,
            "level": "INFO",
            "scope": "export",
            "refId": job_id,
            "message": "export_final:START",
            "data": {
                "jobId": job_id,
                "studentWeight": str(student_weights),
                "exportRoot": str(out_root),
                "roundRoot": str(round_root) if round_root is not None else None,
                "exportConf": float(req.exportConf),
                "mergePass": bool(req.mergePass),
                "passPool": req.passPool,
            },
        }
    )

    try:
        export_pass_fail_final(
            student_weights=student_weights,
            out_root=out_root,
            pass_img_dirs=pass_img_dirs,
            pass_fail_img_dir=pass_fail_img_dir,
            fail_fail_img_dir=fail_fail_img_dir,
            miss_img_dir=miss_img_dir,
            device="0",
            conf_th=float(req.exportConf),
            merge_pass_into_pass_dir=bool(req.mergePass),
        )

        finished_at = _utc_now_iso_z()

        # job DONE update
        try:
            _write_job(
                job_id,
                {
                    "jobId": job_id,
                    "type": "exportFinal",
                    "status": "DONE",
                    "createdAt": created_at,
                    "updatedAt": finished_at,
                    "result": {
                        "exportRoot": str(out_root),
                        "usedStudentWeight": str(student_weights),
                        "roundRoot": str(round_root) if round_root is not None else None,
                    },
                },
            )
        except Exception:
            pass

        log_json(
            {
                "timestamp": finished_at,
                "level": "INFO",
                "scope": "export",
                "refId": job_id,
                "message": "export_final:DONE",
                "data": {
                    "jobId": job_id,
                    "exportRoot": str(out_root),
                    "usedStudentWeight": str(student_weights),
                    "roundRoot": str(round_root) if round_root is not None else None,
                },
            }
        )

    except Exception as e:
        failed_at = _utc_now_iso_z()

        # job FAILED update
        try:
            _write_job(
                job_id,
                {
                    "jobId": job_id,
                    "type": "exportFinal",
                    "status": "FAILED",
                    "createdAt": created_at,
                    "updatedAt": failed_at,
                    "error": str(e),
                    "result": {
                        "exportRoot": str(out_root),
                        "usedStudentWeight": str(student_weights),
                        "roundRoot": str(round_root) if round_root is not None else None,
                    },
                },
            )
        except Exception:
            pass

        log_json(
            {
                "timestamp": failed_at,
                "level": "ERROR",
                "scope": "export",
                "refId": job_id,
                "message": "export_final:FAILED",
                "data": {
                    "jobId": job_id,
                    "error": str(e),
                },
            }
        )
        raise

    # ✅ 응답은 ExportFinalResponse 스펙 유지 + detail에 jobId 포함
    return ExportFinalResponse(
        exportRoot=str(out_root),
        usedStudentWeight=str(student_weights),
        detail={
            "jobId": job_id,
            "roundRoot": str(round_root) if round_root is not None else None,
        },
    )
