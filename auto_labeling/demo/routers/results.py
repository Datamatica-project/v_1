# demo/routers/results.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import json
import zipfile
from typing import Any, Dict, List, Optional

import yaml  # pip install pyyaml

from auto_labeling.demo.routers.ingest import DATA

router = APIRouter(prefix="/api/demo/results", tags=["demo-results"])

# -------------------------------------------------
# paths & config
# -------------------------------------------------
PROJECT_ROOT = DATA.parent
MODEL_YAML = PROJECT_ROOT / "demo" / "configs" / "model.yaml"


def _load_model_yaml() -> dict:
    """
    model.yaml 로드

    목적:
    - 결과(run) 저장 base_dir을 config 기반으로 해석하기 위해 사용
      (output.base_dir or output.baseDir)
    """
    if not MODEL_YAML.exists():
        raise HTTPException(500, f"model.yaml not found: {MODEL_YAML}")
    try:
        return yaml.safe_load(MODEL_YAML.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise HTTPException(500, f"failed to load model.yaml: {e}")


def _resolve_path(p: str) -> Path:
    """
    상대경로/절대경로 모두 지원:
    - 절대경로: 그대로 사용
    - 상대경로: PROJECT_ROOT 기준으로 resolve
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (PROJECT_ROOT / pp).resolve()


def _get_output_base_dir(cfg: dict) -> Path:
    """
    output.base_dir (또는 output.baseDir) 를 읽어서 run 저장 base_dir을 결정한다.

    기본값:
    - "demo/data/demo_runs"  (PROJECT_ROOT 기준 상대경로)
    """
    out = (cfg.get("output") or {})
    base = out.get("base_dir") or out.get("baseDir") or "demo/data/demo_runs"
    return _resolve_path(str(base))


def _run_dir(run_id: str) -> Path:
    """
    run_id -> 실제 run 디렉토리
    """
    cfg = _load_model_yaml()
    base = _get_output_base_dir(cfg)
    return (base / run_id).resolve()


def _safe_join(base: Path, name: str) -> Path:
    """
    path traversal 방지용 safe join

    - 사용자가 name에 "../" 등을 넣어 base_dir 밖 파일을 읽지 못하도록 차단
    """
    p = (base / name).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise HTTPException(400, "invalid name")
    return p


def _read_jsonl(p: Path) -> List[dict]:
    """
    preds.jsonl 읽기 유틸:
    - 라인별 JSON 객체
    - 파싱 실패 라인은 skip
    """
    if not p.exists():
        return []
    out: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _to_abs_path(run_dir: Path, p: Optional[str]) -> Optional[str]:
    """
    resultPath/previewPath를 항상 절대경로로 변환해주는 유틸.
    - p가 None/"" 이면 None
    - 절대경로면 그대로
    - 상대경로면 run_dir 기준으로 resolve
    """
    if not p:
        return None
    pp = Path(str(p))
    if pp.is_absolute():
        return str(pp.resolve())
    return str((run_dir / pp).resolve())


# schemas
class ResultItem(BaseModel):
    imagePath: str = Field(..., description="입력 이미지 경로(원본).")
    boxes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "YOLO 추론 bbox 리스트.\n"
            "각 원소 예: {classId:int, conf:float, xyxy:[x1,y1,x2,y2]}\n"
            "- classId: 클래스 인덱스\n"
            "- conf: confidence\n"
            "- xyxy: 픽셀 좌표 (좌상단 x1,y1 / 우하단 x2,y2)"
        ),
    )
    previewPath: Optional[str] = Field(
        None,
        description="(선택) 미리보기 이미지 경로(원본 기록).",
    )
    resultPath: Optional[str] = Field(
        None,
        description=(
            "박스가 그려진 결과 이미지 경로(원본 기록).\n"
            "PASS만 저장하는 설정이면 PASS인 경우에만 값이 채워질 수 있다."
        ),
    )

    # 추가: 결과 이미지 절대 경로
    resultAbsPath: Optional[str] = Field(
        None,
        description="resultPath를 절대경로로 해석한 값(서버 내부 경로).",
    )

    # (원하면 preview도 같이 절대경로 내려줄 수 있음)
    previewAbsPath: Optional[str] = Field(
        None,
        description="previewPath를 절대경로로 해석한 값(서버 내부 경로).",
    )

    passed: Optional[bool] = Field(
        None,
        description="이미지 단위 PASS/FAIL 판정 결과. True=PASS, False=FAIL",
    )


class GetResultsResponse(BaseModel):
    runId: str = Field(..., description="조회 대상 run ID")
    total: int = Field(..., description="(error row 제외) 전체 결과 개수")
    offset: int = Field(..., description="pagination 시작 index")
    limit: int = Field(..., description="pagination page size")
    items: List[ResultItem] = Field(..., description="결과 리스트 (offset/limit 적용 후)")


class RunInfo(BaseModel):
    runId: str = Field(..., description="run 디렉토리명(고유 ID)")
    runPath: str = Field(..., description="해당 run의 실제 저장 경로(서버 내부 경로)")
    status: Optional[str] = Field(
        None,
        description="run의 상태. status.json이 있을 때만 채워짐. 예: QUEUED/RUNNING/DONE/FAILED",
    )
    createdAt: Optional[str] = Field(
        None,
        description="run 생성 시각. status.json의 createdAt 또는 startedAt",
    )


class ListRunsResponse(BaseModel):
    items: List[RunInfo] = Field(..., description="run 목록 (최근 수정 시각 기준 내림차순)")


# routes
@router.get(
    "/list",
    response_model=ListRunsResponse,
    summary="run 목록 조회 (최근순)",
)
def list_runs(
    limit: int = Query(20, ge=1, le=200),
):
    cfg = _load_model_yaml()
    base = _get_output_base_dir(cfg)
    if not base.exists():
        return ListRunsResponse(items=[])

    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    runs = runs[: int(limit)]

    items: List[RunInfo] = []
    for r in runs:
        st = r / "status.json"
        status = None
        created_at = None
        if st.exists():
            try:
                j = json.loads(st.read_text(encoding="utf-8"))
                status = j.get("status")
                created_at = j.get("createdAt") or j.get("startedAt")
            except Exception:
                pass

        items.append(
            RunInfo(
                runId=r.name,
                runPath=str(r.resolve()),
                status=status,
                createdAt=created_at,
            )
        )

    return ListRunsResponse(items=items)


@router.get(
    "/get",
    response_model=GetResultsResponse,
    summary="특정 run의 결과 조회 (JSON, pagination)",
)
def get_results(
    runId: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5000),
):
    run_dir = _run_dir(runId)
    if not run_dir.exists():
        raise HTTPException(404, f"run not found: {runId}")

    preds_path = run_dir / "preds.jsonl"
    rows = _read_jsonl(preds_path)

    # error row는 제외
    items_raw = [r for r in rows if r.get("type") != "error"]
    total = len(items_raw)

    s = int(offset)
    e = min(s + int(limit), total)
    page = items_raw[s:e]

    items: List[ResultItem] = []
    for r in page:
        result_path = r.get("resultPath")
        preview_path = r.get("previewPath")

        items.append(
            ResultItem(
                imagePath=str(r.get("imagePath", "")),
                boxes=r.get("boxes") or [],
                previewPath=preview_path,
                resultPath=result_path,

                resultAbsPath=_to_abs_path(run_dir, result_path),
                previewAbsPath=_to_abs_path(run_dir, preview_path),

                passed=r.get("pass"),
            )
        )

    return GetResultsResponse(
        runId=runId,
        total=total,
        offset=s,
        limit=int(limit),
        items=items,
    )


@router.get(
    "/preview",
    summary="결과 이미지 1장 미리보기 (FileResponse)",
)
def get_preview(
    runId: str = Query(...),
    name: str = Query(...),
    which: str = Query("pass", pattern="^(pass|fail)$"),
):
    run_dir = _run_dir(runId)
    base = run_dir / "result" / which
    if not base.exists():
        raise HTTPException(404, "result dir not found")

    p = _safe_join(base, name)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")

    return FileResponse(str(p), filename=p.name)


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(src_dir)))


@router.get(
    "/download",
    summary="run 결과 전체 ZIP 다운로드",
)
def download_run(
    runId: str = Query(...),
):
    run_dir = _run_dir(runId)
    if not run_dir.exists():
        raise HTTPException(404, f"run not found: {runId}")

    zip_path = run_dir / f"{runId}_result.zip"
    if not zip_path.exists():
        _zip_dir(run_dir, zip_path)

    return FileResponse(
        str(zip_path),
        filename=zip_path.name,
        media_type="application/zip",
    )
