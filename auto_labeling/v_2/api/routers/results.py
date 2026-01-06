# auto_labeling/v_1/api/routers/results.py
from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple

import cv2
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from auto_labeling.v_1.api.dto.results import ListRunsResponse, RunInfo, Round0ResultResponse

router = APIRouter(prefix="/results")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parents[2]
EXPORT_REG_DIR = ROOT / "logs" / "exports"
EXPORT_REG_DIR.mkdir(parents=True, exist_ok=True)
NAS_SERVED_BASE_URL = os.getenv("V1_EXPORT_NAS_SERVED_BASE_URL", "").strip() or None

DEFAULT_DOWNLOAD_MODE: Literal["baseUrl", "zip"] = (
    "zip" if (os.getenv("V1_RESULTS_DEFAULT_DOWNLOAD_MODE", "baseUrl").strip().lower() == "zip") else "baseUrl"
)

# 서버 로컬에서 exportRelPath를 실제 경로로 resolve 할 때 사용하는 베이스
# 예) V1_EXPORT_LOCAL_ROOT=/workspace 라면 exportRelPath="auto_labeling/v_1/data/round_r0" -> /workspace/auto_labeling/v_1/data/round_r0
V1_EXPORT_LOCAL_ROOT = os.getenv("V1_EXPORT_LOCAL_ROOT", "").strip() or None

# ------------------------------------------------------------
# split policy
#  - physical splits on disk: pass, pass_fail, fail_fail, miss
#  - logical splits exposed to FE: pass(=pass+pass_fail), fail(=fail_fail), miss
# ------------------------------------------------------------
LOGICAL_SPLITS = {"pass", "fail", "miss"}
PHYSICAL_SPLITS = {"pass", "pass_fail", "fail_fail", "miss"}

PASS_FAIL_SUFFIX = "__pf"  # pass_fail source file => <stem>__pf.jpg


# =========================
# Round0 Preview DTOs (3-split)
# =========================
class BuildPreviewSetRequest(BaseModel):
    run_id: str = Field(..., alias="runId", description="대상 runId")
    splits: str = Field("pass,pass_fail,fail_fail,miss", description="comma-separated physical splits")
    overwrite: bool = False
    thickness: int = Field(2, ge=1, le=10)
    max_side: int = Field(0, ge=0, le=4096, alias="maxSide")
    limit: int = Field(0, ge=0, le=200000, description="0이면 전체, >0이면 최대 N장만 (split 당)")

    class Config:
        populate_by_name = True


class PreviewIndexItem(BaseModel):
    split: Literal["pass", "fail", "miss"]
    file_name: str = Field(..., alias="fileName")

    local_path: str = Field(..., alias="localPath")
    src_image_path: Optional[str] = Field(None, alias="srcImagePath")
    label_path: Optional[str] = Field(None, alias="labelPath")
    image_url: Optional[str] = Field(None, alias="imageUrl")

    class Config:
        populate_by_name = True


class PreviewIndexResponse(BaseModel):
    run_id: str = Field(..., alias="runId")
    export_rel_path: str = Field(..., alias="exportRelPath")
    result_rel_path: str = Field(..., alias="resultRelPath")
    total: int
    items: List[PreviewIndexItem]

    class Config:
        populate_by_name = True


# =========================
# FINAL Preview DTOs (3-split)
# =========================
class BuildFinalPreviewSetRequest(BaseModel):
    run_id: str = Field(..., alias="runId", description="대상 runId")
    splits: str = Field("pass,pass_fail,fail_fail,miss", description="comma-separated physical splits")
    rounds: str = Field("", description="comma-separated roundNo list. empty이면 round_r* 자동 스캔")
    overwrite: bool = False
    thickness: int = Field(2, ge=1, le=10)
    max_side: int = Field(0, ge=0, le=4096, alias="maxSide")
    limit: int = Field(0, ge=0, le=200000, description="0이면 전체, >0이면 최대 N장만 (round/split 당)")

    class Config:
        populate_by_name = True


class FinalPreviewItem(BaseModel):
    run_id: str = Field(..., alias="runId")
    round_no: int = Field(..., alias="roundNo")
    split: Literal["pass", "fail", "miss"]
    file_name: str = Field(..., alias="fileName")
    local_path: str = Field(..., alias="localPath")
    src_image_path: Optional[str] = Field(None, alias="srcImagePath")
    label_path: Optional[str] = Field(None, alias="labelPath")
    image_url: Optional[str] = Field(None, alias="imageUrl")

    class Config:
        populate_by_name = True


class FinalRoundSummary(BaseModel):
    round_no: int = Field(..., alias="roundNo")
    round_dir: str = Field(..., alias="roundDir")
    preview_index_path: str = Field(..., alias="previewIndexPath")
    total: int

    class Config:
        populate_by_name = True


class FinalPreviewIndexResponse(BaseModel):
    run_id: str = Field(..., alias="runId")
    export_root: str = Field(..., alias="exportRoot")
    total: int
    items: List[FinalPreviewItem]
    rounds: List[FinalRoundSummary]
    built_at: Optional[str] = Field(None, alias="builtAt")

    class Config:
        populate_by_name = True


# =========================
# bbox helpers
# =========================
def _read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """YOLO txt: cls cx cy w h (normalized)"""
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    out: List[Tuple[int, float, float, float, float]] = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        out.append((cls_id, cx, cy, w, h))
    return out


def _draw_yolo_boxes(img_bgr, labels: List[Tuple[int, float, float, float, float]], thickness: int = 2) -> None:
    H, W = img_bgr.shape[:2]
    for cls_id, cx, cy, w, h in labels:
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(
            img_bgr,
            str(cls_id),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def _resize_keep_aspect(img_bgr, max_side: int) -> Any:
    if not max_side or max_side <= 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr
    scale = max_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# =========================
# misc helpers
# =========================
def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _run_file(run_id: str) -> Path:
    return EXPORT_REG_DIR / f"round0_{run_id}.json"


def _final_file(run_id: str) -> Path:
    return EXPORT_REG_DIR / f"final_{run_id}.json"


def _safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _join_unc(share_root: str, rel_path: str) -> str:
    if not share_root:
        return rel_path
    s = share_root.rstrip("\\/")
    r = rel_path.lstrip("\\/")
    return s + "\\" + r


def _to_served_url(_share_root: str, rel_path: str) -> Optional[str]:
    if not NAS_SERVED_BASE_URL:
        return None
    return NAS_SERVED_BASE_URL.rstrip("/") + "/" + rel_path.replace("\\", "/").lstrip("/")


def _list_registry_files() -> List[Path]:
    files = [p for p in EXPORT_REG_DIR.glob("round0_*.json") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _run_id_from_registry_path(p: Path) -> str:
    name = p.name
    if not name.startswith("round0_") or not name.endswith(".json"):
        return ""
    return name[len("round0_") : -len(".json")]


def _resolve_export_local_dir(export_rel_path: str) -> Optional[Path]:
    """exportRelPath -> 서버 로컬 접근 경로 (V1_EXPORT_LOCAL_ROOT 필요)"""
    if not V1_EXPORT_LOCAL_ROOT:
        return None
    base = Path(V1_EXPORT_LOCAL_ROOT)
    rel = Path(export_rel_path.replace("\\", "/").lstrip("/"))
    return (base / rel).resolve()


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(src_dir)))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(img_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not img_dir.exists() or not img_dir.is_dir():
        return out
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    out.sort()
    return out


def _get_share_root_from_registry(rec: Dict[str, Any]) -> str:
    return str(rec.get("share_root") or rec.get("shareRoot") or rec.get("share_root".upper()) or "")


def _get_export_rel_path_from_round0_registry(rec: Dict[str, Any]) -> str:
    return str(rec.get("exportRelPath") or rec.get("export_rel_path") or rec.get("EXPORT_RELPATH") or "")


def _get_export_rel_path_from_final_registry(rec: Dict[str, Any]) -> str:
    return str(rec.get("exportRelPath") or rec.get("export_rel_path") or rec.get("EXPORT_RELPATH") or "")


def _preview_index_path(export_dir: Path) -> Path:
    return (export_dir / "result" / "preview_index.json").resolve()


def _final_round_preview_index_path(round_dir: Path) -> Path:
    return (round_dir / "result" / "preview_index.json").resolve()


def _final_master_index_path(export_root: Path) -> Path:
    return (export_root / "result" / "final_preview_index.json").resolve()


def _find_src_image_by_stem(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = (img_dir / f"{stem}{ext}").resolve()
        if p.exists():
            return p
    return None


def _resolve_round_dir(export_root: Path, round_no: int) -> Path:
    return (export_root / f"round_r{int(round_no)}").resolve()


def _scan_round_nos(export_root: Path) -> List[int]:
    out: List[int] = []
    if not export_root.exists():
        return out
    for p in export_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^round_r(\d+)$", p.name)
        if not m:
            continue
        out.append(int(m.group(1)))
    out.sort()
    return out


def _validate_physical_splits(splits: str) -> List[str]:
    split_list = [s.strip() for s in (splits or "").split(",") if s.strip()]
    for s in split_list:
        if s not in PHYSICAL_SPLITS:
            raise HTTPException(400, f"invalid physical split: {s}")
    return split_list


def _parse_rounds_csv(rounds_csv: str) -> List[int]:
    if not rounds_csv or not rounds_csv.strip():
        return []
    out: List[int] = []
    for tok in rounds_csv.split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.append(int(t))
        except Exception:
            raise HTTPException(400, f"invalid roundNo: {t}")
    return sorted(set(out))


def _logical_split_from_physical(src_split: str) -> Literal["pass", "fail", "miss"]:
    if src_split in ("pass", "pass_fail"):
        return "pass"
    if src_split == "fail_fail":
        return "fail"
    return "miss"


def _preview_file_name(stem: str, src_split: str) -> str:
    if src_split == "pass_fail":
        return f"{stem}{PASS_FAIL_SUFFIX}.jpg"
    return f"{stem}.jpg"


def _infer_source_split_from_preview_file(logical_split: str, file_name: str) -> str:
    if logical_split == "pass":
        stem = Path(file_name).stem
        if stem.endswith(PASS_FAIL_SUFFIX):
            return "pass_fail"
        return "pass"
    if logical_split == "fail":
        return "fail_fail"
    return "miss"


def _strip_pf_suffix(stem: str) -> str:
    if stem.endswith(PASS_FAIL_SUFFIX):
        return stem[: -len(PASS_FAIL_SUFFIX)]
    return stem


def _safe_body_run_id(body_run_id: Optional[str]) -> str:
    """Swagger 기본값 'string' 같은 거 걸러서 안전하게 runId만 꺼냄"""
    if not body_run_id:
        return ""
    s = str(body_run_id).strip()
    if not s or s.lower() == "string":
        return ""
    return s


def _resolve_final_export_root_from_run(run_id: str) -> Path:
    """
    final registry가 stub(없는데 가리킴)일 수 있으므로:
      1) final registry exportRelPath가 실제로 존재하면 그걸 사용
      2) 아니면 round0 registry extra.exportRoot로 fallback (현재 네 구조는 여기로 떨어짐)
    """
    # 1) final registry 우선
    freg = _safe_read_json(_final_file(run_id)) or {}
    f_rel = _get_export_rel_path_from_final_registry(freg)
    if f_rel:
        p = _resolve_export_local_dir(f_rel)
        if p is not None and p.exists() and p.is_dir():
            return p

    # 2) round0 registry extra.exportRoot fallback
    r0 = _safe_read_json(_run_file(run_id)) or {}
    extra = r0.get("extra") or {}
    export_root_rel = str(extra.get("exportRoot") or "").strip()
    if export_root_rel:
        p2 = _resolve_export_local_dir(export_root_rel)
        if p2 is not None and p2.exists() and p2.is_dir():
            return p2

    raise HTTPException(
        404,
        f"export root not found. final exportRelPath='{f_rel}', round0 extra.exportRoot='{export_root_rel}'",
    )


# =========================
# APIs - list / round0 result / download
# =========================
@router.get(
    "/list",
    response_model=ListRunsResponse,
    summary="run 목록 조회(최근순) - round0 export 레지스트리 기반",
)
def list_runs(limit: int = Query(50, ge=1, le=500)):
    items: List[RunInfo] = []
    files = _list_registry_files()[: int(limit)]

    for p in files:
        run_id = _run_id_from_registry_path(p)
        if not run_id:
            continue

        rec = _safe_read_json(p) or {}
        status = str(rec.get("status", "UNKNOWN") or "UNKNOWN")
        created_at = _parse_dt(rec.get("created_at"))
        updated_at = _parse_dt(rec.get("updated_at"))

        items.append(
            RunInfo(
                run_id=run_id,
                status=status,
                created_at=created_at,
                updated_at=updated_at,
                export_round0_ready=True,
                job_id=None,
                note=None,
            )
        )

    return ListRunsResponse(items=items)


@router.get(
    "/round0",
    response_model=Round0ResultResponse,
    summary="round0 결과 조회",
)
def get_round0_result(runId: str = Query(..., description="조회할 runId")):
    run_id = runId.strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="runId is required")

    rec = _safe_read_json(_run_file(run_id))
    if not rec:
        raise HTTPException(status_code=404, detail="round0 export not found")

    share_root = _get_share_root_from_registry(rec)
    export_rel_path = _get_export_rel_path_from_round0_registry(rec)
    manifest_rel_path = rec.get("manifest_rel_path") or rec.get("manifestRelPath")

    download_base_url = _to_served_url(share_root, export_rel_path) or _join_unc(share_root, export_rel_path)
    manifest_url = None
    if manifest_rel_path:
        manifest_url = _to_served_url(share_root, str(manifest_rel_path)) or _join_unc(share_root, str(manifest_rel_path))

    zip_url = f"/api/v1/results/download?runId={run_id}&mode=zip"

    return Round0ResultResponse(
        run_id=run_id,
        status=str(rec.get("status", "UNKNOWN") or "UNKNOWN"),
        message=rec.get("message"),
        pass_count=int(rec.get("pass_count", rec.get("passCount", 0)) or 0),
        fail_count=int(rec.get("fail_count", rec.get("failCount", 0)) or 0),
        miss_count=int(rec.get("miss_count", rec.get("missCount", 0)) or 0),
        export_rel_path=export_rel_path or None,
        manifest_rel_path=str(manifest_rel_path) if manifest_rel_path else None,
        download_base_url=download_base_url,
        manifest_url=manifest_url,
        zip_url=zip_url,
        created_at=_parse_dt(rec.get("created_at") or rec.get("createdAt")),
        updated_at=_parse_dt(rec.get("updated_at") or rec.get("updatedAt")),
        extra=rec.get("extra") or {},
    )


@router.get(
    "/download",
    summary="round0 결과 다운로드(baseUrl 또는 zip)",
)
def download_round0(
    runId: str = Query(...),
    mode: Literal["baseUrl", "zip"] = Query(DEFAULT_DOWNLOAD_MODE),
):
    run_id = runId.strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="runId is required")

    rec = _safe_read_json(_run_file(run_id))
    if not rec:
        raise HTTPException(status_code=404, detail="round0 export not found")

    export_rel_path = _get_export_rel_path_from_round0_registry(rec)
    if not export_rel_path:
        raise HTTPException(status_code=500, detail="exportRelPath missing in registry")

    share_root = _get_share_root_from_registry(rec)
    download_base_url = _to_served_url(share_root, export_rel_path) or _join_unc(share_root, export_rel_path)

    if mode == "baseUrl":
        return {
            "runId": run_id,
            "mode": "baseUrl",
            "downloadBaseUrl": download_base_url,
            "exportRelPath": export_rel_path,
        }

    export_dir = _resolve_export_local_dir(export_rel_path)
    if export_dir is None:
        raise HTTPException(
            status_code=501,
            detail="zip mode requires V1_EXPORT_LOCAL_ROOT (server-accessible export root).",
        )
    if not export_dir.exists() or not export_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"export dir not found on server: {export_dir}")

    zip_dir = EXPORT_REG_DIR / "zips"
    zip_path = (zip_dir / f"{run_id}_round0.zip").resolve()

    if not zip_path.exists():
        _zip_dir(export_dir, zip_path)

    return FileResponse(str(zip_path), filename=zip_path.name, media_type="application/zip")


# =========================
# Round0 preview (3개 세트)
# =========================
@router.post(
    "/round0/buildPreviewSet",
    summary="round0 프리뷰 세트 생성 + preview_index.json 생성 (노출 split=pass/fail/miss)",
)
def build_preview_set_round0(
    runId: Optional[str] = Query(None, description="대상 runId (query)"),
    splits: str = Query("pass,pass_fail,fail_fail,miss", description="comma-separated physical splits"),
    overwrite: bool = Query(False, description="기존 result 이미지가 있으면 덮어쓸지"),
    thickness: int = Query(2, ge=1, le=10),
    maxSide: int = Query(0, ge=0, le=4096, alias="maxSide"),
    limit: int = Query(0, ge=0, le=200000, description="0이면 전체, >0이면 최대 N장만 (split 당)"),
    body: Optional[BuildPreviewSetRequest] = Body(None),
):
    # ✅ FIX: query 우선, query가 없을 때만 body 사용 + body 'string' 방지
    run_id = (runId or "").strip()
    if not run_id and body is not None:
        run_id = _safe_body_run_id(body.run_id)

    if body is not None:
        # 옵션은 body가 유효할 때만 반영(원하면 query가 비었을 때만 덮도록 더 엄격히 할 수도 있음)
        if body.splits:
            splits = body.splits
        overwrite = bool(body.overwrite)
        thickness = int(body.thickness)
        maxSide = int(body.max_side)
        limit = int(body.limit)

    if not run_id:
        raise HTTPException(400, "runId is required")

    rec = _safe_read_json(_run_file(run_id))
    if not rec:
        raise HTTPException(404, "round0 export not found")

    export_rel_path = _get_export_rel_path_from_round0_registry(rec)
    if not export_rel_path:
        raise HTTPException(500, "exportRelPath missing in registry")

    export_dir = _resolve_export_local_dir(export_rel_path)
    if export_dir is None:
        raise HTTPException(
            status_code=501,
            detail="buildPreviewSet requires V1_EXPORT_LOCAL_ROOT (server-accessible export root).",
        )
    if not export_dir.exists() or not export_dir.is_dir():
        raise HTTPException(404, f"export dir not found on server: {export_dir}")

    physical_splits = _validate_physical_splits(splits)

    # ✅ result는 논리 split 3개 폴더로만 만든다
    result_root = export_dir / "result"
    _ensure_dir(result_root)
    for ls in ("pass", "fail", "miss"):
        _ensure_dir(result_root / ls)

    total_written = 0
    written_by_logical: Dict[str, int] = {"pass": 0, "fail": 0, "miss": 0}
    index_items: List[Dict[str, Any]] = []

    for src_split in physical_splits:
        logical_split = _logical_split_from_physical(src_split)

        img_dir = (export_dir / src_split / "images").resolve()
        lbl_dir = (export_dir / src_split / "labels").resolve()

        # ✅ 폴더 없으면 조용히 스킵 (round마다 pass_fail이 없을 수 있음)
        if not img_dir.exists() or not img_dir.is_dir():
            continue

        out_dir = (result_root / logical_split).resolve()
        _ensure_dir(out_dir)

        imgs = _iter_images(img_dir)
        if limit > 0:
            imgs = imgs[: int(limit)]

        for img_path in imgs:
            file_name = _preview_file_name(img_path.stem, src_split)
            out_path = (out_dir / file_name).resolve()

            if out_path.exists() and not overwrite:
                pass
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                img = _resize_keep_aspect(img, maxSide)

                lbl_path = (lbl_dir / f"{img_path.stem}.txt").resolve()
                labels = _read_yolo_labels(lbl_path)
                _draw_yolo_boxes(img, labels, thickness=thickness)

                ok, enc = cv2.imencode(".jpg", img)
                if not ok:
                    continue
                out_path.write_bytes(enc.tobytes())

            total_written += 1
            written_by_logical[logical_split] = int(written_by_logical.get(logical_split, 0)) + 1

            image_url = (
                f"/api/v1/results/round0/preview/image"
                f"?runId={run_id}&split={logical_split}&fileName={file_name}"
            )

            index_items.append(
                {
                    "split": logical_split,
                    "fileName": file_name,
                    "localPath": str(out_path),
                    "srcImagePath": str(img_path.resolve()),
                    "labelPath": str((lbl_dir / f"{img_path.stem}.txt").resolve()),
                    "imageUrl": image_url,
                }
            )

    result_rel_path = export_rel_path.rstrip("\\/") + "/result"
    idx_path = _preview_index_path(export_dir)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    idx_payload = {
        "runId": run_id,
        "exportRelPath": export_rel_path,
        "resultRelPath": result_rel_path,
        "exportLocalDir": str(export_dir.resolve()),
        "resultLocalDir": str((export_dir / "result").resolve()),
        "total": len(index_items),
        "items": index_items,
        "builtAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "splitPolicy": {"logical": ["pass", "fail", "miss"], "passIncludes": ["pass", "pass_fail"], "failIs": ["fail_fail"]},
    }
    idx_path.write_text(json.dumps(idx_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "runId": run_id,
        "exportRelPath": export_rel_path,
        "resultRelPath": result_rel_path,
        "writtenTotal": total_written,
        "writtenBySplit": written_by_logical,
        "previewIndexPath": str(idx_path),
        "previewItems": len(index_items),
        "splitPolicy": idx_payload["splitPolicy"],
    }


@router.get(
    "/round0/preview",
    response_model=PreviewIndexResponse,
    summary="round0 프리뷰 인덱스(리스트) 조회 (노출 split=pass/fail/miss)",
)
def preview_round0_index(runId: str = Query(..., description="조회할 runId")):
    run_id = runId.strip()
    if not run_id:
        raise HTTPException(400, "runId is required")

    rec = _safe_read_json(_run_file(run_id))
    if not rec:
        raise HTTPException(404, "round0 export not found")

    export_rel_path = _get_export_rel_path_from_round0_registry(rec)
    if not export_rel_path:
        raise HTTPException(500, "exportRelPath missing in registry")

    export_dir = _resolve_export_local_dir(export_rel_path)
    if export_dir is None:
        raise HTTPException(
            status_code=501,
            detail="preview requires V1_EXPORT_LOCAL_ROOT (server-accessible export root).",
        )
    if not export_dir.exists() or not export_dir.is_dir():
        raise HTTPException(404, f"export dir not found on server: {export_dir}")

    idx_path = _preview_index_path(export_dir)
    if not idx_path.exists():
        raise HTTPException(404, "preview index not found. call POST /results/round0/buildPreviewSet first.")

    payload = _safe_read_json(idx_path)
    if not payload:
        raise HTTPException(500, "preview index corrupted")

    items = payload.get("items") or []
    return PreviewIndexResponse(
        runId=payload.get("runId", run_id),
        exportRelPath=payload.get("exportRelPath", export_rel_path),
        resultRelPath=payload.get("resultRelPath", export_rel_path.rstrip("\\/") + "/result"),
        total=int(payload.get("total", len(items)) or len(items)),
        items=[PreviewIndexItem(**it) for it in items],
    )


@router.get(
    "/round0/preview/image",
    summary="round0 단일 이미지 프리뷰(prebuilt 우선, 없으면 즉석 draw) (split=pass/fail/miss)",
)
def preview_round0_image(
    runId: str = Query(..., description="조회할 runId"),
    split: Literal["pass", "fail", "miss"] = Query(..., description="logical split"),
    fileName: str = Query(..., alias="fileName", description="preview 파일명 (예: 000123.jpg 또는 000123__pf.jpg)"),
    thickness: int = Query(2, ge=1, le=10),
    maxSide: int = Query(0, ge=0, le=4096, alias="maxSide"),
):
    run_id = runId.strip()
    if not run_id:
        raise HTTPException(400, "runId is required")

    rec = _safe_read_json(_run_file(run_id))
    if not rec:
        raise HTTPException(404, "round0 export not found")

    export_rel_path = _get_export_rel_path_from_round0_registry(rec)
    if not export_rel_path:
        raise HTTPException(500, "exportRelPath missing in registry")

    export_dir = _resolve_export_local_dir(export_rel_path)
    if export_dir is None:
        raise HTTPException(
            status_code=501,
            detail="preview requires V1_EXPORT_LOCAL_ROOT (server-accessible export root).",
        )
    if not export_dir.exists() or not export_dir.is_dir():
        raise HTTPException(404, f"export dir not found on server: {export_dir}")

    if Path(fileName).suffix.lower() not in IMG_EXTS:
        raise HTTPException(400, f"unsupported image ext: {fileName}")

    prebuilt_path = (export_dir / "result" / split / fileName).resolve()
    if prebuilt_path.exists():
        return Response(content=prebuilt_path.read_bytes(), media_type="image/jpeg")

    src_split = _infer_source_split_from_preview_file(split, fileName)

    stem = Path(fileName).stem
    if split == "pass":
        stem = _strip_pf_suffix(stem)

    img_dir = (export_dir / src_split / "images").resolve()
    lbl_dir = (export_dir / src_split / "labels").resolve()

    src_img = _find_src_image_by_stem(img_dir, stem)
    if src_img is None:
        raise HTTPException(404, f"image not found for stem={stem} in {img_dir}")

    lbl_path = (lbl_dir / f"{stem}.txt").resolve()

    img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "failed to read image")

    img = _resize_keep_aspect(img, maxSide)

    labels = _read_yolo_labels(lbl_path)
    _draw_yolo_boxes(img, labels, thickness=thickness)

    ok, enc = cv2.imencode(".jpg", img)
    if not ok:
        raise HTTPException(500, "failed to encode image")

    return Response(content=enc.tobytes(), media_type="image/jpeg")


# =========================
# FINAL preview (3개 세트)
# =========================
@router.post(
    "/final/buildPreviewSet",
    summary="final 프리뷰 세트 생성 + final_preview_index.json 생성 (노출 split=pass/fail/miss)",
)
def build_preview_set_final(
    runId: Optional[str] = Query(None, description="대상 runId (query)"),
    splits: str = Query("pass,pass_fail,fail_fail,miss", description="comma-separated physical splits"),
    rounds: str = Query("", description="comma-separated roundNo list. empty이면 round_r* 자동 스캔"),
    overwrite: bool = Query(False, description="기존 result 이미지가 있으면 덮어쓸지"),
    thickness: int = Query(2, ge=1, le=10),
    maxSide: int = Query(0, ge=0, le=4096, alias="maxSide"),
    limit: int = Query(0, ge=0, le=200000, description="0이면 전체, >0이면 최대 N장만 (round/split 당)"),
    body: Optional[BuildFinalPreviewSetRequest] = Body(None),
):
    # ✅ FIX: query 우선, query가 없을 때만 body 사용 + body 'string' 방지
    run_id = (runId or "").strip()
    if not run_id and body is not None:
        run_id = _safe_body_run_id(body.run_id)

    if body is not None:
        if body.splits:
            splits = body.splits
        if body.rounds is not None:
            rounds = body.rounds
        overwrite = bool(body.overwrite)
        thickness = int(body.thickness)
        maxSide = int(body.max_side)
        limit = int(body.limit)

    if not run_id:
        raise HTTPException(400, "runId is required")

    # ✅ 핵심: final registry stub일 수 있으니 "실제" export_root를 run 기준으로 resolve
    export_root = _resolve_final_export_root_from_run(run_id)

    physical_splits = _validate_physical_splits(splits)

    round_nos = _parse_rounds_csv(rounds)
    if not round_nos:
        round_nos = _scan_round_nos(export_root)
    if not round_nos:
        raise HTTPException(404, f"no round directories found under export_root: {export_root}")

    (export_root / "result").mkdir(parents=True, exist_ok=True)

    all_items: List[Dict[str, Any]] = []
    round_summaries: List[Dict[str, Any]] = []

    for round_no in round_nos:
        round_dir = _resolve_round_dir(export_root, round_no)
        if not round_dir.exists() or not round_dir.is_dir():
            raise HTTPException(404, f"round dir not found: {round_dir}")

        result_root = (round_dir / "result").resolve()
        _ensure_dir(result_root)
        for ls in ("pass", "fail", "miss"):
            _ensure_dir(result_root / ls)

        round_items: List[Dict[str, Any]] = []

        for src_split in physical_splits:
            logical_split = _logical_split_from_physical(src_split)

            img_dir = (round_dir / src_split / "images").resolve()
            lbl_dir = (round_dir / src_split / "labels").resolve()

            # ✅ 없는 split 폴더는 스킵
            if not img_dir.exists() or not img_dir.is_dir():
                continue

            out_dir = (result_root / logical_split).resolve()

            imgs = _iter_images(img_dir)
            if limit > 0:
                imgs = imgs[: int(limit)]

            for img_path in imgs:
                file_name = _preview_file_name(img_path.stem, src_split)
                out_path = (out_dir / file_name).resolve()

                if out_path.exists() and not overwrite:
                    pass
                else:
                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is None:
                        continue

                    img = _resize_keep_aspect(img, maxSide)

                    lbl_path = (lbl_dir / f"{img_path.stem}.txt").resolve()
                    labels = _read_yolo_labels(lbl_path)
                    _draw_yolo_boxes(img, labels, thickness=thickness)

                    ok, enc = cv2.imencode(".jpg", img)
                    if not ok:
                        continue
                    out_path.write_bytes(enc.tobytes())

                image_url = (
                    f"/api/v1/results/final/preview/image"
                    f"?runId={run_id}&roundNo={int(round_no)}&split={logical_split}&fileName={file_name}"
                )

                item = {
                    "runId": run_id,
                    "roundNo": int(round_no),
                    "split": logical_split,
                    "fileName": file_name,
                    "localPath": str(out_path),
                    "srcImagePath": str(img_path.resolve()),
                    "labelPath": str((lbl_dir / f"{img_path.stem}.txt").resolve()),
                    "imageUrl": image_url,
                }
                round_items.append(item)
                all_items.append(item)

        round_idx_path = _final_round_preview_index_path(round_dir)
        round_idx_payload = {
            "runId": run_id,
            "roundNo": int(round_no),
            "roundDir": str(round_dir.resolve()),
            "total": len(round_items),
            "items": round_items,
            "builtAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "splitPolicy": {"logical": ["pass", "fail", "miss"], "passIncludes": ["pass", "pass_fail"], "failIs": ["fail_fail"]},
        }
        round_idx_path.parent.mkdir(parents=True, exist_ok=True)
        round_idx_path.write_text(json.dumps(round_idx_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        round_summaries.append(
            {
                "roundNo": int(round_no),
                "roundDir": str(round_dir.resolve()),
                "previewIndexPath": str(round_idx_path.resolve()),
                "total": len(round_items),
            }
        )

    master_path = _final_master_index_path(export_root)
    master_payload = {
        "runId": run_id,
        "exportRoot": str(export_root.resolve()),
        "total": len(all_items),
        "items": all_items,
        "rounds": round_summaries,
        "builtAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "splitPolicy": {"logical": ["pass", "fail", "miss"], "passIncludes": ["pass", "pass_fail"], "failIs": ["fail_fail"]},
    }
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_text(json.dumps(master_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "runId": run_id,
        "exportRoot": str(export_root.resolve()),
        "rounds": round_summaries,
        "total": len(all_items),
        "finalPreviewIndexPath": str(master_path.resolve()),
        "splitPolicy": master_payload["splitPolicy"],
    }


@router.get(
    "/final/preview",
    response_model=FinalPreviewIndexResponse,
    summary="final 프리뷰 인덱스(리스트) 조회 (노출 split=pass/fail/miss)",
)
def preview_final_index(
    runId: str = Query(..., description="조회할 runId"),
    offset: int = Query(0, ge=0, description="items offset"),
    limit: int = Query(2000, ge=1, le=200000, description="items limit"),
):
    run_id = runId.strip()
    if not run_id:
        raise HTTPException(400, "runId is required")

    export_root = _resolve_final_export_root_from_run(run_id)

    master_path = _final_master_index_path(export_root)
    if not master_path.exists():
        raise HTTPException(404, "final preview index not found. call POST /results/final/buildPreviewSet first.")

    payload = _safe_read_json(master_path)
    if not payload:
        raise HTTPException(500, "final preview index corrupted")

    items = payload.get("items") or []
    total = int(payload.get("total", len(items)) or len(items))
    off = int(offset)
    lim = int(limit)
    sliced = items[off : off + lim]
    rounds = payload.get("rounds") or []

    return FinalPreviewIndexResponse(
        runId=payload.get("runId", run_id),
        exportRoot=payload.get("exportRoot", str(export_root.resolve())),
        builtAt=payload.get("builtAt"),
        total=total,
        rounds=[FinalRoundSummary(**r) for r in rounds],
        items=[FinalPreviewItem(**it) for it in sliced],
    )


@router.get(
    "/final/preview/image",
    summary="final 단일 이미지 프리뷰(prebuilt 우선, 없으면 즉석 draw) (split=pass/fail/miss)",
)
def final_preview_image(
    runId: str = Query(..., description="조회할 runId"),
    roundNo: int = Query(..., ge=0, alias="roundNo", description="round number (0,1,2,...)"),
    split: Literal["pass", "fail", "miss"] = Query(..., description="logical split"),
    fileName: str = Query(..., alias="fileName", description="preview 파일명 (예: 000123.jpg 또는 000123__pf.jpg)"),
    thickness: int = Query(2, ge=1, le=10),
    maxSide: int = Query(0, ge=0, le=4096, alias="maxSide"),
):
    run_id = runId.strip()
    if not run_id:
        raise HTTPException(400, "runId is required")

    export_root = _resolve_final_export_root_from_run(run_id)

    if Path(fileName).suffix.lower() not in IMG_EXTS:
        raise HTTPException(400, f"unsupported image ext: {fileName}")

    round_dir = _resolve_round_dir(export_root, int(roundNo))
    if not round_dir.exists() or not round_dir.is_dir():
        raise HTTPException(404, f"round dir not found: {round_dir}")

    prebuilt_path = (round_dir / "result" / split / fileName).resolve()
    if prebuilt_path.exists():
        return Response(content=prebuilt_path.read_bytes(), media_type="image/jpeg")

    src_split = _infer_source_split_from_preview_file(split, fileName)

    stem = Path(fileName).stem
    if split == "pass":
        stem = _strip_pf_suffix(stem)

    img_dir = (round_dir / src_split / "images").resolve()
    lbl_dir = (round_dir / src_split / "labels").resolve()

    src_img = _find_src_image_by_stem(img_dir, stem)
    if src_img is None:
        raise HTTPException(404, f"image not found for stem={stem} in {img_dir}")

    lbl_path = (lbl_dir / f"{stem}.txt").resolve()

    img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "failed to read image")

    img = _resize_keep_aspect(img, maxSide)

    labels = _read_yolo_labels(lbl_path)
    _draw_yolo_boxes(img, labels, thickness=thickness)

    ok, enc = cv2.imencode(".jpg", img)
    if not ok:
        raise HTTPException(500, "failed to encode image")

    return Response(content=enc.tobytes(), media_type="image/jpeg")
