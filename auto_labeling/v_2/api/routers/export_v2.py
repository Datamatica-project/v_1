"""
Export API Router (앙상블 방식, v2)

Round별 및 최종 결과 Export
"""

from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import zipfile
import os
import shutil

from auto_labeling.v_2.api.dto import (
    ExportRoundRequest,
    ExportRoundResponse,
    ExportFinalRequest,
    ExportFinalResponse,
)

router = APIRouter(prefix="/api/v2/export", tags=["Export (v2 Ensemble)"])

# 경로 설정
V2_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = V2_ROOT.parent / "data"
RESULTS_ROOT = DATA_ROOT / "results"
EXPORTS_ROOT = DATA_ROOT / "exports"

EXPORTS_ROOT.mkdir(parents=True, exist_ok=True)


# ========================================
# Round별 Export
# ========================================
@router.post("/round", response_model=ExportRoundResponse)
def export_round(
    loop_id: str = Query(..., alias="loopId", description="Loop 식별자"),
    run_number: int = Query(..., alias="runNumber", ge=0, description="Round 번호 (0, 1, 2)")
):
    """
    Round별 결과 Export (ZIP 생성)

    Query Parameters:
    - loopId: loop_abc123
    - runNumber: 0 (Round 번호)

    Response:
    ```json
    {
        "resultCode": "SUCCESS",
        "message": "Round 0 exported successfully",
        "data": {
            "loopId": "loop_abc123",
            "runNumber": 0,
            "zipPath": "exports/loop_abc123/run_0.zip",
            "fileSize": 1048576
        }
    }
    ```

    ZIP 구조:
    ```
    run_0.zip
    ├── PASS_THREE/
    │   ├── images/
    │   └── labels/
    ├── PASS_TWO/
    │   ├── images/
    │   └── labels/
    ├── FAIL/
    │   ├── images/
    │   └── labels/
    └── MISS/
        └── images/
    ```
    """
    # 결과 디렉토리 확인
    result_dir = RESULTS_ROOT / loop_id / f"run_{run_number}"

    if not result_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Round {run_number} results not found for loop: {loop_id}"
        )

    # Export 디렉토리 생성
    export_dir = EXPORTS_ROOT / loop_id
    export_dir.mkdir(parents=True, exist_ok=True)

    zip_path = export_dir / f"run_{run_number}.zip"

    try:
        # ZIP 생성 (기존 파일 덮어쓰기)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for category in ["PASS_THREE", "PASS_TWO", "FAIL", "MISS"]:
                category_dir = result_dir / category

                if not category_dir.exists():
                    continue

                # images/ 추가
                images_dir = category_dir / "images"
                if images_dir.exists():
                    for img_file in images_dir.rglob("*"):
                        if img_file.is_file():
                            arcname = f"{category}/images/{img_file.name}"
                            zipf.write(img_file, arcname)

                # labels/ 추가
                labels_dir = category_dir / "labels"
                if labels_dir.exists():
                    for lbl_file in labels_dir.rglob("*"):
                        if lbl_file.is_file():
                            arcname = f"{category}/labels/{lbl_file.name}"
                            zipf.write(lbl_file, arcname)

        file_size = zip_path.stat().st_size

        return ExportRoundResponse(
            result_code="SUCCESS",
            message=f"Round {run_number} exported successfully",
            data={
                "loopId": loop_id,
                "runNumber": run_number,
                "zipPath": str(zip_path.relative_to(DATA_ROOT)),
                "fileSize": file_size
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {str(e)}"
        )


@router.get("/round/download")
def download_round_export(
    loop_id: str = Query(..., alias="loopId", description="Loop 식별자"),
    run_number: int = Query(..., alias="runNumber", ge=0, description="Round 번호")
):
    """
    Round Export ZIP 다운로드

    Query Parameters:
    - loopId: loop_abc123
    - runNumber: 0

    Returns:
    - ZIP 파일 (application/zip)
    """
    zip_path = EXPORTS_ROOT / loop_id / f"run_{run_number}.zip"

    if not zip_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Export not found. Run POST /api/v2/export/round first."
        )

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"loop_{loop_id}_run_{run_number}.zip"
    )


# ========================================
# 최종 결과 Export
# ========================================
@router.post("/final", response_model=ExportFinalResponse)
def export_final(
    loop_id: str = Query(..., alias="loopId", description="Loop 식별자")
):
    """
    최종 결과 Export (전체 Round 통합)

    Query Parameters:
    - loopId: loop_abc123

    Response:
    ```json
    {
        "resultCode": "SUCCESS",
        "message": "Final results exported successfully",
        "data": {
            "loopId": "loop_abc123",
            "zipPath": "exports/loop_abc123/final.zip",
            "fileSize": 5242880,
            "summary": {
                "totalPass": 960,
                "totalFail": 30,
                "totalMiss": 10
            }
        }
    }
    ```

    ZIP 구조:
    ```
    final.zip
    ├── PASS/         (모든 Round의 PASS_THREE + PASS_TWO)
    │   ├── images/
    │   └── labels/
    ├── FAIL/         (최종 Round의 FAIL)
    │   ├── images/
    │   └── labels/
    └── MISS/         (최종 Round의 MISS)
        └── images/
    ```

    처리:
    - 모든 Round의 PASS_THREE, PASS_TWO를 PASS로 병합
    - 최종 Round의 FAIL을 FAIL로
    - 최종 Round의 MISS를 MISS로
    """
    loop_dir = RESULTS_ROOT / loop_id

    if not loop_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Loop results not found: {loop_id}"
        )

    # 모든 Round 디렉토리 수집
    run_dirs = sorted(loop_dir.glob("run_*"))

    if not run_dirs:
        raise HTTPException(
            status_code=404,
            detail=f"No round results found for loop: {loop_id}"
        )

    # 최종 Round
    final_run_dir = run_dirs[-1]

    # Export 디렉토리
    export_dir = EXPORTS_ROOT / loop_id
    export_dir.mkdir(parents=True, exist_ok=True)

    # 임시 디렉토리 생성
    temp_final_dir = export_dir / "temp_final"
    if temp_final_dir.exists():
        shutil.rmtree(temp_final_dir)
    temp_final_dir.mkdir()

    pass_images_dir = temp_final_dir / "PASS" / "images"
    pass_labels_dir = temp_final_dir / "PASS" / "labels"
    fail_images_dir = temp_final_dir / "FAIL" / "images"
    fail_labels_dir = temp_final_dir / "FAIL" / "labels"
    miss_images_dir = temp_final_dir / "MISS" / "images"

    pass_images_dir.mkdir(parents=True)
    pass_labels_dir.mkdir(parents=True)
    fail_images_dir.mkdir(parents=True)
    fail_labels_dir.mkdir(parents=True)
    miss_images_dir.mkdir(parents=True)

    try:
        total_pass = 0
        total_fail = 0
        total_miss = 0

        # 모든 Round의 PASS_THREE, PASS_TWO 병합
        for run_dir in run_dirs:
            for category in ["PASS_THREE", "PASS_TWO"]:
                category_dir = run_dir / category

                if not category_dir.exists():
                    continue

                # images 복사
                src_images = category_dir / "images"
                if src_images.exists():
                    for img in src_images.iterdir():
                        if img.is_file():
                            shutil.copy2(img, pass_images_dir / img.name)
                            total_pass += 1

                # labels 복사
                src_labels = category_dir / "labels"
                if src_labels.exists():
                    for lbl in src_labels.iterdir():
                        if lbl.is_file():
                            shutil.copy2(lbl, pass_labels_dir / lbl.name)

        # 최종 Round의 FAIL, MISS 복사
        for category, target_img_dir, target_lbl_dir, counter in [
            ("FAIL", fail_images_dir, fail_labels_dir, "total_fail"),
            ("MISS", miss_images_dir, None, "total_miss")
        ]:
            category_dir = final_run_dir / category

            if not category_dir.exists():
                continue

            # images 복사
            src_images = category_dir / "images"
            if src_images.exists():
                for img in src_images.iterdir():
                    if img.is_file():
                        shutil.copy2(img, target_img_dir / img.name)
                        if counter == "total_fail":
                            total_fail += 1
                        else:
                            total_miss += 1

            # labels 복사 (MISS는 labels 없음)
            if target_lbl_dir:
                src_labels = category_dir / "labels"
                if src_labels.exists():
                    for lbl in src_labels.iterdir():
                        if lbl.is_file():
                            shutil.copy2(lbl, target_lbl_dir / lbl.name)

        # ZIP 생성
        zip_path = export_dir / "final.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_final_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_final_dir)
                    zipf.write(file_path, arcname)

        # 임시 디렉토리 삭제
        shutil.rmtree(temp_final_dir)

        file_size = zip_path.stat().st_size

        return ExportFinalResponse(
            result_code="SUCCESS",
            message="Final results exported successfully",
            data={
                "loopId": loop_id,
                "zipPath": str(zip_path.relative_to(DATA_ROOT)),
                "fileSize": file_size,
                "summary": {
                    "totalPass": total_pass,
                    "totalFail": total_fail,
                    "totalMiss": total_miss
                }
            }
        )

    except Exception as e:
        # 임시 디렉토리 정리
        if temp_final_dir.exists():
            shutil.rmtree(temp_final_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"Final export failed: {str(e)}"
        )


@router.get("/final/download")
def download_final_export(
    loop_id: str = Query(..., alias="loopId", description="Loop 식별자")
):
    """
    최종 Export ZIP 다운로드

    Query Parameters:
    - loopId: loop_abc123

    Returns:
    - ZIP 파일 (application/zip)
    """
    zip_path = EXPORTS_ROOT / loop_id / "final.zip"

    if not zip_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Final export not found. Run POST /api/v2/export/final first."
        )

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"loop_{loop_id}_final.zip"
    )
