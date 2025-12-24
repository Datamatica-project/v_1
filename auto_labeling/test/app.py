# app.py
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from test import split_pass_fail


app = FastAPI(title="V1 Demo Pass (viz) API", version="0.2.0")


@app.get("/health")
def health():
    return {"ok": True}


def _extract_zip_to_dir(zip_bytes: bytes, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(dst_dir)


def _zip_dir(src_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                arc = p.relative_to(src_dir).as_posix()
                z.write(p, arcname=arc)
    return buf.getvalue()


@app.post("/splitPassFail")
async def split_pass_fail_api(
    imagesZip: UploadFile = File(...),
    weightsPath: str = Form(default=os.getenv("DEFAULT_WEIGHTS_PATH", "")),

    confThreshold: float = Form(0.7),
    allowEmptyBoxesPass: bool = Form(True),

    device: str = Form("0"),
    imgsz: int = Form(1280),

    copyMode: str = Form("copy"),

    savePassViz: bool = Form(True),

    exportPassLabels: bool = Form(False),
    exportLabelConf: float = Form(0.3),

    saveFailImages: bool = Form(False),
    saveFailList: bool = Form(True),
):
    if not weightsPath:
        raise HTTPException(status_code=400, detail="weightsPath is empty and DEFAULT_WEIGHTS_PATH not set")

    weights = Path(weightsPath)
    if not weights.exists():
        raise HTTPException(status_code=400, detail=f"weights not found: {weights}")

    zip_bytes = await imagesZip.read()

    with tempfile.TemporaryDirectory(prefix="demo_passfail_") as td:
        root = Path(td)
        in_root = root / "input"
        out_root = root / "output"

        _extract_zip_to_dir(zip_bytes, in_root)
        cand_b = in_root / "images"
        src_img_dir = cand_b if cand_b.exists() else in_root

        summary = split_pass_fail(
            weights=weights,
            src_img_dir=src_img_dir,
            out_dir=out_root,
            conf_threshold=float(confThreshold),
            allow_empty_boxes_pass=bool(allowEmptyBoxesPass),
            device=str(device),
            imgsz=int(imgsz),
            copy_mode=str(copyMode),

            export_pass_labels=bool(exportPassLabels),
            export_label_conf=float(exportLabelConf),
            save_pass_viz=bool(savePassViz),
            save_fail_images=bool(saveFailImages),
            save_fail_list=bool(saveFailList),
        )

        # 결과 zip 만들기
        result_zip = _zip_dir(out_root)

    headers = {"X-Summary": json.dumps(summary, ensure_ascii=False)}
    return Response(content=result_zip, media_type="application/zip", headers=headers)
