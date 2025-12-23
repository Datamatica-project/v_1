
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_TXT = LOG_DIR / "v1_fail_loop.log"
LOG_JSONL = LOG_DIR / "v1_fail_loop.jsonl"

JOB_LOG_DIR = LOG_DIR / "jobs"
EXPORT_LOG_DIR = LOG_DIR / "exports"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"[{timestamp()}] {msg}"
    print(line)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_json(data: dict) -> None:
    data["_time"] = timestamp()
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _parse_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _read_json_files(dirp: Path) -> List[Dict[str, Any]]:
    if not dirp.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(dirp.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def load_logs(
    *,
    scope: Optional[str] = None,
    ref_id: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 100,
    since: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    FastAPI logs router에서 호출.
    반환 아이템은 dto.LogItem에 맞춰 매핑될 것으로 가정하고
    가능한 필드들을 통일해서 채운다.

    scope:
      - "loop"   : v1_fail_loop.jsonl
      - "system" : txt/jsonl 모두
      - "ingest" : raw_ingest/status.json 은 현재 여기서 수집 안함(필요하면 확장)
      - "export" : logs/exports/*.json
      - "job" or "loop_job" : logs/jobs/*.json
    """
    scope_norm = (scope or "").strip().lower() if scope else None
    level_norm = (level or "").strip().upper() if level else None

    items: List[Dict[str, Any]] = []

    # 1) loop/system jsonl
    if scope_norm in (None, "loop", "system"):
        for obj in _read_jsonl(LOG_JSONL):
            t = _parse_time(obj.get("_time"))
            if since and t and t <= since:
                continue

            # level 추정(없으면 INFO)
            lv = str(obj.get("level", "INFO")).upper()
            if level_norm and lv != level_norm:
                continue

            # ref_id 필터 (image/job/run_id 등)
            if ref_id:
                rid = str(obj.get("ref_id") or obj.get("run_id") or obj.get("job_id") or obj.get("image") or "")
                if ref_id not in rid:
                    continue

            items.append({
                "time": obj.get("_time"),
                "scope": "loop",
                "level": lv,
                "ref_id": obj.get("run_id") or obj.get("job_id") or obj.get("image"),
                "message": obj.get("event") or obj.get("tag") or "log",
                "data": obj,
            })

    # 2) jobs/*.json
    if scope_norm in (None, "job", "loop_job", "loop"):
        for obj in _read_json_files(JOB_LOG_DIR):
            # updated_at epoch(float) → since 비교는 여기선 skip(필요하면 확장)
            if ref_id and ref_id not in str(obj.get("job_id", "")):
                continue

            items.append({
                "time": obj.get("updated_at"),
                "scope": "job",
                "level": "INFO" if obj.get("status") != "FAILED" else "ERROR",
                "ref_id": obj.get("job_id"),
                "message": f"job_status:{obj.get('status')}",
                "data": obj,
            })

    # 3) exports/*.json
    if scope_norm in (None, "export"):
        for obj in _read_json_files(EXPORT_LOG_DIR):
            if ref_id and ref_id not in str(obj.get("run_id", "")):
                continue
            items.append({
                "time": obj.get("updated_at") or obj.get("created_at"),
                "scope": "export",
                "level": "INFO",
                "ref_id": obj.get("run_id"),
                "message": f"export:{obj.get('status', 'UNKNOWN')}",
                "data": obj,
            })

    # 최신순 정렬(가능한 time key 기반)
    def _sort_key(x: Dict[str, Any]) -> float:
        t = x.get("time")
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, str):
            dt = _parse_time(t)
            if dt:
                return dt.timestamp()
            # isoformat이면 파싱 시도
            try:
                return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0
        return 0.0

    items.sort(key=_sort_key, reverse=True)

    if limit and len(items) > limit:
        items = items[:limit]
    return items
