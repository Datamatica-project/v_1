from pydantic import BaseModel
from typing import Optional, Dict, Any

class RunLoopRequest(BaseModel):
    cfg_path: Optional[str] = None
    student_weight: Optional[str] = None
    teacher_weight: Optional[str] = None
    export_root: Optional[str] = None
    dry_run: bool = False

class RunLoopResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    round_idx: Optional[int] = None
    stats: Dict[str, Any] = {}