"""
Loop 관련 DTO (앙상블 방식)

3모델 앙상블 Loop 시작 및 상태 조회 DTO
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import Field
from .base import CamelModel


class EnsembleLoopRequest(CamelModel):
    """
    앙상블 Loop 시작 요청

    Example:
        {
            "models": ["yolo", "model2", "model3"],
            "configOverride": {
                "maxRounds": 3,
                "confThreshold": 0.5,
                "failThreshold": 0.01,
                "minFailCount": 100,
                "patience": 2,
                "iouThreshold": 0.5
            }
        }
    """
    models: List[str] = Field(
        default=["yolo", "model2", "model3"],
        description="사용할 모델 리스트 (yolo, model2, model3)"
    )

    config_override: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="configOverride",
        description="설정 오버라이드 (maxRounds, confThreshold 등)"
    )


class EnsembleLoopResponse(CamelModel):
    """
    앙상블 Loop 시작 응답

    Example:
        {
            "loopId": "loop_abc123",
            "runId": "run_20250106_120000_xyz789",
            "status": "STARTED",
            "message": "Ensemble loop started"
        }
    """
    loop_id: str = Field(
        ...,
        alias="loopId",
        description="Loop 식별자"
    )

    run_id: str = Field(
        ...,
        alias="runId",
        description="Run 식별자"
    )

    status: str = Field(
        ...,
        description="Loop 상태 (STARTED, RUNNING, COMPLETED, FAILED)"
    )

    message: Optional[str] = Field(
        default=None,
        description="상태 메시지"
    )


class RoundResult(CamelModel):
    """
    Round 결과

    Example:
        {
            "round": 0,
            "total": 1000,
            "passThree": 650,
            "passTwo": 200,
            "fail": 100,
            "miss": 50,
            "failMissRatio": 0.15
        }
    """
    round: int = Field(..., description="Round 번호 (0, 1, 2)")
    total: int = Field(..., description="전체 이미지 수")
    pass_three: int = Field(..., alias="passThree", description="3개 모두 PASS")
    pass_two: int = Field(..., alias="passTwo", description="2개 PASS, 1개 FAIL")
    fail: int = Field(..., description="1개 PASS, 2개 FAIL")
    miss: int = Field(..., description="3개 모두 FAIL")
    fail_miss_ratio: float = Field(..., alias="failMissRatio", description="FAIL+MISS 비율")


class LoopStatusResponse(CamelModel):
    """
    Loop 상태 조회 응답

    Example:
        {
            "loopId": "loop_abc123",
            "runId": "run_20250106_120000_xyz789",
            "status": "RUNNING",
            "stats": {
                "currentRound": 1,
                "totalRounds": 3,
                "roundHistory": [...],
                "latestFailMissRatio": 0.15,
                "results": {
                    "note": "Results are built by /api/v2/results/* endpoints",
                    "suggested": {
                        "roundPreview": "/api/v2/results/round/{round_number}/preview?loopId=xxx",
                        "finalPreview": "/api/v2/results/final/preview?loopId=xxx"
                    }
                }
            }
        }
    """
    loop_id: str = Field(..., alias="loopId", description="Loop 식별자")
    run_id: str = Field(..., alias="runId", description="Run 식별자")
    status: str = Field(..., description="Loop 상태")
    stats: Dict[str, Any] = Field(..., description="Loop 통계")
