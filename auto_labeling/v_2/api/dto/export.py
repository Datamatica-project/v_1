"""
Export 관련 DTO (앙상블 방식)

Round별 및 최종 결과 Export DTO
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import Field
from .base import CamelModel


class ExportRoundRequest(CamelModel):
    """
    Round별 Export 요청

    Example:
        {
            "loopId": "loop_abc123",
            "runNumber": 0
        }
    """
    loop_id: str = Field(
        ...,
        alias="loopId",
        description="Loop 식별자"
    )

    run_number: int = Field(
        ...,
        alias="runNumber",
        ge=0,
        description="Round 번호 (0, 1, 2)"
    )


class ExportFinalRequest(CamelModel):
    """
    최종 결과 Export 요청

    Example:
        {
            "loopId": "loop_abc123"
        }
    """
    loop_id: str = Field(
        ...,
        alias="loopId",
        description="Loop 식별자"
    )


class ExportRoundResponse(CamelModel):
    """
    Round별 Export 응답

    Example:
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
    """
    result_code: str = Field(
        ...,
        alias="resultCode",
        description="결과 코드 (SUCCESS, ERROR)"
    )

    message: str = Field(
        ...,
        description="결과 메시지"
    )

    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Export 결과 데이터"
    )


class ExportFinalResponse(CamelModel):
    """
    최종 결과 Export 응답

    Example:
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
    """
    result_code: str = Field(
        ...,
        alias="resultCode",
        description="결과 코드 (SUCCESS, ERROR)"
    )

    message: str = Field(
        ...,
        description="결과 메시지"
    )

    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Export 결과 데이터"
    )
