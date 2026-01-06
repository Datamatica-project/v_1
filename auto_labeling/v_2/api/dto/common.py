from __future__ import annotations
from typing import Generic, Optional, TypeVar
from pydantic import Field
from .base import CamelModel

T = TypeVar("T")


class ApiEnvelope(CamelModel, Generic[T]):
    resultCode: str = Field(..., description="SUCCESS | FAIL")
    errorCode: Optional[str] = None
    message: Optional[str] = None
    data: Optional[T] = None
