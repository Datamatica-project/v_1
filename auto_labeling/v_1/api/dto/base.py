from pydantic import BaseModel, ConfigDict

def to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])

class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

class ErrorResponse(CamelModel):
    code: str
    message: str
    detail: dict | None = None
