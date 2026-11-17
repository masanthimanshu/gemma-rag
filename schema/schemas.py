import base64
from typing import Any
from beanie import Document
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    chat_id: str
    message: str
    think: bool = False
    image: bytes | None = None

    @field_validator("image", mode="before")
    @classmethod
    def parse_image(cls, v: Any) -> bytes | None:
        if isinstance(v, str) and v.strip():
            b64 = v.split(",", 1)[1] if v.startswith("data:") and "," in v else v
            try:
                return base64.b64decode(b64)
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {e}")
        return v if isinstance(v, bytes) else None

    @classmethod
    def from_raw(cls, raw: str, chat_id: str) -> "ChatRequest":
        try:
            return cls.model_validate_json(raw)
        except Exception:
            return cls(message=raw, chat_id=chat_id)


class ResponseModel(BaseModel):
    status: int = 200
    message: Any


class IngestionRequest(BaseModel):
    chat_id: str
    message: str


class RagHistory(Document):
    chat_id: str
    question: str
    response: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "rag_history"
