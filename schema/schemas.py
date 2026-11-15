from typing import Any
from beanie import Document
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    chat_id: str
    message: str
    think: bool = False
    image: bytes | None = None


class ResponseModel(BaseModel):
    status: int = 200
    message: Any


class IngestionRequest(BaseModel):
    message: str


class RagHistory(Document):
    chat_id: str
    question: str
    response: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "rag_history"
