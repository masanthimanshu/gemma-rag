from typing import Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class RagRequest(BaseModel):
    user: str
    message: str


class ResponseModel(BaseModel):
    status: int = 200
    message: Any
