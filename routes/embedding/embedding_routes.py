from fastapi import APIRouter

from schema import ChatRequest
from services import llm

router = APIRouter(prefix="/embedding", tags=["Embedding"])


@router.post("/create")
def embedding(request: ChatRequest):
    return {"response": llm.embedding(request.message)}
