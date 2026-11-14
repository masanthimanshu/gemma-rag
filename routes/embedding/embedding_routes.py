from fastapi import APIRouter

from schema import ChatRequest, ResponseModel
from services import llm

router = APIRouter(prefix="/embedding", tags=["Embedding"])


@router.post("/create", response_model=ResponseModel)
def embedding(request: ChatRequest):
    return {"message": llm.embedding(request.message)}
