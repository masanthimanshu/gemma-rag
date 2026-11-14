from fastapi import APIRouter

from schema import RagRequest, ResponseModel
from services import rag

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=ResponseModel)
def ingest(request: RagRequest):
    return {"message": rag.ingest(request.message, request.user)}


@router.post("/response", response_model=ResponseModel)
def response(request: RagRequest):
    return {"message": rag.query(request.message, request.user)}
