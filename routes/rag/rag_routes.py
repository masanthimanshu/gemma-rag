from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from schema import ChatRequest, RagHistory, ResponseModel
from services import rag

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=ResponseModel)
def ingest(request: ChatRequest):
    return {"message": rag.ingest(request.message, request.chat_id)}


@router.post("/response", response_model=ResponseModel)
async def response(request: ChatRequest):
    result = rag.query(
        request.message,
        request.chat_id,
        image=request.image,
        think=request.think,
        web_search=request.web_search,
    )

    await RagHistory(
        chat_id=request.chat_id,
        question=request.message,
        response=result["answer"],
    ).insert()

    return {"message": result}


@router.websocket("/stream/{chat_id}")
async def stream(websocket: WebSocket, chat_id: str):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                req = ChatRequest.model_validate_json(raw)
            except Exception:
                req = ChatRequest(message=raw, chat_id=chat_id)

            response = ""
            for chunk in rag.stream_query(
                req.message,
                chat_id,
                image=req.image,
                think=req.think,
                web_search=req.web_search,
            ):
                response += chunk
                await websocket.send_text(chunk)

            await RagHistory(chat_id=chat_id, question=req.message, response=response).insert()

    except WebSocketDisconnect:
        pass
