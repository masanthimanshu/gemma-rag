from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from schema import RagHistory, RagRequest, ResponseModel
from services import rag

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=ResponseModel)
def ingest(request: RagRequest):
    return {"message": rag.ingest(request.message, request.chat_id)}


@router.post("/response", response_model=ResponseModel)
async def response(request: RagRequest):
    result = rag.query(request.message, request.chat_id)
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
            message = await websocket.receive_text()
            full_response = []

            for chunk in rag.stream_query(message, chat_id):
                full_response.append(chunk)
                await websocket.send_text(chunk)

            doc = RagHistory(chat_id=chat_id, question=message,response="".join(full_response))
            await doc.insert()

    except WebSocketDisconnect:
        pass
