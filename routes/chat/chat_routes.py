from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from schema import ChatRequest, RagHistory, ResponseModel
from services import llm, search

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/response", response_model=ResponseModel)
async def chat(request: ChatRequest):
    message = search.prepare_message(request.message)
    result = llm.chat(message, image=request.image, think=request.think)

    await RagHistory(
        chat_id=request.chat_id,
        question=request.message,
        response=result["message"],
    ).insert()

    return result


@router.get("/history/{chat_id}", response_model=ResponseModel)
async def history(chat_id: str):
    data = RagHistory.find(RagHistory.chat_id == chat_id)
    chats = await data.sort(+RagHistory.created_at).to_list()
    return {"message": chats}


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

            message = search.prepare_message(req.message)
            response = ""
            for chunk in llm.stream_chat(message, image=req.image, think=req.think):
                response += chunk
                await websocket.send_text(chunk)

            await RagHistory(chat_id=chat_id, question=req.message, response=response).insert()
    except WebSocketDisconnect:
        pass
