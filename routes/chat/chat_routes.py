from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from schema import ChatRequest, ResponseModel
from services import llm

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/response", response_model=ResponseModel)
def chat(request: ChatRequest):
    return llm.chat(request.message, think=request.think)


@router.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            for chunk in llm.stream_chat(message):
                await websocket.send_text(chunk)
    except WebSocketDisconnect:
        pass
