from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from schema import ChatRequest
from services import llm

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/response")
def chat(request: ChatRequest):
    return llm.chat(request.message)


@router.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            for chunk in llm.stream_chat(message):
                await websocket.send_json(chunk)
    except WebSocketDisconnect:
        pass
