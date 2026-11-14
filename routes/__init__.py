from fastapi import APIRouter

from .chat.chat_routes import router as chat_router
from .health.health_routes import router as health_router
from .embedding.embedding_routes import router as embedding_router


api_router = APIRouter()


api_router.include_router(chat_router)
api_router.include_router(health_router)
api_router.include_router(embedding_router)
