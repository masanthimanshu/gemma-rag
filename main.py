from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from routes import api_router
from schema import RagHistory
from services import db, llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db(document_models=[RagHistory])
    llm.init_engine()
    yield
    llm.close_engine()
    await db.close_db()


app = FastAPI(version="1.0.0", lifespan=lifespan, title="Gemma AI RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3500, reload=True)
