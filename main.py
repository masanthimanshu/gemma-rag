from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

from routes import api_router
from schema import RagHistory
from services import db, llm

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db(document_models=[RagHistory])
    llm.init_engine()
    yield
    llm.close_engine()
    await db.close_db()


app = FastAPI(version="1.0.0", lifespan=lifespan)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3500, reload=True)
