from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from routes import api_router
from services import llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm.init_engine()
    yield
    llm.close_engine()


app = FastAPI(version="1.0.0", lifespan=lifespan)

app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3500, reload=True)
