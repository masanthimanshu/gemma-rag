import os
from beanie import init_beanie
from pymongo import AsyncMongoClient

client = None


async def init_db(document_models=None):
    global client
    client = AsyncMongoClient(os.getenv("MONGO_URI"))
    await init_beanie(
        database=client.get_default_database(default="gemma_rag"),
        document_models=document_models or [],
    )


async def close_db():
    global client
    if client:
        await client.close()
        client = None
