import os
import uuid
import logging
from qdrant_client import QdrantClient, models

from services import llm, search

logger = logging.getLogger(__name__)

COLLECTION = "documents"
user, password = os.getenv("QDRANT_USER"), os.getenv("QDRANT_PASS")
AUTH = (user, password) if user and password else None
client = QdrantClient(url=os.getenv("QDRANT_URL"), auth=AUTH)


def ensure_collection() -> None:
    try:
        if not client.collection_exists(COLLECTION):
            client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            )
    except Exception as e:
        logger.warning(f"Qdrant collection '{COLLECTION}' check/create failed: {e}")


ensure_collection()


def chunk_text(text: str, size: int = 300, overlap: int = 30):
    chunks, step = [], size - overlap
    for i in range(0, len(text), step):
        if c := text[i : i + size].strip():
            chunks.append(c)
        if i + size >= len(text):
            break
    return chunks


def ingest(text: str, chat_id: str):
    ensure_collection()
    chunks = chunk_text(text)
    if chunks:
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=list(llm.embedding(c)),
                payload={"document": c, "chat_id": chat_id},
            )
            for c in chunks
        ]
        client.upsert(COLLECTION, points=points)
    return {"ingested_chunks": len(chunks)}


def retrieve(message: str, chat_id: str, n_results: int = 3):
    try:
        results = client.query_points(
            COLLECTION,
            query=list(llm.embedding(message)),
            limit=n_results,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))]
            ),
        )
        points = results.points or []
    except Exception as e:
        logger.error(f"Error querying Qdrant: {e}")
        points = []

    local = [p.payload.get("document", "") for p in points if p.payload]
    do_web = search.should_search_web(message, points)
    web = search.search(message, limit=3) if do_web else []

    docs = []
    if local:
        docs.append("Local Documents:\n" + "\n\n".join(local))
    if web:
        docs.append("Web Search Results:\n" + "\n\n".join(
            f"{w['title']}: {w['content']} ({w['url']})" for w in web
        ))

    prompt = f"Context:\n{'\n\n'.join(docs)}\n\nQuestion: {message}\nAnswer based on context:"
    return prompt, ({"local": local, "web": web} if do_web else local)


def query(message: str, chat_id: str, image: bytes | None = None, n_results: int = 3, think: bool = False):
    prompt, sources = retrieve(message, chat_id, n_results)
    return {"answer": llm.chat(prompt, image=image, think=think)["message"], "sources": sources}


def stream_query(message: str, chat_id: str, image: bytes | None = None, n_results: int = 3, think: bool = False):
    prompt, _ = retrieve(message, chat_id, n_results)
    yield from llm.stream_chat(prompt, image=image, think=think)
