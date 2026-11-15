import os
import uuid
from qdrant_client import QdrantClient, models

from services import llm, search

COLLECTION = "documents"
AUTH = (os.getenv("QDRANT_USER"), os.getenv("QDRANT_PASS"))
client = QdrantClient(url=os.getenv("QDRANT_URL"), auth=AUTH)

try:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
except Exception:
    pass


def chunk_text(text: str, size: int = 300, overlap: int = 30):
    chunks, step = [], size - overlap
    for i in range(0, len(text), step):
        if c := text[i : i + size].strip():
            chunks.append(c)
        if i + size >= len(text):
            break
    return chunks


def ingest(text: str, chat_id: str):
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


def retrieve(message: str, chat_id: str, n_results: int = 3, web_search: bool = False):
    results = client.query_points(
        COLLECTION,
        query=list(llm.embedding(message)),
        limit=n_results,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))]
        ),
    )

    local = [p.payload["document"] for p in results.points] if results.points else []
    web = search.search(message, limit=3) if web_search else []

    context = []
    if local:
        context.append("Local Documents:\n" + "\n\n".join(local))
    if web:
        context.append("Web Search Results:\n" + "\n\n".join(
            f"{w['title']}: {w['content']} ({w['url']})" for w in web
        ))

    prompt = f"Context:\n{'\n\n'.join(context)}\n\nQuestion: {message}\nAnswer based on context:"
    sources = {"local": local, "web": web} if web_search else local
    return prompt, sources


def query(message: str, chat_id: str, n_results: int = 3, web_search: bool = False, think: bool = False):
    prompt, sources = retrieve(message, chat_id, n_results, web_search)
    return {"answer": llm.chat(prompt, think=think)["message"], "sources": sources}


def stream_query(message: str, chat_id: str, n_results: int = 3, web_search: bool = False, think: bool = False):
    prompt, _ = retrieve(message, chat_id, n_results, web_search)
    yield from llm.stream_chat(prompt, think=think)
