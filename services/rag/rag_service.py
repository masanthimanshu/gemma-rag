import os
import uuid
from qdrant_client import QdrantClient, models

from services import llm


COLLECTION = "documents"
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    auth=(os.getenv("QDRANT_ADMIN"), os.getenv("QDRANT_PASS"))
)

try:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
except Exception:
    pass


def chunk_text(text: str, size: int = 300, overlap: int = 30):
    chunks = []
    step = size - overlap

    for start in range(0, len(text), step):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

    return chunks


def ingest(text: str, chat_id: str):
    chunks = chunk_text(text)
    if not chunks:
        return {"ingested_chunks": 0}

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
    results = client.query_points(
        COLLECTION,
        query=list(llm.embedding(message)),
        limit=n_results,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))]
        ),
    )

    sources = [p.payload["document"] for p in results.points] if results.points else []
    context = "\n\n".join(sources)
    prompt = f"Context:\n{context}\n\nQuestion: {message}\nAnswer based on context:"
    return prompt, sources


def query(message: str, chat_id: str, n_results: int = 3):
    prompt, sources = retrieve(message, chat_id, n_results)
    response = llm.chat(prompt)
    return {"answer": response["message"], "sources": sources}


def stream_query(message: str, chat_id: str, n_results: int = 3):
    prompt, _ = retrieve(message, chat_id, n_results)
    yield from llm.stream_chat(prompt)
