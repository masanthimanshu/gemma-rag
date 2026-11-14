import uuid
import chromadb

from services import llm

client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_or_create_collection(name="documents")


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


def ingest(text: str, user: str):
    chunks = chunk_text(text)
    if not chunks:
        return {"ingested_chunks": 0}

    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = [list(llm.embedding(c)) for c in chunks]
    metadatas = [{"user": user} for _ in chunks]

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return {"ingested_chunks": len(chunks)}


def query(message: str, user: str, n_results: int = 3):
    results = collection.query(
        query_embeddings=[list(llm.embedding(message))],
        n_results=n_results,
        where={"user": user},
    )

    sources = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(sources)

    prompt = f"Context:\n{context}\n\nQuestion: {message}\nAnswer based on context:"
    response = llm.chat(prompt)

    return {"answer": response["message"], "sources": sources}
