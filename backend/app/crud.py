from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from . import models
import numpy as np


# ------------------ Create document ------------------
def create_document(db: Session, title: str, content: str, summary: str, chunks: list):
    """
    Creates a document and its associated chunks with embeddings.
    `chunks` should be a list of dicts: { "text": str, "embedding": list[float] }
    """
    db_doc = models.Document(
        title=title,
        content=content,
        summary=summary
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # Add chunks with embeddings
    for chunk in chunks:
        db_chunk = models.Chunk(
            doc_id=db_doc.id,
            text=chunk["text"],
            embedding=chunk["embedding"]  # already normalized in ai_utils
        )
        db.add(db_chunk)

    db.commit()
    return db_doc


# ------------------ Cosine similarity ------------------
def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------ Search chunks ------------------
def search_chunks(db: Session, query_embedding: list, top_k: int = 5):
    """
    Search the most similar chunks to a query embedding.
    Uses in-memory cosine similarity for now (works fine for small/mid scale).
    If scaling up, switch to pgvector extension in PostgreSQL for efficient ANN search.
    """
    chunks = db.query(models.Chunk).options(joinedload(models.Chunk.document)).all()
    if not chunks:
        return []

    results = []
    for c in chunks:
        score = cosine_similarity(query_embedding, c.embedding)
        results.append({
            "text": c.text,
            "doc_id": c.doc_id,
            "doc_title": c.document.title if c.document else None,
            "score": score
        })

    # sort by similarity
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]
