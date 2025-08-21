from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import inspect
import os

from . import db, crud, ai_utils, schemas

app = FastAPI()

# ------------- CORS -------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # narrow this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Config -------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

# ------------- Health on startup -------------
@app.on_event("startup")
def startup_event():
    try:
        insp = inspect(db.engine)
        print(f"✅ Connected to DB. Found tables: {insp.get_table_names()}")
    except Exception as e:
        print(f"❌ DB connection failed: {e}")

# ------------- Upload -------------
@app.post("/upload", response_model=dict)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    content = ai_utils.extract_text_from_file(file_path)
    if not content.strip():
        raise HTTPException(status_code=400, detail="❌ Could not extract text from file.")

    # optional preview summary
    preview = " ".join(content.split()[:1500])
    summary = ai_utils.generate_summary(preview)

    # chunk + embed (normalized)
    chunk_data = ai_utils.generate_embeddings_in_chunks(content, chunk_size=800, overlap=120)
    if not chunk_data:
        raise HTTPException(status_code=500, detail="❌ Embedding generation failed.")

    # persist (expects your CRUD to insert Document and related Chunks)
    doc = crud.create_document(
        db,
        title=file.filename,
        content=content,
        summary=summary,
        chunks=chunk_data
    )

    return {"id": doc.id, "title": doc.title, "message": "✅ File uploaded & indexed."}

# ------------- Search -------------
@app.get("/search", response_model=schemas.AnswerOut)
def search_documents(q: str = Query(..., description="Search query"), db: Session = Depends(get_db)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="❌ Query cannot be empty.")

    query_emb = ai_utils.generate_embedding(q)

    # Step 1: vector search (top 10)
    # expected shape: [{"text": ..., "doc_id": ..., "doc_title": ..., "score": ...}, ...]
    results = crud.search_chunks(db, query_emb, top_k=10)
    if not results:
        return schemas.AnswerOut(query=q, answer="I could not find the exact exam date.", sources=[])

    # Step 2: re-rank (cross-encoder if available)
    reranked = ai_utils.rerank_candidates(q, results)

    # Step 3: synthesize a clean answer from top contexts
    answer = ai_utils.synthesize_answer(q, reranked[:3]).strip()

    # Step 4: sources (top-2 shown)
    srcs = [{"doc_id": r["doc_id"], "doc_title": r.get("doc_title", "Document")} for r in reranked[:2]]

    return schemas.AnswerOut(query=q, answer=answer, sources=srcs)
