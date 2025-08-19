from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import inspect
import os
import json
from .db import init_db




from . import db, crud, ai_utils

app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Config ------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dependency to get DB session
def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

# ------------------ Startup check ------------------
@app.on_event("startup")
def startup_event():
    try:
        insp = inspect(db.engine)
        tables = insp.get_table_names()
        if tables:
            print(f"✅ Connected to DB. Found tables: {tables}")
        else:
            print("⚠️ Connected to DB but no tables found!")
    except Exception as e:
        print(f"❌ Failed to connect to DB: {e}")


# ------------------ Routes ------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save raw file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    content = ai_utils.extract_text_from_file(file_path)
    if not content.strip():
        raise HTTPException(status_code=400, detail="❌ Could not extract text from file.")

    # Summarize preview (first 1000 words only)
    preview_text = " ".join(content.split()[:1000])
    summary = ai_utils.generate_summary(preview_text)

    # Generate embeddings per chunk
    chunk_data = ai_utils.generate_embeddings_in_chunks(content)
    if not chunk_data:
        raise HTTPException(status_code=500, detail="❌ Embedding generation failed.")

    # Save document + its chunks in DB
    doc = crud.create_document(
        db,
        title=file.filename,
        content=content,
        summary=summary,
        chunks=chunk_data
    )

    return {
        "id": doc.id,
        "title": doc.title,
        "summary": doc.summary,
        "message": "✅ File uploaded and processed successfully"
    }


@app.get("/search")
def search_documents(q: str = Query(..., description="Search query"), db: Session = Depends(get_db)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="❌ Query cannot be empty.")

    query_embedding = ai_utils.generate_embedding(q)

    # 🔹 Step 1: Retrieve top-k chunks by embedding similarity
    results = crud.search_chunks(db, query_embedding, top_k=10)
    if not results:
        return {"query": q, "answer": "No relevant documents found.", "sources": []}

    # results should look like: [{"text": ..., "doc_id": ..., "doc_title": ..., "score": ...}, ...]

    # 🔹 Step 2: Apply reranker (if available in ai_utils)
    reranked = ai_utils.rerank_candidates(q, results)

    # 🔹 Step 3: Stream only the first valid answer
    def answer_stream():
        for chunk in reranked:
            prompt = f"Answer the question based on the context.\nContext: {chunk['text']}\nQuestion: {q}"
            try:
                ans = ai_utils.qa_generator(prompt, max_new_tokens=120, do_sample=False)[0]['generated_text'].strip()
                if ans and "I could not find" not in ans:
                    payload = {
                        "answer": ans,
                        "doc_id": chunk["doc_id"],
                        "doc_title": chunk["doc_title"]
                    }
                    yield json.dumps(payload) + "\n"
                    return
            except Exception as e:
                print(f"⚠️ QA error: {e}")

    return StreamingResponse(answer_stream(), media_type="application/json")



# Create tables on startup
init_db()