from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import docx
import chardet

# ------------------ Load models ------------------
embedder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# Fast summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Stronger generation model for QA
qa_generator = pipeline("text2text-generation", model="google/flan-t5-large")

# Optional reranker
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except:
    reranker = None


# ------------------ Text extraction ------------------
def extract_text_from_file(filepath: str) -> str:
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return " ".join([page.extract_text() or "" for page in reader.pages])

    elif filepath.endswith(".docx"):
        doc = docx.Document(filepath)
        return " ".join([p.text for p in doc.paragraphs])

    elif filepath.endswith(".txt"):
        with open(filepath, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"] or "utf-8"
        with open(filepath, "r", encoding=encoding, errors="ignore") as f:
            return f.read()

    return ""


# ------------------ Chunking ------------------
def chunk_text(text, max_tokens=300):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i+max_tokens])


# ------------------ Summarization ------------------
def generate_summary(text: str) -> str:
    if not text.strip():
        return "No content to summarize."
    words = text.split()
    if len(words) < 50:
        return text

    chunks = list(chunk_text(text, max_tokens=400))
    summaries = []
    for chunk in chunks:
        try:
            result = summarizer(chunk, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
            summaries.append(result)
        except Exception:
            summaries.append(chunk)
    return " ".join(summaries)


# ------------------ Embeddings ------------------
def generate_embedding(text: str) -> list:
    return embedder.encode(text, convert_to_numpy=True).tolist()

def generate_embeddings_in_chunks(text: str, chunk_size=300):
    return [{"text": chunk, "embedding": generate_embedding(chunk)} for chunk in chunk_text(text, max_tokens=chunk_size)]


# ------------------ Reranking ------------------
def rerank_candidates(query: str, candidates: list) -> list:
    if not reranker or not candidates:
        return candidates
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


# ------------------ RAG-style QA ------------------
def answer_query(query: str, retrieved_chunks: list, use_summary: bool = True) -> str:
    """
    Build context + query → LLM → professional explanation
    """
    if not retrieved_chunks:
        return "I couldn't find any relevant information."

    # Take top few chunks as context
    context_texts = [c["text"] for c in retrieved_chunks[:5]]
    context = "\n".join(context_texts)

    if use_summary:
        context = generate_summary(context)

    prompt = f"""You are an expert assistant. 
Use the following retrieved context to answer the question in a clear, professional, and well-explained way.

Context:
{context}

Question: {query}

Answer:"""

    response = qa_generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return response
