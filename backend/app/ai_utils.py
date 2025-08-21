from __future__ import annotations
import re
import math
from typing import List, Dict, Any, Iterable

# --------- NLP / Models ---------
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

import torch

# Optional (better re-ranking); auto-disables if not installed
try:
    from sentence_transformers import CrossEncoder
    _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _RERANKER = None

# PDF / DOCX / TXT extraction (robust fallbacks)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

from PyPDF2 import PdfReader
import docx
import chardet

# =========================
# Models (singletons)
# =========================
_EMBEDDER = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# Try to load a stronger instruct model first (Llama-3 or Mistral), else fallback to Flan-T5
try:
    _TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    _MODEL = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    _QA = pipeline("text-generation", model=_MODEL, tokenizer=_TOKENIZER, max_new_tokens=256)
except Exception:
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        _MODEL = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            device_map="auto",
            torch_dtype=torch.float16
        )
        _QA = pipeline("text-generation", model=_MODEL, tokenizer=_TOKENIZER, max_new_tokens=256)
    except Exception:
        # Final fallback: Flan-T5
        _QA = pipeline("text2text-generation", model="google/flan-t5-large")

# Summarizer
_SUMMARIZER = pipeline("summarization", model="facebook/bart-large-cnn")

# =========================
# Utilities
# =========================
def _normalize(vec: Iterable[float]) -> List[float]:
    s = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / s for v in vec]

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# =========================
# Extraction
# =========================
def extract_text_from_file(filepath: str) -> str:
    if filepath.lower().endswith(".pdf"):
        if pdfplumber is not None:
            try:
                pages = []
                with pdfplumber.open(filepath) as pdf:
                    for p in pdf.pages:
                        pages.append(p.extract_text() or "")
                return clean_text("\n\n".join(pages))
            except Exception:
                pass
        reader = PdfReader(filepath)
        return clean_text(" ".join([(page.extract_text() or "") for page in reader.pages]))

    if filepath.lower().endswith(".docx"):
        doc = docx.Document(filepath)
        return clean_text("\n".join([p.text for p in doc.paragraphs]))

    if filepath.lower().endswith(".txt"):
        with open(filepath, "rb") as f:
            raw = f.read()
            encoding = (chardet.detect(raw)["encoding"] or "utf-8")
        with open(filepath, "r", encoding=encoding, errors="ignore") as f:
            return clean_text(f.read())

    return ""

# =========================
# Chunking
# =========================
def _split_on_separators(text: str, seps: List[str]) -> List[str]:
    parts = [text]
    for sep in seps:
        new_parts = []
        for p in parts:
            new_parts.extend(re.split(sep, p))
        parts = new_parts
    return [p.strip() for p in parts if p.strip()]

def smart_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if not text:
        return []
    blocks = _split_on_separators(text, seps=[r"\n{2,}", r"(?<=[\.\?\!])\s", r"\n", r" - ", r" • "])
    chunks, buff, size = [], [], 0
    for b in blocks:
        if size + len(b) > chunk_size and buff:
            chunks.append(" ".join(buff).strip())
            if overlap > 0:
                over = (" ".join(buff)).split()
                overlap_words = " ".join(over[-overlap // 5:])
                buff, size = [overlap_words, b], len(overlap_words) + len(b)
            else:
                buff, size = [b], len(b)
        else:
            buff.append(b)
            size += len(b)
    if buff:
        chunks.append(" ".join(buff).strip())
    return chunks

# =========================
# Embeddings
# =========================
def generate_embedding(text: str) -> List[float]:
    vec = _EMBEDDER.encode(text)
    return _normalize(vec.tolist())

def generate_embeddings_in_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> List[Dict[str, Any]]:
    return [{"text": c, "embedding": generate_embedding(c)} for c in smart_chunks(text, chunk_size, overlap)]

# =========================
# Re-ranking
# =========================
def rerank_candidates(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    if _RERANKER is None:
        return candidates
    pairs = [(query, c["text"]) for c in candidates]
    scores = _RERANKER.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

# =========================
# Summaries
# =========================
def generate_summary(text: str) -> str:
    if not text or len(text.split()) < 60:
        return text
    try:
        return _SUMMARIZER(text, max_length=160, min_length=60, do_sample=False)[0]["summary_text"]
    except Exception:
        return text

# =========================
# Answer synthesis
# =========================
def extract_date_from_context(context_text: str):
    match = re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})", context_text)
    if match:
        return match.group(1)
    return None

def clean_answer(query: str, answer: str) -> str:
    if not answer or answer.strip() in ["[1]", "[2]", "[3]", "", "Answer in one clear sentence."]:
        return "I could not find the answer in the document."
    if len(answer.split()) > 80:
        return generate_summary(answer)
    return answer

def synthesize_answer(query: str, contexts: List[Dict[str, Any]]) -> str:
    context_text = "\n\n".join([c['text'] for c in contexts[:3]])

    # Handle summaries
    if "summary" in query.lower():
        full_text = " ".join(c['text'] for c in contexts)
        return generate_summary(full_text)

    # Handle exam dates
    if "exam" in query.lower():
        extracted = extract_date_from_context(context_text)
        if extracted:
            return f"The Preliminary Examination is scheduled for {extracted}."

    # Handle last date
    if "last date" in query.lower():
        extracted = extract_date_from_context(context_text)
        if extracted:
            return f"The last date is {extracted}."

    # Fallback: LLM generation
    prompt = f"""
Question: {query}

Context:
{context_text}

Answer only from the context above. 
Reply concisely in one sentence. 
If the answer is not found, reply exactly: "I could not find the answer in the document."
"""
    raw_answer = _QA(prompt, do_sample=False)[0]["generated_text"].strip()
    return clean_answer(query, raw_answer)
