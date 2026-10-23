"""RAG pipeline: PDF -> chunks -> TF-IDF retrieval (no PyTorch) -> Groq answer."""
import os
from dataclasses import dataclass
from typing import List

import fitz  # PyMuPDF
import numpy as np
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_SIZE = 800      # characters
CHUNK_OVERLAP = 150
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a concise, accurate assistant. Answer ONLY from the provided context. "
    "If the answer is not in the context, say \"I don't know.\""
)


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def chunk_pdf(file_bytes: bytes, filename: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_no, page in enumerate(doc, start=1):
            text = " ".join(page.get_text("text").split())
            start = 0
            while start < len(text):
                piece = text[start:start + CHUNK_SIZE].strip()
                if len(piece) > 50:
                    chunks.append(Chunk(piece, filename, page_no))
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class RagIndex:
    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("No text could be extracted from the PDF(s). Scanned PDFs need OCR.")
        self.chunks = chunks
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform([c.text for c in chunks])  # L2-normalised

    def retrieve(self, question: str, k: int = TOP_K) -> List[Chunk]:
        q = self.vec.transform([question])
        scores = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top if scores[i] > 0] or [self.chunks[i] for i in top[:1]]


def answer(question: str, contexts: List[Chunk]) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    context_text = "\n\n".join(
        f"[{c.source} p.{c.page}]\n{c.text}" for c in contexts
    )
    resp = client.chat.completions.create(
        model=os.getenv("RAG_MODEL", "llama-3.1-8b-instant"),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()
