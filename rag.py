"""RAG pipeline: PDF -> chunks -> local embeddings -> FAISS -> Groq answer."""
import os
from dataclasses import dataclass
from typing import List

import faiss
import fitz  # PyMuPDF
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

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


_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(
            os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    return _embedder


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
        emb = get_embedder().encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, dtype="float32"))

    def retrieve(self, question: str, k: int = TOP_K) -> List[Chunk]:
        q = get_embedder().encode([question], normalize_embeddings=True)
        _, idx = self.index.search(np.asarray(q, dtype="float32"), min(k, len(self.chunks)))
        return [self.chunks[i] for i in idx[0] if i != -1]


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
