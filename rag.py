"""RAG pipeline: PDF -> chunks -> pure-Python TF-IDF retrieval -> Groq answer.
No compiled dependencies (works on locked-down Windows laptops)."""
import os
from dataclasses import dataclass
from typing import List

import io
import math
import re
from collections import Counter

from groq import Groq
from pypdf import PdfReader

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


STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its if "
           "then than into can will you your not no do does how what which when where who".split())


def tokens(text: str) -> List[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def chunk_pdf(file_bytes: bytes, filename: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for page_no, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
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
        self.tf = [Counter(tokens(c.text)) for c in chunks]
        df = Counter(t for tf in self.tf for t in tf)
        n = len(chunks)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vecs = [self._vec(tf) for tf in self.tf]

    def _vec(self, tf: Counter) -> dict:
        v = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def retrieve(self, question: str, k: int = TOP_K) -> List[Chunk]:
        q = self._vec(Counter(tokens(question)))
        scores = [sum(w * v.get(t, 0.0) for t, w in q.items()) for v in self.vecs]
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        hits = [self.chunks[i] for i in top if scores[i] > 0]
        return hits or [self.chunks[top[0]]]


def answer(question: str, contexts: List[Chunk]) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    context_text = "\n\n".join(
        f"[{c.source} p.{c.page}]\n{c.text}" for c in contexts
    )
    resp = client.chat.completions.create(
        model=os.getenv("RAG_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()
