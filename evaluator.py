"""RAGAS evaluation using a separate Groq judge model + local embeddings."""
# Compatibility patch: ragas 0.2.x imports langchain_community's Vertex AI classes,
# which newer langchain-community versions no longer ship. Register harmless stand-ins
# BEFORE ragas is imported (same patch as Cell 1 of the notebook).
import sys
import types

_m = types.ModuleType("langchain_community.chat_models.vertexai")
_m.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _m
_m = types.ModuleType("langchain_community.llms.vertexai")
_m.VertexAI = type("VertexAI", (), {})
sys.modules["langchain_community.llms.vertexai"] = _m

import asyncio
import os
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
import zlib

import numpy as np
from rag import tokens
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    FactualCorrectness,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)


class HashingEmbeddings(Embeddings):
    """Stateless keyword-style embeddings (no PyTorch, no downloads, no compiled deps)."""

    DIM = 4096

    def _one(self, text):
        v = np.zeros(self.DIM)
        for t in tokens(text):
            v[zlib.crc32(t.encode()) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._one(t) for t in texts]

    def embed_query(self, text):
        return self._one(text)


def _build():
    judge = LangchainLLMWrapper(
        ChatGroq(
            model=os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
            max_retries=5,
        )
    )
    emb = LangchainEmbeddingsWrapper(HashingEmbeddings())
    return judge, emb


async def _score(metric, sample) -> Optional[float]:
    try:
        return float(await metric.single_turn_ascore(sample))
    except Exception as e:  # keep going if one metric fails (rate limit / bad JSON)
        return f"ERROR: {type(e).__name__}: {str(e)[:200]}"


async def _run(question, answer, contexts, reference):
    judge, emb = _build()
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference or None,
    )

    # Groq does not support n>1 completions -> strictness=1 for ResponseRelevancy.
    metrics = {
        "Faithfulness": Faithfulness(llm=judge),
        "Answer relevancy": ResponseRelevancy(llm=judge, embeddings=emb, strictness=1),
        "Context precision (no reference)": LLMContextPrecisionWithoutReference(llm=judge),
    }
    if reference:
        metrics.update({
            "Context precision (with reference)": LLMContextPrecisionWithReference(llm=judge),
            "Context recall": LLMContextRecall(llm=judge),
            "Factual correctness": FactualCorrectness(llm=judge),
        })

    results: Dict[str, object] = {}
    for name, metric in metrics.items():  # sequential: friendlier to free-tier rate limits
        results[name] = await _score(metric, sample)
    return results


def evaluate_one(question: str, answer: str, contexts: List[str], reference: str = ""):
    return asyncio.run(_run(question, answer, contexts, reference.strip()))
