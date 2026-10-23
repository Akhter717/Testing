"""RAGAS evaluation using a separate Groq judge model + local embeddings."""
import asyncio
import os
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
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


def _build():
    judge = LangchainLLMWrapper(
        ChatGroq(
            model=os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            max_retries=5,
        )
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    )
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
