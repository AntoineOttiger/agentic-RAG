"""Adapter RAGAS derrière l'interface `domain.ports.Judge` (ARCHITECTURE §6).

RAGAS est importé paresseusement. Nécessite un LLM et un embedder configurés côté
RAGAS ; ici on branche le LLM Mistral et l'embedder bge. Optionnel : si RAGAS n'est
pas installé, on lève une erreur explicite et l'appelant retombe sur `NativeJudge`.
"""

from __future__ import annotations

from domain.types import Answer, Query


class RagasJudge:
    """Évalue une réponse via les métriques RAGAS (faithfulness, correctness, ...)."""

    def __init__(self, llm_wrapper=None, embeddings_wrapper=None) -> None:
        self._llm = llm_wrapper
        self._emb = embeddings_wrapper

    def score(self, query: Query, answer: Answer, reference: str | None = None) -> dict[str, float]:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness, context_precision, faithfulness

        sample = {
            "question": [query.text],
            "answer": [answer.text],
            "contexts": [[rc.chunk.text for rc in answer.unique_retrieved[:8]]],
            "ground_truth": [reference or ""],
        }
        dataset = Dataset.from_dict(sample)
        metrics = [faithfulness, answer_correctness, context_precision]
        kwargs = {}
        if self._llm is not None:
            kwargs["llm"] = self._llm
        if self._emb is not None:
            kwargs["embeddings"] = self._emb
        result = evaluate(dataset, metrics=metrics, **kwargs)
        df = result.to_pandas()
        return {
            "answer_correctness": float(df["answer_correctness"].mean()),
            "faithfulness": float(df["faithfulness"].mean()),
            "context_precision": float(df["context_precision"].mean()),
        }
