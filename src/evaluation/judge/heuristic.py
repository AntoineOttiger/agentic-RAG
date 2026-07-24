"""Juge heuristique — sans LLM, déterministe. Pour smoke run offline et tests.

Approxime les métriques par recouvrement lexical. NE remplace PAS un vrai juge pour
le livrable : sert uniquement à valider la plomberie de l'étage 2 sans clé API.
"""

from __future__ import annotations

from domain.types import Answer, Query


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2}


class HeuristicJudge:
    def score(self, query: Query, answer: Answer, reference: str | None = None) -> dict[str, float]:
        ans = _tokens(answer.text)
        ctx = set().union(*[_tokens(rc.chunk.text) for rc in answer.unique_retrieved]) if answer.unique_retrieved else set()
        ref = _tokens(reference or "")

        faithfulness = len(ans & ctx) / len(ans) if ans else 0.0
        correctness = len(ans & ref) / len(ref) if ref else 0.0
        relevance = len(_tokens(query.text) & ctx) / len(_tokens(query.text)) if ctx else 0.0
        return {
            "answer_correctness": round(correctness, 3),
            "faithfulness": round(faithfulness, 3),
            "context_relevance": round(relevance, 3),
        }
