"""RagService — la façade in-process (ARCHITECTURE §1, décision 7).

Un point d'entrée unique `ask(query) -> Answer`, conçu pour un passage HTTP indolore :
la future GUI et un futur endpoint REST réutilisent CETTE façade sans réécriture.
"""

from __future__ import annotations

from core.pipeline import Pipeline
from domain.types import Answer, Query, QuestionType


class RagService:
    """Cache la pipeline derrière une interface minimale et stable."""

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def ask(self, query: str | Query, question_type: QuestionType | None = None) -> Answer:
        q = query if isinstance(query, Query) else Query(text=query, question_type=question_type)
        return self._pipeline.answer(q)
