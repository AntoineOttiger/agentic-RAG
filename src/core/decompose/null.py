"""NullDecomposer — la baseline A (pattern Null Object).

Renvoie la question telle quelle comme unique sous-requête. La trace de raisonnement
est alors dégénérée à un seul hop : aucun cas spécial ailleurs dans la pipeline.
"""

from __future__ import annotations

from domain.types import Query, SubQuery, TokenUsage


class NullDecomposer:
    """Ne décompose pas : 1 question -> 1 sous-requête, zéro appel LLM."""

    def decompose(self, query: Query) -> tuple[list[SubQuery], TokenUsage]:
        return [SubQuery(text=query.text, rationale="baseline: no decomposition")], TokenUsage()
