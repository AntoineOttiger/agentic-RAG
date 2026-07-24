"""Pipeline RAG — compose les ports. Dépend de `domain` uniquement.

Une seule pipeline pour A et A+B : la seule variable qui change est le `Decomposer`
injecté (voir ARCHITECTURE §6). Le chemin `decompose → retrieve → generate` est
identique par construction.
"""

from __future__ import annotations

from domain.ports import Decomposer, Generator, Retriever
from domain.types import (
    Answer,
    HopTrace,
    Query,
    ReasoningTrace,
    RetrievedContext,
    SubQuery,
)


def _build_filters(subquery: SubQuery) -> dict[str, object] | None:
    """Routing par métadonnées : période / entité portées par la sous-requête."""

    filters: dict[str, object] = {}
    if subquery.entity:
        filters["entity"] = subquery.entity
    if subquery.period:
        filters["period"] = subquery.period
    return filters or None


class Pipeline:
    """Orchestre la réponse à une question multi-hop.

    `answer()` cache toute la machinerie derrière une signature triviale
    (module profond, interface simple).
    """

    def __init__(
        self,
        decomposer: Decomposer,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 6,
        use_metadata_filters: bool = True,
    ) -> None:
        self._decomposer = decomposer
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._use_metadata_filters = use_metadata_filters

    def answer(self, query: Query) -> Answer:
        subqueries, decomp_usage = self._decomposer.decompose(query)

        hops: list[HopTrace] = []
        all_contexts: list[RetrievedContext] = []
        seen: set[str] = set()

        for sq in subqueries:
            filters = _build_filters(sq) if self._use_metadata_filters else None
            contexts = self._retriever.retrieve(sq, self._top_k, filters)
            for c in contexts:
                c.from_subquery = sq.text
            hops.append(HopTrace(subquery=sq, retrieved=contexts))
            for c in contexts:
                if c.chunk.id not in seen:
                    seen.add(c.chunk.id)
                    all_contexts.append(c)

        trace = ReasoningTrace(hops=hops)

        answer = self._generator.generate(query, all_contexts)
        # Le générateur produit text/citations/is_answerable/usage ; on rattache
        # la trace complète et le retrieval agrégé (contrat Answer, ARCHITECTURE §5).
        answer.retrieved = all_contexts
        answer.reasoning_trace = trace
        answer.usage = answer.usage + decomp_usage
        return answer
