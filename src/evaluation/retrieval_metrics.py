"""Métriques de retrieval — FONCTIONS PURES, 100 % déterministes, sans LLM.

Étage 1 de l'évaluation (ARCHITECTURE §7.1). On compare mécaniquement les contextes
récupérés à l'evidence en or annotée : Hit Rate, Recall@k, MRR, Precision@k.

Le matching gold ↔ retrieved se fait par **source** (URL prioritaire, sinon titre
normalisé), car l'evidence MultiHop-RAG identifie des articles, pas des chunks.
"""

from __future__ import annotations

from domain.types import Evidence, RetrievedContext


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _source_key(url: str | None, title: str | None) -> str:
    return _norm(url) or _norm(title)


def gold_sources(evidence: list[Evidence]) -> set[str]:
    return {
        key
        for ev in evidence
        if (key := _source_key(ev.source_url, ev.source_title))
    }


def retrieved_sources(contexts: list[RetrievedContext]) -> list[str]:
    """Sources des contextes récupérés, dans l'ordre du ranking (avec doublons)."""

    return [_source_key(rc.chunk.source_url, rc.chunk.source_title) for rc in contexts]


def hit_rate(contexts: list[RetrievedContext], evidence: list[Evidence], k: int) -> float:
    """1.0 si au moins une source or apparaît dans le top-k, sinon 0.0."""

    gold = gold_sources(evidence)
    if not gold:
        return 0.0
    top = set(retrieved_sources(contexts)[:k])
    return 1.0 if gold & top else 0.0


def recall_at_k(contexts: list[RetrievedContext], evidence: list[Evidence], k: int) -> float:
    """Fraction des sources or présentes dans le top-k."""

    gold = gold_sources(evidence)
    if not gold:
        return 0.0
    top = set(retrieved_sources(contexts)[:k])
    return len(gold & top) / len(gold)


def precision_at_k(contexts: list[RetrievedContext], evidence: list[Evidence], k: int) -> float:
    """Fraction des sources or parmi les k premières sources distinctes récupérées."""

    gold = gold_sources(evidence)
    if not gold:
        return 0.0
    seen: list[str] = []
    for src in retrieved_sources(contexts):
        if src and src not in seen:
            seen.append(src)
        if len(seen) >= k:
            break
    if not seen:
        return 0.0
    return len([s for s in seen if s in gold]) / len(seen)


def mrr(contexts: list[RetrievedContext], evidence: list[Evidence]) -> float:
    """Reciprocal rank de la 1re source or récupérée."""

    gold = gold_sources(evidence)
    if not gold:
        return 0.0
    for rank, src in enumerate(retrieved_sources(contexts), start=1):
        if src in gold:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    contexts: list[RetrievedContext],
    evidence: list[Evidence],
    k: int,
) -> dict[str, float]:
    """Toutes les métriques de retrieval pour une question."""

    return {
        "hit_rate": hit_rate(contexts, evidence, k),
        f"recall@{k}": recall_at_k(contexts, evidence, k),
        f"precision@{k}": precision_at_k(contexts, evidence, k),
        "mrr": mrr(contexts, evidence),
    }


def aggregate(per_question: list[dict[str, float]]) -> dict[str, float]:
    """Moyenne des métriques sur un ensemble de questions."""

    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {k: sum(d.get(k, 0.0) for d in per_question) / len(per_question) for k in keys}
