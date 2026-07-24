"""Étage 1 — les métriques de retrieval sont des fonctions pures et testables."""

from domain.types import Chunk, Evidence, RetrievedContext
from evaluation import retrieval_metrics as rm


def _ctx(chunk_id: str, url: str, score: float) -> RetrievedContext:
    return RetrievedContext(
        chunk=Chunk(id=chunk_id, text="x", source_url=url, source_title=url),
        score=score,
    )


GOLD = [Evidence(fact="f", source_url="u1"), Evidence(fact="f", source_url="u2")]


def test_hit_rate_hits_when_gold_in_topk():
    ctxs = [_ctx("a", "u3", 0.9), _ctx("b", "u1", 0.8)]
    assert rm.hit_rate(ctxs, GOLD, k=2) == 1.0


def test_hit_rate_misses_outside_topk():
    ctxs = [_ctx("a", "u3", 0.9), _ctx("b", "u1", 0.8)]
    assert rm.hit_rate(ctxs, GOLD, k=1) == 0.0


def test_recall_counts_fraction_of_gold():
    ctxs = [_ctx("a", "u1", 0.9), _ctx("b", "u4", 0.8)]
    assert rm.recall_at_k(ctxs, GOLD, k=2) == 0.5


def test_mrr_uses_first_gold_rank():
    ctxs = [_ctx("a", "u9", 0.9), _ctx("b", "u2", 0.8), _ctx("c", "u1", 0.7)]
    assert rm.mrr(ctxs, GOLD) == 0.5  # premier gold au rang 2


def test_precision_at_k_distinct_sources():
    ctxs = [_ctx("a", "u1", 0.9), _ctx("b", "u1", 0.8), _ctx("c", "u9", 0.7)]
    # sources distinctes: u1, u9 -> 1 gold sur 2
    assert rm.precision_at_k(ctxs, GOLD, k=2) == 0.5


def test_no_gold_returns_zero():
    ctxs = [_ctx("a", "u1", 0.9)]
    assert rm.hit_rate(ctxs, [], k=2) == 0.0
    assert rm.mrr(ctxs, []) == 0.0


def test_aggregate_averages():
    scores = [{"hit_rate": 1.0}, {"hit_rate": 0.0}]
    assert rm.aggregate(scores)["hit_rate"] == 0.5
