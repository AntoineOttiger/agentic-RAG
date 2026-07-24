"""Smoke de bout en bout en backend mémoire : ingestion -> retrieval -> génération.

Prouve que le système tourne sans Qdrant/Mistral/bge et que la couche B produit bien
plus de sous-requêtes que la baseline sur une question multi-hop.
"""

from adapters.inmemory.llm import ExtractiveLLM
from adapters.inmemory.store import HashEmbedder, InMemoryStore
from adapters.llamaindex.ingestion import ingest_corpus
from apps.composition import build_service
from config.loader import load_config
from core.decompose.agentic import QueryDecomposer
from core.decompose.null import NullDecomposer
from domain.types import Query


def _index_sample(config):
    embedder = HashEmbedder()
    store = InMemoryStore(embedder)
    chunks = ingest_corpus(config.data.corpus_path, config.chunking)
    store.index(chunks)
    return store


def test_ingestion_produces_tagged_chunks():
    config = load_config("configs/smoke_baseline.yaml")
    chunks = ingest_corpus(config.data.corpus_path, config.chunking)
    assert len(chunks) > 0
    assert all(c.source_title for c in chunks)
    assert any(c.entities for c in chunks)


def test_baseline_answers_answerable_question():
    config = load_config("configs/smoke_baseline.yaml")
    store = _index_sample(config)
    service = build_service(config, store, ExtractiveLLM())
    answer = service.ask("What AI chip did TechCorp launch?")
    assert answer.retrieved
    assert len(answer.reasoning_trace.hops) == 1


def test_enriched_decomposes_multihop_question():
    config = load_config("configs/smoke_enriched.yaml")
    store = _index_sample(config)
    service = build_service(config, store, ExtractiveLLM())
    answer = service.ask(
        "Which company launched the Orion chip and how much revenue did NovaChip report?"
    )
    # La couche B doit produire >1 sous-requête sur une question à conjonction.
    assert len(answer.reasoning_trace.hops) >= 2


def test_null_question_flagged_unanswerable():
    config = load_config("configs/smoke_baseline.yaml")
    store = _index_sample(config)
    service = build_service(config, store, ExtractiveLLM())
    # Question sans aucun recouvrement lexical avec le corpus -> non répondable.
    answer = service.ask("Who won the 2050 Martian underwater chess tournament?")
    assert answer.is_answerable is False


def test_decomposer_and_null_share_pipeline_shape():
    # Même pipeline, seule la brique Decomposer change (ARCHITECTURE §6).
    assert hasattr(NullDecomposer(), "decompose")
    assert hasattr(QueryDecomposer(ExtractiveLLM()), "decompose")
