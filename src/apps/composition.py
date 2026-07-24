"""Composition root — l'unique endroit qui assemble config + adapters + cœur.

C'est la couche la plus externe (elle a le droit de tout importer). Elle traduit une
`RunConfig` en un `RagService` prêt à l'emploi et en composants d'évaluation, en
choisissant le backend :

- "real"   : bge + Qdrant + Mistral (prod / vrai livrable).
- "memory" : HashEmbedder + InMemoryStore + ExtractiveLLM (offline, tests, smoke).
- "auto"   : tente "real", retombe sur "memory" si une dépendance/clé manque.

Le choix se fait par la variable d'env AGENTIC_RAG_BACKEND (défaut "auto").
"""

from __future__ import annotations

import os

from config.schema import RunConfig, Variant
from core.decompose.agentic import QueryDecomposer
from core.decompose.null import NullDecomposer
from core.pipeline import Pipeline
from core.service import RagService


def resolve_backend(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("AGENTIC_RAG_BACKEND") or "auto").lower()


# --------------------------------------------------------------------------
# Briques individuelles
# --------------------------------------------------------------------------
def build_embedder(config: RunConfig, backend: str):
    if backend == "memory":
        from adapters.inmemory.store import HashEmbedder

        return HashEmbedder()
    from adapters.embeddings.bge import BgeEmbedder

    return BgeEmbedder(config.embedding)


def build_chat_llm(config: RunConfig, backend: str):
    if backend == "memory":
        from adapters.inmemory.llm import ExtractiveLLM

        return ExtractiveLLM()
    from adapters.llm.mistral import MistralChatLLM

    return MistralChatLLM(config.llm)


def build_store(config: RunConfig, embedder, backend: str):
    """Renvoie un objet qui est à la fois Indexer et Retriever."""

    if backend == "memory":
        from adapters.inmemory.store import InMemoryStore

        return InMemoryStore(embedder)
    from adapters.qdrant.store import QdrantStore

    return QdrantStore(config.vector_store, embedder)


# --------------------------------------------------------------------------
# Assemblages
# --------------------------------------------------------------------------
def build_decomposer(config: RunConfig, chat_llm):
    if config.variant == Variant.ENRICHED:
        return QueryDecomposer(chat_llm)
    return NullDecomposer()


def build_service(config: RunConfig, store, chat_llm) -> RagService:
    from adapters.llm.generator import LLMGenerator

    decomposer = build_decomposer(config, chat_llm)
    generator = LLMGenerator(chat_llm)
    pipeline = Pipeline(
        decomposer=decomposer,
        retriever=store,
        generator=generator,
        top_k=config.retrieval.top_k,
        use_metadata_filters=config.retrieval.use_metadata_filters,
    )
    return RagService(pipeline)


def build_judge(config: RunConfig, chat_llm, backend: str):
    if not config.judge.enabled:
        return None
    if backend == "memory":
        from evaluation.judge.heuristic import HeuristicJudge

        return HeuristicJudge()
    if config.judge.provider == "ragas":
        try:
            from evaluation.judge.ragas_judge import RagasJudge

            return RagasJudge()
        except Exception:
            pass
    from evaluation.judge.native_judge import NativeJudge

    return NativeJudge(chat_llm)


def try_build_real(config: RunConfig):
    """Tente le backend réel ; renvoie None si indisponible (deps/clé manquantes)."""

    try:
        embedder = build_embedder(config, "real")
        chat_llm = build_chat_llm(config, "real")
        store = build_store(config, embedder, "real")
        return embedder, chat_llm, store
    except Exception:
        return None


def build_stack(config: RunConfig, backend: str | None = None):
    """Renvoie (backend_effectif, embedder, chat_llm, store)."""

    backend = resolve_backend(backend)
    if backend == "real":
        embedder = build_embedder(config, "real")
        chat_llm = build_chat_llm(config, "real")
        store = build_store(config, embedder, "real")
        return "real", embedder, chat_llm, store
    if backend == "auto":
        real = try_build_real(config)
        if real is not None:
            return ("real", *real)
        backend = "memory"
    # memory
    embedder = build_embedder(config, "memory")
    chat_llm = build_chat_llm(config, "memory")
    store = build_store(config, embedder, "memory")
    return "memory", embedder, chat_llm, store
