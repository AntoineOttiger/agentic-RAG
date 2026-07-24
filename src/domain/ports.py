"""Ports — les interfaces (Protocols) que `core` connaît.

`core` ne dépend QUE de ces protocoles. Toute implémentation concrète (LlamaIndex,
Qdrant, Mistral, bge) vit dans `adapters/` et se plie à ces contrats. Les Protocols
sont structurels : un adapter n'a pas besoin d'hériter, juste de matcher la signature.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.types import (
    Answer,
    Chunk,
    Query,
    RetrievedContext,
    SubQuery,
    TokenUsage,
)


@runtime_checkable
class ChatLLM(Protocol):
    """Client LLM générique (chat). Port partagé : décomposeur, générateur, juge.

    Renvoie (texte, usage). L'implémentation (Mistral, ...) vit dans adapters/llm.
    """

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[str, TokenUsage]: ...


@runtime_checkable
class Embedder(Protocol):
    """Transforme du texte en vecteurs (bge local)."""

    @property
    def dim(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Retriever(Protocol):
    """Récupère les chunks pertinents pour une sous-requête.

    `filters` autorise le routing par métadonnées (période, entité) — le cœur de
    la couche A enrichie.
    """

    def retrieve(
        self,
        subquery: SubQuery,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedContext]: ...


@runtime_checkable
class Decomposer(Protocol):
    """Découpe une question en sous-requêtes.

    - Baseline A : `NullDecomposer` renvoie une seule sous-requête (la question).
    - A+B : `QueryDecomposer` renvoie plusieurs sous-requêtes ciblées.
    """

    def decompose(self, query: Query) -> tuple[list[SubQuery], TokenUsage]: ...


@runtime_checkable
class Generator(Protocol):
    """Synthétise une réponse ancrée à partir du contexte récupéré."""

    def generate(
        self,
        query: Query,
        contexts: list[RetrievedContext],
    ) -> Answer: ...


@runtime_checkable
class Indexer(Protocol):
    """Indexe des chunks dans le vector store."""

    def index(self, chunks: list[Chunk]) -> None: ...

    def count(self) -> int: ...


@runtime_checkable
class Judge(Protocol):
    """LLM-as-judge derrière une interface stable (adapter RAGAS)."""

    def score(
        self,
        query: Query,
        answer: Answer,
        reference: str | None = None,
    ) -> dict[str, float]: ...
