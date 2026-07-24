"""Adapters en mémoire — font tourner tout le système sans service externe.

- `HashEmbedder` : embeddings déterministes (bag-of-words hashé) — zéro modèle à
  télécharger, parfait pour les tests et un smoke run offline.
- `InMemoryStore` : Indexer + Retriever à similarité cosinus, avec routing par
  métadonnées identique à l'adapter Qdrant.

Ce ne sont PAS des jouets de test cachés dans tests/ : ils implémentent les mêmes
ports que les adapters de prod, ce qui prouve que le cœur est bien découplé.
"""

from __future__ import annotations

import hashlib
import math

from domain.types import Chunk, RetrievedContext, SubQuery


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


class HashEmbedder:
    """Embedding déterministe : chaque token incrémente un bucket haché (TF)."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vecteurs déjà normalisés


class InMemoryStore:
    """Vector store en mémoire : Indexer + Retriever."""

    def __init__(self, embedder: HashEmbedder) -> None:
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    # --- Indexer ---
    def index(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._vectors.extend(self._embedder.embed_texts([c.text for c in chunks]))

    def count(self) -> int:
        return len(self._chunks)

    # --- Retriever ---
    def _passes_filter(self, chunk: Chunk, filters: dict[str, object] | None) -> bool:
        if not filters:
            return True
        entity = filters.get("entity")
        if entity and str(entity) not in chunk.entities:
            return False
        category = filters.get("category")
        if category and chunk.category != category:
            return False
        return True

    def retrieve(
        self,
        subquery: SubQuery,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedContext]:
        qv = self._embedder.embed_query(subquery.text)
        scored = [
            (_cosine(qv, self._vectors[i]), self._chunks[i])
            for i in range(len(self._chunks))
            if self._passes_filter(self._chunks[i], filters)
        ]
        # Repli : si le filtre ne laisse rien, on retente sans filtre (évite les
        # trous de recall dus à un routing trop strict).
        if not scored and filters:
            return self.retrieve(subquery, top_k, filters=None)
        scored.sort(key=lambda p: p[0], reverse=True)
        return [
            RetrievedContext(chunk=chunk, score=float(score), from_subquery=subquery.text)
            for score, chunk in scored[:top_k]
        ]
