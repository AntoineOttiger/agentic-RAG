"""Qdrant — Indexer + Retriever. Anti-corruption layer autour de qdrant-client.

Implémente `domain.ports.Indexer` et `domain.ports.Retriever`. Le routing par
métadonnées (période / entité) se traduit en filtres Qdrant. Dépendance importée
paresseusement.
"""

from __future__ import annotations

from config.schema import VectorStoreConfig
from domain.ports import Embedder
from domain.types import Chunk, RetrievedContext, SubQuery


def _distance(name: str):
    from qdrant_client.models import Distance

    return {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
    }[name.lower()]


def _make_client(config: VectorStoreConfig):
    from qdrant_client import QdrantClient

    if config.location == "local":
        return QdrantClient(path=config.path)
    return QdrantClient(url=config.location)


def _payload(chunk: Chunk) -> dict:
    return {
        "text": chunk.text,
        "source_title": chunk.source_title,
        "source_url": chunk.source_url,
        "author": chunk.author,
        "category": chunk.category,
        "published_at": chunk.published_at,
        "entities": chunk.entities,
        "section": chunk.section,
        "doc_type": chunk.doc_type,
        "chunk_id": chunk.id,
    }


def _chunk_from_payload(payload: dict) -> Chunk:
    return Chunk(
        id=payload.get("chunk_id", ""),
        text=payload.get("text", ""),
        source_title=payload.get("source_title"),
        source_url=payload.get("source_url"),
        author=payload.get("author"),
        category=payload.get("category"),
        published_at=payload.get("published_at"),
        entities=payload.get("entities") or [],
        section=payload.get("section"),
        doc_type=payload.get("doc_type"),
    )


class QdrantStore:
    """Indexer + Retriever partageant une collection Qdrant."""

    def __init__(self, config: VectorStoreConfig, embedder: Embedder) -> None:
        self._config = config
        self._embedder = embedder
        self._client = _make_client(config)

    # --- Indexer ---------------------------------------------------------
    def ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client.models import VectorParams

        exists = self._client.collection_exists(self._config.collection)
        if exists and recreate:
            self._client.delete_collection(self._config.collection)
            exists = False
        if not exists:
            self._client.create_collection(
                collection_name=self._config.collection,
                vectors_config=VectorParams(
                    size=self._embedder.dim,
                    distance=_distance(self._config.distance),
                ),
            )

    def index(self, chunks: list[Chunk]) -> None:
        from qdrant_client.models import PointStruct

        self.ensure_collection(recreate=False)
        vectors = self._embedder.embed_texts([c.text for c in chunks])
        points = [
            PointStruct(id=i, vector=vec, payload=_payload(chunk))
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        # Upsert par batch pour ménager la mémoire.
        for start in range(0, len(points), 256):
            self._client.upsert(
                collection_name=self._config.collection,
                points=points[start : start + 256],
            )

    def count(self) -> int:
        if not self._client.collection_exists(self._config.collection):
            return 0
        return self._client.count(self._config.collection).count

    # --- Retriever -------------------------------------------------------
    def _build_qdrant_filter(self, filters: dict[str, object] | None):
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        conditions = []
        entity = filters.get("entity")
        if entity:
            conditions.append(
                FieldCondition(key="entities", match=MatchAny(any=[str(entity)]))
            )
        # NB : le filtre "period" exact est volontairement souple — on préfère ne
        # pas sur-filtrer une date qui ne matcherait aucun chunk. On le laisse en
        # filtre optionnel côté catégorie si présent.
        category = filters.get("category")
        if category:
            conditions.append(FieldCondition(key="category", match=MatchValue(value=str(category))))
        if not conditions:
            return None
        return Filter(should=conditions)

    def retrieve(
        self,
        subquery: SubQuery,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedContext]:
        vector = self._embedder.embed_query(subquery.text)
        qfilter = self._build_qdrant_filter(filters)
        hits = self._client.query_points(
            collection_name=self._config.collection,
            query=vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        ).points
        return [
            RetrievedContext(
                chunk=_chunk_from_payload(h.payload or {}),
                score=float(h.score),
                from_subquery=subquery.text,
            )
            for h in hits
        ]
