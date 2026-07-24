"""Embedder bge (local) — anti-corruption layer autour de sentence-transformers.

Implémente le port `domain.ports.Embedder`. La dépendance lourde est importée
paresseusement pour que domain/core/tests restent chargeables sans le modèle.
"""

from __future__ import annotations

from config.schema import EmbeddingConfig


class BgeEmbedder:
    """Embeddings BAAI bge via sentence-transformers, en local (CPU/GPU)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self._config = config
        self._model = SentenceTransformer(config.model, device=config.device)
        self._dim = config.dim or self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        # bge recommande un préfixe pour les requêtes de recherche.
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        vector = self._model.encode(
            [prefixed],
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()
