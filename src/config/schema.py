"""Schéma de configuration — la source unique de vérité d'un run.

Un YAML par run capture TOUT ce qui influence le résultat (ARCHITECTURE §8). Zéro
paramètre en dur. La variante A / A+B se choisit ici, jamais dans le code.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Variant(str, Enum):
    """La seule variable qui change entre les deux mesures."""

    BASELINE = "A"  # NullDecomposer
    ENRICHED = "A+B"  # QueryDecomposer


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int | None = None  # None => déduit du modèle au chargement
    device: str = "cpu"
    normalize: bool = True
    batch_size: int = 32


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64


class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    location: str = "local"  # "local" (embedded) ou une URL http://...
    path: str = "artifacts/qdrant"  # stockage embarqué on-disk
    collection: str = "multihop_rag"
    distance: str = "cosine"


class RetrievalConfig(BaseModel):
    top_k: int = 6
    use_metadata_filters: bool = True


class LLMConfig(BaseModel):
    provider: str = "mistral"
    model: str = "mistral-small-latest"
    temperature: float = 0.0
    max_tokens: int = 1024
    # Le nom de la variable d'env qui porte la clé API (jamais la clé elle-même).
    api_key_env: str = "MISTRAL_API_KEY"


class JudgeConfig(BaseModel):
    enabled: bool = True
    provider: str = "ragas"  # "ragas" ou "native"
    metrics: list[str] = Field(
        default_factory=lambda: ["answer_correctness", "faithfulness", "context_precision"]
    )


class DataConfig(BaseModel):
    corpus_path: str = "data/corpus.json"
    questions_path: str = "data/MultiHopRAG.json"
    limit: int | None = None  # limite le nb de questions (dev / smoke test)


class RunConfig(BaseModel):
    """La config complète d'un run, validée et hashable."""

    name: str
    variant: Variant
    seed: int = 42
    prompt_version: str = "decompose-v1"

    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    model_config = {"extra": "forbid"}

    def ingestion_signature(self) -> dict:
        """Sous-ensemble de la config qui détermine l'artefact d'ingestion/index.

        Deux runs qui partagent cette signature partagent le même index Qdrant.
        """

        return {
            "data": self.data.model_dump(exclude={"limit"}),
            "chunking": self.chunking.model_dump(),
            "embedding": self.embedding.model_dump(),
            "vector_store": self.vector_store.model_dump(exclude={"path", "location"}),
        }
