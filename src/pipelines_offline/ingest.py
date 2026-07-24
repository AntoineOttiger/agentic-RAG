"""Étape offline : corpus -> chunks, matérialisée en artefact adressé par contenu.

L'artefact vit dans `artifacts/<ingestion_hash>/chunks.jsonl`. Si la signature
d'ingestion (données + chunking + embedding + vector store) n'a pas changé, l'artefact
est réutilisé tel quel — pas de re-chunking (ARCHITECTURE §8, brique 3).
"""

from __future__ import annotations

import json
from pathlib import Path

from adapters.llamaindex.ingestion import ingest_corpus
from config.loader import ingestion_hash
from config.schema import RunConfig
from domain.types import Chunk

ARTIFACTS_ROOT = Path("artifacts")


def artifact_dir(config: RunConfig) -> Path:
    return ARTIFACTS_ROOT / ingestion_hash(config)


def chunks_path(config: RunConfig) -> Path:
    return artifact_dir(config) / "chunks.jsonl"


def load_cached_chunks(config: RunConfig) -> list[Chunk] | None:
    path = chunks_path(config)
    if not path.exists():
        return None
    chunks = [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return chunks


def run_ingest(config: RunConfig, force: bool = False) -> list[Chunk]:
    """Renvoie les chunks (depuis le cache si possible), écrit l'artefact."""

    if not force:
        cached = load_cached_chunks(config)
        if cached is not None:
            return cached

    chunks = ingest_corpus(config.data.corpus_path, config.chunking)

    out_dir = artifact_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    with chunks_path(config).open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")
    (out_dir / "ingest_manifest.json").write_text(
        json.dumps(
            {
                "ingestion_hash": ingestion_hash(config),
                "n_chunks": len(chunks),
                "signature": config.ingestion_signature(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return chunks
