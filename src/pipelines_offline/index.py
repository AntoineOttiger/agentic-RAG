"""Étape offline : chunks -> index vectoriel, avec cache par signature d'ingestion.

Pour le backend réel (Qdrant on-disk), l'index est persisté et réutilisé entre runs.
Un fichier sentinelle `indexed.json` dans l'artefact marque qu'une collection a déjà
été peuplée pour cette signature.
"""

from __future__ import annotations

import json
from pathlib import Path

from config.schema import RunConfig
from domain.types import Chunk
from pipelines_offline.ingest import artifact_dir, run_ingest


def _sentinel(config: RunConfig) -> Path:
    return artifact_dir(config) / "indexed.json"


def run_index(config: RunConfig, store, force: bool = False) -> int:
    """Indexe les chunks dans `store`. Renvoie le nombre de chunks indexés.

    `store` doit implémenter le port Indexer (index / count). Pour Qdrant, le cache
    on-disk est réutilisé ; pour le backend mémoire, l'index est toujours reconstruit
    (éphémère au process).
    """

    chunks: list[Chunk] = run_ingest(config)

    sentinel = _sentinel(config)
    already = sentinel.exists() and not force

    # Backend mémoire : count() == 0 au démarrage du process -> (ré)indexer.
    if store.count() == 0 or not already:
        # Évite les doublons si la collection Qdrant est déjà remplie.
        if store.count() == 0:
            store.index(chunks)
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text(
                json.dumps({"n_chunks": len(chunks), "count": store.count()}, indent=2),
                encoding="utf-8",
            )

    return store.count()
