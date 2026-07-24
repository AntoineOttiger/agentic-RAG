"""Ingestion & chunking — anti-corruption layer autour de LlamaIndex.

Charge le corpus MultiHop-RAG, découpe chaque article en chunks et **tague chaque
chunk de métadonnées** (titre, source, auteur, catégorie, date, entités) — SCOPE §3.2.

LlamaIndex (SentenceSplitter) est importé paresseusement ; un splitter pur-Python de
repli permet de faire tourner l'ingestion sans la dépendance (tests / mode dégradé).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config.schema import ChunkingConfig
from domain.types import Chunk


def _load_articles(corpus_path: str | Path) -> list[dict]:
    data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("corpus") or data.get("articles") or []
    return data


def _entities(article: dict) -> list[str]:
    """Métadonnées d'entité pour le routing : source + auteur (dédupliqués)."""

    out: list[str] = []
    for key in ("source", "author"):
        val = article.get(key)
        if val and val not in out:
            out.append(str(val))
    return out


def _split_text(text: str, config: ChunkingConfig) -> list[str]:
    try:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception:
        return _fallback_split(text, config)


def _fallback_split(text: str, config: ChunkingConfig) -> list[str]:
    """Découpage par mots (approx. 1 mot ≈ 1.3 token) avec recouvrement."""

    words = text.split()
    if not words:
        return []
    size = max(1, int(config.chunk_size / 1.3))
    overlap = max(0, int(config.chunk_overlap / 1.3))
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + size]).strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(words):
            break
    return chunks


def _chunk_id(url: str, title: str, idx: int) -> str:
    base = f"{url or title}#{idx}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def ingest_corpus(corpus_path: str | Path, config: ChunkingConfig) -> list[Chunk]:
    """corpus -> chunks taggés. Fonction déterministe (artefact reproductible)."""

    articles = _load_articles(corpus_path)
    chunks: list[Chunk] = []
    for article in articles:
        body = article.get("body") or article.get("text") or ""
        title = article.get("title")
        url = article.get("url")
        entities = _entities(article)
        for idx, piece in enumerate(_split_text(body, config)):
            chunks.append(
                Chunk(
                    id=_chunk_id(url or "", title or "", idx),
                    text=piece,
                    source_title=title,
                    source_url=url,
                    author=article.get("author"),
                    category=article.get("category"),
                    published_at=article.get("published_at"),
                    entities=entities,
                    section=article.get("source"),
                    doc_type="news_article",
                )
            )
    return chunks
