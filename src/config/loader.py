"""Chargement de config : YAML -> RunConfig validée + hash déterministe.

Le hash de config identifie un run de façon reproductible (ARCHITECTURE §8, brique 2).
Le hash d'ingestion identifie l'artefact index (brique 3).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from config.schema import RunConfig


def _load_dotenv() -> None:
    """Charge un éventuel fichier `.env` dans os.environ (secrets, ex. MISTRAL_API_KEY).

    Sans effet si python-dotenv n'est pas installé ou si aucun `.env` n'existe.
    Les variables déjà présentes dans l'environnement ne sont pas écrasées.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _stable_hash(obj: object, length: int = 12) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def load_config(path: str | Path) -> RunConfig:
    _load_dotenv()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw or {})


def config_hash(config: RunConfig, length: int = 12) -> str:
    """Hash de l'intégralité de la config (identité du run)."""

    return _stable_hash(config.model_dump(mode="json"), length)


def ingestion_hash(config: RunConfig, length: int = 12) -> str:
    """Hash de la seule signature d'ingestion (identité de l'artefact index)."""

    return _stable_hash(config.ingestion_signature(), length)
