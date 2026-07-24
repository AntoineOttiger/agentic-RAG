"""Reproductibilité : hash de config déterministe, signature d'ingestion stable."""

from config.loader import config_hash, ingestion_hash, load_config
from config.schema import RunConfig, Variant


def test_load_smoke_config():
    cfg = load_config("configs/smoke_baseline.yaml")
    assert cfg.variant == Variant.BASELINE
    assert cfg.data.corpus_path.endswith("sample_corpus.json")


def test_config_hash_deterministic():
    a = load_config("configs/smoke_baseline.yaml")
    b = load_config("configs/smoke_baseline.yaml")
    assert config_hash(a) == config_hash(b)


def test_variant_changes_config_hash_but_not_ingestion():
    base = load_config("configs/smoke_baseline.yaml")
    enriched = load_config("configs/smoke_enriched.yaml")
    # La variante change l'identité du run...
    assert config_hash(base) != config_hash(enriched)
    # ...mais PAS l'index (même corpus/chunking/embedding) -> artefact réutilisable.
    assert ingestion_hash(base) == ingestion_hash(enriched)


def test_extra_field_forbidden():
    import pytest

    with pytest.raises(Exception):
        RunConfig.model_validate({"name": "x", "variant": "A", "unknown_field": 1})
