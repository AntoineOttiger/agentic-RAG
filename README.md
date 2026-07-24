# RAG à raisonnement temporel multi-hop

Système RAG capable de répondre à des questions **multi-hop** (croisant plusieurs
documents, dont temporelles) sur un corpus de presse, et de **prouver par la mesure**
qu'il n'hallucine pas. Architecture **hexagonale**, comparaison chiffrée **A vs A+B**.

Voir `docs/SCOPE.md` (le quoi/pourquoi) et `docs/ARCHITECTURE.md` (le comment).
`IMPLEMENTATION.md` documente ce qui a été construit et les écarts assumés.

## Démarrage rapide (offline, zéro dépendance lourde)

Le backend `memory` (embeddings hachés + LLM extractif déterministe) fait tourner tout
le système sans Qdrant, sans Mistral et sans télécharger de modèle.

```bash
pip install -e .              # installe pydantic + pyyaml, ajoute src/ au path
python -m pytest -q           # 21 tests, ~0.1s

# Éval A vs A+B sur le jeu d'exemple
python -m evaluation.run_eval configs/smoke_baseline.yaml --backend memory
python -m evaluation.run_eval configs/smoke_enriched.yaml --backend memory
python -m evaluation.compare  # écrit runs/REPORT_A_vs_AB.md

# Poser une question
python -m apps.cli --config configs/smoke_enriched.yaml --backend memory \
  "Which company launched the Orion chip and how much revenue did NovaChip report?"
```

## Livrable réel (bge + Qdrant + Mistral)

```bash
pip install -e .[real]                 # + .[ragas] pour le juge RAGAS
python scripts/prepare_data.py          # télécharge corpus.json + MultiHopRAG.json
export MISTRAL_API_KEY=...              # clé API (jamais dans le code/YAML)

python -m evaluation.run_eval configs/baseline.yaml --backend real
python -m evaluation.run_eval configs/enriched.yaml --backend real
python -m evaluation.compare

# Valider le juge à la main (~20 exemples) avant de lui faire confiance
python -m evaluation.judge.manual_validation export runs/<run-A+B> --out runs/judge.csv
#   ... remplir les colonnes human_* ...
python -m evaluation.judge.manual_validation agree runs/judge.csv
```

## Carte du dépôt

| Chemin | Rôle |
|---|---|
| `src/domain/` | Types purs + ports (Protocols). Zéro dépendance externe. |
| `src/core/` | Pipeline, décomposeurs (A/A+B), façade `RagService`. |
| `src/adapters/` | bge, Qdrant, Mistral, LlamaIndex + adapters `inmemory` (offline). |
| `src/config/` | Schéma Pydantic + loader YAML + hashing (repro). |
| `src/pipelines_offline/` | Ingestion & indexation adressées par contenu (cache). |
| `src/evaluation/` | Métriques retrieval (pures), juge, driver, comparatif A vs A+B. |
| `src/apps/` | CLI + composition root. |
| `configs/` | Un YAML par run (baseline / enriched / smoke). |
| `runs/`, `artifacts/` | Runs immuables & artefacts cachés (générés, gitignorés). |
```
