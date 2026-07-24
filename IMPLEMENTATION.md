# Rapport d'implémentation

> Ce que j'ai construit à partir de `docs/SCOPE.md` et `docs/ARCHITECTURE.md`, comment
> le vérifier, et les écarts assumés. Périmètre : implémentation **complète et fonctionnelle**
> de l'architecture décrite, tournant de bout en bout dès maintenant.

## 1. État en un coup d'œil

- **41 fichiers Python** (~2 380 lignes) organisés selon la carte des modules de l'ARCHITECTURE §3.
- **21 tests unitaires + intégration**, verts en ~0.1 s (`python -m pytest -q`).
- **Système exécutable de bout en bout** sans aucune dépendance externe (backend `memory`) :
  ingestion → décomposition → retrieval → génération → éval A vs A+B → rapport chiffré.
- **Backend réel prêt à brancher** (bge + Qdrant + Mistral + RAGAS) via `pip install -e .[real,ragas]`.

La règle de dépendance unidirectionnelle (`apps/evaluation → core → domain ← adapters`,
ARCHITECTURE §2) est respectée à la lettre : `domain` n'importe rien, `core` n'importe que
`domain`, et **toute** dépendance externe est confinée dans `adapters/`.

## 2. Correspondance avec l'ARCHITECTURE

| Décision (ARCHITECTURE §1) | Implémentation |
|---|---|
| 1. Hexagonal + pipeline-à-artefacts | `src/{domain,core,adapters}` + `src/pipelines_offline/{ingest,index}.py` (artefacts adressés par contenu) |
| 2. LlamaIndex confiné aux adapters ; décomposition écrite à la main | `adapters/llamaindex/ingestion.py` ; `core/decompose/agentic.py` (JSON parsé maison, zéro framework agentique) |
| 3. Une seule pipeline, couche B = étape injectée | `core/pipeline.py` unique ; le `Decomposer` est le seul point de variation |
| 4. `Answer` riche et typé, trace en 1re classe | `domain/types.py::Answer` (+ `ReasoningTrace`, `HopTrace`, `Citation`, `TokenUsage`) |
| 5. Config unique + runs immuables hashés + artefacts par contenu | `config/loader.py` (`config_hash`, `ingestion_hash`) ; `runs/<ts>-<hash>/` ; `artifacts/<hash>/` |
| 6. Éval = contexte séparé ; métriques pures ; juge derrière `Judge` | `evaluation/retrieval_metrics.py` (pures) ; `evaluation/judge/*` derrière `domain.ports.Judge` |
| 7. Façade `RagService` GUI-ready | `core/service.py` ; réutilisée telle quelle par `apps/cli.py` |

**Contrat `Answer` (ARCHITECTURE §5)** : implémenté intégralement — `text`, `citations`,
`retrieved`, `reasoning_trace`, `is_answerable`, `usage`. La baseline A produit bien une
trace dégénérée à un hop via le `NullDecomposer` (pattern Null Object), sans cas spécial
ailleurs dans la pipeline.

**Variation A vs A+B (ARCHITECTURE §6)** : le choix se fait **uniquement en config**
(`variant: A` / `A+B` dans le YAML). `config_hash` diffère entre les deux runs, mais
`ingestion_hash` est **identique** → le même index Qdrant est réutilisé, garantissant
qu'une seule variable change entre les deux mesures. C'est testé (`test_config.py::
test_variant_changes_config_hash_but_not_ingestion`).

## 3. Correspondance avec le SCOPE

| Item SCOPE | Implémentation |
|---|---|
| 3.1 Données MultiHop-RAG | `scripts/prepare_data.py` télécharge corpus + questions ; `evaluation/data.py` charge questions + evidence en or |
| 3.2 Chunking + tagging de métadonnées | `adapters/llamaindex/ingestion.py` : chunk + entités (source/auteur), catégorie, date, section, type |
| 3.2 Qdrant + embeddings bge | `adapters/qdrant/store.py` + `adapters/embeddings/bge.py` |
| 3.3 Couche A — retrieval enrichi de métadonnées | routing par filtres (entité/catégorie) dans `pipeline.py` + `qdrant/store.py` |
| 3.4 Couche B — agent de décomposition | `core/decompose/agentic.py` : question → 2-4 sous-requêtes ciblées (entité/période) |
| 3.4 Null queries | `adapters/llm/generator.py` impose « preuve insuffisante » → `is_answerable=False` |
| 3.5 Éval étage 1 (retrieval) | `retrieval_metrics.py` : Hit Rate, Recall@k, Precision@k, MRR — **fonctions pures** |
| 3.5 Éval étage 2 (génération) | `judge/native_judge.py` (LLM-as-judge) + `judge/ragas_judge.py` (RAGAS) |
| 3.5 Validation manuelle du juge | `judge/manual_validation.py` : export CSV → annotation → accord (Pearson + MAE) |
| 3.6 Livrable analytique A vs A+B | `evaluation/compare.py` → rapport Markdown, global + 4 catégories + null detection |

**Détection des null queries** : mesurée explicitement (precision/recall/F1) en confrontant
`Answer.is_answerable` au type de question or (`null_query`). Les null queries sont exclues
du juge (leur réponse or est « insuffisant »).

## 4. Reproductibilité (ARCHITECTURE §8) — implémentée

1. **Config = source unique.** `config/schema.py` avec `extra="forbid"` (aucun champ en dur
   ne passe inaperçu). Seed, versions de prompt, modèle, top-k, variante : tout est capturé.
2. **Runs immuables.** Chaque run écrit `runs/<ts>-<config_hash>/` : `config.json` figée,
   `manifest.json` (git SHA, `pip freeze`, version Python, plateforme, seed, backend,
   n_chunks), `results.jsonl` (réponses brutes loggées → run auditable), `metrics.json`.
3. **Artefacts adressés par contenu.** `artifacts/<ingestion_hash>/chunks.jsonl` + sentinelle
   `indexed.json` ; réutilisés si la signature d'ingestion n'a pas bougé.

**Limite assumée reprise du SCOPE §6.1** : le LLM (API) n'est pas déterministe bit-à-bit.
Mitigation implémentée : température 0 partout + logging des réponses dans `results.jsonl`.

## 5. Testabilité (ARCHITECTURE §9) — implémentée

- `domain` + métriques retrieval : tests purs (`test_retrieval_metrics.py`).
- `core` : testé contre des **fakes** des ports (`test_pipeline.py`) — pas de Qdrant ni LLM.
- Bout en bout : `test_end_to_end.py` via le backend `memory`.
- Config/repro : `test_config.py`.

## 6. Écarts assumés vs les documents

1. **Emplacement des dossiers.** `evaluation/` et `apps/` vivent sous `src/` (et non à la
   racine comme dans la carte ARCHITECTURE §3). Motif : un seul `package-dir = src`, imports
   propres (`from evaluation… import`), zéro bricolage de `sys.path`. **La règle de dépendance
   reste strictement respectée** — c'est elle qui compte, pas le niveau de dossier.
2. **Adapters `inmemory/` ajoutés.** Non prévus par les docs, mais ils implémentent les
   *mêmes ports* que la prod. Double bénéfice : (a) ils **prouvent** que le cœur est découplé ;
   (b) ils rendent le système démontrable et testable offline, sans clé API ni Docker.
3. **Port `ChatLLM` ajouté à `domain.ports`.** Les docs listent `Generator`/`Judge` mais le
   décomposeur (couche B, dans `core`) a lui aussi besoin d'un LLM. Plutôt que de coupler
   `core` à un adapter, j'ai introduit un port LLM générique partagé — fidèle à l'esprit
   « adapters/llm = port LLM partagé » (ARCHITECTURE §3).
4. **Filtre temporel `period` volontairement souple.** Un filtre de date exact risquerait de
   vider le top-k. Le routing par entité/catégorie est actif ; la période est portée par la
   sous-requête et exploitable, mais non imposée en filtre dur (documenté dans `qdrant/store.py`).
5. **Juge par défaut = `native` (pas RAGAS).** RAGAS est branché et sélectionnable
   (`provider: ragas`) mais lourd à configurer ; le juge natif derrière la même interface
   `Judge` est le chemin fiable par défaut. Le **biais juge≈générateur (SCOPE §6.1)** est
   rappelé dans le rapport `compare.py` et adressable via la validation manuelle.

## 7. Vérification effectuée

```
python -m pytest -q                        → 21 passed
python -m evaluation.run_eval  smoke_baseline.yaml --backend memory  → run écrit
python -m evaluation.run_eval  smoke_enriched.yaml --backend memory  → run écrit
python -m evaluation.compare               → runs/REPORT_A_vs_AB.md
python -m apps.cli  (question multi-hop)   → réponse + citations + 2 sous-requêtes
```

Extrait du rapport généré sur le jeu d'exemple (backend `memory`, illustratif) : retrieval
Hit Rate 0.75 / MRR 0.75 (A) ; la couche B produit bien la décomposition multi-hop attendue
(2 sous-requêtes sur une question à conjonction). Les chiffres réels s'obtiennent en
relançant avec `--backend real`.

## 8. Critères de succès (SCOPE §7)

- [x] Couche A tourne et est mesurée (retrieval + génération).
- [x] Couche B tourne, gère les null queries, et est mesurée.
- [x] Comparaison A vs A+B produite, chiffres reproductibles (hash config/ingestion).
- [x] Outillage de validation manuelle du juge fourni (à exécuter sur le vrai jeu).
- [x] Ce rapport + `README.md` expliquent méthode, résultats et limites.

## 9. Étapes restantes pour le livrable final (exécution, pas code)

1. `pip install -e .[real,ragas]`, `python scripts/prepare_data.py`, `export MISTRAL_API_KEY`.
2. Lancer les deux runs en `--backend real` (attention aux rate limits free tier — SCOPE §6.3 ;
   utiliser `data.limit` pour un run partiel si besoin).
3. Valider le juge à la main sur ~20 exemples (`judge/manual_validation.py`).
4. Publier `runs/REPORT_A_vs_AB.md` comme livrable analytique.
```
