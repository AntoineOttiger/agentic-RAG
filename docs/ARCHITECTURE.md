# Architecture — RAG raisonnement temporel multi-hop

> Document de référence du *system design*. Complète `SCOPE.md` (le quoi/pourquoi) en
> décrivant le **comment** : découpage en modules, règles de dépendance, flux de données,
> mécanisme de reproductibilité.

## 0. Fil conducteur

Le SCOPE met l'UI et le déploiement **hors V1**. L'architecture n'ajoute donc *aucune*
infra web maintenant — mais elle est conçue pour que l'ajout d'une **interface graphique**
plus tard soit **peu coûteux**. C'est la contrainte qui justifie plusieurs choix ci-dessous.

Deux « produits » partagent **un seul cœur** :

1. **Banc d'évaluation** (offline, batch) — pilote le cœur sur les ~2 556 questions en mode
   A *et* A+B, produit le rapport chiffré (livrable du Défi D).
2. **Application interactive** (GUI plus tard) — une question → une réponse, en A+B.

## 1. Principes (décisions actées)

| # | Décision | Raison |
|---|----------|--------|
| 1 | **Hexagonal** (runtime) + **pipeline-à-artefacts** (offline) | Modules profonds & GUI-ready ; étapes lentes cacheables/reproductibles |
| 2 | **LlamaIndex confiné aux adapters** ; décomposition (couche B) écrite à la main | Cœur non couplé au framework, testable ; la contribution reste auditable |
| 3 | **Une seule pipeline**, couche B = étape optionnelle injectée | Comparaison A vs A+B rigoureuse : une seule variable change |
| 4 | **`Answer` riche et typé**, trace de raisonnement en 1re classe | Un contrat unique nourrit l'éval *et* la GUI |
| 5 | **Config unique/run** + runs immuables hashés + artefacts adressés par contenu | Reproductibilité mesurée (Défi D) |
| 6 | **Évaluation = contexte séparé** ; métriques retrieval = fonctions pures ; juge derrière `Judge` | Cœur ignorant de sa mesure ; juge remplaçable |
| 7 | **Façade `RagService`** in-process, conçue pour un passage HTTP indolore | GUI = adapter de plus, zéro réécriture |

## 2. Règle de dépendance (l'unique règle anti-spaghetti)

```
apps / evaluation  ─────►  core  ─────►  domain  ◄─────  adapters
```

- Les flèches ne pointent **jamais** vers l'extérieur.
- `domain` ne dépend de **rien** (types purs, aucune lib externe).
- `core` ne fait **jamais** `import llama_index` ni `import qdrant` : il ne connaît que les
  ports de `domain`.
- Toute dépendance externe (LlamaIndex, Qdrant, Mistral, bge) vit **uniquement** dans
  `adapters/` (anti-corruption layer).

## 3. Carte des modules

```
src/
  domain/            # Types purs, zéro dépendance externe. Le vocabulaire stable.
    types.py         #   Chunk, Query, SubQuery, RetrievedContext,
                     #   Citation, ReasoningTrace, Answer, Evidence, TokenUsage
    ports.py         #   Protocols : Retriever, Decomposer, Generator,
                     #   Indexer, Embedder, Judge

  core/              # Le cœur RAG. Dépend de domain/ uniquement.
    pipeline.py      #   Pipeline.answer(query) -> Answer   (compose les ports)
    decompose/       #   Contribution (couche B), écrite à la main
      null.py        #     NullDecomposer   (baseline A : trace à un seul hop)
      agentic.py     #     QueryDecomposer  (A+B)
    service.py       #   RagService.ask(query) -> Answer   (la façade)

  adapters/          # Détails. Traduisent domain <-> monde extérieur.
    llamaindex/      #   ingestion, chunking, indexation
    qdrant/          #   Retriever + Indexer sur Qdrant
    embeddings/      #   Embedder bge (local)
    llm/             #   Generator + client Mistral (port LLM partagé)

  config/
    schema.py        #   Modèles Pydantic validés
    loader.py        #   YAML -> config validée + calcul du hash de config

  pipelines_offline/ # Étapes matérialisées en artefacts (le volet reproductible)
    ingest.py        #   corpus -> chunks        (artefact adressé par contenu)
    index.py         #   chunks -> index Qdrant  (artefact adressé par contenu)

evaluation/          # Contexte séparé. Consomme le cœur, ne le modifie pas.
  retrieval_metrics.py  # FONCTIONS PURES : Hit Rate, Recall@k, MRR
  judge/
    ragas_judge.py      #   adapter RAGAS derrière l'interface domain.Judge
  run_eval.py           # driving adapter : boucle questions, A puis A+B

apps/                # Driving adapters entrants
  cli.py             #   V1 : appelle RagService
  # ui/  (plus tard) #   Streamlit/Gradio, appelle le MÊME RagService

configs/             # Un YAML par run (baseline.yaml, enriched.yaml, ...)
runs/                # Runs immuables : <ts>-<hash>/ {config, manifest, results}
artifacts/           # Artefacts adressés par contenu : <hash>/ {chunks, index}
tests/
```

## 4. Types de domaine (vocabulaire stable)

Modèles Pydantic (sérialisables → JSON gratuit, prêt pour une future API HTTP).

- **`Query`** — la question de l'utilisateur.
- **`SubQuery`** — une sous-question produite par la couche B (par entité / période / hop).
- **`Chunk`** — passage indexé + métadonnées (entité, date/période, section, type).
- **`RetrievedContext`** — chunk récupéré + score + provenance (quelle sous-requête).
- **`Citation`** — lie une affirmation de la réponse à un passage source.
- **`ReasoningTrace`** — sous-requêtes + retrieval par sous-requête (A = trace à un hop).
- **`Answer`** — l'objet de retour riche (voir §5).
- **`Evidence`** — l'evidence en or annotée (utilisée par l'éval uniquement).
- **`TokenUsage`** — coût/latence, pour le debug et l'analyse d'éval.

## 5. Le contrat `Answer`

Objet de retour unique, consommé par les deux produits :

```python
class Answer:
    text: str
    citations: list[Citation]          # passage -> claim
    retrieved: list[RetrievedContext]  # ce que le retrieval a ramené
    reasoning_trace: ReasoningTrace     # sous-requêtes + retrieval (couche B)
    is_answerable: bool                 # gestion des null queries
    usage: TokenUsage                   # coût/latence
```

- L'**éval** lit `retrieved` (vs `Evidence`), `text` (vs juge), `is_answerable` (null queries).
- La **GUI** lit `text`, `citations`, `reasoning_trace` (« voir l'agent raisonner »).
- La baseline **A** remplit `reasoning_trace` avec une trace dégénérée à un seul hop —
  pas de cas spécial, juste un `NullDecomposer` (pattern Null Object).

## 6. La variation A vs A+B

Une seule pipeline ; la couche B est un `Decomposer` injecté :

- **A (baseline)** : `{ retriever: metadata, decomposer: NullDecomposer }`
- **A+B (enrichi)** : `{ retriever: metadata, decomposer: QueryDecomposer }`

Tout le reste du chemin (`retrieve → generate → answer`) est **identique par construction**.
Le choix se fait **en config**, jamais dans le code → une seule variable change entre les
deux mesures.

## 7. Les deux flux

### 7.1 Banc d'éval (offline, A vs A+B)

```
configs/baseline.yaml ─┐
configs/enriched.yaml ─┤
                       ▼
run_eval → pour chaque question : RagService.ask() → Answer
        → retrieval_metrics (pures) : Answer.retrieved vs Evidence
        → Judge : Answer.text + contexte
        → runs/<ts-hash>/results.json → rapport A vs A+B (4 catégories)
```

- **Étage 1 (retrieval)** : fonctions pures, 100 % déterministes, testables sans LLM.
- **Étage 2 (génération)** : LLM-as-judge derrière l'interface `Judge` (adapter RAGAS).
  Le juge est **validé manuellement sur ~20 exemples** avant qu'on lui fasse confiance
  (voir SCOPE §3.5 et risque §6.2).

### 7.2 Application interactive (A+B, GUI plus tard)

```
utilisateur → apps/cli (puis apps/ui) → RagService.ask(query) → Answer
           → affiche text + citations + reasoning_trace
```

Même façade, même `Answer`. La GUI est un fichier de plus dans `apps/`.

## 8. Reproductibilité

Trois briques :

1. **Config = source unique de vérité.** Un YAML par run capture *tout* ce qui influence le
   résultat : modèle d'embedding, taille de chunk, top-k, modèle LLM, température, seed,
   variante (A / A+B), version de prompt. Zéro paramètre en dur.
2. **Runs immuables.** Chaque exécution écrit `runs/<timestamp>-<hash_config>/` contenant :
   la config figée, le hash du corpus, `pip freeze`, le git SHA, les résultats. Deux runs
   identiques sont détectables par leur `hash_config`.
3. **Artefacts adressés par contenu + cache.** Les étapes lentes et déterministes (chunking,
   embeddings, index Qdrant) sont matérialisées dans `artifacts/<hash>/` et réutilisées si
   la config d'ingestion n'a pas changé. Seeds fixées partout.

**Limite assumée** — le LLM (générateur/juge via API) n'est pas déterministe bit-à-bit, même
à température 0. Mitigation : température 0 partout + **logging des réponses LLM brutes** dans
le run (donc le run est rejouable/auditable) + documentation de la limite dans le rapport
(voir SCOPE §6.1).

## 9. Testabilité

- `domain` et les métriques de retrieval : **fonctions/types purs**, tests unitaires triviaux.
- `core` : testé contre des **fakes** des ports (faux `Retriever`, faux `Generator`), sans
  Qdrant ni LLM.
- `adapters` : tests d'intégration ciblés (Qdrant en Docker, un appel LLM réel minimal).
- `Judge` : validé manuellement (~20 exemples) avant usage automatisé.

## 10. Ce que l'architecture garantit

- **Modules profonds / interfaces simples** — `RagService.ask(query) -> Answer` cache toute
  la machinerie ; `Pipeline`, `Retriever`, `Judge` idem.
- **Pas de spaghetti** — une seule règle de dépendance, unidirectionnelle.
- **Comparaison A vs A+B rigoureuse** — une seule variable change (le `Decomposer` injecté).
- **Reproductible** — config → hash → run immuable ; artefacts cachés ; seeds ; LLM loggé.
- **GUI-ready** — la GUI réutilise `RagService` tel quel ; passage HTTP = un adapter trivial
  (types déjà sérialisables).

## Références

- `SCOPE.md` — objectif, données, scope V1, risques, critères de succès.
- MultiHop-RAG : https://github.com/yixuantt/MultiHop-RAG/
