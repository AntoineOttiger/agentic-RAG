# Graph Report - .  (2026-07-23)

## Corpus Check
- Corpus is ~2,059 words - fits in a single context window. You may not need a graph.

## Summary
- 85 nodes · 121 edges · 10 communities (8 shown, 2 thin omitted)
- Extraction: 82% EXTRACTED · 18% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.5)
- Token cost: 30,000 input · 13,741 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Decomposition & AB Answers|Decomposition & A/B Answers]]
- [[_COMMUNITY_Evaluation & Judging|Evaluation & Judging]]
- [[_COMMUNITY_Ports & Adapters (EmbeddingsVector)|Ports & Adapters (Embeddings/Vector)]]
- [[_COMMUNITY_Hexagonal Architecture & Multi-hop Challenge|Hexagonal Architecture & Multi-hop Challenge]]
- [[_COMMUNITY_App Facade & Entry Points|App Facade & Entry Points]]
- [[_COMMUNITY_Config & Reproducibility|Config & Reproducibility]]
- [[_COMMUNITY_Offline Ingestion Pipeline|Offline Ingestion Pipeline]]
- [[_COMMUNITY_LLM Generator (Mistral)|LLM Generator (Mistral)]]
- [[_COMMUNITY_Isolated Evaluation Context|Isolated Evaluation Context]]
- [[_COMMUNITY_Query|Query]]

## God Nodes (most connected - your core abstractions)
1. `Answer` - 9 edges
2. `RAG raisonnement temporel multi-hop` - 7 edges
3. `domain/ports.py` - 7 edges
4. `RagService` - 7 edges
5. `run_eval` - 7 edges
6. `Banc d'évaluation` - 6 edges
7. `core` - 6 edges
8. `adapters` - 6 edges
9. `Pipeline` - 6 edges
10. `ragas_judge` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Risque rate limits Mistral` --threatens--> `Banc d'évaluation`  [INFERRED]
  SCOPE.md → ARCHITECTURE.md
- `Banc d'évaluation` --runs--> `Étage 1 — Retrieval`  [INFERRED]
  ARCHITECTURE.md → SCOPE.md
- `QueryDecomposer` --performs--> `Gestion des null queries`  [INFERRED]
  ARCHITECTURE.md → SCOPE.md
- `Étage 1 — Retrieval` --implemented_by--> `retrieval_metrics`  [INFERRED]
  SCOPE.md → ARCHITECTURE.md
- `Retrieval enrichi de métadonnées` --retrieves_from--> `Chunk`  [INFERRED]
  SCOPE.md → ARCHITECTURE.md

## Import Cycles
- None detected.

## Communities (10 total, 2 thin omitted)

### Community 0 - "Decomposition & A/B Answers"
Cohesion: 0.17
Nodes (16): Comparaison A vs A+B, core/decompose, Retrieval enrichi de métadonnées, NullDecomposer, Gestion des null queries, Decomposer, QueryDecomposer, Answer riche et typé (+8 more)

### Community 1 - "Evaluation & Judging"
Cohesion: 0.19
Nodes (15): configs/, Banc d'évaluation, Rapport d'évaluation chiffré, Étage 1 — Retrieval, Étage 2 — Génération, evaluation, Application interactive, Validation manuelle du juge (+7 more)

### Community 2 - "Ports & Adapters (Embeddings/Vector)"
Cohesion: 0.19
Nodes (13): adapters, BAAI bge, Règle de dépendance, domain, adapters/embeddings, LlamaIndex confiné aux adapters, Embedder, Indexer (+5 more)

### Community 3 - "Hexagonal Architecture & Multi-hop Challenge"
Cohesion: 0.18
Nodes (11): Corpus d'articles de presse, Défi B, Défi D, Architecture hexagonale, LlamaIndex, adapters/llamaindex, MultiHop-RAG, Pipeline-à-artefacts (+3 more)

### Community 4 - "App Facade & Entry Points"
Cohesion: 0.31
Nodes (9): apps, apps/cli, core, Façade RagService GUI-ready, Hors scope V1, Pipeline, RagService, apps/ui (+1 more)

### Community 5 - "Config & Reproducibility"
Cohesion: 0.29
Nodes (8): hash_config, config, Limite de déterminisme LLM, config/loader.py, Config unique + runs immuables, Reproductibilité, runs/, config/schema.py

### Community 6 - "Offline Ingestion Pipeline"
Cohesion: 0.53
Nodes (6): artifacts/, index.py, ingest.py, Tagging de métadonnées, pipelines_offline, Chunk

### Community 7 - "LLM Generator (Mistral)"
Cohesion: 0.40
Nodes (5): adapters/llm, Mistral Small, Generator, Risque rate limits Mistral, Risque biais d'auto-préférence

## Knowledge Gaps
- **22 isolated node(s):** `Architecture hexagonale`, `Pipeline-à-artefacts`, `LlamaIndex confiné aux adapters`, `Une seule pipeline`, `Answer riche et typé` (+17 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_eval` connect `Evaluation & Judging` to `App Facade & Entry Points`, `Config & Reproducibility`?**
  _High betweenness centrality (0.215) - this node is a cross-community bridge._
- **Why does `Answer` connect `Decomposition & A/B Answers` to `Evaluation & Judging`, `App Facade & Entry Points`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **Why does `retrieval_metrics` connect `Evaluation & Judging` to `Hexagonal Architecture & Multi-hop Challenge`?**
  _High betweenness centrality (0.177) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Answer` (e.g. with `Gestion des null queries` and `ragas_judge`) actually correct?**
  _`Answer` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RagService` (e.g. with `Façade RagService GUI-ready` and `Pipeline`) actually correct?**
  _`RagService` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Architecture hexagonale`, `Pipeline-à-artefacts`, `LlamaIndex confiné aux adapters` to the rest of the system?**
  _22 weakly-connected nodes found - possible documentation gaps or missing edges._