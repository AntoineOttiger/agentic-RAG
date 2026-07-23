# Scope — RAG raisonnement temporel multi-hop

## 1. Objectif

Construire un système RAG capable de répondre à des questions **multi-hop** (croisant plusieurs documents) sur un corpus d'articles de presse, et **prouver par la mesure** qu'il ne produit pas d'hallucinations.

- **Défi B** — raisonnement multi-hop, en particulier temporel (croiser des documents de périodes différentes).
- **Défi D** — fiabilité *mesurée* : évaluation automatisée de la justesse, de la fidélité (groundedness) et de la détection des questions sans réponse.

**Cas d'usage simulé** : veille média / competitive intelligence.
**Finalité** : projet portfolio démontrant une maîtrise du RAG au-delà du tutoriel.

## 2. Résultat attendu (definition of done)

Un système fonctionnel + un **rapport d'évaluation chiffré** qui compare une baseline (couche A) à la version enrichie (couche A+B) et démontre, preuves à l'appui, l'apport de la décomposition de requête.

## 3. Dans le scope (V1)

### 3.1 Données
- Corpus + questions issus de **MultiHop-RAG** (COLM 2024) : articles de presse, ~2 556 questions multi-hop, 4 catégories (inférence, comparaison, temporel, null), avec **evidence en or** annotée.
- Aucune rédaction manuelle de questions/réponses.

### 3.2 Ingestion & indexation
- Chunking des articles.
- **Tagging de métadonnées** par chunk : entité, date/période, section, type.
- Indexation dans **Qdrant** avec embeddings **BAAI bge** (local).

### 3.3 Couche A — baseline
- Retrieval enrichi de métadonnées (filtrage/routing par période et entité).
- Reproduction de la meilleure baseline du repo MultiHop-RAG comme point de départ chiffré.

### 3.4 Couche B — contribution
- **Agent de décomposition de requête** : découpe une question multi-hop en sous-requêtes (par entité / période / sous-question), lance un retrieval ciblé pour chacune, puis synthétise.
- **Gestion des null queries** : le système doit pouvoir répondre « preuve insuffisante / je ne sais pas » quand aucune réponse n'est étayée.

### 3.5 Évaluation (2 étages)
- **Étage 1 — Retrieval** : comparaison mécanique à l'evidence en or → Hit Rate, Recall@k, MRR (`retrieval_evaluate.py`).
- **Étage 2 — Génération** : LLM-as-judge sur Answer Correctness, Faithfulness/Groundedness, Citation accuracy (`qa_evaluate.py` + RAGAS).
- **Validation manuelle du juge** sur ~20 exemples avant de lui faire confiance.
- Éval lancée en batch.

### 3.6 Livrable analytique
- Comparaison chiffrée **A vs A+B** sur les deux étages et les 4 catégories de questions.
- Le résultat peut être positif *ou* négatif — l'objectif est de mesurer, pas de prouver un a priori.

## 4. Stack technique

| Composant | Choix |
|---|---|
| Langage | Python |
| Orchestration | LlamaIndex |
| Vector store | Qdrant (local, Docker) |
| Embeddings | BAAI bge (`bge-large` / `bge-m3`), local |
| LLM (générateur + juge) | Mistral Small (API, free tier) — *provisoire, à réévaluer* |
| Éval | RAGAS + scripts natifs MultiHop-RAG |

## 5. Hors scope (V1)

- Domaine finance / 10-K (abandonné au profit de la presse).
- Rédaction manuelle d'un jeu de questions/réponses.
- Comparatif multi-entreprises / multi-corpus (extension future).
- Graphe de connaissances temporel (extension future).
- Fine-tuning de modèle.
- UI / mise en production / déploiement.
- Benchmark TEMPO (optionnel, seulement si le temporel de MultiHop-RAG est insuffisant).

## 6. Risques & dettes assumés

1. **Juge = générateur (même modèle)** → biais d'auto-préférence : le taux d'hallucination mesuré peut être trop optimiste. À documenter ; réévaluer un juge distinct plus tard.
2. **LLM-as-judge non validé** → doit être calibré à la main avant usage.
3. **Rate limits** du free tier Mistral → risque de blocage sur l'éval en batch.

## 7. Critères de succès

- [ ] La couche A tourne et est mesurée (retrieval + génération).
- [ ] La couche B tourne, gère les null queries, et est mesurée.
- [ ] Comparaison A vs A+B produite avec des chiffres reproductibles.
- [ ] Le juge LLM est validé manuellement sur ~20 exemples.
- [ ] Rapport final expliquant méthode, résultats et limites.

## Références
- MultiHop-RAG : https://github.com/yixuantt/MultiHop-RAG/ · https://openreview.net/forum?id=t4eB3zYWBK
- TEMPO (optionnel) : https://arxiv.org/pdf/2601.09523
