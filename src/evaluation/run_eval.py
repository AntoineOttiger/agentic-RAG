"""Driving adapter d'évaluation : pilote le cœur sur le jeu de questions.

Un appel = une config = une variante (A ou A+B). Produit un run immuable
`runs/<timestamp>-<config_hash>/` contenant la config figée, un manifest de
reproductibilité, les résultats par question et les métriques agrégées
(ARCHITECTURE §7.1 & §8).

Deux étages d'évaluation :
- Étage 1 (retrieval) : fonctions pures vs evidence en or (déterministe, sans LLM).
- Étage 2 (génération) : juge (LLM-as-judge ou heuristique) sur la réponse.

Le comparatif A vs A+B se fait ensuite via `evaluation/compare.py`.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apps.composition import build_judge, build_service, build_stack
from config.loader import config_hash, load_config
from config.schema import RunConfig
from evaluation import retrieval_metrics as rm
from evaluation.data import load_questions
from pipelines_offline.index import run_index

RUNS_ROOT = Path("runs")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, text=True
        )
        return out.splitlines()
    except Exception:
        return []


def _run_dir(config: RunConfig) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RUNS_ROOT / f"{ts}-{config_hash(config)}"


def run_evaluation(config_path: str, backend: str | None = None, progress: bool = True) -> Path:
    config = load_config(config_path)
    eff_backend, embedder, chat_llm, store = build_stack(config, backend)

    # Ingestion + indexation (avec cache par signature).
    n_indexed = run_index(config, store)

    service = build_service(config, store, chat_llm)
    judge = build_judge(config, chat_llm, eff_backend)

    questions = load_questions(config.data.questions_path, config.data.limit)

    results: list[dict] = []
    retrieval_scores: list[dict[str, float]] = []
    judge_scores: list[dict[str, float]] = []
    null_stats = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}  # pour la détection des null queries

    k = config.retrieval.top_k
    for i, q in enumerate(questions):
        if progress and i % 25 == 0:
            print(f"  [{i}/{len(questions)}] {config.name}...", file=sys.stderr)

        answer = service.ask(q.to_query())

        # Étage 1 — retrieval (déterministe).
        r = rm.evaluate_retrieval(answer.retrieved, q.evidence, k)
        retrieval_scores.append(r)

        # Détection des null queries : is_answerable vs question_type NULL.
        is_null_gold = q.question_type.value == "null_query"
        predicted_null = not answer.is_answerable
        if is_null_gold and predicted_null:
            null_stats["tp"] += 1
        elif is_null_gold and not predicted_null:
            null_stats["fn"] += 1
        elif not is_null_gold and predicted_null:
            null_stats["fp"] += 1
        else:
            null_stats["tn"] += 1

        # Étage 2 — juge (sauf null queries, où la réponse or est "insufficient").
        js: dict[str, float] = {}
        if judge is not None and not is_null_gold:
            js = judge.score(q.to_query(), answer, reference=q.answer)
            judge_scores.append(js)

        results.append(
            {
                "query": q.query,
                "question_type": q.question_type.value,
                "gold_answer": q.answer,
                "answer": answer.text,
                "is_answerable": answer.is_answerable,
                "n_subqueries": len(answer.reasoning_trace.hops),
                "n_retrieved": len(answer.retrieved),
                "retrieval": r,
                "judge": js,
                "usage": answer.usage.model_dump(),
                "subqueries": [h.subquery.text for h in answer.reasoning_trace.hops],
            }
        )

    metrics = _aggregate_all(retrieval_scores, judge_scores, null_stats, results)
    out_dir = _write_run(config, config_path, eff_backend, n_indexed, results, metrics)
    print(f"✔ Run écrit : {out_dir}", file=sys.stderr)
    return out_dir


def _aggregate_all(retrieval_scores, judge_scores, null_stats, results) -> dict:
    overall_retrieval = rm.aggregate(retrieval_scores)
    overall_judge = rm.aggregate(judge_scores) if judge_scores else {}

    # Ventilation par catégorie de question.
    per_category: dict[str, dict] = {}
    by_cat: dict[str, list[dict]] = {}
    for res in results:
        by_cat.setdefault(res["question_type"], []).append(res)
    for cat, items in by_cat.items():
        r_scores = [it["retrieval"] for it in items]
        j_scores = [it["judge"] for it in items if it["judge"]]
        per_category[cat] = {
            "n": len(items),
            "retrieval": rm.aggregate(r_scores),
            "judge": rm.aggregate(j_scores) if j_scores else {},
            "avg_subqueries": sum(it["n_subqueries"] for it in items) / len(items),
        }

    tp, fp, tn, fn = null_stats["tp"], null_stats["fp"], null_stats["tn"], null_stats["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "retrieval": overall_retrieval,
        "judge": overall_judge,
        "null_detection": {
            **null_stats,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        },
        "per_category": per_category,
        "n_questions": len(results),
    }


def _write_run(config, config_path, backend, n_indexed, results, metrics) -> Path:
    out_dir = _run_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config figée.
    (out_dir / "config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Manifest de reproductibilité.
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config_hash": config_hash(config),
                "config_source": str(config_path),
                "variant": config.variant.value,
                "backend": backend,
                "n_chunks_indexed": n_indexed,
                "n_questions": len(results),
                "git_sha": _git_sha(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "seed": config.seed,
                "pip_freeze": _pip_freeze(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Résultats par question (logging des réponses LLM brutes -> run auditable).
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    # Métriques agrégées.
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Lance un run d'évaluation RAG (A ou A+B).")
    parser.add_argument("config", help="Chemin d'un YAML de config (ex: configs/baseline.yaml)")
    parser.add_argument("--backend", choices=["auto", "real", "memory"], default=None)
    args = parser.parse_args(argv)
    run_evaluation(args.config, backend=args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
