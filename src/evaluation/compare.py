"""Comparatif A vs A+B — le livrable analytique (SCOPE §3.6).

Lit deux runs (baseline A et enrichi A+B), produit un rapport Markdown chiffré :
retrieval + génération, global et ventilé sur les 4 catégories de questions, plus la
détection des null queries. Le résultat peut être positif OU négatif — on mesure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS_ROOT = Path("runs")


def _load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def find_latest_by_variant(variant: str) -> Path | None:
    candidates = []
    for d in sorted(RUNS_ROOT.glob("*")):
        if not (d / "manifest.json").exists():
            continue
        if _load_manifest(d).get("variant") == variant:
            candidates.append(d)
    return candidates[-1] if candidates else None


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def _delta(a: float | None, b: float | None) -> str:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "—"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _metric_table(name: str, keys: list[str], a: dict, b: dict) -> list[str]:
    lines = [f"### {name}", "", "| Métrique | A (baseline) | A+B (enrichi) | Δ |", "|---|---|---|---|"]
    for key in keys:
        av, bv = a.get(key), b.get(key)
        lines.append(f"| {key} | {_fmt(av)} | {_fmt(bv)} | {_delta(av, bv)} |")
    lines.append("")
    return lines


def build_report(baseline_dir: Path, enriched_dir: Path) -> str:
    a_metrics = _load_metrics(baseline_dir)
    b_metrics = _load_metrics(enriched_dir)
    a_man = _load_manifest(baseline_dir)
    b_man = _load_manifest(enriched_dir)

    L: list[str] = []
    L.append("# Rapport d'évaluation — A vs A+B")
    L.append("")
    L.append(f"- **Baseline (A)** : `{baseline_dir.name}` — backend `{a_man.get('backend')}`, "
             f"{a_man.get('n_questions')} questions, git `{(a_man.get('git_sha') or '—')[:8]}`")
    L.append(f"- **Enrichi (A+B)** : `{enriched_dir.name}` — backend `{b_man.get('backend')}`, "
             f"{b_man.get('n_questions')} questions, git `{(b_man.get('git_sha') or '—')[:8]}`")
    L.append("")
    if a_man.get("backend") == "memory" or b_man.get("backend") == "memory":
        L.append("> ⚠️ **Backend `memory`** : chiffres produits avec les adapters en mémoire "
                 "(embeddings hachés + LLM extractif déterministe). Ils valident la plomberie "
                 "et la méthodologie, **pas** la performance réelle du système. Relancer avec "
                 "`--backend real` (bge + Qdrant + Mistral) pour le livrable final.")
        L.append("")

    # Étage 1 — retrieval
    r_keys = sorted(set(a_metrics.get("retrieval", {})) | set(b_metrics.get("retrieval", {})))
    L += _metric_table("Étage 1 — Retrieval (global)", r_keys, a_metrics.get("retrieval", {}), b_metrics.get("retrieval", {}))

    # Étage 2 — génération
    j_keys = sorted(set(a_metrics.get("judge", {})) | set(b_metrics.get("judge", {})))
    if j_keys:
        L += _metric_table("Étage 2 — Génération / juge (global)", j_keys, a_metrics.get("judge", {}), b_metrics.get("judge", {}))

    # Null detection
    a_null, b_null = a_metrics.get("null_detection", {}), b_metrics.get("null_detection", {})
    L += _metric_table("Détection des null queries", ["precision", "recall", "f1"], a_null, b_null)

    # Par catégorie
    L.append("## Ventilation par catégorie")
    L.append("")
    a_cat, b_cat = a_metrics.get("per_category", {}), b_metrics.get("per_category", {})
    cats = sorted(set(a_cat) | set(b_cat))
    for cat in cats:
        ac, bc = a_cat.get(cat, {}), b_cat.get(cat, {})
        n = ac.get("n") or bc.get("n")
        L.append(f"### {cat} (n={n})")
        L.append("")
        L.append(f"- Sous-requêtes moyennes : A={_fmt(ac.get('avg_subqueries'))} → "
                 f"A+B={_fmt(bc.get('avg_subqueries'))}")
        rk = sorted(set(ac.get("retrieval", {})) | set(bc.get("retrieval", {})))
        L += _metric_table("Retrieval", rk, ac.get("retrieval", {}), bc.get("retrieval", {}))
        jk = sorted(set(ac.get("judge", {})) | set(bc.get("judge", {})))
        if jk:
            L += _metric_table("Génération", jk, ac.get("judge", {}), bc.get("judge", {}))

    L.append("## Lecture")
    L.append("")
    L.append("- Δ > 0 : la couche B (décomposition) améliore la métrique.")
    L.append("- Δ < 0 : la décomposition dégrade — résultat négatif, à documenter tel quel.")
    L.append("- Biais assumé : juge = même famille de modèle que le générateur (SCOPE §6.1) ; "
             "valider le juge à la main sur ~20 exemples avant de conclure.")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Comparatif A vs A+B à partir de deux runs.")
    parser.add_argument("--baseline", help="dossier run A (défaut: dernier run variant A)")
    parser.add_argument("--enriched", help="dossier run A+B (défaut: dernier run variant A+B)")
    parser.add_argument("--out", default="runs/REPORT_A_vs_AB.md")
    args = parser.parse_args(argv)

    baseline = Path(args.baseline) if args.baseline else find_latest_by_variant("A")
    enriched = Path(args.enriched) if args.enriched else find_latest_by_variant("A+B")
    if not baseline or not enriched:
        print("Runs introuvables : lance d'abord run_eval sur baseline.yaml et enriched.yaml.", file=sys.stderr)
        return 1

    report = build_report(baseline, enriched)
    out = Path(args.out)
    out.write_text(report, encoding="utf-8")
    # NB : on n'imprime PAS le rapport sur stdout (glyphes Δ/emoji incompatibles avec
    # certaines consoles Windows cp1252). Le fichier UTF-8 est le livrable.
    sys.stderr.write(f"Rapport ecrit : {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
