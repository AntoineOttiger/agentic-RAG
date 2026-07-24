"""Validation manuelle du juge (SCOPE §3.5, critère de succès).

Avant de faire confiance au LLM-as-judge, on le calibre à la main sur ~20 exemples.

Flux en 2 temps :
1. `export` : échantillonne N questions d'un run, produit un CSV à annoter à la main
   (colonnes `human_correctness`, `human_faithful` à remplir dans [0,1] ou {0,1}).
2. `agree`  : recharge le CSV annoté et calcule l'accord juge ↔ humain (corrélation +
   erreur absolue moyenne). Un accord faible = juge non fiable, à revoir.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def export_template(run_dir: str | Path, out_csv: str | Path, n: int = 20) -> Path:
    run_dir = Path(run_dir)
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Échantillon régulier couvrant tout le run.
    step = max(1, len(rows) // n)
    sample = rows[::step][:n]

    out_csv = Path(out_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["query", "gold_answer", "answer", "judge_correctness", "judge_faithful",
             "human_correctness", "human_faithful"]
        )
        for r in sample:
            j = r.get("judge", {})
            writer.writerow(
                [
                    r["query"], r.get("gold_answer", ""), r["answer"],
                    j.get("answer_correctness", ""), j.get("faithfulness", ""),
                    "", "",  # à remplir à la main
                ]
            )
    print(f"✔ Modèle d'annotation écrit : {out_csv} ({len(sample)} exemples)", file=sys.stderr)
    return out_csv


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def agreement(annotated_csv: str | Path) -> dict:
    rows = list(csv.DictReader(Path(annotated_csv).open(encoding="utf-8")))
    pairs_c, pairs_f = [], []
    for r in rows:
        try:
            if r["human_correctness"] != "":
                pairs_c.append((float(r["judge_correctness"]), float(r["human_correctness"])))
            if r["human_faithful"] != "":
                pairs_f.append((float(r["judge_faithful"]), float(r["human_faithful"])))
        except (ValueError, KeyError):
            continue

    def _summary(pairs):
        if not pairs:
            return {"n": 0}
        j = [p[0] for p in pairs]
        h = [p[1] for p in pairs]
        mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
        return {"n": len(pairs), "pearson": round(_corr(j, h), 3), "mae": round(mae, 3)}

    result = {"correctness": _summary(pairs_c), "faithfulness": _summary(pairs_f)}
    print(json.dumps(result, indent=2), file=sys.stderr)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Valide le juge LLM à la main.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="Génère un CSV à annoter.")
    ex.add_argument("run_dir")
    ex.add_argument("--out", default="runs/judge_validation.csv")
    ex.add_argument("-n", type=int, default=20)

    ag = sub.add_parser("agree", help="Calcule l'accord juge ↔ humain.")
    ag.add_argument("csv")

    args = parser.parse_args(argv)
    if args.cmd == "export":
        export_template(args.run_dir, args.out, args.n)
    else:
        agreement(args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
