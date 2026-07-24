"""apps/cli — driving adapter interactif (V1). Appelle le MÊME `RagService`.

La future GUI (Streamlit/Gradio) sera un fichier de plus dans apps/ réutilisant cette
façade sans réécriture (ARCHITECTURE §7.2).

Usage :
    python -m apps.cli --config configs/enriched.yaml "Ta question ?"
    python -m apps.cli --config configs/enriched.yaml        # mode interactif
"""

from __future__ import annotations

import argparse
import sys

from apps.composition import build_service, build_stack
from config.loader import load_config
from config.schema import RunConfig
from pipelines_offline.index import run_index


def _print_answer(answer) -> None:
    print("\n" + "=" * 70)
    print("RÉPONSE" + ("" if answer.is_answerable else "  (⚠ preuve insuffisante)"))
    print("=" * 70)
    print(answer.text)
    if answer.citations:
        print("\nCitations :")
        for c in answer.citations:
            title = f" — {c.source_title}" if c.source_title else ""
            print(f"  • [{c.chunk_id}]{title} : {c.claim[:120]}")
    if answer.reasoning_trace.is_decomposed:
        print("\nRaisonnement (sous-requêtes) :")
        for i, hop in enumerate(answer.reasoning_trace.hops, 1):
            print(f"  {i}. {hop.subquery.text}  ({len(hop.retrieved)} passages)")
    print(f"\n[usage] {answer.usage.llm_calls} appel(s) LLM, "
          f"{answer.usage.total_tokens} tokens, {answer.usage.latency_s:.2f}s")


def build_from_config(config: RunConfig, backend: str | None):
    eff_backend, embedder, chat_llm, store = build_stack(config, backend)
    run_index(config, store)  # s'assure que l'index existe (cache si déjà fait)
    service = build_service(config, store, chat_llm)
    return eff_backend, service


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pose une question au système RAG (A+B).")
    parser.add_argument("--config", default="configs/enriched.yaml")
    parser.add_argument("--backend", choices=["auto", "real", "memory"], default=None)
    parser.add_argument("question", nargs="*", help="La question (sinon mode interactif).")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    eff_backend, service = build_from_config(config, args.backend)
    print(f"[backend: {eff_backend} | variante: {config.variant.value}]", file=sys.stderr)

    if args.question:
        _print_answer(service.ask(" ".join(args.question)))
        return 0

    print("Mode interactif — Ctrl-C ou 'quit' pour sortir.")
    try:
        while True:
            q = input("\n❓ ").strip()
            if q.lower() in {"quit", "exit", ""}:
                break
            _print_answer(service.ask(q))
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
