"""Juge LLM natif — implémente `domain.ports.Judge` via le port `ChatLLM`.

Étage 2 de l'évaluation (ARCHITECTURE §7.1). Le juge note :
- answer_correctness : la réponse correspond-elle à la référence or ?
- faithfulness : chaque affirmation est-elle étayée par le contexte récupéré ?
- context_relevance : le contexte est-il pertinent pour la question ?

⚠️ Biais assumé (SCOPE §6.1) : si le juge partage le modèle du générateur, il y a
auto-préférence. À documenter dans le rapport et à valider à la main sur ~20 exemples
(voir `evaluation/judge/manual_validation.py`).
"""

from __future__ import annotations

import json
import re

from domain.ports import ChatLLM
from domain.types import Answer, Query

_SYSTEM = """Tu es un évaluateur impartial de réponses de systèmes RAG. On te donne une \
question, une réponse candidate, la réponse de référence (or) et les passages qui ont \
servi de contexte. Note la réponse candidate sur 3 axes, chacun dans [0, 1] :

- "answer_correctness" : proximité factuelle avec la réponse de référence (1 = équivalente).
- "faithfulness" : proportion d'affirmations réellement étayées par les passages (1 = tout étayé).
- "context_relevance" : pertinence des passages pour répondre (1 = très pertinents).

Réponds STRICTEMENT en JSON : {"answer_correctness": x, "faithfulness": x, "context_relevance": x}.
"""

_USER_TMPL = """Question : {question}

Réponse de référence (or) : {reference}

Réponse candidate : {candidate}

Passages fournis au système :
{context}

Note maintenant en JSON."""


def _extract_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def _clamp(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


class NativeJudge:
    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm

    def score(self, query: Query, answer: Answer, reference: str | None = None) -> dict[str, float]:
        context = "\n\n".join(
            f"[{rc.chunk.id}] {rc.chunk.text[:500]}" for rc in answer.unique_retrieved[:8]
        )
        user = _USER_TMPL.format(
            question=query.text,
            reference=reference or "(non fournie)",
            candidate=answer.text,
            context=context or "(aucun)",
        )
        try:
            raw, _ = self._llm.complete(_SYSTEM, user, temperature=0.0)
            data = _extract_json(raw)
        except Exception:
            return {"answer_correctness": 0.0, "faithfulness": 0.0, "context_relevance": 0.0}
        return {
            "answer_correctness": _clamp(data.get("answer_correctness")),
            "faithfulness": _clamp(data.get("faithfulness")),
            "context_relevance": _clamp(data.get("context_relevance")),
        }
