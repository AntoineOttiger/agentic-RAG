"""QueryDecomposer — la contribution (couche B), écrite à la main.

Découpe une question multi-hop en sous-questions atomiques (par entité / période /
hop). Écrit à la main, sans framework agentique, pour rester auditable
(ARCHITECTURE §1, décision 2). Dépend uniquement du port `ChatLLM` de `domain`.

Robustesse : si le LLM échoue ou renvoie du JSON invalide, on retombe sur la
question originale (dégradation gracieuse vers le comportement baseline).
"""

from __future__ import annotations

import json
import re

from domain.ports import ChatLLM
from domain.types import Query, SubQuery, TokenUsage

_PROMPT_VERSION = "decompose-v1"

_SYSTEM = """Tu es un assistant de recherche documentaire. On te donne une question \
complexe qui exige de croiser plusieurs documents (multi-hop), parfois de périodes \
différentes. Découpe-la en 2 à 4 sous-questions ATOMIQUES, chacune répondable par un \
seul document. Chaque sous-question doit être autonome (pas de pronom ambigu).

Réponds STRICTEMENT en JSON, sans texte autour, au format :
{"subqueries": [{"text": "...", "entity": "... ou null", "period": "... ou null", \
"rationale": "..."}]}

- "entity" : l'entité principale (personne, organisation, produit) si identifiable, sinon null.
- "period" : une période ou date si la question est temporelle (ex: "2023-Q1", "septembre 2023"), sinon null.
- Si la question est déjà atomique, renvoie une seule sous-question égale à la question.
"""

_USER_TMPL = 'Question : "{q}"\n\nDécompose-la maintenant.'


def _extract_json(raw: str) -> dict:
    """Extrait le premier objet JSON d'une réponse LLM (tolère le bavardage)."""

    raw = raw.strip()
    # Retire d'éventuels fences ```json ... ```
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class QueryDecomposer:
    """Décomposition agentique de requête via un LLM (couche B)."""

    prompt_version = _PROMPT_VERSION

    def __init__(self, llm: ChatLLM, max_subqueries: int = 4) -> None:
        self._llm = llm
        self._max = max_subqueries

    def decompose(self, query: Query) -> tuple[list[SubQuery], TokenUsage]:
        user = _USER_TMPL.format(q=query.text)
        try:
            raw, usage = self._llm.complete(_SYSTEM, user, temperature=0.0)
            data = _extract_json(raw)
            items = data.get("subqueries", [])
            subs: list[SubQuery] = []
            for item in items[: self._max]:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                subs.append(
                    SubQuery(
                        text=text,
                        rationale=item.get("rationale"),
                        entity=(item.get("entity") or None),
                        period=(item.get("period") or None),
                    )
                )
            if not subs:
                raise ValueError("empty decomposition")
            return subs, usage
        except Exception:
            # Dégradation gracieuse : retombe sur la baseline (1 hop).
            return (
                [SubQuery(text=query.text, rationale="fallback: decomposition failed")],
                TokenUsage(),
            )
