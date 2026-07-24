"""LLMGenerator — synthèse ancrée + gestion des null queries.

Implémente `domain.ports.Generator` par-dessus le port `ChatLLM`. Le prompt impose :
- une réponse UNIQUEMENT étayée par le contexte fourni (faithfulness) ;
- l'aveu explicite « preuve insuffisante » quand rien n'étaye la réponse
  (`is_answerable=False`, gestion des null queries, SCOPE §3.4) ;
- des citations reliant chaque affirmation à un id de chunk (contrat Answer).
"""

from __future__ import annotations

import json
import re

from domain.ports import ChatLLM
from domain.types import Answer, Citation, Query, RetrievedContext, TokenUsage

_PROMPT_VERSION = "generate-v1"

_SYSTEM = """Tu es un analyste de veille média rigoureux. Tu réponds à une question \
UNIQUEMENT à partir des passages numérotés fournis. Règles absolues :

1. N'invente rien. N'utilise aucune connaissance externe aux passages.
2. Si les passages ne contiennent pas de quoi répondre, tu réponds \
"is_answerable": false et "answer": "Preuve insuffisante dans les documents fournis.".
3. Chaque affirmation de ta réponse doit être appuyée par au moins un passage, cité \
par son identifiant (le [ID] entre crochets).

Réponds STRICTEMENT en JSON, sans texte autour :
{"is_answerable": true|false, "answer": "...", "citations": [{"claim": "...", "chunk_id": "ID"}]}
"""

_USER_TMPL = """Question : {question}

Passages :
{context}

Réponds en JSON maintenant."""


def _format_context(contexts: list[RetrievedContext]) -> str:
    lines = []
    for rc in contexts:
        meta = []
        if rc.chunk.source_title:
            meta.append(rc.chunk.source_title)
        if rc.chunk.published_at:
            meta.append(rc.chunk.published_at)
        header = f"[{rc.chunk.id}]" + (f" ({' — '.join(meta)})" if meta else "")
        lines.append(f"{header}\n{rc.chunk.text}")
    return "\n\n".join(lines)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class LLMGenerator:
    prompt_version = _PROMPT_VERSION

    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm

    def generate(self, query: Query, contexts: list[RetrievedContext]) -> Answer:
        if not contexts:
            return Answer(
                text="Preuve insuffisante dans les documents fournis.",
                is_answerable=False,
                usage=TokenUsage(),
            )

        titles = {rc.chunk.id: rc.chunk.source_title for rc in contexts}
        user = _USER_TMPL.format(question=query.text, context=_format_context(contexts))

        try:
            raw, usage = self._llm.complete(_SYSTEM, user, temperature=0.0)
            data = _extract_json(raw)
        except Exception:
            return Answer(
                text="Erreur de génération : réponse du modèle non exploitable.",
                is_answerable=False,
                usage=TokenUsage(llm_calls=1),
            )

        is_answerable = bool(data.get("is_answerable", True))
        text = (data.get("answer") or "").strip()
        citations = [
            Citation(
                claim=(c.get("claim") or "").strip(),
                chunk_id=str(c.get("chunk_id", "")).strip(),
                source_title=titles.get(str(c.get("chunk_id", "")).strip()),
            )
            for c in data.get("citations", [])
            if c.get("chunk_id")
        ]
        return Answer(
            text=text or "Preuve insuffisante dans les documents fournis.",
            citations=citations,
            is_answerable=is_answerable and bool(text),
            usage=usage,
        )
