"""LLM en mémoire — un `ChatLLM` sans réseau, pour tests et smoke run offline.

`ExtractiveLLM` ne « raisonne » pas : il applique des heuristiques déterministes qui
respectent le contrat JSON attendu par le décomposeur et le générateur. Cela permet
de valider la plomberie de bout en bout sans clé API ni non-déterminisme.
"""

from __future__ import annotations

import json
import re

from domain.types import TokenUsage

_STOP = {
    "what", "which", "who", "when", "where", "how", "the", "a", "an", "of", "and",
    "or", "to", "in", "on", "for", "did", "does", "is", "are", "was", "were", "that",
    "this", "with", "as", "by", "at", "from", "after", "before", "between",
}


class ExtractiveLLM:
    """Répond de façon extractive/déterministe au format JSON attendu."""

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[str, TokenUsage]:
        usage = TokenUsage(
            prompt_tokens=len(system.split()) + len(user.split()),
            completion_tokens=32,
            total_tokens=len(system.split()) + len(user.split()) + 32,
            latency_s=0.0,
            llm_calls=1,
        )
        if "Découpe" in system or "sous-questions" in system:
            return self._decompose(user), usage
        return self._generate(user), usage

    # --- décomposition naïve : coupe sur "and"/","/"compared to" ---
    def _decompose(self, user: str) -> str:
        m = re.search(r'"(.+?)"', user, flags=re.DOTALL)
        question = m.group(1) if m else user
        parts = re.split(r"\band\b|\bcompared to\b|,", question)
        parts = [p.strip(" ?.") for p in parts if len(p.strip()) > 8]
        if len(parts) < 2:
            parts = [question.strip()]
        subs = [{"text": p, "entity": None, "period": None, "rationale": "split"} for p in parts[:4]]
        return json.dumps({"subqueries": subs})

    # --- génération extractive : reprend la phrase du passage la plus couvrante ---
    def _generate(self, user: str) -> str:
        qmatch = re.search(r"Question\s*:\s*(.+)", user)
        question = qmatch.group(1).strip() if qmatch else ""
        qtokens = {t for t in _tokenize(question) if t not in _STOP}

        # Repère les passages "[ID]\ntext"
        blocks = re.findall(r"\[([0-9a-fA-F]+)\]([^\[]*)", user.split("Passages :", 1)[-1])
        best_id, best_text, best_overlap = None, "", 0
        for cid, body in blocks:
            # Retire l'entête de métadonnées "(Titre — date)" en tête de bloc.
            body = re.sub(r"^\s*\([^)]*\)\s*", "", body.strip())
            btokens = set(_tokenize(body))
            overlap = len(qtokens & btokens)
            if overlap > best_overlap:
                best_id, best_text, best_overlap = cid, body.strip(), overlap

        if not best_id or best_overlap == 0:
            return json.dumps(
                {
                    "is_answerable": False,
                    "answer": "Preuve insuffisante dans les documents fournis.",
                    "citations": [],
                }
            )
        sentence = re.split(r"(?<=[.!?])\s+", best_text)[0][:300]
        return json.dumps(
            {
                "is_answerable": True,
                "answer": sentence,
                "citations": [{"claim": sentence, "chunk_id": best_id}],
            }
        )


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]
