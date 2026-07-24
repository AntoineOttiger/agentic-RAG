"""Chargement du jeu d'évaluation MultiHop-RAG (questions + evidence en or)."""

from __future__ import annotations

import json
from pathlib import Path

from domain.types import Evidence, EvalQuestion, QuestionType

_TYPE_MAP = {
    "inference_query": QuestionType.INFERENCE,
    "comparison_query": QuestionType.COMPARISON,
    "temporal_query": QuestionType.TEMPORAL,
    "null_query": QuestionType.NULL,
    "null": QuestionType.NULL,
}


def _map_type(raw: str | None) -> QuestionType:
    if not raw:
        return QuestionType.INFERENCE
    return _TYPE_MAP.get(raw.strip().lower(), QuestionType.INFERENCE)


def load_questions(path: str | Path, limit: int | None = None) -> list[EvalQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or []

    questions: list[EvalQuestion] = []
    for item in data:
        evidence = [
            Evidence(
                fact=ev.get("fact", ""),
                source_title=ev.get("title"),
                source_url=ev.get("url"),
                published_at=ev.get("published_at"),
                category=ev.get("category"),
            )
            for ev in item.get("evidence_list", [])
        ]
        questions.append(
            EvalQuestion(
                query=item.get("query", ""),
                answer=item.get("answer"),
                question_type=_map_type(item.get("question_type")),
                evidence=evidence,
            )
        )
    if limit is not None:
        questions = questions[:limit]
    return questions
