"""Types de domaine — vocabulaire stable, sérialisable, sans dépendance externe.

Ces modèles sont le contrat partagé entre le banc d'évaluation et (plus tard) la
GUI. Ils s'appuient uniquement sur Pydantic (sérialisation JSON gratuite, prête
pour une future API HTTP). Aucun `import llama_index`, `qdrant`, etc. ici.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Les 4 catégories de questions de MultiHop-RAG."""

    INFERENCE = "inference_query"
    COMPARISON = "comparison_query"
    TEMPORAL = "temporal_query"
    NULL = "null_query"


class Chunk(BaseModel):
    """Passage indexé + métadonnées (entité, date/période, section, type)."""

    id: str
    text: str
    source_title: str | None = None
    source_url: str | None = None
    author: str | None = None
    category: str | None = None
    published_at: str | None = None  # ISO 8601 quand disponible
    entities: list[str] = Field(default_factory=list)
    section: str | None = None
    doc_type: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class Query(BaseModel):
    """La question de l'utilisateur."""

    text: str
    question_type: QuestionType | None = None


class SubQuery(BaseModel):
    """Une sous-question produite par la couche B (par entité / période / hop)."""

    text: str
    rationale: str | None = None
    entity: str | None = None
    period: str | None = None


class RetrievedContext(BaseModel):
    """Chunk récupéré + score + provenance (quelle sous-requête l'a ramené)."""

    chunk: Chunk
    score: float
    from_subquery: str | None = None  # texte de la sous-requête d'origine


class Citation(BaseModel):
    """Lie une affirmation de la réponse à un passage source."""

    claim: str
    chunk_id: str
    source_title: str | None = None


class HopTrace(BaseModel):
    """Une étape de raisonnement : une sous-requête et ce qu'elle a récupéré."""

    subquery: SubQuery
    retrieved: list[RetrievedContext] = Field(default_factory=list)


class ReasoningTrace(BaseModel):
    """Sous-requêtes + retrieval par sous-requête.

    La baseline A produit une trace dégénérée à un seul hop (Null Object).
    """

    hops: list[HopTrace] = Field(default_factory=list)

    @property
    def is_decomposed(self) -> bool:
        return len(self.hops) > 1


class TokenUsage(BaseModel):
    """Coût / latence, pour le debug et l'analyse d'éval."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    llm_calls: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            latency_s=self.latency_s + other.latency_s,
            llm_calls=self.llm_calls + other.llm_calls,
        )


class Answer(BaseModel):
    """L'objet de retour riche — consommé par l'éval ET la GUI."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedContext] = Field(default_factory=list)
    reasoning_trace: ReasoningTrace = Field(default_factory=ReasoningTrace)
    is_answerable: bool = True
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @property
    def unique_retrieved(self) -> list[RetrievedContext]:
        """Contextes dédupliqués par chunk id, meilleur score en tête."""

        best: dict[str, RetrievedContext] = {}
        for rc in self.retrieved:
            cur = best.get(rc.chunk.id)
            if cur is None or rc.score > cur.score:
                best[rc.chunk.id] = rc
        return sorted(best.values(), key=lambda r: r.score, reverse=True)


class Evidence(BaseModel):
    """L'evidence en or annotée (utilisée par l'éval uniquement)."""

    fact: str
    source_title: str | None = None
    source_url: str | None = None
    published_at: str | None = None
    category: str | None = None


class EvalQuestion(BaseModel):
    """Une question du jeu MultiHop-RAG : question + réponse or + evidence or."""

    query: str
    answer: str | None = None
    question_type: QuestionType
    evidence: list[Evidence] = Field(default_factory=list)

    def to_query(self) -> Query:
        return Query(text=self.query, question_type=self.question_type)
