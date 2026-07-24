"""Le cœur se teste contre des fakes des ports — sans Qdrant ni LLM (ARCHITECTURE §9)."""

from core.decompose.agentic import QueryDecomposer
from core.decompose.null import NullDecomposer
from core.pipeline import Pipeline
from domain.types import Answer, Chunk, Query, RetrievedContext, SubQuery, TokenUsage


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, subquery, top_k, filters=None):
        self.calls.append((subquery.text, filters))
        return [
            RetrievedContext(
                chunk=Chunk(id=f"c-{subquery.text[:3]}", text=subquery.text),
                score=0.9,
            )
        ]


class FakeGenerator:
    def generate(self, query, contexts):
        return Answer(text=f"answer:{len(contexts)}", is_answerable=bool(contexts), usage=TokenUsage(llm_calls=1))


class ScriptedLLM:
    """Renvoie une décomposition JSON fixe."""

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user, temperature=0.0, max_tokens=None):
        return self.payload, TokenUsage(llm_calls=1)


def test_null_decomposer_single_hop():
    subs, usage = NullDecomposer().decompose(Query(text="q?"))
    assert len(subs) == 1
    assert usage.llm_calls == 0


def test_pipeline_baseline_one_hop():
    retriever = FakeRetriever()
    pipe = Pipeline(NullDecomposer(), retriever, FakeGenerator(), top_k=3)
    answer = pipe.answer(Query(text="single question"))
    assert len(answer.reasoning_trace.hops) == 1
    assert not answer.reasoning_trace.is_decomposed
    assert answer.is_answerable


def test_pipeline_enriched_multi_hop_and_dedup():
    llm = ScriptedLLM('{"subqueries": [{"text": "sub one"}, {"text": "sub two"}]}')
    retriever = FakeRetriever()
    pipe = Pipeline(QueryDecomposer(llm), retriever, FakeGenerator(), top_k=3)
    answer = pipe.answer(Query(text="complex multi hop"))
    assert answer.reasoning_trace.is_decomposed
    assert len(answer.reasoning_trace.hops) == 2
    assert len(retriever.calls) == 2


def test_decomposer_fallback_on_bad_json():
    llm = ScriptedLLM("not json at all")
    subs, _ = QueryDecomposer(llm).decompose(Query(text="q?"))
    assert len(subs) == 1  # dégradation gracieuse vers la baseline


def test_metadata_filters_passed_when_entity_present():
    llm = ScriptedLLM('{"subqueries": [{"text": "s", "entity": "TechCorp"}]}')
    retriever = FakeRetriever()
    pipe = Pipeline(QueryDecomposer(llm), retriever, FakeGenerator(), top_k=3, use_metadata_filters=True)
    pipe.answer(Query(text="q"))
    assert retriever.calls[0][1] == {"entity": "TechCorp"}
