"""Unit tests for `dataset_scout.embedding_fit`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from dataset_scout.cache import Cache
from dataset_scout.context import ScoutContext
from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    Intent,
    Scorecard,
)
from dataset_scout.embedding_fit import (
    EMBEDDING_FIT_VERSION,
    _compose_candidate_text,
    _compose_intent_text,
    _cosine,
    _embedding_cache_key,
    assess_label_intent_fit,
)
from dataset_scout.errors import LLMError, SourceUnsupportedError

pytestmark = pytest.mark.unit


# ─── helpers ─────────────────────────────────────────────────────────


def _ctx_with_embedding(**overrides: Any) -> ScoutContext:
    base = {
        "aoai_endpoint": "https://my-aoai.openai.azure.com",
        "aoai_deployment": "gpt-4o-mini",
        "aoai_embedding_deployment": "text-embedding-3-small",
    }
    base.update(overrides)
    return ScoutContext(**base)


def _candidate(source: str = "huggingface", id_: str = "alice/x") -> Candidate:
    return Candidate(
        source=source,
        id=id_,
        revision="abc",
        metadata=CandidateMetadata(
            description="A toxic-comment classification dataset.",
            card_url="https://huggingface.co/datasets/alice/x",
            tags=["toxicity", "nlp"],
            task_categories=["text-classification"],
        ),
        streamable=True,
        surfaced_by=[],
    )


def _intent() -> Intent:
    return Intent(
        raw_brief="Find labeled toxicity datasets",
        detection_target="comment toxicity",
        threat_families=["toxicity"],
        languages=["en"],
    )


class _FakeSource:
    name: str

    def __init__(self, name: str, rows: list[dict[str, Any]] | None = None,
                 raise_unsupported: bool = False) -> None:
        self.name = name
        self._rows = rows or []
        self._raise = raise_unsupported

    def search(self, *args: Any, **kwargs: Any) -> Iterator[Candidate]:  # pragma: no cover
        yield from ()

    def fetch_metadata(self, candidate: Candidate) -> dict[str, Any]:  # pragma: no cover
        return {}

    def stream_sample(self, candidate: Candidate, n: int, seed: int) -> Iterator[dict[str, Any]]:
        yield from self._rows[:n]

    def stream_rows(
        self,
        candidate: Candidate,
        *,
        config: str | None = None,
        split: str = "train",
        take: int | None = None,
        seed: int = 42,
    ) -> Iterator[dict[str, Any]]:
        if self._raise:
            raise SourceUnsupportedError("test")
        yield from (self._rows[:take] if take else self._rows)

    def card_url(self, candidate: Candidate) -> str:  # pragma: no cover
        return candidate.metadata.card_url

    def terms_check(self, intent: Intent) -> list[Any]:  # pragma: no cover
        return []


class _FakeEmbeddingResponse:
    """Mimics litellm.embedding's response shape."""

    def __init__(self, vector: list[float]) -> None:
        # Either attribute or dict access works in real responses.
        self.data = [{"embedding": vector}]


class _FakeEmbedder:
    """Test double for the Embedder protocol.

    Records every call to ``embed`` and returns canned vectors. Tests
    inject this directly via ``embedder=`` to bypass the real backend
    factory and avoid mocking litellm / torch internals.
    """

    name: str = "fake"

    def __init__(
        self,
        vectors: list[list[float]] | None = None,
        *,
        model: str = "fake-model",
        on_call: Any | None = None,
    ) -> None:
        self.model = model
        self.calls: list[list[str]] = []
        self._vectors = vectors or [[1.0, 0.0, 0.0]]
        self._idx = 0
        # Optional callback (call_count -> vector | raises) to vary
        # behavior per call. When set, overrides ``vectors``.
        self._on_call = on_call

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for _ in texts:
            if self._on_call is not None:
                vec = self._on_call(len(self.calls))
            elif self._idx < len(self._vectors):
                vec = self._vectors[self._idx]
                self._idx += 1
            else:
                vec = self._vectors[-1]
            out.append(vec)
        return out


# ─── pure-function tests ────────────────────────────────────────────


def test_cosine_identical_vectors_is_one():
    assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_is_negative_one():
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_zero():
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_mismatched_lengths():
    assert _cosine([1.0, 2.0], [1.0]) == 0.0


def test_compose_intent_text_includes_brief_and_directions():
    from dataset_scout.core import DecompositionDirection

    intent = _intent()
    directions = [
        DecompositionDirection(
            name="d1",
            rationale="Refusal benchmarks.",
            keywords=["a", "b"],
            threat_families=[],
            expected_finds="Labeled refusal data",
        ),
    ]
    text = _compose_intent_text(intent, directions)
    assert "Find labeled toxicity datasets" in text
    assert "comment toxicity" in text
    assert "toxicity" in text
    assert "d1" in text
    assert "Refusal benchmarks." in text
    assert "Labeled refusal data" in text


def test_compose_candidate_text_stable_ordering():
    """Same candidate + same rows → same bytes."""
    cand = _candidate()
    rows = [{"b": 2, "a": 1}, {"text": "hi", "label": "ok"}]
    a = _compose_candidate_text(cand, rows)
    b = _compose_candidate_text(cand, list(rows))
    assert a == b
    # Sorted-key serialisation: 'a=' should appear before 'b=' in the row.
    assert a.find("a=1") < a.find("b=2")


def test_compose_candidate_text_caps_length():
    cand = _candidate()
    huge = "x" * 50_000
    rows = [{"col": huge}]
    text = _compose_candidate_text(cand, rows)
    assert len(text) <= 4001  # 4000 + ellipsis


def test_extract_embedding_dict_shape():
    from dataset_scout.embedder import _extract_embeddings

    response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    assert _extract_embeddings(response, expected=1) == [[0.1, 0.2, 0.3]]


def test_extract_embedding_attr_shape():
    from dataset_scout.embedder import _extract_embeddings

    response = _FakeEmbeddingResponse([0.4, 0.5])
    assert _extract_embeddings(response, expected=1) == [[0.4, 0.5]]


def test_extract_embedding_malformed():
    from dataset_scout.embedder import _extract_embeddings

    with pytest.raises(LLMError):
        _extract_embeddings({}, expected=1)
    with pytest.raises(LLMError):
        _extract_embeddings({"data": []}, expected=1)
    with pytest.raises(LLMError):
        _extract_embeddings({"data": [{}]}, expected=1)


def test_embedding_cache_key_stable():
    a = _embedding_cache_key(
        "hello world", embedder_name="aoai", embedder_model="text-embedding-3-small"
    )
    b = _embedding_cache_key(
        "hello world", embedder_name="aoai", embedder_model="text-embedding-3-small"
    )
    assert a == b
    assert len(a) == 64


def test_embedding_cache_key_changes_with_text_or_backend():
    a = _embedding_cache_key("a", embedder_name="aoai", embedder_model="dep1")
    assert _embedding_cache_key("b", embedder_name="aoai", embedder_model="dep1") != a
    assert _embedding_cache_key("a", embedder_name="aoai", embedder_model="dep2") != a
    # Same text+model but different backend → different key (sbert vs aoai
    # produce different-dim vectors that must not collide).
    assert _embedding_cache_key("a", embedder_name="sbert", embedder_model="dep1") != a


# ─── pipeline-stage integration tests ───────────────────────────────


def test_assess_noops_when_no_embedder_available():
    """Backend "none" → no calls, no scorecard mutation."""
    ctx = ScoutContext(
        aoai_endpoint="https://x",
        aoai_deployment="gpt",
        aoai_embedding_deployment=None,
        embedding_backend="none",
    )
    sc = Scorecard(candidate=_candidate())
    n = assess_label_intent_fit([sc], _intent(), ctx=ctx)
    assert n == 0
    assert sc.label_intent_fit is None


def test_assess_noops_when_aoai_backend_chosen_without_endpoint():
    """``embedding_backend=aoai`` with no AOAI deployment → no-op."""
    ctx = ScoutContext(
        aoai_endpoint=None,
        aoai_embedding_deployment="emb",
        embedding_backend="aoai",
    )
    sc = Scorecard(candidate=_candidate())
    n = assess_label_intent_fit([sc], _intent(), ctx=ctx)
    assert n == 0
    assert sc.label_intent_fit is None


def test_assess_populates_subscore_with_injected_embedder(tmp_path):
    """Happy path: injected embedder returns vectors, scorecard gets a SubScore."""
    ctx = _ctx_with_embedding()
    sc = Scorecard(candidate=_candidate())
    src = _FakeSource(
        "huggingface",
        rows=[{"text": "this comment is toxic", "label": "toxic"}],
    )
    cache = Cache(tmp_path / "cache.db")

    embedder = _FakeEmbedder(vectors=[[1.0, 0.0, 0.0]])
    n = assess_label_intent_fit(
        [sc],
        _intent(),
        ctx=ctx,
        source_index={"huggingface": src},
        cache=cache,
        embedder=embedder,
    )
    cache.close()

    assert n == 1
    assert sc.label_intent_fit is not None
    assert sc.label_intent_fit.value == pytest.approx(1.0, rel=1e-3)
    assert sc.label_intent_fit.status == "ok"
    assert sc.label_intent_fit.probe_version == EMBEDDING_FIT_VERSION
    assert sc.label_intent_fit.evidence
    assert "cosine=" in sc.label_intent_fit.evidence[0].detail
    assert "fake/fake-model" in sc.label_intent_fit.evidence[0].detail


def test_assess_uses_cache_on_second_call(tmp_path):
    ctx = _ctx_with_embedding()
    sc = Scorecard(candidate=_candidate())
    src = _FakeSource("huggingface", rows=[{"text": "x", "label": "y"}])
    cache = Cache(tmp_path / "cache.db")

    embedder = _FakeEmbedder(vectors=[[1.0, 0.0]])
    assess_label_intent_fit(
        [sc], _intent(), ctx=ctx, source_index={"huggingface": src}, cache=cache,
        embedder=embedder,
    )
    first = len(embedder.calls)
    # Reset scorecard, run again — every embedding (intent + 1 candidate)
    # should come from cache.
    sc.label_intent_fit = None
    assess_label_intent_fit(
        [sc], _intent(), ctx=ctx, source_index={"huggingface": src}, cache=cache,
        embedder=embedder,
    )
    second = len(embedder.calls)
    cache.close()
    assert first == 2  # intent + candidate
    assert second == first  # second run hit cache for both


def test_assess_low_confidence_when_no_sample_rows(tmp_path):
    ctx = _ctx_with_embedding()
    cand = _candidate()
    sc = Scorecard(candidate=cand)
    # Source returns no rows.
    src = _FakeSource("huggingface", rows=[])
    cache = Cache(tmp_path / "cache.db")
    embedder = _FakeEmbedder(vectors=[[1.0, 0.0]])
    assess_label_intent_fit(
        [sc], _intent(), ctx=ctx, source_index={"huggingface": src}, cache=cache,
        embedder=embedder,
    )
    cache.close()
    assert sc.label_intent_fit is not None
    assert sc.label_intent_fit.status == "low_confidence"


def test_assess_handles_unsupported_source_gracefully(tmp_path):
    """A SourceUnsupportedError on row fetch must not crash the stage."""
    ctx = _ctx_with_embedding()
    cand = _candidate(source="kaggle")
    sc = Scorecard(candidate=cand)
    src = _FakeSource("kaggle", raise_unsupported=True)
    cache = Cache(tmp_path / "cache.db")
    embedder = _FakeEmbedder(vectors=[[0.5, 0.5]])
    n = assess_label_intent_fit(
        [sc], _intent(), ctx=ctx, source_index={"kaggle": src}, cache=cache,
        embedder=embedder,
    )
    cache.close()
    # Stage still updated the scorecard (with low_confidence: no rows).
    assert n == 1
    assert sc.label_intent_fit is not None
    assert sc.label_intent_fit.status == "low_confidence"


def test_assess_per_candidate_failure_isolated(tmp_path):
    """If an embedding call fails for one candidate, others still get scored."""
    ctx = _ctx_with_embedding()
    sc1 = Scorecard(candidate=_candidate(id_="a/one"))
    sc2 = Scorecard(candidate=_candidate(id_="a/two"))
    src = _FakeSource("huggingface", rows=[{"text": "x", "label": "y"}])
    cache = Cache(tmp_path / "cache.db")

    def _on_call(call_count: int) -> list[float]:
        # Intent embedding (call 1) and second candidate (call 3) succeed;
        # first candidate (call 2) fails.
        if call_count == 2:
            raise RuntimeError("simulated embedding failure")
        return [1.0, 0.0]

    embedder = _FakeEmbedder(on_call=_on_call)
    assess_label_intent_fit(
        [sc1, sc2],
        _intent(),
        ctx=ctx,
        source_index={"huggingface": src},
        cache=cache,
        embedder=embedder,
    )
    cache.close()
    assert sc1.label_intent_fit is not None
    assert sc1.label_intent_fit.value is None
    assert sc1.label_intent_fit.status == "low_confidence"
    assert sc2.label_intent_fit is not None
    assert sc2.label_intent_fit.value is not None


def test_assess_returns_zero_for_empty_input():
    ctx = _ctx_with_embedding()
    n = assess_label_intent_fit([], _intent(), ctx=ctx)
    assert n == 0


def test_score_clamped_to_nonnegative(tmp_path):
    """Cosine ∈ [-1, 1] is clamped to [0, 1] in the SubScore.value."""
    ctx = _ctx_with_embedding()
    sc = Scorecard(candidate=_candidate())
    src = _FakeSource("huggingface", rows=[{"text": "x"}])
    cache = Cache(tmp_path / "cache.db")

    # Make intent and candidate embeddings opposite → cosine = -1.
    def _on_call(call_count: int) -> list[float]:
        if call_count == 1:
            return [1.0, 0.0]
        return [-1.0, 0.0]

    embedder = _FakeEmbedder(on_call=_on_call)
    assess_label_intent_fit(
        [sc], _intent(), ctx=ctx, source_index={"huggingface": src}, cache=cache,
        embedder=embedder,
    )
    cache.close()
    assert sc.label_intent_fit is not None
    assert sc.label_intent_fit.value == 0.0
    # Raw cosine still recorded in evidence.
    assert "cosine=-1." in sc.label_intent_fit.evidence[0].detail
