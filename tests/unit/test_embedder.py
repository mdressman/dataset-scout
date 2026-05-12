"""Unit tests for dataset_scout.embedder — the embedding backend factory."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from dataset_scout.context import ScoutContext
from dataset_scout.embedder import (
    AOAIEmbedder,
    SbertEmbedder,
    _extract_embeddings,
    build_embedder,
)
from dataset_scout.errors import LLMError

pytestmark = pytest.mark.unit


# ─── _extract_embeddings ────────────────────────────────────────────


def test_extract_embeddings_dict_shape() -> None:
    response = {
        "data": [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
    }
    assert _extract_embeddings(response, expected=2) == [[0.1, 0.2], [0.3, 0.4]]


def test_extract_embeddings_attr_shape() -> None:
    class _Item:
        def __init__(self, vec: list[float]) -> None:
            self.embedding = vec

    class _Resp:
        def __init__(self, items: list[_Item]) -> None:
            self.data = items

    resp = _Resp([_Item([0.5, 0.6])])
    assert _extract_embeddings(resp, expected=1) == [[0.5, 0.6]]


def test_extract_embeddings_raises_on_empty() -> None:
    with pytest.raises(LLMError):
        _extract_embeddings({"data": []}, expected=1)


def test_extract_embeddings_raises_on_too_few_items() -> None:
    response = {"data": [{"embedding": [0.1]}]}
    with pytest.raises(LLMError):
        _extract_embeddings(response, expected=3)


# ─── SbertEmbedder ──────────────────────────────────────────────────


def test_sbert_embedder_friendly_error_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without sentence-transformers installed, instantiation must
    raise an LLMError with an actionable install hint, not an
    ImportError stack trace."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(LLMError, match="sentence-transformers is not installed"):
        SbertEmbedder()


# ─── AOAIEmbedder ───────────────────────────────────────────────────


def test_aoai_embedder_requires_aoai_config() -> None:
    ctx = ScoutContext()  # no AOAI fields
    with pytest.raises(LLMError, match="AOAIEmbedder requires"):
        AOAIEmbedder(ctx)


def test_aoai_embedder_routes_through_litellm_with_token_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_embedding(**kwargs: Any) -> dict[str, list[dict[str, list[float]]]]:
        captured.update(kwargs)
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    fake_litellm = MagicMock()
    fake_litellm.embedding.side_effect = fake_embedding

    def fake_token_provider() -> object:
        return lambda: "fake-token"

    monkeypatch.setattr("dataset_scout.llm_client.import_litellm", lambda: fake_litellm)
    monkeypatch.setattr(
        "dataset_scout.llm_client.make_token_provider", fake_token_provider
    )

    ctx = ScoutContext(
        aoai_endpoint="https://my-aoai.openai.azure.com",
        aoai_deployment="gpt-4o",
        aoai_embedding_deployment="text-embedding-3-small",
    )
    emb = AOAIEmbedder(ctx)
    vecs = emb.embed(["hello"])

    assert vecs == [[0.1, 0.2, 0.3]]
    assert captured["model"] == "azure/text-embedding-3-small"
    assert captured["api_base"] == "https://my-aoai.openai.azure.com"
    assert captured["api_version"] == "2024-10-21"
    assert callable(captured["azure_ad_token_provider"])


def test_aoai_embedder_empty_input_short_circuits() -> None:
    ctx = ScoutContext(
        aoai_endpoint="https://x",
        aoai_deployment="d",
        aoai_embedding_deployment="emb",
    )
    emb = AOAIEmbedder(ctx)
    assert emb.embed([]) == []


# ─── build_embedder factory ─────────────────────────────────────────


def test_build_embedder_returns_none_for_backend_none() -> None:
    ctx = ScoutContext(embedding_backend="none")
    assert build_embedder(ctx) is None


def test_build_embedder_returns_none_for_aoai_without_endpoint() -> None:
    ctx = ScoutContext(embedding_backend="aoai")
    assert build_embedder(ctx) is None


def test_build_embedder_returns_aoai_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = ScoutContext(
        aoai_endpoint="https://x",
        aoai_deployment="d",
        aoai_embedding_deployment="emb",
        embedding_backend="aoai",
    )
    emb = build_embedder(ctx)
    assert emb is not None
    assert emb.name == "aoai"
    assert emb.model == "emb"


def test_build_embedder_returns_none_when_sbert_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default sbert backend with sentence-transformers missing → None
    (cleanly skips the embedding-fit stage)."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    ctx = ScoutContext(embedding_backend="sbert")
    assert build_embedder(ctx) is None


def test_build_embedder_uses_custom_sbert_model_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DATASET_SCOUT_EMBEDDING_MODEL`` flows through to SbertEmbedder."""
    captured_model: list[str] = []

    class _FakeST:
        def __init__(self, model: str) -> None:
            captured_model.append(model)

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

    fake_module = MagicMock()
    fake_module.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    ctx = ScoutContext(
        embedding_backend="sbert", embedding_model="BAAI/bge-small-en-v1.5"
    )
    emb = build_embedder(ctx)
    assert emb is not None
    assert emb.name == "sbert"
    assert emb.model == "BAAI/bge-small-en-v1.5"
    assert captured_model == ["BAAI/bge-small-en-v1.5"]


# ─── regression: mixed-provider AOAI embeddings ─────────────────────


def test_aoai_embedder_works_without_chat_deployment() -> None:
    """Mixed-provider regression. Pre-fix: AOAIEmbedder gated on
    ``ctx.aoai_configured`` which requires both ``aoai_endpoint`` AND
    ``aoai_deployment`` (the *chat* deployment). A user who runs chat
    on github_copilot/ but embeddings on AOAI would only set the
    embedding deployment and the endpoint — and got locked out of AOAI
    embeddings with no useful error. Embeddings only need endpoint +
    embedding deployment.
    """
    ctx = ScoutContext(
        aoai_endpoint="https://x.openai.azure.com",
        aoai_embedding_deployment="text-embedding-3-small",
        # No aoai_deployment — chat is being routed to a different provider.
        model="github_copilot/gpt-5-mini",
    )
    emb = AOAIEmbedder(ctx)
    assert emb.name == "aoai"
    assert emb.model == "text-embedding-3-small"


def test_build_embedder_aoai_works_without_chat_deployment() -> None:
    """Same regression at the factory level."""
    ctx = ScoutContext(
        aoai_endpoint="https://x.openai.azure.com",
        aoai_embedding_deployment="text-embedding-3-small",
        embedding_backend="aoai",
        model="github_copilot/gpt-5-mini",
    )
    emb = build_embedder(ctx)
    assert emb is not None
    assert emb.name == "aoai"
