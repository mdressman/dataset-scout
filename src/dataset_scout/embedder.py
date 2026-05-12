"""Embedding-backend abstraction.

Lets `embedding_fit.py` use either a CPU-friendly sentence-transformers
model (default: ``all-MiniLM-L6-v2``, 384-dim) or the legacy Azure
OpenAI embedding deployment without caring about which one. Picked
once per run by :func:`build_embedder` based on
``ScoutContext.embedding_backend``.

Why a Protocol and not a base class?
- The two implementations have nothing in common at runtime — sbert
  hits a local PyTorch model, AOAI hits a network endpoint with Entra
  auth. A Protocol lets each be a dataclass-shaped value object
  without forcing inheritance.
- Tests can inject a trivial ``FakeEmbedder`` with two attributes and
  one method; no mock of litellm or PyTorch needed.

Caching policy lives in ``embedding_fit.py`` (cache namespace
``embedding``, key includes ``embedder.name`` + ``embedder.model``).
The Embedder itself is dumb: given a list of texts, return a list of
vectors. No retries, no caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dataset_scout.errors import LLMError

if TYPE_CHECKING:
    from dataset_scout.context import ScoutContext


_DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """Minimal contract for an embedding backend.

    Attributes
    ----------
    name:
        Short backend tag (``"sbert"`` / ``"aoai"`` / test fakes).
        Goes into the cache key so a backend switch can't serve stale
        vectors from the previous backend.
    model:
        Model identifier. For sbert this is the HF repo id (e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"``); for AOAI this is
        the deployment name. Also goes into the cache key.
    """

    name: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in input order."""
        ...


class SbertEmbedder:
    """Local CPU embeddings via the ``sentence-transformers`` library.

    Lazy import: ``sentence_transformers`` (and its ~1.5GB of torch
    deps) is only imported when this class is instantiated, so users on
    the AOAI or no-LLM path don't pay the cost. Raises a friendly
    :class:`LLMError` with the install hint if the package is missing.
    """

    name: str = "sbert"

    def __init__(self, model: str = _DEFAULT_SBERT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = (
                "sentence-transformers is not installed. Either install "
                "the optional extra (`pip install dataset-scout[local-"
                "embeddings]`), set DATASET_SCOUT_EMBEDDING_BACKEND=aoai "
                "(requires AZURE_OPENAI_EMBEDDING_DEPLOYMENT), or set "
                "DATASET_SCOUT_EMBEDDING_BACKEND=none to skip the "
                "embedding-fit stage entirely."
            )
            raise LLMError(msg) from exc

        self.model = model
        # Wrap model-load failures (offline first run, bad HF repo id,
        # hub auth, incompatible torch) in LLMError so build_embedder's
        # ``except LLMError`` cleanly degrades to "no embedder available"
        # instead of leaking a torch / urllib3 traceback to the user.
        try:
            self._st = SentenceTransformer(model)
        except Exception as exc:
            msg = (
                f"sentence-transformers failed to load model {model!r}: "
                f"{exc}. Check the HF repo id, network connectivity, or "
                "set DATASET_SCOUT_EMBEDDING_BACKEND=none to skip the "
                "embedding-fit stage."
            )
            raise LLMError(msg) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [[float(v) for v in row] for row in vecs]


class AOAIEmbedder:
    """Azure OpenAI embeddings via litellm + Entra token provider.

    Wraps the same auth path used elsewhere in the codebase. Uses the
    shared ``llm_client`` helpers so token acquisition is consistent.
    """

    name: str = "aoai"

    def __init__(self, ctx: ScoutContext, *, timeout_s: float = 30.0) -> None:
        # Gate on the *embedding* prerequisites only — endpoint plus
        # embedding deployment. Don't require ctx.aoai_deployment (the
        # CHAT deployment), because users can legitimately mix providers
        # (e.g. chat via github_copilot/, embeddings via AOAI). Requiring
        # both would silently lock those users out of AOAI embeddings.
        if not ctx.aoai_endpoint or not ctx.aoai_embedding_deployment:
            msg = (
                "AOAIEmbedder requires AZURE_OPENAI_ENDPOINT + "
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT (and Entra auth via "
                "`az login`)."
            )
            raise LLMError(msg)
        self._ctx = ctx
        self.model = ctx.aoai_embedding_deployment
        self._timeout_s = timeout_s

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Imports are inside embed() so import-time cost stays tiny and
        # tests that mock at ``dataset_scout.llm_client.import_litellm``
        # still work.
        from dataset_scout.llm_client import (
            import_litellm,
            make_token_provider,
        )

        litellm = import_litellm()
        token_provider = make_token_provider()
        response = litellm.embedding(
            model=f"azure/{self.model}",
            input=texts,
            api_base=self._ctx.aoai_endpoint,
            api_version=self._ctx.aoai_api_version,
            azure_ad_token_provider=token_provider,
            timeout=self._timeout_s,
        )
        return _extract_embeddings(response, expected=len(texts))


def _extract_embeddings(response: Any, *, expected: int) -> list[list[float]]:
    """Pull the embedding vectors from a litellm.embedding response.

    Tolerates both attribute access (``.data[i].embedding``) and dict
    access (``response["data"][i]["embedding"]``); litellm has shipped
    both shapes at various points. Raises :class:`LLMError` on shape
    surprises so callers can degrade cleanly.
    """
    try:
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        if not data or len(data) < expected:
            msg = "embedding response missing data items"
            raise LLMError(msg)
        out: list[list[float]] = []
        for item in data[:expected]:
            embedding = getattr(item, "embedding", None)
            if embedding is None and isinstance(item, dict):
                embedding = item.get("embedding")
            if not embedding:
                msg = "embedding response item missing 'embedding'"
                raise LLMError(msg)
            out.append([float(v) for v in embedding])
        return out
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        msg = f"unexpected embedding response shape: {exc}"
        raise LLMError(msg) from exc


def build_embedder(ctx: ScoutContext) -> Embedder | None:
    """Pick the embedding backend declared by ``ctx``.

    Returns None when:
      - ``embedding_backend == "none"`` (explicit opt-out), OR
      - ``embedding_backend == "aoai"`` but no AOAI embedding
        deployment is configured, OR
      - ``embedding_backend == "sbert"`` but ``sentence-transformers``
        is not installed.

    A None return cleanly skips the embedding-fit stage in
    :func:`embedding_fit.assess_label_intent_fit` — the rest of the
    pipeline still runs.
    """
    backend = ctx.embedding_backend
    if backend == "none":
        return None
    if backend == "aoai":
        # See AOAIEmbedder.__init__ — embeddings only need endpoint +
        # embedding deployment, not the chat deployment. Mixed-provider
        # setups (e.g. chat via github_copilot/, embeddings via AOAI)
        # must keep working.
        if not (ctx.aoai_endpoint and ctx.aoai_embedding_deployment):
            return None
        return AOAIEmbedder(ctx)
    if backend == "sbert":
        try:
            return SbertEmbedder(ctx.embedding_model or _DEFAULT_SBERT_MODEL)
        except LLMError:
            # sentence-transformers missing — let embedding_fit no-op.
            return None
    return None


__all__ = [
    "AOAIEmbedder",
    "Embedder",
    "SbertEmbedder",
    "build_embedder",
]
