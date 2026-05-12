"""ScoutContext — explicit configuration & auth carrier.

Every public library entry point (`recon`, `inspect`, `curate`) takes a
`ScoutContext`. The CLI populates it from environment variables and the
TOML config file; a future HTTP server would populate it from request
context.

This keeps the library free of global state so it stays trivially
re-entrant and trivially wrappable behind an API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable


def _default_cache_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "dataset-scout" / "cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "dataset-scout"
    return Path.home() / ".cache" / "dataset-scout"


def _default_config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "dataset-scout"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "dataset-scout"
    return Path.home() / ".config" / "dataset-scout"


class SourceConfig(BaseModel):
    """Per-source enablement and auth posture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True


class ScoutContext(BaseModel):
    """Explicit runtime context. No global state.

    Construct via `ScoutContext.from_env()` for normal use, or build one
    directly in tests / API callers.

    LLM auth supports two postures:

    1. **Azure OpenAI + Entra** (legacy default): set `aoai_endpoint`
       and `aoai_deployment`; bearer tokens come from
       `DefaultAzureCredential`. No API key stored.
    2. **Universal model id** (recommended for non-Azure backends):
       set `model` to a litellm-style id like `github_copilot/gpt-5-mini`,
       `github/gpt-4o-mini`, `openai/gpt-4o-mini`, or
       `anthropic/claude-sonnet-4-5`. litellm handles the per-provider
       auth flow (e.g. OAuth device-code for `github_copilot/`,
       `GITHUB_TOKEN` env var for `github/`).

    When both are set, `model` wins. When neither is set, LLM stages
    degrade to no-ops (metadata-only mode) per `llm_configured`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    config_dir: Path = Field(default_factory=_default_config_dir)
    out_dir: Path = Field(default_factory=lambda: Path("datascout-out"))

    sources: Mapping[str, SourceConfig] = Field(
        default_factory=lambda: {
            "huggingface": SourceConfig(enabled=True),
            "kaggle": SourceConfig(enabled=True),  # quietly skipped without creds
            # "pwc": SourceConfig(enabled=False),  # not yet wired
        }
    )

    # Source-specific tokens (HF, Kaggle). LLM auth is provider-specific
    # and lives in `model` (universal) or the AOAI fields (legacy).
    api_keys: Mapping[str, str] = Field(default_factory=dict)

    # ─── Universal LLM model id ──────────────────────────────────
    # litellm-style: "github_copilot/gpt-5-mini", "github/gpt-4o-mini",
    # "openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5", or
    # "azure/<deployment>". When None, falls back to synthesized
    # "azure/<aoai_deployment>" if AOAI is configured.
    model: str | None = None

    # ─── Azure OpenAI (legacy / Entra-auth path) ─────────────────
    aoai_endpoint: str | None = None
    aoai_deployment: str | None = None
    aoai_api_version: str = "2024-10-21"
    # Optional: an embeddings deployment name (e.g. text-embedding-3-small).
    # When unset, the embedding-fit pipeline stage no-ops gracefully.
    aoai_embedding_deployment: str | None = None

    # ─── Embeddings (label_intent_fit stage) ─────────────────────
    # "sbert"  → local CPU sentence-transformers (default; needs the
    #            `dataset-scout[local-embeddings]` extra installed).
    # "aoai"   → Azure OpenAI embeddings via aoai_embedding_deployment.
    # "none"   → skip the embedding-fit stage entirely (still no-ops if
    #            the chosen backend isn't actually available).
    # The factory build_embedder() honors this AND returns None if the
    # backend isn't actually available — so misconfiguration just
    # degrades cleanly to skipping the stage.
    embedding_backend: Literal["aoai", "sbert", "none"] = "sbert"
    # Override the model name for the chosen backend. For sbert this is
    # an HF repo id; for aoai it falls back to aoai_embedding_deployment
    # when None.
    embedding_model: str | None = None

    is_tty: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        is_tty: bool | None = None,
    ) -> Self:
        """Build a ScoutContext from environment variables.

        Recognized variables:
            DATASET_SCOUT_CACHE_DIR
            DATASET_SCOUT_CONFIG_DIR
            DATASET_SCOUT_OUT_DIR
            HUGGINGFACE_HUB_TOKEN / HF_TOKEN
            KAGGLE_USERNAME / KAGGLE_KEY

            DATASET_SCOUT_MODEL         — universal litellm-style model id,
                                          e.g. "github_copilot/gpt-5-mini",
                                          "github/gpt-4o-mini", "openai/gpt-4o".
                                          Wins over the AOAI fields when set.

            DATASET_SCOUT_EMBEDDING_BACKEND  — "sbert" (default), "aoai",
                                          or "none".
            DATASET_SCOUT_EMBEDDING_MODEL    — override the model name
                                          for the chosen embedding
                                          backend. For sbert: an HF
                                          repo id (default
                                          "sentence-transformers/all-
                                          MiniLM-L6-v2").

            AZURE_OPENAI_ENDPOINT       — e.g. https://my-aoai.openai.azure.com/
            AZURE_OPENAI_DEPLOYMENT     — deployment name (e.g. gpt-4o-mini)
            AZURE_OPENAI_API_VERSION    — overrides default
        """
        e = env if env is not None else os.environ
        api_keys: dict[str, str] = {}
        for k in (
            "HUGGINGFACE_HUB_TOKEN",
            "HF_TOKEN",
            "KAGGLE_USERNAME",
            "KAGGLE_KEY",
        ):
            if v := e.get(k):
                api_keys[k] = v

        kwargs: dict[str, object] = {"api_keys": api_keys}
        if v := e.get("DATASET_SCOUT_CACHE_DIR"):
            kwargs["cache_dir"] = Path(v)
        if v := e.get("DATASET_SCOUT_CONFIG_DIR"):
            kwargs["config_dir"] = Path(v)
        if v := e.get("DATASET_SCOUT_OUT_DIR"):
            kwargs["out_dir"] = Path(v)
        if v := e.get("DATASET_SCOUT_MODEL"):
            kwargs["model"] = v
        if v := e.get("DATASET_SCOUT_EMBEDDING_BACKEND"):
            normalized = v.strip().lower()
            if normalized in {"sbert", "aoai", "none"}:
                kwargs["embedding_backend"] = normalized
        if v := e.get("DATASET_SCOUT_EMBEDDING_MODEL"):
            kwargs["embedding_model"] = v
        if v := e.get("AZURE_OPENAI_ENDPOINT"):
            kwargs["aoai_endpoint"] = v.rstrip("/")
        if v := e.get("AZURE_OPENAI_DEPLOYMENT"):
            kwargs["aoai_deployment"] = v
        if v := e.get("AZURE_OPENAI_API_VERSION"):
            kwargs["aoai_api_version"] = v
        if v := e.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
            kwargs["aoai_embedding_deployment"] = v
        if is_tty is not None:
            kwargs["is_tty"] = is_tty

        return cls(**kwargs)

    def enabled_sources(self) -> Iterable[str]:
        """Names of enabled sources, in stable order."""
        return [name for name, cfg in self.sources.items() if cfg.enabled]

    @property
    def aoai_configured(self) -> bool:
        """True iff Azure-specific endpoint + deployment are present.

        Bearer-token availability (e.g., `az login` having been run) is
        checked lazily by the LLM call site — it requires a network
        round-trip we don't want to pay here.

        This property only reflects the legacy Azure path. For "is *any*
        LLM provider configured?", use `llm_configured` instead.
        """
        return bool(self.aoai_endpoint) and bool(self.aoai_deployment)

    @property
    def llm_configured(self) -> bool:
        """True iff *some* LLM provider is configured.

        True when either:
          - `model` is set (any litellm-supported provider), or
          - the legacy Azure fields are set.

        Provider-specific credential availability (Entra token, GitHub
        device code, GITHUB_TOKEN, OPENAI_API_KEY) is checked lazily at
        call time — this is a no-network probe.
        """
        return bool(self.model) or self.aoai_configured
