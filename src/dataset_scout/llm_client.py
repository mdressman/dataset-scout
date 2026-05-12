"""Shared LLM client plumbing — provider-agnostic.

dataset-scout's LLM call sites (decompose, strategy, coverage, judge,
keyword expansion, embedding fit) all route through this module. We
centralize three things:

1. **Provider dispatch** (`resolve_llm_params`): given a litellm-style
   model id like ``azure/<deployment>``, ``github_copilot/gpt-5-mini``,
   ``github/gpt-4o-mini``, ``openai/gpt-4o``, ``anthropic/claude-...``,
   return the right kwargs dict to hand to ``litellm.completion``.

2. **Effective-model resolution** (`effective_model_id`): centralizes
   the precedence (per-call override > ``ctx.model`` > synthesized
   ``azure/<deployment>``) so call sites and cache keys agree.

3. **Lazy heavy imports** (`import_litellm`, `make_token_provider`):
   keep ``import dataset_scout`` cheap; the LLM stack is only paid for
   when an LLM call actually runs.

Auth model per provider:

- ``azure/<deployment>`` — Entra. We attach ``azure_ad_token_provider``
  built from ``DefaultAzureCredential`` (chains ``az login``, managed
  identity, env-var SPN, etc.).
- ``github_copilot/<model>`` — litellm runs an OAuth device-code flow
  on first use, caches the token under
  ``~/.config/litellm/github_copilot/`` (override via
  ``GITHUB_COPILOT_TOKEN_DIR``), and refreshes silently. **No env var
  required.**
- ``github/<model>`` — litellm reads ``GITHUB_API_KEY`` or
  ``GITHUB_TOKEN`` from the environment (free GitHub Models tier).
- ``openai/<model>`` — litellm reads ``OPENAI_API_KEY``.
- ``anthropic/<model>`` — litellm reads ``ANTHROPIC_API_KEY``.

For non-Azure providers we deliberately return *just* ``{"model": ...}``
plus the message/timeout/response_format and let litellm do everything
else; mirroring the proven pattern in ml-intern's
``agent/core/llm_params.py:_resolve_llm_params``.

Network-free at import time. Heavy imports (``litellm``,
``azure-identity``) are deferred into the function bodies so the
metadata-only path stays fast and unit tests don't pay for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataset_scout.errors import LLMError

if TYPE_CHECKING:
    from dataset_scout.context import ScoutContext


# Entra token scope for Azure Cognitive Services (covers AOAI).
AOAI_SCOPE = "https://cognitiveservices.azure.com/.default"

# Provider-prefix sentinel used by `resolve_llm_params` to decide
# whether to attach the Entra token provider. Other litellm prefixes
# (github_copilot/, github/, openai/, anthropic/, bedrock/, ...) are
# passed through unmodified — litellm's per-provider handlers know how
# to authenticate themselves (env vars or device-code flow).
_AZURE_PREFIX = "azure/"


def make_token_provider() -> Any:
    """Build a fresh bearer-token provider via `DefaultAzureCredential`.

    Only used by the ``azure/`` branch of `resolve_llm_params`. The
    credential chain checks: env-var service-principal,
    workload-identity, managed identity, shared-token-cache, Azure CLI
    (`az login`), and interactive browser. Tokens are cached internally
    so we don't add an extra app-level cache.

    Tests stub this whole function — see decompose / strategy test
    files for the `fake_token_provider` fixture pattern.
    """
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except Exception as exc:
        raise LLMError("azure-identity is required for AOAI Entra auth: " + str(exc)) from exc
    return get_bearer_token_provider(DefaultAzureCredential(), AOAI_SCOPE)


def effective_model_id(ctx: ScoutContext, override: str | None = None) -> str | None:
    """Resolve the model id a call should use.

    Precedence:
        1. ``override`` (e.g. CLI flag passed to a single command)
        2. ``ctx.model`` (env var or programmatic context setting)
        3. ``"azure/<aoai_deployment>"`` synthesized from legacy AOAI
           config (back-compat: existing users with only AZURE_OPENAI_*
           env vars set keep working unchanged)
        4. ``None`` (no provider configured — caller should degrade)

    Centralizing this means every cache key and every call site agrees
    on what was actually executed against, which avoids cross-provider
    cache pollution.
    """
    if override:
        return override
    if ctx.model:
        return ctx.model
    if ctx.aoai_endpoint and ctx.aoai_deployment:
        return f"{_AZURE_PREFIX}{ctx.aoai_deployment}"
    return None


def resolve_llm_params(
    model: str,
    *,
    ctx: ScoutContext,
    messages: list[dict[str, str]],
    response_format: type | None = None,
    timeout_s: float = 30.0,
    token_provider: Any | None = None,
) -> dict[str, Any]:
    """Build the litellm.completion(**kwargs) for `model`, dispatching by prefix.

    Pattern lifted from ml-intern's ``agent/core/llm_params.py``
    ``_resolve_llm_params``. Each branch returns the *minimal* kwargs
    set litellm needs for that provider; we deliberately do NOT pass
    ``api_base`` / ``api_key`` for non-Azure providers because litellm's
    per-provider handlers know the right defaults and reading env vars
    they own (``GITHUB_API_KEY``, ``OPENAI_API_KEY``, …) is *their* job.

    `response_format`: when not None, we send the JSON-mode hint
    ``{"type": "json_object"}`` rather than the Pydantic class itself.
    All providers we route to support this; Azure OpenAI's strict-schema
    validator rejects some Pydantic-generated schemas (e.g. with
    ``dict[str, Literal[...]]``), and the GitHub Copilot proxy is more
    permissive but still happiest in plain json_object mode. Callers
    are expected to post-parse with ``MyModel.model_validate(...)`` and
    retry on ``ValidationError``.
    """
    base: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout_s,
    }
    if response_format is not None:
        base["response_format"] = {"type": "json_object"}

    if model.startswith(_AZURE_PREFIX):
        if not ctx.aoai_endpoint:
            raise LLMError(
                "Model id starts with 'azure/' but ctx.aoai_endpoint is "
                "unset. Set AZURE_OPENAI_ENDPOINT or use a different "
                "model id (e.g. github_copilot/<model>)."
            )
        provider = token_provider if token_provider is not None else make_token_provider()
        base["api_base"] = ctx.aoai_endpoint
        base["api_version"] = ctx.aoai_api_version
        base["azure_ad_token_provider"] = provider
        return base

    # github_copilot/, github/, openai/, anthropic/, and any other
    # litellm-supported prefix all follow the same shape: just hand the
    # model id to litellm and let it do its thing. We don't fabricate
    # api_base or api_key — litellm's own provider handlers read the
    # right env vars (or run OAuth device flow for github_copilot).
    return base


def build_completion_kwargs(
    ctx: ScoutContext,
    *,
    messages: list[dict[str, str]],
    response_format: type | None = None,
    timeout_s: float = 30.0,
    token_provider: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Construct the litellm.completion(**kwargs) for the configured provider.

    Resolves the effective model via `effective_model_id` (per-call
    `model` override > ``ctx.model`` > synthesized ``azure/<deployment>``),
    then dispatches via `resolve_llm_params`.

    Raises `LLMError` if no provider is configured (no model override,
    no ``ctx.model``, no AOAI config). The caller should check
    ``ctx.llm_configured`` before reaching this if it wants to degrade
    gracefully instead.
    """
    resolved = effective_model_id(ctx, model)
    if resolved is None:
        raise LLMError(
            "No LLM provider configured. Set DATASET_SCOUT_MODEL "
            "(e.g. 'github_copilot/gpt-5-mini' or 'github/gpt-4o-mini'), "
            "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT (and run "
            "`az login` for Entra auth)."
        )
    return resolve_llm_params(
        resolved,
        ctx=ctx,
        messages=messages,
        response_format=response_format,
        timeout_s=timeout_s,
        token_provider=token_provider,
    )


def import_litellm() -> Any:
    """Import `litellm` lazily, translating import failures into LLMError.

    Also sets `suppress_debug_info = True` on first import so framework
    decoration around exceptions doesn't leak to user stderr.
    """
    try:
        import litellm
    except Exception as exc:
        raise LLMError(f"litellm not importable: {exc}") from exc
    if hasattr(litellm, "suppress_debug_info"):
        litellm.suppress_debug_info = True
    return litellm


def extract_content(response: Any) -> str:
    """Pull the JSON string from a litellm completion response.

    OpenAI-style: `response.choices[0].message.content`. We avoid
    importing litellm response types so the module stays light and
    test mocks can be plain objects.
    """
    try:
        choice = response.choices[0]
        message = choice.message
        content = message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected LLM response shape: {exc}") from exc
    if not isinstance(content, str):
        raise LLMError("LLM response message.content was not a string")
    return content


__all__ = [
    "AOAI_SCOPE",
    "build_completion_kwargs",
    "effective_model_id",
    "extract_content",
    "import_litellm",
    "make_token_provider",
    "resolve_llm_params",
]
