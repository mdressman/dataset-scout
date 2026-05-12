"""ScoutContext behavior tests."""

from __future__ import annotations

import pytest

from dataset_scout import ScoutContext

pytestmark = pytest.mark.unit


def test_default_context_has_expected_sources():
    ctx = ScoutContext.from_env(env={})
    names = list(ctx.sources)
    assert "huggingface" in names
    assert "kaggle" in names
    # Kaggle is enabled-by-default; the factory quietly skips it when
    # no creds are present, so the user-visible behaviour is "if you
    # configure Kaggle, it just works".
    assert ctx.sources["kaggle"].enabled is True
    assert ctx.sources["huggingface"].enabled is True


def test_enabled_sources_filters():
    ctx = ScoutContext.from_env(env={})
    enabled = list(ctx.enabled_sources())
    assert "huggingface" in enabled
    assert "kaggle" in enabled


def test_from_env_picks_up_api_keys():
    """Source-specific tokens are captured; LLM auth is not API-key based."""
    ctx = ScoutContext.from_env(
        env={
            "HUGGINGFACE_HUB_TOKEN": "hf_test",
            "KAGGLE_KEY": "kg_test",
            "UNRELATED_VAR": "ignored",
        }
    )
    assert ctx.api_keys.get("HUGGINGFACE_HUB_TOKEN") == "hf_test"
    assert ctx.api_keys.get("KAGGLE_KEY") == "kg_test"
    assert "UNRELATED_VAR" not in ctx.api_keys
    # OPENAI/ANTHROPIC keys are no longer captured — Entra is the LLM auth.
    assert "OPENAI_API_KEY" not in ctx.api_keys


def test_from_env_overrides_paths(tmp_path):
    ctx = ScoutContext.from_env(
        env={
            "DATASET_SCOUT_CACHE_DIR": str(tmp_path / "c"),
            "DATASET_SCOUT_OUT_DIR": str(tmp_path / "o"),
        }
    )
    assert ctx.cache_dir == tmp_path / "c"
    assert ctx.out_dir == tmp_path / "o"


def test_from_env_picks_up_aoai_config():
    ctx = ScoutContext.from_env(
        env={
            "AZURE_OPENAI_ENDPOINT": "https://my-aoai.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT": "gpt-4o-mini",
            "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
        }
    )
    # Trailing slash trimmed for consistency with litellm's api_base.
    assert ctx.aoai_endpoint == "https://my-aoai.openai.azure.com"
    assert ctx.aoai_deployment == "gpt-4o-mini"
    assert ctx.aoai_api_version == "2024-08-01-preview"
    assert ctx.aoai_configured is True
    # Universal LLM signal also true via the AOAI back-compat path.
    assert ctx.llm_configured is True


def test_aoai_configured_requires_both_fields():
    assert ScoutContext().aoai_configured is False
    assert ScoutContext(aoai_endpoint="https://x.openai.azure.com").aoai_configured is False
    assert ScoutContext(aoai_deployment="gpt-4o-mini").aoai_configured is False
    assert (
        ScoutContext(
            aoai_endpoint="https://x.openai.azure.com",
            aoai_deployment="gpt-4o-mini",
        ).aoai_configured
        is True
    )


# ─── universal model field + llm_configured ─────────────────────────


def test_from_env_picks_up_universal_model():
    ctx = ScoutContext.from_env(env={"DATASET_SCOUT_MODEL": "github_copilot/gpt-5-mini"})
    assert ctx.model == "github_copilot/gpt-5-mini"
    # LLM is configured via the universal path even though no AOAI
    # endpoint / deployment exist.
    assert ctx.llm_configured is True
    # ...but the AOAI-specific check stays False.
    assert ctx.aoai_configured is False


def test_llm_configured_false_when_nothing_set():
    assert ScoutContext().llm_configured is False


def test_llm_configured_true_when_only_model_set():
    """github_copilot / github / openai / anthropic users don't need
    any AOAI config — setting ctx.model alone must satisfy the
    'is some LLM provider configured?' check."""
    assert ScoutContext(model="github_copilot/gpt-5-mini").llm_configured is True
    assert ScoutContext(model="github/gpt-4o-mini").llm_configured is True
    assert ScoutContext(model="openai/gpt-4o").llm_configured is True
    assert ScoutContext(model="anthropic/claude-sonnet-4-5").llm_configured is True


def test_llm_configured_true_via_aoai_back_compat():
    """Existing users with only AZURE_OPENAI_* set must keep being
    detected as configured — no breakage on upgrade."""
    ctx = ScoutContext(
        aoai_endpoint="https://x.openai.azure.com",
        aoai_deployment="gpt-4o-mini",
    )
    assert ctx.llm_configured is True
    assert ctx.model is None  # not set explicitly


def test_context_is_frozen():
    import pydantic

    ctx = ScoutContext.from_env(env={})
    with pytest.raises(pydantic.ValidationError):
        ctx.aoai_endpoint = "https://other.openai.azure.com"  # type: ignore[misc]
