"""LLM-driven brief decomposition.

Given an `Intent`, ask the configured LLM provider for 3-7 related
search directions adjacent to the user's stated target. The pipeline
uses this output to widen the search net before scoring; reframing
(strategy assessment) lives separately in M2b.

Auth model: provider-agnostic. The default path is Azure OpenAI +
Entra (via `DefaultAzureCredential`), but any litellm-supported model
id works — set ``ctx.model`` (env: ``DATASET_SCOUT_MODEL``) to e.g.
``github_copilot/gpt-5-mini``, ``github/gpt-4o-mini``,
``openai/gpt-4o``, etc. Provider auth is delegated to litellm
(OAuth device-code flow for github_copilot, env vars for others).

Network-free at import time. Heavy imports (`litellm`, `azure-identity`)
are deferred into call sites so unit tests don't pay for them and the
metadata-only path stays fast.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dataset_scout.core import DecompositionDirection, Intent
from dataset_scout.errors import LLMError
from dataset_scout.llm_client import (
    build_completion_kwargs,
    effective_model_id,
    extract_content,
    import_litellm,
)

if TYPE_CHECKING:
    from dataset_scout.cache import Cache
    from dataset_scout.context import ScoutContext

# Bumped when prompt or response handling changes in a way that would
# invalidate cached decomposition results. v4 introduced effective-
# model-id keying (was ctx.aoai_deployment) to prevent cross-provider
# cache pollution.
DECOMPOSE_VERSION = "4"

# Hard upper bound on directions returned (mirrors the prompt).
_MAX_DIRECTIONS = 7


class DecomposeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Step-1 output of the two-step prompt. Stored on the response so
    # callers (review flow, report rendering) can show users which
    # research communities the model considered. Optional so older
    # cached responses still parse.
    adjacent_disciplines: list[str] = Field(default_factory=list)
    directions: list[DecompositionDirection]


# ─── prompt rendering ───────────────────────────────────────────────


# The literal JSON-schema block contains `{` and `}`, which precludes
# str.format / f-strings. Use a sentinel-replacement instead so the
# template stays readable inline. Sentinels are unambiguous strings
# we know won't appear in the surrounding prose.
_PROMPT_TEMPLATE = """\
You are helping an AI security engineer find public datasets related
to their detection target. Their stated brief is narrow; we want to
widen the search net so reframings of related work (proxy positives,
hard negatives, benign baselines, subset extractions, label remappings)
become discoverable.

Brief: <<RAW_BRIEF>>
Detection target: <<DETECTION_TARGET>>
Threat families: <<THREAT_FAMILIES>>
Deployment context: <<DEPLOYMENT_CONTEXT>>

This is a TWO-STEP task. Step 1 forces breadth across communities;
Step 2 grounds each direction in one of them so the search query
draws from a real data ecosystem rather than abstract ML vocabulary.

STEP 1 — ADJACENT DISCIPLINES
============================
Before listing any directions, identify 5-8 RESEARCH COMMUNITIES,
PROFESSIONAL FIELDS, or DATA ECOSYSTEMS adjacent to this brief that
have already accumulated relevant data. These should span distinct
parent fields, not multiple framings of the same one.

What makes a good entry:
  - A specific community or ecosystem ("clinical-psychology research"
    not "psychology"; "trust-and-safety policy teams" not "safety").
  - At least 3 of them must come from DIFFERENT parent fields
    (e.g., one from clinical sciences, one from social sciences,
    one from CS/NLP). Don't list 5 sub-areas of NLP.
  - Each one must plausibly have published, released, or indexed
    datasets a researcher could find.

Examples — for an arbitrary brief about content moderation:
  - "trust-and-safety operations teams (released moderation logs)"
  - "academic NLP toxicity-detection research"
  - "platform-policy analysts and digital-rights orgs"
  - "social-psychology studies of online harassment"
  - "linguistics-of-deception researchers"

Examples — for an arbitrary brief about hallucination detection:
  - "fact-checking organizations"
  - "NLP question-answering benchmark builders"
  - "knowledge-base / linked-data communities"
  - "journalism-research labs"
  - "argument-mining / discourse-analysis academics"

Be SPECIFIC. "AI research" alone is too broad. Each entry should
hint at the *kind of data* that community produces.

STEP 2 — DIRECTIONS GROUNDED IN DISCIPLINES
============================================
Now propose 3-7 search directions. Each direction MUST cite which
discipline from Step 1 it taps into. This forces breadth: you cannot
list 5 directions that all draw from the same parent field.

Each direction's `keywords` should be SHORT, LEXICAL search terms a
substring keyword search engine would match. HF's search is literal
substring matching — short 2-3 word phrases ("prompt injection",
"jailbreak prompts") work well; long phrases ("Python AttributeError
missing method definition") almost never hit. Avoid full sentences.

Where the keyword would be ambiguous on its own (e.g. "wrong answer",
which would match every MCQA dataset on the Hub), prefer a
domain-anchored variant ("code wrong answer", "API misuse").

Return JSON matching this schema:
{
  "adjacent_disciplines": [
    "<discipline 1>",
    "<discipline 2>",
    ...
  ],
  "directions": [
    {
      "name": "snake_case_short_name",
      "rationale": "1-2 sentences. MUST start with: 'Drawing on <one of the disciplines from Step 1>:' so the grounding is visible.",
      "keywords": ["term", "term", "term"],
      "threat_families": ["family", ...],
      "expected_finds": "1 sentence on what useful data we'd hope to find"
    },
    ...
  ]
}

Constraints:
  - 5-8 adjacent disciplines. Pick from at least 3 different parent fields.
  - 3-7 directions. Each direction's rationale starts with
    "Drawing on <discipline>:" and that discipline must appear in
    the adjacent_disciplines list.
  - At least 3 of the directions must draw on DIFFERENT disciplines
    (don't ground 5 directions in the same one).
  - Be conservative-but-creative. Do NOT include directions you
    cannot defend.
"""


def _none_or_csv(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def render_decompose_prompt(intent: Intent) -> str:
    """Render the exact prompt sent to the model. No I/O.

    Empty / None fields render as ``(none)`` so the prompt stays
    well-formed and snapshot-stable across briefs.
    """
    return (
        _PROMPT_TEMPLATE.replace("<<RAW_BRIEF>>", intent.raw_brief or "(none)")
        .replace("<<DETECTION_TARGET>>", intent.detection_target or "(none)")
        .replace("<<THREAT_FAMILIES>>", _none_or_csv(intent.threat_families))
        .replace(
            "<<DEPLOYMENT_CONTEXT>>",
            intent.deployment_context or "(none)",
        )
    )


# ─── capability detection ───────────────────────────────────────────


def llm_available(ctx: ScoutContext) -> bool:
    """Cheap, no-network probe: is *some* LLM provider configured?

    True when either ``ctx.model`` is set (any litellm provider —
    ``github_copilot/...``, ``github/...``, ``openai/...``,
    ``anthropic/...``, ``azure/...``) or the legacy AOAI endpoint +
    deployment fields are set. Provider-specific credential
    availability (``az login`` having been run, GITHUB_TOKEN exported,
    GitHub Copilot device-code flow completed) is NOT checked here —
    that requires a network round-trip we don't want to pay on every
    recon. If the credential can't be acquired the `decompose_intent`
    call will fail with `LLMError` and the pipeline will fall back to
    metadata-only mode.

    Importantly, this function does NOT import `litellm` (~10s) or
    `azure-identity`. Users without an LLM configured pay nothing.
    """
    return ctx.llm_configured


# ─── decomposition ──────────────────────────────────────────────────


def _parse_response(content: str) -> DecomposeResponse:
    """Parse and validate. Raises ValidationError on schema mismatch,
    json.JSONDecodeError on malformed JSON; both treated as retryable.
    """
    payload = json.loads(content)
    return DecomposeResponse.model_validate(payload)


def decompose_intent(
    intent: Intent,
    *,
    ctx: ScoutContext,
    timeout_s: float = 60.0,
    cache: Cache | None = None,
    model: str | None = None,
) -> list[DecompositionDirection]:
    """Ask the configured LLM provider for 3-7 related search directions.

    Single completion call; one retry on Pydantic validation failure
    using the same prompt. Any other failure (network, missing provider
    config, repeated validation failure) raises `LLMError` so the
    pipeline can fall back to metadata-only mode.

    Result is clipped at 7 directions; an empty list is returned
    cleanly when the model honestly reports no useful adjacencies.

    `model` overrides ``ctx.model`` (and the legacy AOAI synthesis) for
    this call only — useful for the ``--model`` CLI flag. When None,
    falls back to ``ctx.model`` then to ``azure/<aoai_deployment>``.

    When `cache` is provided, identical (prompt, DECOMPOSE_VERSION,
    effective-model-id) inputs return without an LLM call. Cache hits
    skip the litellm import entirely. The cache key includes the
    effective model id so switching backends (Azure ↔ GitHub Copilot)
    does not serve stale cross-provider responses.
    """
    resolved_model = effective_model_id(ctx, model)
    if resolved_model is None:
        raise LLMError(
            "No LLM provider configured. Set DATASET_SCOUT_MODEL "
            "(e.g. 'github_copilot/gpt-5-mini' or 'github/gpt-4o-mini'), "
            "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT (and run "
            "`az login` for Entra auth)."
        )

    prompt = render_decompose_prompt(intent)

    # Cache check before any heavy import. Key on the rendered prompt
    # plus the version sentinel plus the resolved model id — that
    # captures intent + template + provider/model in one place, and
    # prevents stale Azure entries being served on a github_copilot/
    # run (or vice versa).
    cache_key: str | None = None
    if cache is not None:
        cache_key = hashlib.sha256(
            (DECOMPOSE_VERSION + "\n" + resolved_model + "\n" + prompt).encode("utf-8")
        ).hexdigest()
        cached = cache.get_json("decompose", cache_key)
        if cached is not None:
            try:
                payload = DecomposeResponse.model_validate(cached)
            except ValidationError:
                # Cache was written by an older/incompatible version —
                # treat as a miss.
                pass
            else:
                return list(payload.directions[:_MAX_DIRECTIONS])

    litellm = import_litellm()

    messages = [{"role": "user", "content": prompt}]
    completion_kwargs = build_completion_kwargs(
        ctx,
        messages=messages,
        response_format=DecomposeResponse,
        timeout_s=timeout_s,
        model=resolved_model,
    )

    last_parse_error: Exception | None = None
    parsed: DecomposeResponse | None = None
    for attempt in range(2):
        try:
            response = litellm.completion(**completion_kwargs)
        except Exception as exc:
            # Retry once on timeout-class errors (some providers under
            # load can take 30-60s for some prompts). Other errors fail
            # fast.
            err_text = str(exc).lower()
            if attempt == 0 and ("timeout" in err_text or "timed out" in err_text):
                continue
            raise LLMError(f"LLM call failed: {exc}") from exc

        content = extract_content(response)
        try:
            parsed = _parse_response(content)
            break
        except (ValidationError, json.JSONDecodeError) as exc:
            last_parse_error = exc
            parsed = None
            continue

    if parsed is None:
        msg = f"LLM returned invalid JSON twice: {last_parse_error}"
        raise LLMError(msg) from last_parse_error

    if cache is not None and cache_key is not None:
        cache.set_json("decompose", cache_key, parsed.model_dump(mode="json"))

    return list(parsed.directions[:_MAX_DIRECTIONS])
