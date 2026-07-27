"""LLM-described coverage report (M2b).

After the per-candidate strategy assessor runs, we ask the model one
final question: what aspects of the user's detection target are NOT
addressed by any candidate (even via reframing), and what concrete
next steps would close the gap?

Single litellm call. Inputs: intent, decomposition directions, and
the best (or top-2 close-confidence) strategies per shortlisted
candidate. Output: list[CoverageGap].

Provider-agnostic via `llm_client`: routes through whichever provider
``ctx.model`` (or the legacy AOAI fields) configures.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from dataset_scout.core import (
    CoverageGap,
    DecompositionDirection,
    Intent,
    Scorecard,
    Strategy,
)
from dataset_scout.errors import LLMError
from dataset_scout.llm_client import (
    build_completion_kwargs,
    effective_model_id,
    extract_content,
    import_litellm,
    make_token_provider,
)

if TYPE_CHECKING:
    from dataset_scout.cache import Cache
    from dataset_scout.context import ScoutContext


# Back-compat shim: tests monkeypatch this name to stub the Entra
# credential acquisition. Coverage no longer invokes it eagerly — it's
# now reached lazily via llm_client.resolve_llm_params only on the
# Azure branch — but the symbol stays so existing test patches don't
# break.
_make_token_provider = make_token_provider


# Bumped when prompt/response handling changes. v2 introduces
# effective-model-id keying (was ctx.aoai_deployment) so cross-provider
# runs don't pollute each other.
COVERAGE_VERSION = "2"

# Maximum candidates included in the prompt. Coverage prompt is a
# single LLM call so we cap the size of the candidate summary to keep
# context use reasonable; per-candidate detail is in the report.
_MAX_CANDIDATES_IN_PROMPT = 20

# Confidence gap below which we include the second-best strategy too —
# per duck guidance, "best-strategy-only throws away useful uncertainty
# when a candidate's second-best is still viable."
_TOP2_CLOSE_CONFIDENCE_DELTA = 0.15


class CoverageResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gaps: list[CoverageGap]


# ─── prompt rendering ───────────────────────────────────────────────


_PROMPT_TEMPLATE = """\
You are helping an AI security engineer assess whether the candidates
they've found cover their detection target adequately, or leave gaps.

USER INTENT
-----------
Brief: <<RAW_BRIEF>>
Detection target: <<DETECTION_TARGET>>
Threat families: <<THREAT_FAMILIES>>
Deployment context: <<DEPLOYMENT_CONTEXT>>

DECOMPOSITION DIRECTIONS WE EXPLORED
-------------------------------------
<<DIRECTIONS>>

CANDIDATES + BEST STRATEGY (or top-2 when close)
-------------------------------------------------
<<CANDIDATE_SUMMARY>>

YOUR TASK
---------
Identify ASPECTS of the user's detection target that NONE of the
candidates above address — even via reframing — and propose concrete
next steps. Be honest; if coverage is genuinely good, return an
empty list. Do NOT invent gaps to fill space.

Each gap should reference something specific the user said they
care about (a sub-task, a deployment surface, a language, an attack
style) that the candidate set doesn't cover. Suggestions should be
actionable: a search direction, a synthesis approach, a partner
corpus to look for.

Return JSON matching this schema:
{
  "gaps": [
    {
      "aspect": "short snake_case or hyphenated label",
      "description": "1-2 sentences on what the candidates miss",
      "suggestion": "1 sentence concrete next step"
    },
    ...
  ]
}
"""


def _none_or_csv(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _format_directions(directions: list[DecompositionDirection]) -> str:
    if not directions:
        return "(none)"
    lines: list[str] = []
    for d in directions:
        lines.append(f"- {d.name}: {d.rationale}")
    return "\n".join(lines)


def _strategies_for_coverage(scorecard: Scorecard) -> list[Strategy]:
    """Return the strongest strategy or strongest two when close."""
    if not scorecard.strategies:
        return []
    sorted_strats = sorted(scorecard.strategies, key=lambda s: s.confidence, reverse=True)
    best = sorted_strats[0]
    if len(sorted_strats) == 1:
        return [best]
    second = sorted_strats[1]
    if best.confidence - second.confidence <= _TOP2_CLOSE_CONFIDENCE_DELTA:
        return [best, second]
    return [best]


def _format_candidate_summary(scorecards: list[Scorecard]) -> str:
    if not scorecards:
        return "(none — no candidates were assessed)"
    lines: list[str] = []
    for sc in scorecards[:_MAX_CANDIDATES_IN_PROMPT]:
        cand = sc.candidate
        strats = _strategies_for_coverage(sc)
        if not strats:
            line = f"- {cand.source}:{cand.id} — (no strategy assessed)"
        else:
            parts = [f"{s.kind.value} ({s.confidence:.2f}): {s.rationale}" for s in strats]
            line = f"- {cand.source}:{cand.id} — " + "; ".join(parts)
        if cand.metadata.description:
            desc = cand.metadata.description.strip().splitlines()[0][:140]
            line += f" [card: {desc}]"
        lines.append(line)
    if len(scorecards) > _MAX_CANDIDATES_IN_PROMPT:
        lines.append(
            f"...and {len(scorecards) - _MAX_CANDIDATES_IN_PROMPT} more candidate(s) "
            "not shown here."
        )
    return "\n".join(lines)


def render_coverage_prompt(
    intent: Intent,
    directions: list[DecompositionDirection],
    scorecards: list[Scorecard],
) -> str:
    """Render the exact prompt sent to the model. No I/O."""
    return (
        _PROMPT_TEMPLATE.replace("<<RAW_BRIEF>>", intent.raw_brief or "(none)")
        .replace("<<DETECTION_TARGET>>", intent.detection_target or "(none)")
        .replace("<<THREAT_FAMILIES>>", _none_or_csv(intent.threat_families))
        .replace(
            "<<DEPLOYMENT_CONTEXT>>",
            intent.deployment_context or "(none)",
        )
        .replace("<<DIRECTIONS>>", _format_directions(directions))
        .replace("<<CANDIDATE_SUMMARY>>", _format_candidate_summary(scorecards))
    )


# ─── coverage call ──────────────────────────────────────────────────


def _parse_response(content: str) -> CoverageResponse:
    payload = json.loads(content)
    return CoverageResponse.model_validate(payload)


def build_coverage_report(
    intent: Intent,
    directions: list[DecompositionDirection],
    scorecards: list[Scorecard],
    *,
    ctx: ScoutContext,
    timeout_s: float = 30.0,
    cache: Cache | None = None,
) -> list[CoverageGap]:
    """Ask the LLM to identify coverage gaps in the candidate set.

    Single completion call, one retry on Pydantic validation failure.
    Empty list returned cleanly when coverage is honestly good or when
    no scorecards were assessed.

    When ``cache`` is provided, identical (rendered prompt,
    COVERAGE_VERSION, effective-model-id) inputs return without an LLM
    call — useful when the user is iterating on report rendering.
    """
    if not ctx.llm_configured:
        raise LLMError(
            "No LLM provider configured. Set DATASET_SCOUT_MODEL "
            "(e.g. 'github_copilot/gpt-5-mini' or 'github/gpt-4o-mini'), "
            "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT (and run "
            "`az login` for Entra auth)."
        )

    prompt = render_coverage_prompt(intent, directions, scorecards)

    cache_key: str | None = None
    if cache is not None:
        resolved = effective_model_id(ctx) or ""
        cache_key = hashlib.sha256(
            (COVERAGE_VERSION + "\n" + resolved + "\n" + prompt).encode("utf-8")
        ).hexdigest()
        cached = cache.get_json("coverage", cache_key)
        if cached is not None:
            try:
                payload = CoverageResponse.model_validate(cached)
            except ValidationError:
                pass
            else:
                return list(payload.gaps)

    litellm = import_litellm()
    completion_kwargs = build_completion_kwargs(
        ctx,
        messages=[{"role": "user", "content": prompt}],
        response_format=CoverageResponse,
        timeout_s=timeout_s,
    )

    last_parse_error: Exception | None = None
    parsed: CoverageResponse | None = None
    for _attempt in range(2):
        try:
            response = litellm.completion(**completion_kwargs)
        except Exception as exc:
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
        cache.set_json("coverage", cache_key, parsed.model_dump(mode="json"))

    return list(parsed.gaps)
