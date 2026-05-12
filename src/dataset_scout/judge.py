"""LLM-as-judge label rescue.

Promotes per-row labels from "unknown / weakly inferred" to "judged with
stated confidence" by calling the configured LLM provider per row,
parsing a strict JSON verdict into a :class:`JudgeBlock`, and applying
an explicit-gap promotion rule: only ``positive`` / ``negative`` verdicts
at-or-above the configured threshold rewrite the row's label.

Provider-agnostic via ``llm_client``: routes through whichever provider
``ctx.model`` (or the legacy AOAI fields) configures. The CLI verb is
a thin wrapper over :func:`run_judge`.

Network-free at import time. Heavy imports (``litellm``, ``azure-identity``)
are deferred into the chat-client call site so unit tests never pay for
them and the test suite can inject a fake client via the ``chat_client``
parameter on :func:`run_judge`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import statistics
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dataset_scout.core import JudgeBlock, LabelKind, NormalizedRecord
from dataset_scout.errors import DatasetScoutError, LLMError
from dataset_scout.llm_client import (
    build_completion_kwargs,
    effective_model_id,
    extract_content,
    import_litellm,
)

if TYPE_CHECKING:
    from dataset_scout.context import ScoutContext

# Bumped only when scout's prompt template (or response post-processing)
# changes in a way that would invalidate cached verdicts. Scout-internal;
# never coordinated with any other tool. v2 introduces effective-model-id
# keying via _resolve_model_name (was azure-openai/<deployment>) so
# cross-provider runs don't pollute each other.
JUDGE_TEMPLATE_VERSION = "2"

# Per-batch checkpoint cadence (rows per checkpoint flush). The figure
# is a tradeoff between resumability granularity and IO churn.
DEFAULT_BATCH_SIZE = 50

# Default per-row cap on judge JSON output. Plenty of room for a
# rationale; bounds runaway responses.
_MAX_TOKENS = 600

_log = logging.getLogger(__name__)

# Verdict label → corpus label. ``negative`` means "this row does NOT
# satisfy the axis question"; we promote it to ``benign``. Scout's
# ``hard_negative`` label is reserved for recon-side strategy outputs
# and is not produced by the judge.
_VERDICT_TO_LABEL: dict[str, Literal["positive", "benign", "hard_negative"]] = {
    "positive": "positive",
    "negative": "benign",
}


# ─── prompt rendering ───────────────────────────────────────────────


_PROMPT_TEMPLATE = """\
You are an LLM judge labeling rows of a public dataset for an AI
security engineer. Answer ONE labeling question (the "axis") about
the row's text, with a strict JSON verdict.

AXIS
----
<<AXIS>>

RUBRIC
------
<<RUBRIC>>

ROW TEXT
--------
<<TEXT>>

INSTRUCTIONS
------------
1. Decide whether the row's text is, with respect to the axis above,
   POSITIVE (clearly matches), NEGATIVE (clearly does not), or
   AMBIGUOUS (genuinely unclear or partial). Be honest — engineers
   prefer an explicit "ambiguous" to a confidently-wrong call.
2. Pick a SHORT kebab-case subcategory describing the row's flavor
   under the axis (e.g. "self-harm-ideation", "benign-emotion-talk",
   "borderline-distress"). Use a stable vocabulary.
3. Self-rate your CONFIDENCE in the verdict on a 0.0-1.0 scale.
   0.9+ = "I would bet on this"; 0.7-0.9 = "fairly sure"; below 0.7
   = "calling it but uncertain".
4. Write a ONE-SENTENCE rationale.
5. Respond with STRICT JSON ONLY — no prose, no markdown fences,
   no leading/trailing whitespace beyond the JSON object itself:

{
  "verdict": "positive" | "negative" | "ambiguous",
  "subcategory": "<short kebab-case>",
  "confidence": <float 0.0-1.0>,
  "rationale": "<one sentence>"
}
"""

_RETRY_REMINDER = (
    "Your previous reply was not valid JSON in the required shape. "
    "Reply again with STRICT JSON ONLY in the exact schema specified, "
    "no prose, no markdown fences."
)


def render_judge_prompt(*, axis: str, rubric: str | None, text: str) -> str:
    """Render the exact prompt sent to the judge model. Pure; no I/O.

    ``rubric`` is optional free text. When ``None``, an explicit
    ``(no rubric supplied — apply the axis name strictly)`` placeholder
    is rendered so the prompt stays well-formed and snapshot-stable.
    """
    rubric_text = (
        rubric.strip()
        if rubric and rubric.strip()
        else ("(no rubric supplied — apply the axis name strictly and be conservative)")
    )
    return (
        _PROMPT_TEMPLATE.replace("<<AXIS>>", axis)
        .replace("<<RUBRIC>>", rubric_text)
        .replace("<<TEXT>>", text)
    )


# ─── chat-client protocol ───────────────────────────────────────────


class _ChatClient:
    """Minimal single-shot chat-completion interface used by the judge.

    Pulled out so tests inject a deterministic fake without monkey-
    patching ``litellm``. ``call`` returns the model's raw string
    content. Implementations must raise :class:`LLMError` on transport
    failures and :class:`ContentFilterError` on content-filter blocks.
    """

    def call(self, *, messages: list[dict[str, str]], timeout_s: float) -> str:
        raise NotImplementedError


class ContentFilterError(LLMError):
    """Raised when the provider's content-safety filter rejected the row.

    Soft per-row failure; the run continues. Counted in
    ``stats.n_content_filter_blocked``. Currently only Azure OpenAI's
    policy filter and the Copilot proxy raise this shape, but any
    provider returning a recognisable content-filter response will
    surface here.
    """


class JudgeParseError(LLMError):
    """Raised when the judge response cannot be parsed into a JudgeBlock."""


@dataclass
class _LiteLLMChatClient(_ChatClient):
    """Default ``_ChatClient`` that routes through ``litellm``.

    Re-uses the shared ``llm_client.build_completion_kwargs`` helper so
    provider routing is identical to decompose / strategy / coverage —
    Azure OpenAI when ``ctx.model`` is unset, GitHub Copilot / GitHub
    Models / OpenAI / Anthropic when ``ctx.model`` is set with the
    appropriate prefix.

    ``model_override`` carries the per-call ``run_judge(model=...)``
    flag through to the actual completion call. Without it the cache
    key + audit trail (built via ``_resolve_model_name``) would record
    one model while the call ran under another — silent provider
    mismatch.
    """

    ctx: ScoutContext
    token_provider: Any | None = None
    model_override: str | None = None

    def call(self, *, messages: list[dict[str, str]], timeout_s: float) -> str:
        litellm = import_litellm()
        kwargs = build_completion_kwargs(
            self.ctx,
            messages=messages,
            response_format=BaseModel,  # triggers json_object mode
            timeout_s=timeout_s,
            token_provider=self.token_provider,
            model=self.model_override,
        )
        kwargs["max_tokens"] = _MAX_TOKENS
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "content_filter" in msg or "responsibleaipolicyviolation" in msg:
                raise ContentFilterError(str(exc)) from exc
            raise LLMError(f"judge LLM call failed: {exc}") from exc
        return extract_content(response)


# ─── response parsing ───────────────────────────────────────────────


class _JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: Literal["positive", "negative", "ambiguous"]
    subcategory: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


def _parse_judge_content(content: str) -> _JudgeResponse:
    payload = json.loads(content)
    return _JudgeResponse.model_validate(payload)


# ─── cache ───────────────────────────────────────────────────────────


def judge_cache_dir(ctx: ScoutContext) -> Path:
    """Per-workspace judge cache root.

    Distinct from any other cache; never shared across tools (the
    template_version is scout-internal — see design doc §6).
    """
    return ctx.cache_dir / "judge"


def _cache_key(*, prompt: str, axis: str, model: str, template_version: str) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(axis.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(template_version.encode("utf-8"))
    return h.hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    # Two-char shard so a single dir doesn't grow unbounded on big runs.
    return cache_dir / key[:2] / f"{key}.json"


def _read_cache(path: Path) -> _JudgeResponse | None:
    """Read a cached judge response. Corrupt files return ``None`` and
    are deleted so the next call regenerates the entry."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        return _JudgeResponse.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        import contextlib

        with contextlib.suppress(OSError):
            path.unlink()
        return None


def _write_cache(path: Path, resp: _JudgeResponse) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{secrets.token_hex(4)}")
    tmp.write_text(resp.model_dump_json(), encoding="utf-8")
    os.replace(tmp, path)


# ─── single-call wrapper ────────────────────────────────────────────


def _one_judge_call(
    *,
    prompt: str,
    chat_client: _ChatClient,
    timeout_s: float,
) -> _JudgeResponse:
    """One judge call with one parse-error retry. Raises on terminal failure."""
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    last_error: Exception | None = None
    for attempt in range(2):
        content = chat_client.call(messages=messages, timeout_s=timeout_s)
        try:
            return _parse_judge_content(content)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt == 0:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": _RETRY_REMINDER},
                ]
                continue
            raise JudgeParseError(f"judge returned invalid JSON twice: {exc}") from exc
    raise JudgeParseError(f"judge returned invalid JSON: {last_error}")


def _judged_block(
    resps: list[_JudgeResponse],
    *,
    axis: str,
    model: str,
    template_version: str,
    n_judges: int,
    agreement_mode: Literal["single", "majority", "unanimous"],
) -> tuple[JudgeBlock, float]:
    """Aggregate ``len(resps)`` judge responses into one ``JudgeBlock``
    plus the derived ``label_confidence`` per design doc §5.

    For ``majority``/``unanimous`` the chosen verdict + subcategory +
    rationale come from the agreeing-judges majority. ``confidence`` on
    the block is the ``mean_confidence_of_agreeing`` (the raw self-rated
    figure, before agreement weighting); the *derived* ``label_confidence``
    is the second tuple element.
    """
    verdicts = [r.verdict for r in resps]
    if agreement_mode == "single":
        chosen = resps[0]
        derived = chosen.confidence
        return (
            JudgeBlock(
                axis=axis,
                verdict=chosen.verdict,
                subcategory=chosen.subcategory,
                confidence=chosen.confidence,
                rationale=chosen.rationale,
                model=model,
                template_version=template_version,
                n_judges=n_judges,
                agreement="single",
            ),
            derived,
        )
    if agreement_mode == "unanimous":
        if len(set(verdicts)) != 1:
            chosen = resps[0]
            mean_conf = round(statistics.fmean(r.confidence for r in resps), 6)
            return (
                JudgeBlock(
                    axis=axis,
                    verdict="ambiguous",
                    subcategory="multi-judge-disagreement",
                    confidence=mean_conf,
                    rationale=(f"unanimous required but verdicts split: {', '.join(verdicts)}"),
                    model=model,
                    template_version=template_version,
                    n_judges=n_judges,
                    agreement="unanimous",
                ),
                0.0,
            )
        agreeing = resps
        mean_conf = round(statistics.fmean(r.confidence for r in agreeing), 6)
        chosen = agreeing[0]
        return (
            JudgeBlock(
                axis=axis,
                verdict=chosen.verdict,
                subcategory=chosen.subcategory,
                confidence=mean_conf,
                rationale=chosen.rationale,
                model=model,
                template_version=template_version,
                n_judges=n_judges,
                agreement="unanimous",
            ),
            mean_conf,
        )
    # majority
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    top_verdict, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count <= n_judges // 2:
        chosen = resps[0]
        mean_conf = round(statistics.fmean(r.confidence for r in resps), 6)
        return (
            JudgeBlock(
                axis=axis,
                verdict="ambiguous",
                subcategory="multi-judge-no-majority",
                confidence=mean_conf,
                rationale=(f"no majority among {n_judges} judges: {', '.join(verdicts)}"),
                model=model,
                template_version=template_version,
                n_judges=n_judges,
                agreement="majority",
            ),
            0.0,
        )
    agreeing = [r for r in resps if r.verdict == top_verdict]
    mean_conf = round(statistics.fmean(r.confidence for r in agreeing), 6)
    chosen = agreeing[0]
    derived = round((len(agreeing) / n_judges) * mean_conf, 6)
    return (
        JudgeBlock(
            axis=axis,
            verdict=chosen.verdict,
            subcategory=chosen.subcategory,
            confidence=mean_conf,
            rationale=chosen.rationale,
            model=model,
            template_version=template_version,
            n_judges=n_judges,
            agreement="majority",
        ),
        derived,
    )


# ─── promotion ──────────────────────────────────────────────────────


def _promote(
    record: NormalizedRecord,
    block: JudgeBlock,
    label_confidence: float,
    *,
    threshold: float,
) -> NormalizedRecord:
    """Apply the explicit-gap promotion rule (design doc §5).

    Always attaches ``block`` to the row so reviewers see *why* a
    promotion was declined; only updates ``label`` / ``label_kind`` /
    ``label_confidence`` when the verdict is positive/negative AND the
    derived confidence is at-or-above ``threshold``.
    """
    if block.verdict in ("positive", "negative") and label_confidence >= threshold:
        new_label = _VERDICT_TO_LABEL[block.verdict]
        return record.model_copy(
            update={
                "label": new_label,
                "label_kind": LabelKind.JUDGED,
                "label_confidence": label_confidence,
                "judge": block,
            }
        )
    # below threshold, ambiguous, or unknown verdict: keep label,
    # attach block + derived confidence so the audit trail is complete.
    return record.model_copy(update={"judge": block, "label_confidence": label_confidence})


# ─── corpus IO ──────────────────────────────────────────────────────


_CORPUS_FILES: tuple[str, ...] = ("train.jsonl", "val.jsonl", "test.jsonl")


def _resolve_corpus_files(target: Path) -> list[Path]:
    """Return JSONL files that make up the corpus rooted at ``target``.

    Directory: any subset of ``train.jsonl``/``val.jsonl``/``test.jsonl``
    that exist (in that order). Single ``.jsonl`` file: just that file.
    """
    if target.is_file() and target.suffix == ".jsonl":
        return [target]
    if not target.is_dir():
        raise DatasetScoutError(f"judge target {target} is neither a directory nor a .jsonl file")
    files = [target / name for name in _CORPUS_FILES if (target / name).is_file()]
    if not files:
        # Fall back: any *.jsonl in the directory, sorted for stability.
        files = sorted(target.glob("*.jsonl"))
    if not files:
        raise DatasetScoutError(f"no JSONL files found under {target}")
    return files


def _iter_records(path: Path) -> Iterator[NormalizedRecord]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield NormalizedRecord.model_validate(json.loads(line))


def _write_records(path: Path, records: Iterable[NormalizedRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
            n += 1
    return n


# ─── checkpoint ─────────────────────────────────────────────────────


_CHECKPOINT_NAME = ".judge_state.json"


@dataclass
class _Checkpoint:
    axis: str
    completed_row_ids: set[str] = field(default_factory=set)
    in_flight_batch: list[str] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "completed_row_ids": sorted(self.completed_row_ids),
            "in_flight_batch": self.in_flight_batch,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _Checkpoint:
        return cls(
            axis=str(data.get("axis", "")),
            completed_row_ids=set(data.get("completed_row_ids") or []),
            in_flight_batch=list(data["in_flight_batch"])
            if data.get("in_flight_batch") is not None
            else None,
        )


def _load_checkpoint(out_dir: Path, axis: str) -> _Checkpoint:
    path = out_dir / _CHECKPOINT_NAME
    if not path.is_file():
        return _Checkpoint(axis=axis)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        cp = _Checkpoint.from_json(data)
    except (json.JSONDecodeError, OSError, ValidationError):
        # Corrupt checkpoint: start fresh; old verdicts come back from
        # the disk cache anyway (regenerable).
        return _Checkpoint(axis=axis)
    if cp.axis != axis:
        # Different axis: treat as fresh run for this axis. We don't
        # support multi-axis checkpoints in one out_dir.
        return _Checkpoint(axis=axis)
    return cp


def _save_checkpoint(out_dir: Path, cp: _Checkpoint) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _CHECKPOINT_NAME
    tmp = path.with_suffix(path.suffix + f".tmp-{secrets.token_hex(4)}")
    tmp.write_text(json.dumps(cp.to_json(), indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ─── result types ───────────────────────────────────────────────────


@dataclass
class JudgeStats:
    """Counts emitted by a judge run; mirrors the lockfile ``stats`` block."""

    n_input: int = 0
    n_judged: int = 0
    n_promoted_positive: int = 0
    n_promoted_negative: int = 0
    n_left_unknown: int = 0
    n_cache_hits: int = 0
    n_skipped: int = 0
    n_content_filter_blocked: int = 0
    n_parse_errors: int = 0
    n_api_errors: int = 0
    n_resumed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "n_input": self.n_input,
            "n_judged": self.n_judged,
            "n_promoted_positive": self.n_promoted_positive,
            "n_promoted_negative": self.n_promoted_negative,
            "n_left_unknown": self.n_left_unknown,
            "n_cache_hits": self.n_cache_hits,
            "n_skipped": self.n_skipped,
            "n_content_filter_blocked": self.n_content_filter_blocked,
            "n_parse_errors": self.n_parse_errors,
            "n_api_errors": self.n_api_errors,
            "n_resumed": self.n_resumed,
        }


@dataclass
class JudgeResult:
    """Public result of :func:`run_judge`. Mirrored into report + lockfile."""

    out_dir: Path
    axis: str
    rubric: str | None
    model: str
    template_version: str
    n_judges: int
    agreement: Literal["single", "majority", "unanimous"]
    threshold: float
    cache_dir: Path
    stats: JudgeStats
    failures: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    files_written: list[Path] = field(default_factory=list)
    calibration: dict[str, Any] | None = None
    dry_run: bool = False
    estimated_calls: int = 0


# ─── core engine ────────────────────────────────────────────────────


def _judge_one_record(
    rec: NormalizedRecord,
    *,
    axis: str,
    rubric: str | None,
    model: str,
    template_version: str,
    judges: int,
    agreement_mode: Literal["single", "majority", "unanimous"],
    cache_dir: Path,
    chat_client: _ChatClient,
    timeout_s: float,
    stats: JudgeStats,
) -> tuple[JudgeBlock | None, float]:
    """Judge one row. Returns ``(block, derived_confidence)`` or
    ``(None, 0.0)`` on soft failure (with stats updated)."""
    prompt = render_judge_prompt(axis=axis, rubric=rubric, text=rec.text)
    base_key = _cache_key(prompt=prompt, axis=axis, model=model, template_version=template_version)
    responses: list[_JudgeResponse] = []
    cache_hits = 0
    for j in range(judges):
        # Salt the cache key with the judge index so a 3-judge run
        # caches three distinct verdicts (independent calls), not one
        # call replayed three times. Stable across runs given the same
        # (prompt, axis, model, template_version, j).
        salted = (
            _cache_key(
                prompt=prompt,
                axis=axis,
                model=f"{model}#j{j}",
                template_version=template_version,
            )
            if judges > 1
            else base_key
        )
        cpath = _cache_path(cache_dir, salted)
        cached = _read_cache(cpath)
        if cached is not None:
            responses.append(cached)
            cache_hits += 1
            continue
        try:
            resp = _one_judge_call(prompt=prompt, chat_client=chat_client, timeout_s=timeout_s)
        except ContentFilterError as exc:
            stats.n_content_filter_blocked += 1
            stats.n_skipped += 1
            _log.warning("content filter blocked row %s: %s", rec.stable_id, exc)
            return None, 0.0
        except JudgeParseError as exc:
            stats.n_parse_errors += 1
            stats.n_skipped += 1
            _log.warning("judge parse error on row %s: %s", rec.stable_id, exc)
            return None, 0.0
        except LLMError as exc:
            stats.n_api_errors += 1
            stats.n_skipped += 1
            _log.warning("judge API error on row %s: %s", rec.stable_id, exc)
            return None, 0.0
        _write_cache(cpath, resp)
        responses.append(resp)
    stats.n_cache_hits += cache_hits
    block, derived = _judged_block(
        responses,
        axis=axis,
        model=model,
        template_version=template_version,
        n_judges=judges,
        agreement_mode=agreement_mode,
    )
    return block, derived


def _resolve_chat_client(
    ctx: ScoutContext,
    chat_client: _ChatClient | None,
    *,
    model: str | None = None,
) -> _ChatClient:
    if chat_client is not None:
        return chat_client
    if not ctx.llm_configured and not model:
        raise LLMError(
            "No LLM provider configured. Set DATASET_SCOUT_MODEL "
            "(e.g. 'github_copilot/gpt-5-mini' or 'github/gpt-4o-mini'), "
            "or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT (and run "
            "`az login` for Entra auth)."
        )
    return _LiteLLMChatClient(ctx=ctx, model_override=model)


def _resolve_model_name(ctx: ScoutContext, override: str | None) -> str:
    """Stable model identifier for cache keys + audit trail.

    Reflects the actual provider/model the call goes to. Cache keys
    embed this so a backend switch (Azure ↔ GitHub Copilot) can't
    serve stale verdicts.
    """
    if override:
        return override
    return effective_model_id(ctx) or "unconfigured"


def _eligible_for_judging(rec: NormalizedRecord, *, only_unknown: bool) -> bool:
    """Whether a row should be sent to the judge.

    With ``only_unknown=True`` (the default), rows already labelled with
    ``GROUND_TRUTH`` keep their authoritative labels untouched — judge
    rescue is for rows where we don't already trust the label. Rows
    that have already been ``JUDGED`` are skipped to avoid double-judge
    on resume; pass ``only_unknown=False`` (i.e. ``--re-judge-all``) to
    override.
    """
    if only_unknown:
        if rec.label_kind == LabelKind.GROUND_TRUTH:
            return False
        if rec.label_kind == LabelKind.JUDGED:
            return False
    return True


def _write_judge_lockfile(
    path: Path,
    *,
    result: JudgeResult,
    target: Path,
    started_iso: str,
) -> None:
    """Emit the design-doc §4.3 lockfile slice for a judge run.

    The output is YAML (matching curate's lockfile aesthetic) and lives
    next to the judged corpus at ``<out_dir>/judge.lock.yaml``. It is
    additive: re-running over an existing curate corpus does NOT alter
    that corpus's ``recipe.lock.yaml``.
    """
    import yaml  # type: ignore[import-untyped]

    payload: dict[str, Any] = {
        "judge_template_version": JUDGE_TEMPLATE_VERSION,
        "started_at": started_iso,
        "elapsed_seconds": result.elapsed_seconds,
        "target": str(target),
        "out_dir": str(result.out_dir),
        "judge": {
            "axis": result.axis,
            "rubric": result.rubric,
            "model": result.model,
            "template_version": result.template_version,
            "n_judges": result.n_judges,
            "agreement": result.agreement,
            "threshold": result.threshold,
            "cache_dir": str(result.cache_dir),
            "calibration": result.calibration,
            "stats": result.stats.as_dict(),
        },
        "failures": list(result.failures),
        "files_written": [str(p) for p in result.files_written],
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _sample_judged_rows(files: list[Path], *, top_n: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Bucket and sample judged rows for the report.

    Returns a dict with three keys: ``promoted_positive``, ``promoted_negative``,
    and ``ambiguous``. Each list holds the top-``top_n`` highest-confidence
    rows in that bucket, with ``text`` truncated for display. Reads only
    files that exist (some splits may be empty).
    """
    promoted_pos: list[dict[str, Any]] = []
    promoted_neg: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                judge = row.get("judge")
                if not isinstance(judge, dict):
                    continue
                verdict = judge.get("verdict")
                conf = float(judge.get("confidence") or 0.0)
                bucket: list[dict[str, Any]] | None
                if row.get("label_kind") == "judged" and verdict == "positive":
                    bucket = promoted_pos
                elif row.get("label_kind") == "judged" and verdict == "negative":
                    bucket = promoted_neg
                elif verdict == "ambiguous":
                    bucket = ambiguous
                else:
                    bucket = None
                if bucket is None:
                    continue
                bucket.append(
                    {
                        "confidence": conf,
                        "subcategory": judge.get("subcategory") or "",
                        "rationale": judge.get("rationale") or "",
                        "source": row.get("source") or "",
                        "stable_id": (
                            f"{row.get('source', '')}::"
                            f"{row.get('source_config') or '_'}::"
                            f"{row.get('source_split') or '_'}::"
                            f"{row.get('source_row_id', '')}"
                        ),
                        "text": row.get("text") or "",
                    }
                )
    for bucket in (promoted_pos, promoted_neg, ambiguous):
        bucket.sort(key=lambda r: r["confidence"], reverse=True)
        del bucket[top_n:]
    return {
        "promoted_positive": promoted_pos,
        "promoted_negative": promoted_neg,
        "ambiguous": ambiguous,
    }


def _format_sample_section(title: str, rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [f"### {title}", "", "_(none)_", ""]
    out: list[str] = [f"### {title}", ""]
    for r in rows:
        text = r["text"]
        if len(text) > 240:
            text = text[:237] + "..."
        text = text.replace("\n", " ").replace("\r", " ")
        out += [
            f"- **conf {r['confidence']:.2f}** · `{r['subcategory']}`",
            f"  - source: `{r['source']}`",
            f"  - stable_id: `{r['stable_id']}`",
            f"  - rationale: {r['rationale']}",
            f"  - text: > {text}",
            "",
        ]
    return out


def _write_judge_report(path: Path, *, result: JudgeResult, target: Path) -> None:
    """Emit a short markdown report next to the judged corpus."""
    s = result.stats
    lines: list[str] = [
        f"# Judged corpus — axis `{result.axis}`",
        "",
        f"- **Source:** `{target}`",
        f"- **Model:** `{result.model}` (template {result.template_version})",
        f"- **Judges:** {result.n_judges} ({result.agreement} agreement)",
        f"- **Threshold:** {result.threshold:.2f}",
        f"- **Promoted positive:** {s.n_promoted_positive}",
        f"- **Promoted negative:** {s.n_promoted_negative}",
        f"- **Left unknown:** {s.n_left_unknown}",
        f"- **Skipped (soft fail):** {s.n_skipped} "
        f"(api={s.n_api_errors} · parse={s.n_parse_errors} · "
        f"content_filter={s.n_content_filter_blocked})",
        f"- **Cache hits:** {s.n_cache_hits} / {max(s.n_judged * result.n_judges, 1)}",
        f"- **Wall-clock:** {result.elapsed_seconds:.2f}s",
    ]
    if result.calibration is not None:
        c = result.calibration
        lines += [
            "",
            "## Calibration",
            "",
            f"- **Against:** `{c.get('against')}`",
            f"- **Sample size:** {c.get('n_sampled')}",
            f"- **Precision:** {c.get('precision', 0.0):.3f}",
            f"- **Recall:** {c.get('recall', 0.0):.3f}",
            f"- **F1:** {c.get('f1', 0.0):.3f}",
        ]

    # Sample top-3 rows per bucket so reviewers can eyeball the run without
    # rg'ing the JSONL. Cheap (one stream pass over already-written files).
    samples = _sample_judged_rows(result.files_written, top_n=3)
    if any(samples.values()):
        lines += [
            "",
            "## Sample rows",
            "",
            "_Top 3 by judge confidence in each bucket. Use these for "
            "eyeball validation before trusting the run; full provenance "
            "lives on every row's `judge` block in the JSONL._",
            "",
        ]
        lines += _format_sample_section(
            "Highest-confidence promoted POSITIVES", samples["promoted_positive"]
        )
        lines += _format_sample_section(
            "Highest-confidence promoted NEGATIVES", samples["promoted_negative"]
        )
        lines += _format_sample_section(
            "Highest-confidence AMBIGUOUS (not promoted)", samples["ambiguous"]
        )

    lines += [
        "",
        "---",
        "",
        "_Promotion rule: explicit gap. Rows with judge verdict "
        "`positive` / `negative` at-or-above threshold are written with "
        "`label_kind: judged`; lower-confidence verdicts attach the "
        "JudgeBlock for review but keep the original label._",
        "",
    ]
    if (path.parent / _CHECKPOINT_NAME).exists():
        lines += [
            f"_Tip: re-running `datascout judge` with the same "
            f"`--out {path.parent}` resumes from the per-batch checkpoint "
            f"({_CHECKPOINT_NAME}); already-judged rows are served from "
            f"the disk cache rather than re-judged._",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_calibration(
    *,
    gold: Path,
    axis: str,
    rubric: str | None,
    judges: int,
    agreement: Literal["single", "majority", "unanimous"],
    threshold: float,
    model_name: str,
    cache_dir: Path,
    chat_client: _ChatClient,
    timeout_s: float,
    seed_n: int,
    seed: int,
    floor: float | None,
    proceed: bool,
) -> dict[str, Any]:
    """Sample ``seed_n`` ground-truth gold rows, judge them, and return a
    calibration report (precision / recall / F1 / confusion).

    Mirrors the shape recorded under ``judge.calibration`` in the
    lockfile (design doc §4.3). Raises :class:`DatasetScoutError` when
    ``floor`` is set, calibrated precision falls below it, and
    ``proceed`` is False.
    """
    import random
    from dataclasses import asdict

    gold_rows: list[NormalizedRecord] = [
        rec for rec in _iter_records_anywhere(gold) if rec.label_kind == LabelKind.GROUND_TRUTH
    ]
    if not gold_rows:
        return {
            "against": str(gold),
            "seed_n": seed_n,
            "n_sampled": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage": 0.0,
            "notes": ["no ground_truth rows found in gold corpus"],
        }
    rng = random.Random(seed)
    sample = list(gold_rows) if len(gold_rows) <= seed_n else rng.sample(gold_rows, seed_n)
    cm_tp = cm_fp = cm_fn = cm_tn = 0
    n_promoted = 0
    cal_stats = JudgeStats()
    for rec in sample:
        block, derived = _judge_one_record(
            rec,
            axis=axis,
            rubric=rubric,
            model=model_name,
            template_version=JUDGE_TEMPLATE_VERSION,
            judges=judges,
            agreement_mode=agreement,
            cache_dir=cache_dir,
            chat_client=chat_client,
            timeout_s=timeout_s,
            stats=cal_stats,
        )
        if block is None:
            continue
        if not (block.verdict in ("positive", "negative") and derived >= threshold):
            continue
        n_promoted += 1
        gold_class = "positive" if rec.label == "positive" else "negative"
        judged_class = "positive" if block.verdict == "positive" else "negative"
        if judged_class == "positive" and gold_class == "positive":
            cm_tp += 1
        elif judged_class == "positive":
            cm_fp += 1
        elif gold_class == "positive":
            cm_fn += 1
        else:
            cm_tn += 1
    precision = cm_tp / (cm_tp + cm_fp) if (cm_tp + cm_fp) else 0.0
    recall = cm_tp / (cm_tp + cm_fn) if (cm_tp + cm_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    block_payload = {
        "against": str(gold),
        "seed_n": seed_n,
        "n_sampled": len(sample),
        "n_promoted": n_promoted,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "confusion": {
            "true_positive": cm_tp,
            "false_positive": cm_fp,
            "false_negative": cm_fn,
            "true_negative": cm_tn,
        },
        "stats": asdict(cal_stats) if False else cal_stats.as_dict(),
        "threshold": threshold,
    }
    if floor is not None and precision < floor and not proceed:
        raise DatasetScoutError(
            f"calibrated precision {precision:.3f} < floor {floor:.3f}; "
            "pass --proceed to override or refine the rubric."
        )
    return block_payload


def _iter_records_anywhere(target: Path) -> Iterator[NormalizedRecord]:
    """Helper: iterate records from a directory or single jsonl file."""
    for p in _resolve_corpus_files(target):
        yield from _iter_records(p)


def run_judge(
    ctx: ScoutContext,
    target: Path,
    *,
    axis: str,
    rubric: str | None = None,
    judges: int = 1,
    agreement: Literal["single", "majority", "unanimous"] = "single",
    threshold: float = 0.8,
    out_dir: Path | None = None,
    only_unknown: bool = True,
    re_judge_all: bool = False,
    dry_run: bool = False,
    model: str | None = None,
    timeout_s: float = 30.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chat_client: _ChatClient | None = None,
    progress: Callable[[int, int], None] | None = None,
    calibrate_against: Path | None = None,
    calibration_seed_n: int = 100,
    calibration_floor: float | None = None,
    proceed: bool = False,
    calibration_seed: int = 1729,
) -> JudgeResult:
    """Run the LLM-as-judge label-rescue pass over a corpus.

    ``target`` is a directory (containing ``train.jsonl``/``val.jsonl``/
    ``test.jsonl``, or any other ``*.jsonl`` files) or a single
    ``.jsonl`` file. ``out_dir`` defaults to ``<target>/judged`` (or, for
    single-file inputs, ``<file>.judged/``); the original corpus is
    never overwritten in this slice.

    Resumability: the per-run checkpoint at ``<out_dir>/.judge_state.json``
    records ``stable_id`` of completed rows. Re-running with the same
    ``out_dir`` resumes; the disk cache (``judge_cache_dir(ctx)``)
    eliminates re-paying for already-judged rows even without a checkpoint.

    Soft failures: the run completes regardless of per-row failures
    (API errors, parse errors twice in a row, content-filter rejections,
    cache corruption). Each is counted in ``result.stats``.
    """
    import time

    started = time.monotonic()
    started_iso = datetime.now(UTC).isoformat()

    if judges < 1:
        raise DatasetScoutError("judges must be >= 1")
    if judges > 1 and agreement == "single":
        raise DatasetScoutError(f"agreement={agreement!r} requires judges=1; got judges={judges}")
    if judges == 1 and agreement != "single":
        raise DatasetScoutError(f"agreement={agreement!r} requires judges>=2; got judges=1")
    if agreement == "majority" and judges < 3:
        raise DatasetScoutError("majority agreement requires at least 3 judges")
    if agreement == "unanimous" and judges < 3:
        raise DatasetScoutError("unanimous agreement requires at least 3 judges")
    if not 0.0 <= threshold <= 1.0:
        raise DatasetScoutError("threshold must be in [0.0, 1.0]")

    files = _resolve_corpus_files(target)
    resolved_out: Path
    if out_dir is None:
        if target.is_file():
            resolved_out = target.with_suffix(target.suffix + ".judged")
        else:
            resolved_out = target / "judged"
    else:
        resolved_out = out_dir
    resolved_out.mkdir(parents=True, exist_ok=True)

    cache_dir = judge_cache_dir(ctx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_name = _resolve_model_name(ctx, model)
    stats = JudgeStats()
    failures: list[dict[str, Any]] = []
    files_written: list[Path] = []

    only_unknown_eff = only_unknown and not re_judge_all
    chat = None if dry_run else _resolve_chat_client(ctx, chat_client, model=model)

    calibration_block: dict[str, Any] | None = None
    if calibrate_against is not None and not dry_run:
        assert chat is not None
        calibration_block = _run_calibration(
            gold=calibrate_against,
            axis=axis,
            rubric=rubric,
            judges=judges,
            agreement=agreement,
            threshold=threshold,
            model_name=model_name,
            cache_dir=cache_dir,
            chat_client=chat,
            timeout_s=timeout_s,
            seed_n=calibration_seed_n,
            seed=calibration_seed,
            floor=calibration_floor,
            proceed=proceed,
        )

    cp = _load_checkpoint(resolved_out, axis)
    stats.n_resumed = len(cp.completed_row_ids)
    if stats.n_resumed:
        # Visible-on-the-wire signal that resume kicked in. Without this the
        # only hint is the post-run "resumed: N" panel row, by which point
        # the user has already wondered if their run was a duplicate.
        _log.info(
            "resuming axis=%r from prior run: %d row(s) already completed; "
            "they'll be re-applied from cache rather than re-judged.",
            axis,
            stats.n_resumed,
        )

    estimated_calls = 0
    pending: list[tuple[Path, NormalizedRecord]] = []
    out_records: dict[Path, list[NormalizedRecord]] = {f: [] for f in files}
    for src in files:
        out_path = resolved_out / src.name
        for rec in _iter_records(src):
            stats.n_input += 1
            out_records[src].append(rec)
            if not _eligible_for_judging(rec, only_unknown=only_unknown_eff):
                continue
            if rec.stable_id in cp.completed_row_ids:
                # Already judged on a prior run; the cached verdict will
                # be re-applied below from disk cache.
                pass
            estimated_calls += judges
            pending.append((src, rec))

    if dry_run:
        elapsed = time.monotonic() - started
        return JudgeResult(
            out_dir=resolved_out,
            axis=axis,
            rubric=rubric,
            model=model_name,
            template_version=JUDGE_TEMPLATE_VERSION,
            n_judges=judges,
            agreement=agreement,
            threshold=threshold,
            cache_dir=cache_dir,
            stats=stats,
            failures=failures,
            elapsed_seconds=round(elapsed, 3),
            files_written=files_written,
            dry_run=True,
            estimated_calls=estimated_calls,
        )

    assert chat is not None  # narrowed for type-checkers
    judged_by_id: dict[str, tuple[JudgeBlock, float]] = {}
    total = len(pending)
    batch_buf: list[str] = []
    for n_done, (_src, rec) in enumerate(pending, start=1):
        block, derived = _judge_one_record(
            rec,
            axis=axis,
            rubric=rubric,
            model=model_name,
            template_version=JUDGE_TEMPLATE_VERSION,
            judges=judges,
            agreement_mode=agreement,
            cache_dir=cache_dir,
            chat_client=chat,
            timeout_s=timeout_s,
            stats=stats,
        )
        if block is None:
            failures.append(
                {
                    "stable_id": rec.stable_id,
                    "source": rec.source,
                    "source_row_id": rec.source_row_id,
                    "category": "judge_failure",
                }
            )
        else:
            judged_by_id[rec.stable_id] = (block, derived)
            cp.completed_row_ids.add(rec.stable_id)
            stats.n_judged += 1
        batch_buf.append(rec.stable_id)
        if progress is not None:
            progress(n_done, total)
        if len(batch_buf) >= batch_size:
            cp.in_flight_batch = None
            _save_checkpoint(resolved_out, cp)
            batch_buf = []
    cp.in_flight_batch = None
    _save_checkpoint(resolved_out, cp)

    # Apply promotions and write outputs.
    for src in files:
        out_path = resolved_out / src.name
        promoted: list[NormalizedRecord] = []
        for rec in out_records[src]:
            judged = judged_by_id.get(rec.stable_id)
            if judged is None:
                promoted.append(rec)
                continue
            block, derived = judged
            new_rec = _promote(rec, block, derived, threshold=threshold)
            if new_rec.label_kind == LabelKind.JUDGED:
                if new_rec.label == "positive":
                    stats.n_promoted_positive += 1
                else:
                    stats.n_promoted_negative += 1
            else:
                stats.n_left_unknown += 1
            promoted.append(new_rec)
        _write_records(out_path, promoted)
        files_written.append(out_path)

    elapsed = time.monotonic() - started
    result = JudgeResult(
        out_dir=resolved_out,
        axis=axis,
        rubric=rubric,
        model=model_name,
        template_version=JUDGE_TEMPLATE_VERSION,
        n_judges=judges,
        agreement=agreement,
        threshold=threshold,
        cache_dir=cache_dir,
        stats=stats,
        failures=failures,
        elapsed_seconds=round(elapsed, 3),
        files_written=files_written,
        estimated_calls=estimated_calls,
        calibration=calibration_block,
    )
    _write_judge_lockfile(
        resolved_out / "judge.lock.yaml",
        result=result,
        target=target,
        started_iso=started_iso,
    )
    _write_judge_report(resolved_out / "judge.report.md", result=result, target=target)
    return result


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "JUDGE_TEMPLATE_VERSION",
    "ContentFilterError",
    "JudgeParseError",
    "JudgeResult",
    "JudgeStats",
    "_ChatClient",
    "judge_cache_dir",
    "render_judge_prompt",
    "run_judge",
]
