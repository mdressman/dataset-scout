"""HuggingFace Hub source plugin.

Reads from the public HF API via `huggingface_hub`. Yields `Candidate`
objects with a populated `CandidateMetadata` envelope so probes can
consume a single normalized shape.

Search strategy in M1a: combine the original Intent's brief and any
explicit threat-family / language hints into one `search` query. Returns
candidates in HF's relevance order (which is the only ranking we trust
in M1a — embedding fit and the strategy assessor land later).

This module is intentionally thin: it owns the
`HfApi -> CandidateMetadata` translation and nothing else. Caching,
rate-limiting middleware, and a per-run `Budget` enforcement live
upstream in the pipeline (added in M1b).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    DecompositionDirection,
    Intent,
)
from dataset_scout.licenses import guess_spdx
from dataset_scout.sources.base import Budget, Obligation

# Splits a CamelCase-only string into its component words. Handles:
#   PersonaChat        -> ["Persona", "Chat"]
#   GPT4              -> ["GPT", "4"]
#   IntimacyBench     -> ["Intimacy", "Bench"]
#   HHRLHFData        -> ["HHRLHF", "Data"]   (acronym runs stay together)
# Returns the input as a single-element list if no boundary is found.
_CAMEL_BOUNDARY = re.compile(
    r"(?<=[a-z])(?=[A-Z])"           # lowerUpper
    r"|(?<=[A-Z])(?=[A-Z][a-z])"     # ABCdef -> AB|Cdef
    r"|(?<=[A-Za-z])(?=\d)"          # letter|digit
    r"|(?<=\d)(?=[A-Za-z])"          # digit|letter
)


def _camel_split(s: str) -> list[str]:
    parts = _CAMEL_BOUNDARY.split(s)
    return [p for p in parts if p]

if TYPE_CHECKING:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import DatasetInfo


_HF_DATASET_URL = "https://huggingface.co/datasets/{id}"
_DEFAULT_LIMIT = 50


def _card_data_to_dict(card_data: Any) -> dict[str, Any]:
    """Coerce a `DatasetCardData` (or None / dict) into a plain dict.

    `huggingface_hub`'s `DatasetCardData` exposes attribute-style access
    and `.to_dict()` but not `.keys()` directly. Normalize defensively so
    the rest of this module can treat it as a regular dict.
    """
    if card_data is None:
        return {}
    if isinstance(card_data, dict):
        return card_data
    to_dict = getattr(card_data, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return result
    return {}


def _coerce_languages(value: Any) -> list[str]:
    """Card YAML may declare `language:` as a string or a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, (str, int))]
    return []


def _coerce_task_categories(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _coerce_dt(value: Any) -> datetime | None:
    """huggingface_hub returns datetimes already; pass through, accept
    ISO strings as a fallback."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _build_metadata(info: DatasetInfo) -> CandidateMetadata:
    """Translate a single HF DatasetInfo into the source-agnostic envelope."""
    card = _card_data_to_dict(info.card_data)
    license_raw = card.get("license")
    if isinstance(license_raw, list):
        # HF allows a list of licenses; keep the first as the canonical raw
        # and stash the rest in extras for visibility.
        license_raw_str = str(license_raw[0]) if license_raw else None
    elif license_raw is None:
        license_raw_str = None
    else:
        license_raw_str = str(license_raw)

    languages = _coerce_languages(card.get("language") or card.get("languages"))
    task_categories = _coerce_task_categories(card.get("task_categories"))

    main_size = getattr(info, "main_size", None)
    bytes_value: int | None = None
    if isinstance(main_size, int):
        bytes_value = main_size
    elif isinstance(main_size, dict):
        # HF reports {"size_in_bytes": ...} in some forms.
        for key in ("size_in_bytes", "total_size", "bytes"):
            v = main_size.get(key)
            if isinstance(v, int):
                bytes_value = v
                break

    extras: dict[str, Any] = {}
    if isinstance(license_raw, list) and len(license_raw) > 1:
        extras["additional_licenses"] = [str(x) for x in license_raw[1:]]
    citation = getattr(info, "citation", None)
    if citation:
        extras["citation"] = citation
    paperswithcode_id = getattr(info, "paperswithcode_id", None)
    if paperswithcode_id:
        extras["paperswithcode_id"] = paperswithcode_id

    return CandidateMetadata(
        description=getattr(info, "description", None) or None,
        card_url=_HF_DATASET_URL.format(id=info.id),
        homepage_url=card.get("homepage") if isinstance(card.get("homepage"), str) else None,
        license_raw=license_raw_str,
        license_spdx=guess_spdx(license_raw_str),
        languages_declared=languages,
        uploaded_at=_coerce_dt(getattr(info, "created_at", None)),
        last_modified=_coerce_dt(getattr(info, "last_modified", None)),
        content_date_range=None,  # HF cards rarely declare this; leave None.
        rows=None,  # row counts come from datasets-server (M1b).
        bytes=bytes_value,
        downloads=getattr(info, "downloads", None),
        likes=getattr(info, "likes", None),
        columns=[],  # filled by datasets-server columns endpoint in M1b.
        label_column_guess=None,
        text_column_guess=None,
        card_fields_present=frozenset(card.keys()),
        requires_auth=bool(getattr(info, "private", False)),
        gated=bool(getattr(info, "gated", False)),
        tags=list(getattr(info, "tags", []) or []),
        task_categories=task_categories,
        extras=extras,
    )


def _build_search_query(intent: Intent) -> str:
    """Construct the lexical query HF's `search=` parameter expects.

    HF's `search=` does keyword matching, not natural-language search.
    Long free-text briefs return nothing. So:

    - If the user (or heuristic parser) extracted threat_families, use
      those as the query. They're the highest-precision signal we have.
    - Otherwise pass the brief through verbatim.
    """
    if intent.threat_families:
        return " ".join(f.replace("_", " ") for f in intent.threat_families)
    return intent.raw_brief.strip()


def _direction_queries(direction: DecompositionDirection) -> list[str]:
    """Build per-keyword HF lexical queries for a decomposition direction.

    The decomposer returns 3-5 short academic-style keywords per direction
    in `direction.keywords`. The keyword-expansion stage (when an LLM
    provider is available) also populates `direction.dataset_keywords`
    with HF-uploader- style compound-noun phrases ("mental health chat",
    "counseling dialogue") that complement the academic terms.
    Empirically these expanded phrases catch datasets that the abstract
    keywords miss entirely (uploaders don't write "parasocial bonds"
    in dataset ids).

    Strategy: highest-precision first (recalled named benchmarks),
    then HF-uploader-style compound nouns, then academic-style as
    breadth. Cap at 7 per direction so the per-recon HF query budget
    stays bounded.

    Each phrase becomes its own HF `search=` query — substring matching
    means that joining all phrases into one query returns near-zero hits
    because no single dataset card contains every term in sequence.
    Round-robin across queries lives upstream in `search()`.

    HF's lexical search is multi-word AND — every whitespace-separated
    token must appear in the dataset id/card. Empirically:
        "mental health" → 3 hits   "mental health dialogues" → 0 hits
        "companion"     → 3 hits   "companion chatbot"      → 0 hits
    Even semantically-correct compound nouns from the LLM frequently
    fail this AND constraint when the third token doesn't happen to
    appear next to the first two in any uploader's id. To compensate,
    any phrase of 3+ tokens ALSO emits its 2-token prefix as a sibling
    query (deduped). Recalled named benchmarks (e.g. "PersonaChat",
    "INTIMA", "XSTest") are issued AS-IS — they're proper nouns and
    shortening would destroy the match. Empirically these are the
    highest-precision queries: they catch named research lines that
    compound nouns miss entirely (`google/Synthetic-Persona-Chat`,
    `AI-companionship/INTIMA`).

    CamelCase tokenization: HF dataset uploaders write the same
    benchmark as ``persona-chat`` or ``Persona-Chat`` while the LLM
    recalls it as ``PersonaChat``. ``PersonaChat`` (one word) does
    NOT match ``google/Synthetic-Persona-Chat`` on HF's tokenized
    search. So for any recalled name with 2+ CamelCase parts, we
    ALSO emit the space-separated variant as a sibling query.
    """
    pool: list[str] = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        norm = phrase.strip()
        if not norm:
            return
        key = norm.lower()
        if key in seen:
            return
        seen.add(key)
        pool.append(norm)

    def _add_with_shortenings(phrase: str) -> None:
        norm = phrase.strip()
        if not norm:
            return
        _add(norm)
        # Auto-shorten 3+ word phrases to their 2-word prefix so HF's
        # AND-search still hits something. "mental health dialogues" ->
        # "mental health"; "elder care conversation" -> "elder care".
        tokens = norm.split()
        if len(tokens) >= 3:
            _add(" ".join(tokens[:2]))

    def _add_recalled_name(name: str) -> None:
        """Add a recalled benchmark name plus its CamelCase split if any.

        ``PersonaChat`` -> emits both ``PersonaChat`` and ``Persona Chat``.
        ``INTIMA`` -> just ``INTIMA`` (no CamelCase boundary).
        ``Anthropic/hh-rlhf`` -> just ``Anthropic/hh-rlhf`` (already has
        separators).
        """
        norm = name.strip()
        if not norm:
            return
        _add(norm)
        # CamelCase split only when there are no whitespace/hyphen
        # separators (avoid mangling already-tokenised names).
        if " " in norm or "-" in norm or "_" in norm or "/" in norm:
            return
        parts = _camel_split(norm)
        if len(parts) >= 2:
            _add(" ".join(parts))

    # Recalled named benchmarks first — highest precision.
    for name in direction.recalled_dataset_names:
        _add_recalled_name(name)
    # HF-uploader-style compound nouns second.
    for kw in direction.dataset_keywords:
        _add_with_shortenings(kw)
    # Academic-style as breadth.
    for kw in direction.keywords:
        _add_with_shortenings(kw)
    if pool:
        return pool[:7]
    # No keywords at all; degrade to direction name.
    return [direction.name.replace("_", " ")]


class HuggingFaceSource:
    """Concrete `Source` plugin for the HuggingFace Hub.

    Holds an `HfApi` client. The token is optional — public datasets are
    accessible anonymously, but rate limits are friendlier with a token.
    """

    name: str = "huggingface"

    def __init__(self, *, token: str | None = None, limit: int = _DEFAULT_LIMIT) -> None:
        # Defer the import so `dataset_scout.sources` is cheap to import
        # even in environments where huggingface_hub isn't installed.
        from huggingface_hub import HfApi

        self._api: HfApi = HfApi(token=token)
        self._limit = limit

    def search(
        self,
        intent: Intent,
        directions: list[DecompositionDirection],
        *,
        budget: Budget,
    ) -> Iterator[Candidate]:
        """Yield candidates from the original Intent plus each decomposition direction.

        The yield order is **round-robin across queries**, not
        sequential. Without round-robin, a high-recall first direction
        (e.g. "prompt injection" returning 50+ datasets) would saturate
        the candidate budget in the pipeline and starve every later
        direction. With round-robin, each query contributes one
        candidate per pass, so a 50-candidate budget split across 1
        Intent query + 6 directions x 3 keywords (~19 queries) lets
        every direction land 2-3 hits.

        Each candidate's `surfaced_by` is set to `[direction.name]` for
        direction-derived hits and `[]` for hits from the original Intent.
        Candidates may appear in multiple direction queries; the pipeline
        is responsible for deduping and merging surfaced_by lists.
        """
        # Materialise all per-query result lists upfront so we can
        # interleave them. This means we issue all the API calls in
        # one batch, then yield round-robin across the results. The
        # alternative (lazy iterators) would require advancing each
        # `list_datasets` call one item at a time, which the HF API
        # doesn't support cleanly.
        per_query_results: list[list[Candidate]] = []

        original_query = _build_search_query(intent)
        if original_query:
            per_query_results.append(list(self._search_one(original_query, surfaced_by=[])))

        for direction in directions:
            for query in _direction_queries(direction):
                if query:
                    per_query_results.append(
                        list(self._search_one(query, surfaced_by=[direction.name]))
                    )

        # Round-robin yield: one candidate from each query per pass.
        idx = 0
        any_left = True
        while any_left:
            any_left = False
            for results in per_query_results:
                if idx < len(results):
                    yield results[idx]
                    any_left = True
            idx += 1

    def _search_one(self, query: str, *, surfaced_by: list[str]) -> Iterator[Candidate]:
        infos = self._api.list_datasets(search=query, limit=self._limit, full=True)
        for info in infos:
            yield Candidate(
                source=self.name,
                id=info.id,
                revision=getattr(info, "sha", None),
                metadata=_build_metadata(info),
                streamable=True,
                surfaced_by=list(surfaced_by),
            )

    def fetch_metadata(self, candidate: Candidate) -> dict[str, Any]:
        """Return the full HF DatasetInfo as a plain dict.

        Used by `inspect` (M3). For M1a, candidates already carry the
        envelope from search(), so this is rarely called.
        """
        info = self._api.dataset_info(candidate.id, revision=candidate.revision)
        return {
            "id": info.id,
            "sha": getattr(info, "sha", None),
            "card_data": _card_data_to_dict(info.card_data),
            "tags": list(getattr(info, "tags", []) or []),
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "gated": getattr(info, "gated", False),
            "private": getattr(info, "private", False),
            "created_at": getattr(info, "created_at", None),
            "last_modified": getattr(info, "last_modified", None),
        }

    def stream_sample(
        self,
        candidate: Candidate,
        n: int,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        """Stream a small sample of rows. Implementation lands in M1b."""
        raise NotImplementedError("HuggingFaceSource.stream_sample lands in M1b")

    def stream_rows(
        self,
        candidate: Candidate,
        *,
        config: str | None = None,
        split: str = "train",
        take: int | None = None,
        seed: int = 42,
    ) -> Iterator[dict[str, Any]]:
        """Stream rows from an HF dataset for full materialisation.

        Uses `datasets.load_dataset(streaming=True)` so memory stays
        flat for arbitrarily large corpora. The `datasets` import is
        lazy: callers that only need search + cheap probes never pay
        for it.

        Many HF datasets require an explicit `config` (subset name) or
        non-default split — both are passed through from the recipe.
        Errors from `datasets` (gated repo, missing config, etc.) are
        propagated; the caller decides how to surface them.
        """
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(
            candidate.id,
            name=config,
            split=split,
            streaming=True,
            revision=candidate.revision,
        )
        for i, row in enumerate(ds):
            if take is not None and i >= take:
                break
            yield dict(row)

    def card_url(self, candidate: Candidate) -> str:
        return candidate.metadata.card_url or _HF_DATASET_URL.format(id=candidate.id)

    def terms_check(self, intent: Intent) -> list[Obligation]:
        """No blanket obligations; per-dataset gating is reflected in
        `CandidateMetadata.gated` and surfaced at curate time."""
        return []


__all__ = ["HuggingFaceSource"]
