"""Academic-paper discovery channel.

Adds a discovery surface beyond the dataset-platform Source plugins:
papers from leading ML / security venues that may *cite* relevant
datasets. The pipeline runs this stage between probes and the strategy
assessor and:

  1. Records the relevant papers (title, abstract, link) in
     `ReconResult.papers` so the report can render a "Related papers
     and dataset citations" section.
  2. Promotes any HuggingFace / Kaggle dataset URL extracted from a
     paper's abstract into the existing candidate pool, with
     `surfaced_by` carrying the paper id. Existing strategy /
     coverage flow then runs over those candidates normally.

Why a stage and not a Source plugin? Papers ≠ datasets. A Source's
contract is to yield Candidates that are datasets; mixing papers in
breaks the strategy assessor (no rows to stream, no transform). Papers
are referrers — different shape, separate render section.

Two backends:

  - **Semantic Scholar Graph API** (free, no auth required for moderate
    use). Default for every query; covers all configured venues with
    abstracts and citation counts. Subject to aggressive rate-limiting
    on the free tier — handled with jittered exponential backoff that
    honors `Retry-After`.
  - **arXiv API** (free, no auth, generous limits). Targeted fallback:
    fired in addition to Semantic Scholar for queries that target
    named benchmarks (proper nouns from a decomposition direction's
    `recalled_dataset_names`). Catches frontier preprints that S2's
    indexing pipeline may not have ingested yet. See
    `arxiv_search.py` for the implementation.

Cross-backend dedupe is keyed on `PaperReference.arxiv_id` when
populated, falling back to the source-native paper id.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    DecompositionDirection,
    ExtractedDataset,
    Intent,
    PaperReference,
)

if TYPE_CHECKING:
    from dataset_scout.cache import Cache


_log = logging.getLogger(__name__)

# Bumped when query construction or extraction policy changes in a way
# that would invalidate cached search results.
PAPER_SEARCH_VERSION = "2"

_S2_BULK_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_S2_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "venue",
        "year",
        "authors",
        "citationCount",
        "url",
        "externalIds",
    ]
)

# Default per-query result cap. The user-facing per-recon cap is much
# smaller; this is just the page size we ask S2 for.
_DEFAULT_LIMIT = 50

# Default ceiling on papers retained per recon, after dedupe across
# queries and venues. Keeps the report scannable.
_DEFAULT_MAX_PAPERS = 20

# Window of recent years to query.
_DEFAULT_YEAR_WINDOW = 4

# Per-direction keyword cap, mirroring the HF / Kaggle plugins.
_KEYWORDS_PER_DIRECTION = 3

# Cap on total named-benchmark queries that fire arXiv alongside S2.
# Each arXiv query costs 3s (rate gate) + up to 8s (timeout) = 11s worst case.
# Capping at 5 keeps worst-case arXiv time under 55s.
_MAX_ARXIV_NAMED_QUERIES = 5

_DEFAULT_TIMEOUT = 15.0


# ─── venue normalisation ────────────────────────────────────────────


# S2 indexes papers under canonical venue strings; passing aliases
# (short and long forms) maximises recall without requiring users to
# know which form S2 uses for a given conference. The order doesn't
# matter — S2's `venue` parameter is a comma-separated OR.
_VENUE_ALIASES: dict[str, list[str]] = {
    "NeurIPS": [
        "NeurIPS",
        "Neural Information Processing Systems",
        "Advances in Neural Information Processing Systems",
    ],
    "ICML": [
        "ICML",
        "International Conference on Machine Learning",
    ],
    "ICLR": [
        "ICLR",
        "International Conference on Learning Representations",
    ],
    "SaTML": [
        "SaTML",
        "IEEE Conference on Secure and Trustworthy Machine Learning",
    ],
    # NLP venues — added because chat / dialogue / counseling / safety
    # corpora frequently land at NLP conferences before (or instead of)
    # ML venues. NeurIPS-only filtering misses MentalChat16K-class work.
    "ACL": [
        "ACL",
        "Annual Meeting of the Association for Computational Linguistics",
    ],
    "EMNLP": [
        "EMNLP",
        "Conference on Empirical Methods in Natural Language Processing",
    ],
    "NAACL": [
        "NAACL",
        "North American Chapter of the Association for Computational Linguistics",
    ],
    # Ethics / fairness / HCI / general-AI venues. AI-safety briefs
    # frequently surface here before (or instead of) the core ML venues:
    # parasocial / companionship / harms-of-AI work especially.
    "FAccT": [
        "FAccT",
        "ACM Conference on Fairness, Accountability, and Transparency",
        "Conference on Fairness, Accountability, and Transparency",
    ],
    "AIES": [
        "AIES",
        "AAAI/ACM Conference on AI, Ethics, and Society",
    ],
    "AAAI": [
        "AAAI",
        "AAAI Conference on Artificial Intelligence",
    ],
    "CHI": [
        "CHI",
        "Conference on Human Factors in Computing Systems",
        "ACM CHI Conference on Human Factors in Computing Systems",
    ],
    "COLM": [
        "COLM",
        "Conference on Language Modeling",
    ],
    # arXiv preprints — much frontier AI-safety work lives here before
    # any peer-reviewed venue. S2 indexes them with venue="arXiv.org".
    "arXiv": ["arXiv.org"],
}


# Reverse map: any alias → canonical short name. Used when an S2
# response carries a long-form venue and we want a tidy display label.
_VENUE_CANONICAL: dict[str, str] = {
    alias.lower(): short for short, aliases in _VENUE_ALIASES.items() for alias in aliases
}


# Special token: when 'all' appears in the venue selection, the venue
# filter is dropped entirely. Lets users cast the widest net (catches
# health journals, ethics workshops, niche conferences) at the cost of
# some lexical-noise risk. The query itself still constrains relevance.
ALL_VENUES_SENTINEL = "all"


DEFAULT_VENUES: tuple[str, ...] = (
    "NeurIPS",
    "ICML",
    "ICLR",
    "SaTML",
    "ACL",
    "EMNLP",
    "NAACL",
    "FAccT",
    "AIES",
    "AAAI",
    "CHI",
    "COLM",
    "arXiv",
)


def canonical_venue(raw: str) -> str:
    """Map a venue string returned by S2 onto our canonical short form.

    Falls back to the raw value when no alias matches.
    """
    return _VENUE_CANONICAL.get(raw.strip().lower(), raw.strip())


def venue_filter_value(venues: Iterable[str]) -> str:
    """Build the comma-separated `venue` parameter for S2's search/bulk.

    Returns an empty string when the special `ALL_VENUES_SENTINEL`
    appears in the input — caller is responsible for omitting the
    `venue` param entirely from the request.
    """
    venues_list = list(venues)
    if any(v.lower() == ALL_VENUES_SENTINEL for v in venues_list):
        return ""
    expanded: list[str] = []
    for v in venues_list:
        aliases = _VENUE_ALIASES.get(v, [v])
        expanded.extend(aliases)
    # De-dupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for a in expanded:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return ",".join(out)


def is_all_venues(venues: Iterable[str]) -> bool:
    """True iff the venue selection is the broaden-everything sentinel."""
    return any(v.lower() == ALL_VENUES_SENTINEL for v in venues)


# ─── extraction ─────────────────────────────────────────────────────


# HF dataset slugs accept word chars, dots, hyphens; ids may be either
# `<namespace>/<slug>` or just `<slug>` for legacy datasets.
_RE_HF_DATASET = re.compile(
    r"https?://huggingface\.co/datasets/(?P<id>[A-Za-z0-9_\-.]+(?:/[A-Za-z0-9_\-.]+)?)",
    re.IGNORECASE,
)
_RE_KAGGLE_DATASET = re.compile(
    r"https?://(?:www\.)?kaggle\.com/datasets/(?P<id>[A-Za-z0-9_\-.]+/[A-Za-z0-9_\-.]+)",
    re.IGNORECASE,
)
# A loose match for GitHub repos sometimes mentioned as data releases.
# Conservative: only match when the URL ends with `/data`, `-data`,
# `-dataset`, or contains the word "dataset" in the path.
_RE_GITHUB_DATA = re.compile(
    r"https?://github\.com/(?P<id>[A-Za-z0-9_\-.]+/[A-Za-z0-9_\-.]+)"
    r"(?P<suffix>(?:[/-]data(?:set)?|/[A-Za-z0-9_\-.]*dataset[A-Za-z0-9_\-.]*)?)",
    re.IGNORECASE,
)


def extract_dataset_references(text: str | None) -> list[ExtractedDataset]:
    """Pull explicit dataset URLs out of free-text (an abstract).

    v1 is regex-only over HF + Kaggle URLs (high precision) plus a
    conservative GitHub data-repo match. LLM-based "what dataset does
    this paper rely on?" extraction is deferred. De-duped by url.
    """
    if not text:
        return []
    out: list[ExtractedDataset] = []
    seen: set[str] = set()

    for m in _RE_HF_DATASET.finditer(text):
        ident = m.group("id").rstrip("/.")
        url = f"https://huggingface.co/datasets/{ident}"
        if url not in seen:
            seen.add(url)
            out.append(
                ExtractedDataset(
                    source="huggingface",
                    identifier=ident,
                    url=url,
                    confidence="explicit_url",
                )
            )
    for m in _RE_KAGGLE_DATASET.finditer(text):
        ident = m.group("id").rstrip("/.")
        url = f"https://www.kaggle.com/datasets/{ident}"
        if url not in seen:
            seen.add(url)
            out.append(
                ExtractedDataset(
                    source="kaggle",
                    identifier=ident,
                    url=url,
                    confidence="explicit_url",
                )
            )
    for m in _RE_GITHUB_DATA.finditer(text):
        suffix = m.group("suffix") or ""
        if not suffix:
            continue  # skip generic GitHub repo refs
        ident = m.group("id").rstrip("/.")
        url = f"https://github.com/{ident}{suffix}"
        if url not in seen:
            seen.add(url)
            out.append(
                ExtractedDataset(
                    source="github",
                    identifier=ident,
                    url=url,
                    confidence="explicit_url",
                )
            )
    return out


# ─── query construction ────────────────────────────────────────────


def _build_intent_query(intent: Intent) -> str:
    """One-shot query string for the original brief.

    Mirrors HuggingFaceSource: prefer threat_families when populated
    (high-precision lexical signal); otherwise fall back to the brief.
    """
    if intent.threat_families:
        return " ".join(f.replace("_", " ") for f in intent.threat_families)
    return intent.raw_brief.strip()


def _direction_queries(direction: DecompositionDirection) -> list[str]:
    """Per-direction S2 queries.

    Pull the LLM-recalled named benchmarks first (proper-noun queries
    like "PersonaChat", "INTIMA" — paper titles often contain these
    verbatim), then the original keywords for breadth. Cap so a single
    direction with many recalled names doesn't dominate the per-query
    budget across all directions.
    """
    seen: set[str] = set()
    out: list[str] = []
    for name in direction.recalled_dataset_names:
        norm = name.strip()
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
        if len(out) >= _KEYWORDS_PER_DIRECTION:
            break
    for kw in direction.keywords:
        norm = kw.strip()
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
        if len(out) >= _KEYWORDS_PER_DIRECTION:
            break
    if out:
        return out
    return [direction.name.replace("_", " ")]


def default_year_range(window: int = _DEFAULT_YEAR_WINDOW) -> tuple[int, int]:
    """Last `window` years, inclusive of the current year."""
    now = datetime.now(UTC).year
    return now - window + 1, now


# ─── public entry point ────────────────────────────────────────────


def find_papers_and_promote(
    intent: Intent,
    directions: list[DecompositionDirection],
    *,
    venues: Iterable[str] = DEFAULT_VENUES,
    year_range: tuple[int, int] | None = None,
    max_papers: int = _DEFAULT_MAX_PAPERS,
    cache: Cache | None = None,
    client: httpx.Client | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT,
) -> tuple[list[PaperReference], list[Candidate]]:
    """Search target venues for relevant papers; extract dataset citations.

    Returns ``(papers, promoted_candidates)``:

    - ``papers`` — list of `PaperReference` capped at ``max_papers``,
      deduped by paper_id (and by arxiv_id across backends), in
      original-query order (round-robin across brief + per-direction
      queries) so every direction contributes.
    - ``promoted_candidates`` — `Candidate` objects derived from
      explicit HF / Kaggle URLs in the abstracts. The pipeline merges
      these into the existing pool by (source, id) — see
      `pipeline._merge_or_register` — so paper provenance shows up as
      `surfaced_by` on whichever candidate already carries the data.

    For queries that target named benchmarks (proper nouns from a
    direction's ``recalled_dataset_names``), arXiv runs alongside
    Semantic Scholar to catch frontier preprints S2 hasn't indexed
    yet. See `arxiv_search.search_arxiv`.

    Failures (network, parse, persistent rate limits) log a warning
    and return ``([], [])`` — recon must not be blocked by
    paper-discovery infrastructure.
    """
    # Local import keeps the module-import graph free of a cycle:
    # arxiv_search imports the shared retry helper from this module.
    from dataset_scout.arxiv_search import search_arxiv

    yr = year_range or default_year_range()
    queries = _enumerate_queries(intent, directions)
    if not queries:
        return [], []

    own_client = False
    if client is None:
        client = httpx.Client(timeout=timeout_s)
        own_client = True
    try:
        per_query: list[list[PaperReference]] = []
        for surfaced_by, q, is_named in queries:
            s2_results = _fetch_one(
                q,
                venues=venues,
                year_range=yr,
                surfaced_by=surfaced_by,
                client=client,
                cache=cache,
                timeout_s=timeout_s,
            )
            if is_named:
                arxiv_results = search_arxiv(
                    q,
                    year_range=yr,
                    surfaced_by=surfaced_by,
                    client=client,
                    cache=cache,
                    timeout_s=timeout_s,
                )
            else:
                arxiv_results = []
            per_query.append(s2_results + arxiv_results)
    finally:
        if own_client:
            client.close()

    # Round-robin merge so every query / direction contributes a
    # paper before any single query saturates the budget.
    deduped = _round_robin_dedupe(per_query, cap=max_papers)

    # Promote extracted HF/Kaggle datasets into Candidates.
    candidates = _promote_datasets(deduped)
    return deduped, candidates


def _enumerate_queries(
    intent: Intent, directions: list[DecompositionDirection]
) -> list[tuple[list[str], str, bool]]:
    """Build the (surfaced_by, query, is_named_benchmark) tuples.

    `is_named_benchmark` is True when the query string came verbatim
    from `direction.recalled_dataset_names` (a proper-noun benchmark
    name like INTIMA or AnthroBench). These queries also fire arXiv
    in addition to S2: preprint indexing matters most for named
    benchmarks where S2 may lag arXiv by days or weeks.

    Named-benchmark queries are capped at `_MAX_ARXIV_NAMED_QUERIES`
    to bound arXiv time (each costs 3s rate gate + timeout).
    """
    out: list[tuple[list[str], str, bool]] = []
    named_count = 0
    intent_q = _build_intent_query(intent)
    if intent_q:
        out.append(([], intent_q, False))
    for d in directions:
        recalled_set = {n.strip().lower() for n in d.recalled_dataset_names if n.strip()}
        for q in _direction_queries(d):
            if not q:
                continue
            is_named = q.strip().lower() in recalled_set
            if is_named and named_count >= _MAX_ARXIV_NAMED_QUERIES:
                is_named = False
            if is_named:
                named_count += 1
            out.append(([d.name], q, is_named))
    return out


def _round_robin_dedupe(
    per_query: list[list[PaperReference]],
    *,
    cap: int,
) -> list[PaperReference]:
    """Interleave per-query results, deduping across queries and backends.

    Primary key is `paper_id` (which differs across S2 and arXiv).
    A second key on `arxiv_id` collapses cross-backend duplicates so
    a paper that surfaces from both S2 and arXiv counts once. When
    the same paper appears in multiple queries, surfaced_by is merged.
    """
    by_paper_id: dict[str, PaperReference] = {}
    by_arxiv_id: dict[str, str] = {}  # arxiv_id -> primary paper_id
    out: list[PaperReference] = []
    idx = 0
    any_left = True
    while any_left and len(out) < cap:
        any_left = False
        for results in per_query:
            if idx >= len(results):
                continue
            any_left = True
            p = results[idx]
            primary_id: str | None = None
            if p.arxiv_id and p.arxiv_id in by_arxiv_id:
                primary_id = by_arxiv_id[p.arxiv_id]
            elif p.paper_id in by_paper_id:
                primary_id = p.paper_id
            if primary_id is None:
                by_paper_id[p.paper_id] = p
                if p.arxiv_id:
                    by_arxiv_id[p.arxiv_id] = p.paper_id
                out.append(p)
                if len(out) >= cap:
                    break
                continue
            existing = by_paper_id[primary_id]
            merged = list(existing.surfaced_by)
            for s in p.surfaced_by:
                if s not in merged:
                    merged.append(s)
            if merged != existing.surfaced_by:
                # Pydantic frozen check: ReconResult papers list
                # holds non-frozen models, so direct mutation works.
                existing.surfaced_by = merged
            # Adopt new dataset references from the duplicate so a
            # paper found via both backends keeps the union of
            # promotable URLs.
            existing_urls = {d.url for d in existing.referenced_datasets}
            for d in p.referenced_datasets:
                if d.url not in existing_urls:
                    existing.referenced_datasets.append(d)
                    existing_urls.add(d.url)
        idx += 1
    return out


def _promote_datasets(papers: list[PaperReference]) -> list[Candidate]:
    """Convert HF / Kaggle dataset citations into Candidates with paper provenance.

    GitHub citations are kept on the paper but not promoted to candidates
    — the dataset-scout candidate pool is keyed on plugin sources and
    GitHub isn't one. The paper still carries the GitHub link for the
    reader.
    """
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for p in papers:
        for d in p.referenced_datasets:
            if d.source not in ("huggingface", "kaggle"):
                continue
            key = (d.source, d.identifier)
            if key in seen:
                continue
            seen.add(key)
            tag = f"paper:{canonical_venue(p.venue)}-{p.year}-{p.paper_id}"
            candidates.append(
                Candidate(
                    source=d.source,
                    id=d.identifier,
                    revision=None,
                    metadata=CandidateMetadata(
                        description=p.title,
                        card_url=d.url,
                    ),
                    streamable=(d.source == "huggingface"),
                    surfaced_by=[tag],
                )
            )
    return candidates


# ─── HTTP fetch + cache ────────────────────────────────────────────


def _cache_key(query: str, venues: Iterable[str], year_range: tuple[int, int]) -> str:
    import hashlib

    canonical = (
        f"{PAPER_SEARCH_VERSION}\n"
        f"{','.join(sorted(set(venues)))}\n"
        f"{year_range[0]}-{year_range[1]}\n"
        f"{query}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fetch_one(
    query: str,
    *,
    venues: Iterable[str],
    year_range: tuple[int, int],
    surfaced_by: list[str],
    client: httpx.Client,
    cache: Cache | None,
    timeout_s: float,
) -> list[PaperReference]:
    """Fetch one query against S2 (with cache); return PaperReferences.

    Hits the cache first. On miss, calls the API; on success, caches a
    JSON-serialised list. Any failure returns [] and logs.
    """
    venues_list = list(venues)
    key = _cache_key(query, venues_list, year_range)
    if cache is not None:
        cached = cache.get_json("papers", key)
        if isinstance(cached, list):
            try:
                return [_model_validate_with_surfaced_by(item, surfaced_by) for item in cached]
            except Exception:  # pragma: no cover - defensive
                pass

    params: dict[str, str] = {
        "query": query,
        "year": f"{year_range[0]}-{year_range[1]}",
        "fields": _S2_FIELDS,
        "limit": str(_DEFAULT_LIMIT),
    }
    # Omit the `venue` param entirely when the user opts into "all":
    # S2 then searches every indexed venue, including health journals
    # and niche workshops where AI-safety-adjacent work occasionally
    # lives. The query alone constrains relevance.
    if not is_all_venues(venues_list):
        venue_param = venue_filter_value(venues_list)
        if venue_param:
            params["venue"] = venue_param
    response = http_get_with_retry(
        client,
        _S2_BULK_SEARCH,
        params=params,
        timeout_s=timeout_s,
        label=f"semantic-scholar query={query!r}",
    )
    if response is None:
        return []
    try:
        payload: Any = response.json()
    except ValueError as exc:
        _log.warning("paper search parse failed for query=%r: %s", query, exc)
        return []

    raw_data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(raw_data, list):
        return []

    records: list[dict[str, Any]] = []
    parsed: list[PaperReference] = []
    for raw in raw_data:
        if not isinstance(raw, dict):
            continue
        paper = _normalise_s2_paper(raw)
        if paper is None:
            continue
        # Persist a surfaced_by-stripped variant in cache so the same
        # cached result can be re-attributed across different runs.
        records.append(_to_cacheable(paper))
        # The per-call return carries this query's surfaced_by tag.
        attributed = paper.model_copy(update={"surfaced_by": list(surfaced_by)})
        parsed.append(attributed)

    if cache is not None and records:
        cache.set_json("papers", key, records)
    return parsed


def _to_cacheable(paper: PaperReference) -> dict[str, Any]:
    """Drop surfaced_by before caching — it depends on the calling query."""
    payload = paper.model_dump(mode="json")
    payload["surfaced_by"] = []
    return payload


def _model_validate_with_surfaced_by(
    payload: dict[str, Any], surfaced_by: list[str]
) -> PaperReference:
    """Rehydrate a cached record and overlay this call's surfaced_by tag."""
    paper = PaperReference.model_validate(payload)
    return paper.model_copy(update={"surfaced_by": list(surfaced_by)})


def _normalise_s2_paper(raw: dict[str, Any]) -> PaperReference | None:
    """Translate an S2 paper payload into our typed PaperReference.

    Returns None when essential fields are missing (no title, no
    paper_id, or no year). S2 occasionally returns sparse records.
    """
    paper_id = raw.get("paperId")
    title = raw.get("title")
    year = raw.get("year")
    venue = raw.get("venue") or ""
    if not isinstance(paper_id, str) or not isinstance(title, str) or not isinstance(year, int):
        return None

    abstract = raw.get("abstract") if isinstance(raw.get("abstract"), str) else None
    citation_count = raw.get("citationCount") if isinstance(raw.get("citationCount"), int) else None

    authors_raw = raw.get("authors") or []
    authors: list[str] = []
    if isinstance(authors_raw, list):
        for a in authors_raw:
            if isinstance(a, dict) and isinstance(a.get("name"), str):
                authors.append(a["name"])

    url = _resolve_paper_url(raw, paper_id)
    references = extract_dataset_references(abstract)
    arxiv_id = _extract_s2_arxiv_id(raw)

    return PaperReference(
        paper_id=paper_id,
        title=title,
        authors=authors,
        venue=canonical_venue(venue) if venue else "",
        year=year,
        url=url,
        abstract=abstract,
        citation_count=citation_count,
        referenced_datasets=references,
        surfaced_by=[],  # filled in by caller
        arxiv_id=arxiv_id,
    )


def _extract_s2_arxiv_id(raw: dict[str, Any]) -> str | None:
    """Pull S2's `externalIds.ArXiv` for cross-backend dedupe.

    Normalises by stripping any version suffix (`v1`, `v2`, ...) so a
    paper that arXiv returns as `2401.12345v3` matches what S2 stored
    as `2401.12345`.
    """
    ext = raw.get("externalIds")
    if not isinstance(ext, dict):
        return None
    val = ext.get("ArXiv")
    if not isinstance(val, str) or not val.strip():
        return None
    return _normalise_arxiv_id(val.strip())


def _normalise_arxiv_id(value: str) -> str:
    """Strip a trailing `vN` version suffix from an arXiv ID, if any."""
    return re.sub(r"v\d+$", "", value, flags=re.IGNORECASE)


def _resolve_paper_url(raw: dict[str, Any], paper_id: str) -> str:
    """Prefer the externally-resolvable URL when S2 provides one."""
    url = raw.get("url")
    if isinstance(url, str) and url:
        return url
    ext = raw.get("externalIds")
    if isinstance(ext, dict):
        if isinstance(ext.get("ArXiv"), str):
            return f"https://arxiv.org/abs/{ext['ArXiv']}"
        if isinstance(ext.get("DOI"), str):
            return f"https://doi.org/{ext['DOI']}"
    return f"https://www.semanticscholar.org/paper/{paper_id}"


# ─── shared HTTP retry helper ──────────────────────────────────────


# Tunables for the retry helper. Caller can override per-call. These
# defaults survived a real-world six-axis parallel recon that hit S2
# with sustained 429 floods — recon completed without losing the
# paper channel.
_RETRY_MAX_ATTEMPTS = 4
_RETRY_BASE_WAIT = 2.0
_RETRY_MAX_WAIT = 30.0


def http_get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout_s: float,
    label: str,
    max_attempts: int = _RETRY_MAX_ATTEMPTS,
    base_wait: float = _RETRY_BASE_WAIT,
    max_wait: float = _RETRY_MAX_WAIT,
) -> httpx.Response | None:
    """GET with jittered exponential backoff + Retry-After support.

    Used by both paper-search backends (Semantic Scholar, arXiv) so
    they share rate-limit handling. Returns the response on 2xx, or
    `None` if all attempts are exhausted or the request raised. Logs
    a single warning on persistent failure — paper search is
    non-blocking for recon so we don't surface to the caller.

    Backoff per 429:
      - Honor `Retry-After` header when present (seconds form).
      - Otherwise wait `base_wait * 2^attempt + uniform(0,1)` seconds,
        capped at `max_wait`.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = client.get(url, params=params, timeout=timeout_s)
        except httpx.HTTPError as exc:
            last_exc = exc
            break
        if response.status_code == 429:
            if attempt == max_attempts - 1:
                _log.warning(
                    "%s rate-limited (429) after %d attempts; giving up",
                    label,
                    max_attempts,
                )
                return None
            wait = _retry_after_seconds(response.headers.get("Retry-After"))
            if wait is None:
                wait = base_wait * (2**attempt) + random.uniform(0, 1)
            wait = min(wait, max_wait)
            time.sleep(wait)
            continue
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            last_exc = exc
            break
        return response
    if last_exc is not None:
        _log.warning("%s failed: %s", label, last_exc)
    return None


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header's seconds form. None when absent or HTTP-date.

    Per RFC 9110, Retry-After can be either a delay-seconds integer or
    an HTTP-date. We support the seconds form (what S2 sends in
    practice) and fall through to our exponential backoff for dates.
    """
    if not value:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


__all__ = [
    "DEFAULT_VENUES",
    "PAPER_SEARCH_VERSION",
    "canonical_venue",
    "default_year_range",
    "extract_dataset_references",
    "find_papers_and_promote",
    "http_get_with_retry",
    "venue_filter_value",
]
