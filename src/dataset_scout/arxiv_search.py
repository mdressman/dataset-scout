"""arXiv API as a targeted-fallback paper source.

Runs alongside Semantic Scholar (`paper_search.py`) for queries that
target named benchmarks — proper nouns from a decomposition direction's
``recalled_dataset_names``. The motivation: Semantic Scholar has a
multi-day-to-multi-week indexing lag for fresh arXiv preprints, and
the free-tier rate-limit gets hostile under parallel recons. arXiv's
public API is free, unauthenticated, and has more permissive limits;
hitting it directly catches frontier work S2 hasn't ingested yet.

Only fired for named-benchmark queries because:

  - arXiv's query syntax is stricter than S2's bulk search; broad
    keyword queries return noisy results.
  - Proper nouns (PersonaChat, INTIMA, AnthroBench, ELEPHANT) appear
    verbatim in paper titles/abstracts — ideal for arXiv's search.
  - arXiv's 3-second-per-request guideline means we can't afford to
    fire it for every query in a recon.

Cross-backend dedupe with S2 is handled in
``paper_search._round_robin_dedupe`` via ``PaperReference.arxiv_id``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import httpx

from dataset_scout.core import PaperReference
from dataset_scout.paper_search import (
    PAPER_SEARCH_VERSION,
    _normalise_arxiv_id,
    extract_dataset_references,
    http_get_with_retry,
)

if TYPE_CHECKING:
    from dataset_scout.cache import Cache


_log = logging.getLogger(__name__)


# ─── HTTP / parsing constants ──────────────────────────────────────


_ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"

# Atom 1.0 + arXiv extension namespaces used by `query` responses.
_ATOM_NS = "http://www.w3.org/2005/Atom"
_NS = {"atom": _ATOM_NS}

_DEFAULT_LIMIT = 20
_DEFAULT_TIMEOUT = 8.0

# arXiv's API guidance: "no more than one request every 3 seconds".
# We enforce a global gate (module-level lock) so concurrent recons
# in the same process don't violate the limit.
_RATE_LIMIT_SECONDS = 3.0
_RATE_LOCK = threading.Lock()
_LAST_CALL_AT: list[float] = [0.0]


# Pull the canonical arXiv ID out of an Atom `<id>` URL like
# `http://arxiv.org/abs/2401.12345v3`. Strip any version suffix.
_ARXIV_ID_FROM_URL = re.compile(
    r"arxiv\.org/abs/(?P<id>[A-Za-z0-9._/-]+?)(?:v\d+)?(?:[?#].*)?$",
    re.IGNORECASE,
)


# ─── public entry point ────────────────────────────────────────────


def search_arxiv(
    query: str,
    *,
    year_range: tuple[int, int],
    surfaced_by: list[str],
    client: httpx.Client,
    cache: Cache | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT,
    max_results: int = _DEFAULT_LIMIT,
) -> list[PaperReference]:
    """Search arXiv for one query; return PaperReferences within year range.

    Cached per (query, year_range, version) in the ``papers`` namespace
    (14-day TTL — see ``cache._NAMESPACE_TTL_DEFAULTS``). Failures
    return ``[]`` and log a single warning so recon stays unblocked.
    Empty results are negative-cached too — repeated query expansions
    that miss shouldn't burn the rate-limit budget.
    """
    if not query.strip():
        return []
    key = _cache_key(query, year_range)
    if cache is not None:
        cached = cache.get_json("papers", key)
        if isinstance(cached, list):
            try:
                return [
                    PaperReference.model_validate(item).model_copy(
                        update={"surfaced_by": list(surfaced_by)}
                    )
                    for item in cached
                ]
            except Exception:  # pragma: no cover - defensive
                pass

    _arxiv_rate_gate()
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = http_get_with_retry(
        client,
        _ARXIV_QUERY_URL,
        params=params,
        timeout_s=timeout_s,
        label=f"arxiv search query={query!r}",
    )
    if response is None:
        return []

    try:
        papers = _parse_atom(response.text, year_range)
    except ET.ParseError as exc:
        _log.warning("arxiv parse failed for query=%r: %s", query, exc)
        return []

    if cache is not None:
        # Negative-cache empty results too — saves the 3-second gate
        # next time the same query expansion comes through.
        cache.set_json("papers", key, [_to_cacheable(p) for p in papers])
    return [p.model_copy(update={"surfaced_by": list(surfaced_by)}) for p in papers]


# ─── internals ─────────────────────────────────────────────────────


def _arxiv_rate_gate() -> None:
    """Sleep until at least `_RATE_LIMIT_SECONDS` have passed since the last call."""
    with _RATE_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_CALL_AT[0]
        if elapsed < _RATE_LIMIT_SECONDS:
            time.sleep(_RATE_LIMIT_SECONDS - elapsed)
        _LAST_CALL_AT[0] = time.monotonic()


def _cache_key(query: str, year_range: tuple[int, int]) -> str:
    canonical = f"arxiv:{PAPER_SEARCH_VERSION}\n{year_range[0]}-{year_range[1]}\n{query}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _to_cacheable(paper: PaperReference) -> dict[str, Any]:
    """Strip surfaced_by before caching — it depends on the calling query."""
    payload = paper.model_dump(mode="json")
    payload["surfaced_by"] = []
    return payload


def _parse_atom(xml_text: str, year_range: tuple[int, int]) -> list[PaperReference]:
    """Parse arXiv's Atom response into PaperReferences within year range.

    Strips XML namespaces (atom, arxiv) and applies the year filter
    client-side via the ``<published>`` date. Skips entries missing
    essentials (title, id, year).
    """
    root = ET.fromstring(xml_text)
    out: list[PaperReference] = []
    for entry in root.findall("atom:entry", _NS):
        paper = _parse_entry(entry)
        if paper is None:
            continue
        if not (year_range[0] <= paper.year <= year_range[1]):
            continue
        out.append(paper)
    return out


def _parse_entry(entry: ET.Element) -> PaperReference | None:
    id_elem = entry.find("atom:id", _NS)
    title_elem = entry.find("atom:title", _NS)
    summary_elem = entry.find("atom:summary", _NS)
    published_elem = entry.find("atom:published", _NS)
    if id_elem is None or title_elem is None or published_elem is None:
        return None

    id_text = (id_elem.text or "").strip()
    if not id_text:
        return None
    arxiv_id = _extract_arxiv_id_from_url(id_text)
    if arxiv_id is None:
        return None

    title = _normalise_text(title_elem.text)
    if not title:
        return None
    abstract = _normalise_text(summary_elem.text) if summary_elem is not None else None

    try:
        year = int((published_elem.text or "")[:4])
    except (ValueError, TypeError):
        return None

    authors: list[str] = []
    for a in entry.findall("atom:author/atom:name", _NS):
        name = _normalise_text(a.text)
        if name:
            authors.append(name)

    references = extract_dataset_references(abstract)

    return PaperReference(
        # `arxiv:` prefix on paper_id keeps it distinct from S2's
        # opaque paperIds even when arxiv_id is also stored. The
        # round-robin dedupe collapses by arxiv_id when S2 also
        # carries it; this prefix is just for human-readable logs.
        paper_id=f"arxiv:{arxiv_id}",
        title=title,
        authors=authors,
        venue="arXiv",
        year=year,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        abstract=abstract,
        citation_count=None,
        referenced_datasets=references,
        surfaced_by=[],  # filled in by caller
        arxiv_id=arxiv_id,
    )


def _extract_arxiv_id_from_url(id_text: str) -> str | None:
    m = _ARXIV_ID_FROM_URL.search(id_text)
    if not m:
        return None
    return _normalise_arxiv_id(m.group("id"))


def _normalise_text(text: str | None) -> str:
    """Collapse whitespace / strip — Atom titles+summaries hard-wrap with newlines."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


__all__ = ["search_arxiv"]
