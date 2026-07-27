"""dataset-scout CLI — thin shell over the library.

Two equivalent entry points: `dataset-scout` and `datascout`. Same
command tree (see [docs/cli.md]). M0 ships the verb skeletons; later
milestones light up real work behind each.
"""

from __future__ import annotations

# ─── early environment hygiene ─────────────────────────────────────
# Set BEFORE any huggingface_hub / datasets import (which happens lazily
# inside command implementations). These are quality-of-life muting of
# warnings that don't carry actionable signal in our use-case:
#   * HF symlink-unsupported warning is harmless; on Windows it's emitted
#     once per dataset cache, flooding logs.
#   * HF telemetry is opt-out by policy here — no need to advertise.
#   * The HF "unauthenticated requests" warning is real signal but already
#     surfaced at the report layer; muting the per-call repetition.
# Users who want the warnings back can set the env vars to "0" explicitly
# (we use setdefault so we never override).
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=r".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=r".*Repo card metadata block was not found.*")

import sys  # noqa: E402  (intentional: env setdefault above must happen first)
from pathlib import Path  # noqa: E402
from typing import Annotated, Any  # noqa: E402

import click  # noqa: E402
import typer  # noqa: E402
from rich.console import Console  # noqa: E402

from dataset_scout import __version__  # noqa: E402

app = typer.Typer(
    name="dataset-scout",
    help="Reconnaissance, reframing, and curation of public datasets for AI detection engineers.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

err = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"dataset-scout {__version__}")
        raise typer.Exit()


def _load_dotenv() -> Path | None:
    """Load `.env` from the CWD if present. Returns the path loaded, or None.

    Uses `override=False` so explicitly-set environment variables always
    win over `.env` — explicit shell config trumps dotfiles.
    """
    from dotenv import load_dotenv

    candidate = Path.cwd() / ".env"
    if candidate.is_file():
        load_dotenv(candidate, override=False)
        return candidate
    return None


def _interactive_review_decomposition(
    *,
    brief: str,
    intent: Any,
    ctx: Any,
    out: Path,
) -> list[Any] | None:
    """Pause after decomposition; show directions; let user approve/edit/abort.

    Returns the (possibly edited) direction list, or None if the user
    aborted. The decomposition is written to `<out>/decomposition.yaml`
    so a subsequent run can resume via `--decomposition-from`.

    Cross-platform editor handling via `click.edit()` (which honours
    $EDITOR / $VISUAL or falls back to a sensible per-OS default).
    """
    from dataset_scout.decompose import decompose_intent
    from dataset_scout.decomposition_io import load_decomposition, write_decomposition
    from dataset_scout.errors import LLMError

    out.mkdir(parents=True, exist_ok=True)

    # Run decompose. Failures fall through to no-review path: print the
    # error and return None so caller can abort.
    err.print("[dim]Running brief decomposition…[/dim]")
    try:
        directions = decompose_intent(intent, ctx=ctx)
    except LLMError as exc:
        err.print(f"[yellow]Decomposition failed: {exc}[/yellow]")
        err.print("[yellow]Cannot enter --review mode without an LLM. Aborting.[/yellow]")
        return None

    if not directions:
        err.print("[yellow]Decomposer returned 0 directions.[/yellow]")
        return None

    # Write to canonical location so the user can edit it.
    yaml_path = write_decomposition(directions, out)
    if yaml_path is None:
        err.print("[yellow]Could not write decomposition.yaml; aborting review.[/yellow]")
        return None

    while True:
        _print_directions_for_review(directions)
        err.print(
            f"\n[bold]decomposition.yaml:[/bold] {yaml_path}\n"
            "[bold]Press Enter to continue · [e] to edit in $EDITOR · [a] to abort[/bold]"
        )
        choice = typer.prompt(">", default="", show_default=False).strip().lower()
        if choice in {"", "c", "continue", "y", "yes"}:
            return directions
        if choice in {"a", "abort", "n", "no", "q", "quit"}:
            err.print(
                f"[yellow]Aborted.[/yellow] Decomposition saved to {yaml_path}; "
                "resume later with `datascout recon ... --decomposition-from "
                f"{yaml_path}`."
            )
            return None
        if choice in {"e", "edit"}:
            edited_text = click.edit(yaml_path.read_text(encoding="utf-8"))
            if edited_text is None:
                # Editor was closed without saving (or unavailable).
                err.print("[yellow]No changes detected; using original directions.[/yellow]")
                return directions
            yaml_path.write_text(edited_text, encoding="utf-8")
            try:
                directions = load_decomposition(yaml_path)
            except Exception as exc:
                err.print(
                    f"[red]Edited YAML failed to parse:[/red] {exc}\n"
                    "Fix the file (it's still on disk) and choose [e] again."
                )
                continue
            err.print(
                f"[green]✓ Reloaded {len(directions)} direction(s).[/green]"
            )
            # Show the updated set and prompt again.
            continue
        err.print(f"[yellow]Unrecognised choice: {choice!r}. Try again.[/yellow]")


def _print_directions_for_review(directions: list[Any]) -> None:
    """Render the decomposition for terminal review. Compact, scannable."""
    err.print("\n[bold]Proposed search directions[/bold]\n")
    for i, d in enumerate(directions, start=1):
        err.print(f"  [bold cyan]{i}. {d.name}[/bold cyan]")
        rationale = d.rationale.strip()
        if len(rationale) > 200:
            rationale = rationale[:197] + "…"
        err.print(f"     {rationale}")
        if d.keywords:
            err.print(f"     [dim]keywords:[/dim] {', '.join(d.keywords)}")
        if d.dataset_keywords:
            err.print(f"     [dim]dataset terms:[/dim] {', '.join(d.dataset_keywords[:6])}")
        err.print("")


@app.callback()
def _root(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """dataset-scout — find and curate public datasets for detection work."""
    import contextlib

    # Make stdout/stderr UTF-8 so emoji/unicode in reports render on
    # Windows consoles (cp1252 by default).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8")
    _load_dotenv()


# ─── recon ──────────────────────────────────────────────────────────


@app.command(help="Find candidates and assess how each could be used.")
def recon(
    brief: Annotated[
        str, typer.Argument(help="Natural-language brief describing the detection target.")
    ],
    detection_target: Annotated[
        str | None,
        typer.Option("--detection-target", help="Override the parsed detection target."),
    ] = None,
    deployment_context: Annotated[str | None, typer.Option("--deployment-context")] = None,
    language: Annotated[
        list[str] | None,
        typer.Option("--language", help="Languages to require (repeatable)."),
    ] = None,
    license: Annotated[
        list[str] | None,
        typer.Option("--license", help="License allowlist (repeatable)."),
    ] = None,
    min_strategy_confidence: Annotated[
        float, typer.Option("--min-strategy-confidence", min=0.0, max=1.0)
    ] = 0.5,
    decomposition_from: Annotated[
        Path | None,
        typer.Option(
            "--decomposition-from",
            help=(
                "Reuse a hand-edited decomposition.yaml instead of paying for "
                "a fresh LLM decompose call. Lets you iterate on directions "
                "without re-running the full decompose step."
            ),
        ),
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("datascout-out"),
    # Hidden expert escape hatches — not in --help. Useful for debugging
    # and forward compatibility; production users should let the brief +
    # decomposer carry the signal.
    threat_families: Annotated[
        list[str] | None,
        typer.Option("--threat-families", hidden=True),
    ] = None,
    no_explore: Annotated[
        bool,
        typer.Option("--no-explore", hidden=True, help="Skip LLM decomposition (debug)."),
    ] = False,
    no_papers: Annotated[
        bool,
        typer.Option(
            "--no-papers",
            help="Skip the academic-paper discovery stage entirely.",
        ),
    ] = False,
    venues: Annotated[
        str | None,
        typer.Option(
            "--venues",
            help=(
                "Comma-separated venue list for paper discovery. "
                "Defaults to a curated set of 13 ML / NLP / ethics / HCI venues "
                "(NeurIPS, ICML, ICLR, SaTML, ACL, EMNLP, NAACL, FAccT, AIES, "
                "AAAI, CHI, COLM, arXiv). Use `--venues all` to drop the venue "
                "filter entirely (catches health journals + niche workshops at "
                "the cost of more noise). Use `--venues NeurIPS,ICML` to narrow."
            ),
        ),
    ] = None,
    no_arxiv: Annotated[
        bool,
        typer.Option(
            "--no-arxiv",
            help="Exclude arXiv preprints from paper discovery (peer-reviewed only).",
        ),
    ] = False,
    review: Annotated[
        bool,
        typer.Option(
            "--review",
            "-r",
            help=(
                "Pause after decomposition to review or edit the search "
                "directions before paying for the full recon. Opens "
                "decomposition.yaml in $EDITOR; press Enter to continue, "
                "[e] to edit, [a] to abort. The 5-second human override "
                "is often the difference between finding the right "
                "datasets and missing them."
            ),
        ),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=(
                "Override the LLM model id (litellm-style). Examples: "
                "'github_copilot/gpt-5-mini' (uses your GitHub Copilot "
                "subscription via OAuth device-code flow on first run), "
                "'github/gpt-4o-mini' (uses GITHUB_TOKEN against the free "
                "GitHub Models tier), 'openai/gpt-4o' (uses OPENAI_API_KEY), "
                "'azure/<deployment>' (uses Entra). Falls back to "
                "$DATASET_SCOUT_MODEL, then to the AZURE_OPENAI_* config. "
                "When unset and no Azure config is present, recon runs in "
                "metadata-only mode (no decompose / strategy / coverage)."
            ),
        ),
    ] = None,
) -> None:
    from dataset_scout.context import ScoutContext
    from dataset_scout.decomposition_io import load_decomposition, write_decomposition
    from dataset_scout.errors import DatasetScoutError
    from dataset_scout.pipeline import run_recon
    from dataset_scout.recipe_draft import write_recipe_draft
    from dataset_scout.render import (
        write_recon_report,
        write_recon_report_html,
        write_results_json,
    )

    overrides: dict[str, object] = {"min_strategy_confidence": min_strategy_confidence}
    if detection_target:
        overrides["detection_target"] = detection_target
    if deployment_context:
        overrides["deployment_context"] = deployment_context
    if language:
        overrides["language"] = list(language)
    if license:
        overrides["license"] = list(license)
    if threat_families:
        overrides["threat_families"] = list(threat_families)

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    if model:
        # CLI flag wins. ScoutContext is frozen Pydantic so we copy.
        ctx = ctx.model_copy(update={"model": model})

    directions_override = None
    if decomposition_from is not None:
        try:
            directions_override = load_decomposition(decomposition_from)
        except Exception as e:
            err.print(f"[red]error:[/red] failed to load decomposition: {e}")
            raise typer.Exit(code=1) from e

    # Human-in-the-loop review: run decompose now, show the directions,
    # let the user edit / approve, then pass the result as a
    # directions_override so run_recon skips its own decompose call.
    # Skipped when --no-explore or --decomposition-from is set, since
    # both already provide the directions explicitly.
    if review and not no_explore and directions_override is None:
        from dataset_scout.intent import HeuristicIntentParser

        intent = HeuristicIntentParser().parse(brief, **overrides)
        directions_override = _interactive_review_decomposition(
            brief=brief, intent=intent, ctx=ctx, out=out
        )
        if directions_override is None:
            # User aborted.
            raise typer.Exit(code=130)

    # Build the paper-search callable with the user's venue selection
    # baked in (or False to disable entirely).
    paper_search_fn: object
    if no_papers:
        paper_search_fn = False
    elif venues is None and not no_arxiv:
        paper_search_fn = None  # default behaviour
    else:
        from dataset_scout.paper_search import (
            DEFAULT_VENUES,
            find_papers_and_promote,
        )

        if venues:
            selected_venues = tuple(v.strip() for v in venues.split(",") if v.strip())
        else:
            selected_venues = DEFAULT_VENUES
        if no_arxiv:
            selected_venues = tuple(v for v in selected_venues if v.lower() != "arxiv")
        if not selected_venues:
            err.print(
                "[yellow]--venues / --no-arxiv combination produced an empty venue "
                "list; skipping paper discovery.[/yellow]"
            )
            paper_search_fn = False
        else:
            def _ps(intent: Any, directions: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("venues", selected_venues)
                return find_papers_and_promote(intent, directions, **kwargs)

            paper_search_fn = _ps

    try:
        result = run_recon(
            brief,
            ctx=ctx,
            parser_overrides=overrides,
            explore=not no_explore,
            directions_override=directions_override,
            paper_search_fn=paper_search_fn,
        )
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    json_path = write_results_json(result, out)
    md_path = write_recon_report(result, out)
    html_path = write_recon_report_html(result, out)
    recipe_path = write_recipe_draft(result, out)
    decomposition_path = (
        write_decomposition(result.coverage.decomposition, out)
        if result.coverage and result.coverage.decomposition
        else None
    )

    err.print(
        f"[green]✔[/green] {len(result.candidates)} candidate(s) "
        f"from {', '.join(result.sources_searched) or '(no source)'} "
        f"in {result.elapsed_seconds:.2f}s"
    )
    err.print(f"  - results:       {json_path}")
    err.print(f"  - report:        {md_path}")
    err.print(f"  - report (html): {html_path}")
    if decomposition_path is not None:
        err.print(f"  - decomposition: {decomposition_path}")
    if recipe_path is not None:
        err.print(f"  - recipe:        {recipe_path}")
    if result.notices:
        for n in result.notices:
            err.print(f"  [yellow]![/yellow] {n}")


# ─── decompose ──────────────────────────────────────────────────────


@app.command(help="Cheap brief-iteration: just decompose, don't search or assess.")
def decompose(
    brief: Annotated[str, typer.Argument(help="Natural-language brief.")],
    detection_target: Annotated[str | None, typer.Option("--detection-target")] = None,
    deployment_context: Annotated[str | None, typer.Option("--deployment-context")] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Optional path to write decomposition.yaml; otherwise stdout-only.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=(
                "Override the LLM model id (litellm-style). Examples: "
                "'github_copilot/gpt-5-mini' (uses your GitHub Copilot "
                "subscription via OAuth device-code flow on first run), "
                "'github/gpt-4o-mini' (uses GITHUB_TOKEN against the free "
                "GitHub Models tier), 'openai/gpt-4o' (uses OPENAI_API_KEY), "
                "'azure/<deployment>' (uses Entra). Falls back to "
                "$DATASET_SCOUT_MODEL, then to the AZURE_OPENAI_* config."
            ),
        ),
    ] = None,
) -> None:
    """Run only the LLM decomposition step.

    ~5 seconds and one LLM call. Use this to iterate on a brief
    cheaply: see what directions the model proposes, refine, repeat.
    Once happy, pass --decomposition-from to `recon` to skip
    re-paying the decompose cost.
    """
    from dataset_scout.context import ScoutContext
    from dataset_scout.decompose import decompose_intent, llm_available
    from dataset_scout.decomposition_io import write_decomposition
    from dataset_scout.errors import DatasetScoutError, LLMError
    from dataset_scout.intent import HeuristicIntentParser

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    if model:
        # CLI flag wins. ScoutContext is frozen Pydantic so we copy.
        ctx = ctx.model_copy(update={"model": model})

    if not llm_available(ctx):
        err.print(
            "[red]error:[/red] No LLM provider configured. Set "
            "[bold]--model[/bold] (e.g. 'github_copilot/gpt-5-mini'), or "
            "[bold]DATASET_SCOUT_MODEL[/bold], or the legacy "
            "[bold]AZURE_OPENAI_ENDPOINT[/bold] + "
            "[bold]AZURE_OPENAI_DEPLOYMENT[/bold] env vars (and run "
            "`az login`)."
        )
        raise typer.Exit(code=1)

    overrides: dict[str, object] = {}
    if detection_target:
        overrides["detection_target"] = detection_target
    if deployment_context:
        overrides["deployment_context"] = deployment_context
    intent = HeuristicIntentParser().parse(brief, **overrides)

    try:
        directions = decompose_intent(intent, ctx=ctx)
    except (LLMError, DatasetScoutError) as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    if not directions:
        err.print("[yellow]The model returned no decomposition directions.[/yellow]")
        return

    # Print to stdout so it pipes cleanly.
    typer.echo(f"# Decomposition for: {brief}\n")
    for d in directions:
        typer.echo(f"## {d.name}\n")
        typer.echo(f"{d.rationale}\n")
        if d.keywords:
            typer.echo(f"- keywords: `{', '.join(d.keywords)}`")
        if d.expected_finds:
            typer.echo(f"- expected: {d.expected_finds}")
        typer.echo("")

    if out is not None:
        path = write_decomposition(directions, out)
        if path is not None:
            err.print(f"[green]✔[/green] {len(directions)} direction(s) written to {path}")
            err.print(
                "  Pass [cyan]--decomposition-from " + str(path) + "[/cyan] to "
                "[cyan]datascout recon[/cyan] to reuse."
            )


# ─── render ─────────────────────────────────────────────────────────


@app.command(help="Re-render report.html and report.md from an existing results.json (no recon, no API calls).")
def render(
    out_dir: Annotated[
        Path,
        typer.Argument(
            help="Recon output folder containing results.json (e.g. datascout-out/<run>).",
        ),
    ],
    html_only: Annotated[
        bool,
        typer.Option("--html-only", help="Skip the Markdown report."),
    ] = False,
    md_only: Annotated[
        bool,
        typer.Option("--md-only", help="Skip the HTML report."),
    ] = False,
) -> None:
    """Re-render reports from a previous recon's persisted results.

    Useful for iterating on report styling (CSS, layout, copy) without
    re-paying the LLM/HTTP cost of running recon. Reads
    ``<out_dir>/results.json`` and rewrites ``report.html`` and
    ``report.md`` in place.
    """
    import json

    from dataset_scout.core import ReconResult
    from dataset_scout.render import (
        write_recon_report,
        write_recon_report_html,
    )

    if html_only and md_only:
        err.print("[red]error:[/red] --html-only and --md-only are mutually exclusive.")
        raise typer.Exit(code=2)

    results_path = out_dir / "results.json"
    if not results_path.exists():
        err.print(f"[red]error:[/red] {results_path} not found.")
        raise typer.Exit(code=2)

    try:
        result = ReconResult.model_validate(
            json.loads(results_path.read_text(encoding="utf-8"))
        )
    except (json.JSONDecodeError, ValueError) as exc:
        err.print(f"[red]error:[/red] could not load {results_path}: {exc}")
        raise typer.Exit(code=2) from exc

    written: list[Path] = []
    if not md_only:
        written.append(write_recon_report_html(result, out_dir))
    if not html_only:
        written.append(write_recon_report(result, out_dir))

    for path in written:
        err.print(f"[green]✔[/green] {path}")


@app.command(help="Deep-dive on one candidate.")
def inspect(
    target: Annotated[
        str,
        typer.Argument(
            help="<source>:<id>[@revision], e.g. huggingface:deepset/prompt-injections."
        ),
    ],
    intent_from: Annotated[
        Path | None,
        typer.Option(
            "--intent-from",
            help="Re-use the most recent recon's Intent (point at results.json).",
        ),
    ] = None,
    brief: Annotated[
        str | None,
        typer.Option(
            "--brief",
            help="One-off brief for strategy assessment when no recon results.json exists.",
        ),
    ] = None,
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            help="Rows to stream for the schema/label-distribution/sample sections.",
            min=1,
            max=1000,
        ),
    ] = 50,
) -> None:
    from dataset_scout.context import ScoutContext
    from dataset_scout.errors import DatasetScoutError
    from dataset_scout.inspect_ import make_intent, run_inspect
    from dataset_scout.render import render_inspect

    if intent_from is not None and brief is not None:
        err.print("[red]error:[/red] pass either --intent-from or --brief, not both.")
        raise typer.Exit(code=1)

    try:
        intent = make_intent(brief=brief, intent_from=intent_from)
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())

    try:
        result = run_inspect(
            target,
            ctx=ctx,
            intent=intent,
            sample_size=sample_size,
        )
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    # Inspect prints to stdout (so it works in pipes).
    typer.echo(render_inspect(result))
    if result.notices:
        for n in result.notices:
            err.print(f"[dim]note:[/dim] {n}")


# ─── compose ────────────────────────────────────────────────────────


@app.command(help="Compose multiple recipes into one merged corpus blueprint.")
def compose(
    recipes: Annotated[
        list[Path],
        typer.Argument(
            help="Two or more recipe.yaml files to merge.",
            metavar="RECIPE...",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Path to write the merged recipe.yaml."),
    ],
    intent_brief: Annotated[
        str | None,
        typer.Option(
            "--intent-brief",
            help="Override the merged intent's brief (default: keep the first input's).",
        ),
    ] = None,
) -> None:
    """Merge multiple recipes (e.g. from multi-detection-program runs).

    Components dedupe by (source, source_id) — higher
    `strategy_confidence` wins on conflict. Output recipe is fed
    straight to `datascout curate`.
    """
    from dataset_scout.curate import load_recipe
    from dataset_scout.errors import DatasetScoutError
    from dataset_scout.recipe import RecipeIntent
    from dataset_scout.recipe_compose import compose_recipes, write_composed_recipe

    if len(recipes) < 2:
        err.print(
            f"[red]error:[/red] compose needs at least two input recipes (got {len(recipes)})."
        )
        raise typer.Exit(code=1)

    try:
        loaded = [load_recipe(p) for p in recipes]
    except Exception as e:
        err.print(f"[red]error:[/red] failed to load a recipe: {e}")
        raise typer.Exit(code=1) from e

    intent_override = None
    if intent_brief is not None:
        intent_override = RecipeIntent(brief=intent_brief)

    try:
        merged, notices = compose_recipes(loaded, intent_override=intent_override)
    except (ValueError, DatasetScoutError) as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    target = write_composed_recipe(merged, out)
    err.print(
        f"[green]✔[/green] merged {len(loaded)} recipe(s) into {target} — "
        f"{len(merged.components)} components, {len(merged.declined)} declined"
    )
    for n in notices:
        err.print(f"  [yellow]![/yellow] {n}")


# ─── curate ─────────────────────────────────────────────────────────


@app.command(help="Build a schema-normalized corpus from a recipe.")
def curate(
    from_: Annotated[Path, typer.Option("--from", help="Path to recipe.yaml.")],
    out: Annotated[Path, typer.Option("--out", help="Output corpus directory.")] = Path("mycorpus"),
    min_strategy_confidence: Annotated[
        float | None, typer.Option("--min-strategy-confidence", min=0.0, max=1.0)
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the recipe's seed.")] = None,
    max_rows_per_component: Annotated[
        int | None,
        typer.Option(
            "--max-rows-per-component",
            min=1,
            help=(
                "Cap rows materialized per component for this run. Lowers but "
                "never raises the recipe's `take` value. Useful for fast "
                "iteration on heavy code/text corpora without hand-editing "
                "the recipe."
            ),
        ),
    ] = None,
    max_concurrency: Annotated[
        int,
        typer.Option(
            "--max-concurrency",
            min=1,
            max=16,
            help=(
                "Number of components materialized in parallel. Default 4. "
                "Most of the per-component cost is HuggingFace `load_dataset` "
                "setup overhead, which parallelises near-linearly. Lower to 1 "
                "for sequential builds (e.g. when debugging a hang); raise "
                "carefully to avoid HF rate limits, especially without HF_TOKEN."
            ),
        ),
    ] = 4,
) -> None:
    from dataset_scout.context import ScoutContext
    from dataset_scout.curate import load_recipe, run_curate
    from dataset_scout.errors import DatasetScoutError

    try:
        recipe = load_recipe(from_)
    except Exception as e:
        err.print(f"[red]error:[/red] failed to load recipe: {e}")
        raise typer.Exit(code=1) from e

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())

    try:
        result = run_curate(
            recipe,
            out,
            ctx=ctx,
            seed_override=seed,
            min_strategy_confidence_override=min_strategy_confidence,
            max_rows_per_component=max_rows_per_component,
            max_concurrency=max_concurrency,
        )
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    materialised_with_rows = result.components_kept - result.components_zero_row
    total_components = result.components_kept + result.components_skipped + result.components_failed
    err.print(
        f"[green]✔[/green] {result.total_rows} row(s) written to {result.out_dir} "
        f"({materialised_with_rows} of {total_components} component(s) materialised with rows"
        + (f", {result.components_zero_row} produced 0 rows" if result.components_zero_row else "")
        + (f", {result.components_skipped} dropped" if result.components_skipped else "")
        + (f", {result.components_failed} failed" if result.components_failed else "")
        + f") in {result.elapsed_seconds:.2f}s"
    )
    splits_str = " · ".join(f"{n}={c}" for n, c in result.rows_per_split.items())
    err.print(f"  - splits: {splits_str}")
    err.print(f"  - fingerprint: {result.fingerprint[:16]}...")
    if result.components_zero_row:
        err.print(
            f"  [yellow]![/yellow] {result.components_zero_row} component(s) "
            "materialised but produced 0 rows — usually a recipe column "
            "mismatch. See report.md → Components for the per-component count."
        )
    if result.failures:
        err.print(
            f"  [yellow]![/yellow] {len(result.failures)} component(s) skipped "
            "due to upstream errors — see report.md / recipe.lock.yaml "
            "→ failed_components for hints:"
        )
        for f in result.failures[:5]:
            err.print(f"    - {f['id']} [{f['category']}]: {f['hint']}")
        if len(result.failures) > 5:
            err.print(f"    ... and {len(result.failures) - 5} more in the lockfile.")
    err.print("  [green]✔[/green] audit-ready: leakage-aware splits + filter DSL applied.")


# ─── cache ──────────────────────────────────────────────────────────

cache_app = typer.Typer(help="Inspect and manage the dataset-scout cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("info")
def cache_info() -> None:
    """Print cache stats: size, entries, per-namespace breakdown."""
    from dataset_scout.cache import Cache
    from dataset_scout.context import ScoutContext

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    cache = Cache.open(ctx.cache_dir)
    try:
        stats = cache.info()
    finally:
        cache.close()
    err.print(f"[bold]Cache:[/bold] {stats['db_path']}")
    err.print(
        f"  total: {stats['total_entries']} entries, "
        f"{stats['total_bytes'] / 1024 / 1024:.1f} MiB / "
        f"{stats['max_bytes'] / 1024 / 1024 / 1024:.1f} GiB cap"
    )
    if not stats["by_namespace"]:
        err.print("  [dim](empty)[/dim]")
        return
    for ns in stats["by_namespace"]:
        err.print(
            f"  - {ns['namespace']:<12} {ns['entries']:>6} entries  "
            f"{ns['bytes'] / 1024 / 1024:>8.2f} MiB"
        )


@cache_app.command("prune")
def cache_prune() -> None:
    """Remove expired entries (TTL elapsed)."""
    from dataset_scout.cache import Cache
    from dataset_scout.context import ScoutContext

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    cache = Cache.open(ctx.cache_dir)
    try:
        removed = cache.prune()
    finally:
        cache.close()
    err.print(f"Removed [bold]{removed}[/bold] expired entrie(s).")


@cache_app.command("clear")
def cache_clear(
    namespace: Annotated[
        str | None,
        typer.Option(
            "--namespace", "-n",
            help="If provided, clear only this namespace (e.g. decompose, strategy).",
        ),
    ] = None,
) -> None:
    """Remove cache entries — all by default, or one namespace."""
    from dataset_scout.cache import Cache
    from dataset_scout.context import ScoutContext

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    cache = Cache.open(ctx.cache_dir)
    try:
        removed = cache.clear(namespace)
    finally:
        cache.close()
    where = f"namespace '{namespace}'" if namespace else "all namespaces"
    err.print(f"Removed [bold]{removed}[/bold] entrie(s) from {where}.")


# ─── sources ────────────────────────────────────────────────────────

sources_app = typer.Typer(help="List and toggle source plugins.", no_args_is_help=True)
app.add_typer(sources_app, name="sources")


@sources_app.command("list")
def sources_list() -> None:
    from dataset_scout.context import ScoutContext

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    for name, cfg in ctx.sources.items():
        state = "[green]enabled[/green]" if cfg.enabled else "[dim]disabled[/dim]"
        err.print(f"  {name:<14} {state}")


@sources_app.command("enable")
def sources_enable(name: str) -> None:
    err.print(
        f"[yellow]sources enable {name} is not implemented yet "
        f"(editing config.toml lands in M1).[/yellow]"
    )
    raise typer.Exit(code=2)


@sources_app.command("disable")
def sources_disable(name: str) -> None:
    err.print(
        f"[yellow]sources disable {name} is not implemented yet "
        f"(editing config.toml lands in M1).[/yellow]"
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":  # pragma: no cover
    app()


# ─── judge ──────────────────────────────────────────────────────────


@app.command(help="Promote weak labels to high-confidence ones with an LLM judge.")
def judge(
    target: Annotated[
        Path,
        typer.Argument(
            help=(
                "Corpus directory (containing train/val/test.jsonl) or a single "
                ".jsonl file produced by datascout curate."
            ),
        ),
    ],
    axis: Annotated[
        str,
        typer.Option("--axis", help="Labeling question, e.g. 'psych_harm'."),
    ],
    rubric: Annotated[
        Path | None,
        typer.Option(
            "--rubric",
            help="Optional rubric YAML / text file describing the axis in detail.",
        ),
    ] = None,
    judges: Annotated[
        int,
        typer.Option(
            "--judges",
            min=1,
            help="Number of independent judge calls per row. Default 1.",
        ),
    ] = 1,
    agreement: Annotated[
        str,
        typer.Option(
            "--agreement",
            help="Multi-judge aggregation rule: single | majority | unanimous.",
        ),
    ] = "single",
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            max=1.0,
            help="Promotion threshold on derived label_confidence. Default 0.8.",
        ),
    ] = 0.8,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Output directory. Defaults to <target>/judged/.",
        ),
    ] = None,
    only_unknown: Annotated[
        bool,
        typer.Option(
            "--only-unknown/--re-judge-all",
            help=(
                "Only judge rows that aren't already ground_truth or judged "
                "(default), or re-judge every row."
            ),
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Estimate the call count without invoking the LLM."),
    ] = False,
    calibrate_against: Annotated[
        Path | None,
        typer.Option(
            "--calibrate-against",
            help=(
                "Gold corpus to sample for a calibration pass before the full run. "
                "Reports P/R/F1 against ground_truth labels on the same axis."
            ),
        ),
    ] = None,
    calibration_seed_n: Annotated[
        int,
        typer.Option("--calibration-seed-n", min=1, help="Calibration sample size."),
    ] = 100,
    calibration_floor: Annotated[
        float | None,
        typer.Option(
            "--calibration-floor",
            min=0.0,
            max=1.0,
            help=(
                "Required minimum calibrated precision. With this set, the run "
                "aborts if calibration is below the floor unless --proceed."
            ),
        ),
    ] = None,
    proceed: Annotated[
        bool,
        typer.Option("--proceed", help="Override --calibration-floor."),
    ] = False,
) -> None:
    from dataset_scout.context import ScoutContext
    from dataset_scout.errors import DatasetScoutError, LLMError
    from dataset_scout.judge import run_judge
    from dataset_scout.render import render_judge_panel

    if agreement not in ("single", "majority", "unanimous"):
        err.print(
            f"[red]error:[/red] --agreement must be one of single|majority|unanimous "
            f"(got {agreement!r})"
        )
        raise typer.Exit(code=1)

    rubric_text: str | None = None
    if rubric is not None:
        try:
            rubric_text = rubric.read_text(encoding="utf-8")
        except OSError as e:
            err.print(f"[red]error:[/red] cannot read rubric file: {e}")
            raise typer.Exit(code=1) from e

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())

    try:
        result = run_judge(
            ctx,
            target,
            axis=axis,
            rubric=rubric_text,
            judges=judges,
            agreement=agreement,  # type: ignore[arg-type]
            threshold=threshold,
            out_dir=out,
            only_unknown=only_unknown,
            re_judge_all=not only_unknown,
            dry_run=dry_run,
            calibrate_against=calibrate_against,
            calibration_seed_n=calibration_seed_n,
            calibration_floor=calibration_floor,
            proceed=proceed,
        )
    except (DatasetScoutError, LLMError) as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    render_judge_panel(result, console=err)


# ─── eval ───────────────────────────────────────────────────────────


@app.command(name="eval", help="Score a judged corpus against a gold corpus.")
def eval_(
    judged: Annotated[Path, typer.Argument(help="Judged corpus directory or .jsonl file.")],
    against: Annotated[
        Path,
        typer.Option(
            "--against", help="Gold corpus directory or .jsonl file (ground_truth labels)."
        ),
    ],
    axis: Annotated[
        str | None,
        typer.Option("--axis", help="Restrict scoring to a single axis."),
    ] = None,
) -> None:
    from dataset_scout.context import ScoutContext
    from dataset_scout.errors import DatasetScoutError
    from dataset_scout.eval_ import run_eval
    from dataset_scout.render import render_eval_panel

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    try:
        result = run_eval(ctx, judged, gold=against, axis=axis)
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    render_eval_panel(result, console=err)


if __name__ == "__main__":  # pragma: no cover
    app()
