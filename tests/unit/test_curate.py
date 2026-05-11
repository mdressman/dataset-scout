"""End-to-end curate tests using FakeSource."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    DatasetScoutError,
    NormalizedRecord,
    ScoutContext,
    StrategyKind,
)
from dataset_scout.curate import load_recipe, run_curate
from dataset_scout.recipe import (
    Recipe,
    RecipeComponent,
    RecipeIntent,
    RecipeSplits,
    RecipeTransform,
)
from tests._fakes.fake_source import FakeSource

pytestmark = pytest.mark.unit


# ─── helpers ────────────────────────────────────────────────────────


def _two_class_rows(n: int = 100) -> list[dict[str, object]]:
    """Half positive, half benign with stable text + integer id."""
    rows: list[dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "id": i,
                "text": f"prompt content row {i}",
                "label": 1 if i % 2 == 0 else 0,
            }
        )
    return rows


def _make_recipe(
    *,
    components: list[RecipeComponent],
    splits: RecipeSplits | None = None,
    seed: int = 42,
    threshold: float = 0.5,
) -> Recipe:
    return Recipe(
        intent=RecipeIntent(
            brief="prompt injection corpora",
            detection_target="prompt injection",
            threat_families=["prompt_injection"],
        ),
        min_strategy_confidence=threshold,
        seed=seed,
        splits=splits or RecipeSplits(),
        components=components,
    )


def _component(
    cid: str = "fake_org_x",
    *,
    strategy: StrategyKind = StrategyKind.DIRECT_USE,
    confidence: float = 0.9,
    take: int | str = "all",
    text_column: str = "text",
    label_column: str | None = "label",
    label_value_map: dict[str, str] | None = None,
    label_kind_map: dict[str, str] | None = None,
    filter_str: str | None = None,
) -> RecipeComponent:
    return RecipeComponent(
        id=cid,
        source="fake",
        source_id="org/x",
        revision="r1",
        source_split="train",
        strategy=strategy,
        strategy_confidence=confidence,
        transform=RecipeTransform(
            text_column=text_column,
            label_column=label_column,
            label_value_map=(
                label_value_map if label_value_map is not None else {"1": "positive", "0": "benign"}
            ),
            label_kind_map=label_kind_map or {},
            filter=filter_str,
            take=take,
        ),
    )


def _ctx() -> ScoutContext:
    return ScoutContext.from_env(env={})


def _fake(rows: list[dict[str, object]]) -> FakeSource:
    cand = Candidate(
        source="fake",
        id="org/x",
        revision="r1",
        metadata=CandidateMetadata(),
    )
    return FakeSource([cand], samples={"org/x": rows})


# ─── load_recipe ────────────────────────────────────────────────────


def test_load_recipe_round_trip(tmp_path: Path):
    recipe = _make_recipe(components=[_component()])
    target = tmp_path / "recipe.yaml"
    target.write_text(yaml.safe_dump(recipe.model_dump(mode="json")), encoding="utf-8")
    loaded = load_recipe(target)
    assert loaded.intent.brief == "prompt injection corpora"
    assert len(loaded.components) == 1
    assert loaded.components[0].source_id == "org/x"


# ─── full materialisation ───────────────────────────────────────────


def test_run_curate_writes_all_artefacts(tmp_path: Path):
    rows = _two_class_rows(n=80)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])
    out = tmp_path / "corpus"

    result = run_curate(recipe, out, ctx=_ctx(), sources_override=[fake])

    assert result.total_rows == 80
    assert result.components_kept == 1
    # All expected files written.
    for fname in (
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "recipe.yaml",
        "recipe.lock.yaml",
        "manifest.json",
        "report.md",
        "fingerprint.txt",
        "usage.md",
    ):
        assert (out / fname).exists(), f"missing {fname}"

    # Leakage-aware MinHash splits give wider variance on small N
    # because each cluster is assigned wholesale; relax the band but
    # keep the total invariant.
    train_lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (out / "val.jsonl").read_text(encoding="utf-8").splitlines()
    test_lines = (out / "test.jsonl").read_text(encoding="utf-8").splitlines()
    assert 40 <= len(train_lines) <= 80
    assert 0 <= len(val_lines) <= 30
    assert 0 <= len(test_lines) <= 30
    assert len(train_lines) + len(val_lines) + len(test_lines) == 80


def test_run_curate_records_are_valid_normalized(tmp_path: Path):
    rows = _two_class_rows(n=20)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])

    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])

    # Every line in train.jsonl validates as NormalizedRecord.
    for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines():
        rec = NormalizedRecord.model_validate_json(line)
        assert rec.source == "fake:org/x"
        assert rec.label in {"positive", "benign", "hard_negative"}
        assert rec.strategy == StrategyKind.DIRECT_USE


def test_run_curate_label_kind_propagates(tmp_path: Path):
    rows = _two_class_rows(n=20)
    fake = _fake(rows)
    component = _component(
        strategy=StrategyKind.SIGNAL_PROXY,
        label_kind_map={"1": "proxy", "0": "ground_truth"},
    )
    recipe = _make_recipe(components=[component])

    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])

    seen_kinds: set[str] = set()
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        for line in (tmp_path / split).read_text(encoding="utf-8").splitlines():
            seen_kinds.add(NormalizedRecord.model_validate_json(line).label_kind.value)
    assert "proxy" in seen_kinds


# ─── filter hard-fail (M4a) ─────────────────────────────────────────


def test_run_curate_applies_filter_expression(tmp_path: Path):
    """Non-null filter is now compiled and applied (M4b filter DSL)."""
    rows = _two_class_rows(n=10)
    fake = _fake(rows)
    # Filter to label == 1 only — half the rows
    component = _component(filter_str="label == 1")
    recipe = _make_recipe(components=[component])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    total = 0
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        total += len((tmp_path / split).read_text(encoding="utf-8").splitlines())
    # Of 10 rows, 5 had label==1 (i % 2 == 0 → labels 1,0,1,0,...). Hmm
    # actually our _two_class_rows uses i % 2 == 0 → label 1, so 5 pass.
    assert total == 5


def test_run_curate_rejects_invalid_filter_syntax(tmp_path: Path):
    """Compile-time errors are surfaced clearly to the user."""
    fake = _fake(_two_class_rows(n=10))
    component = _component(filter_str="this is not valid python")
    recipe = _make_recipe(components=[component])
    with pytest.raises(DatasetScoutError, match="invalid filter expression"):
        run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])


def test_run_curate_rejects_disallowed_filter_function(tmp_path: Path):
    """Whitelist enforcement: __import__, attribute access, etc. blocked."""
    fake = _fake(_two_class_rows(n=10))
    component = _component(filter_str="__import__('os').system('rm -rf /')")
    recipe = _make_recipe(components=[component])
    with pytest.raises(DatasetScoutError, match="invalid filter expression"):
        run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])


# ─── min_strategy_confidence ─────────────────────────────────────────


def test_run_curate_filters_below_threshold(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    weak = _component(cid="weak", confidence=0.3)
    strong = _component(cid="strong", confidence=0.8)
    recipe = _make_recipe(
        components=[strong, weak],
        threshold=0.5,
    )
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])

    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    kept_ids = {c["id"] for c in lock["components"]}
    declined_ids = {c["id"] for c in lock["declined_components"]}
    assert kept_ids == {"strong"}
    assert "weak" in declined_ids


def test_run_curate_min_confidence_override_recorded(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    component = _component(confidence=0.7)
    recipe = _make_recipe(components=[component], threshold=0.5)
    # Override threshold to 0.8 — component now drops.
    with pytest.raises(DatasetScoutError, match="No components remain"):
        run_curate(
            recipe,
            tmp_path,
            ctx=_ctx(),
            sources_override=[fake],
            min_strategy_confidence_override=0.8,
        )


def test_run_curate_seed_override_recorded(tmp_path: Path):
    fake = _fake(_two_class_rows(n=20))
    recipe = _make_recipe(components=[_component()], seed=42)
    run_curate(
        recipe,
        tmp_path,
        ctx=_ctx(),
        sources_override=[fake],
        seed_override=99,
    )
    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    assert lock["seed"]["recipe"] == 42
    assert lock["seed"]["effective"] == 99
    assert lock["seed"]["overridden_by_cli"] is True


# ─── lockfile / manifest / report shape ─────────────────────────────


def test_lockfile_has_audit_readiness_flag(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    assert lock["audit_readiness"] == "ready"
    assert any(
        "leakage-aware" in n.lower() or "minhash" in n.lower()
        for n in lock["audit_readiness_notes"]
    )
    # Splits block records the dedup parameters.
    assert lock["splits"]["method"] == "minhash_lsh"
    assert "num_perm" in lock["splits"]
    assert "clusters_total" in lock["splits"]


def test_report_carries_audit_ready_banner(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "audit-ready" in report.lower()
    assert "leakage-aware" in report.lower()


def test_manifest_is_json_round_trip(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit_readiness"] == "ready"
    assert manifest["recipe_version"] == "1"


def test_fingerprint_is_deterministic_for_same_seed(tmp_path: Path):
    """Same recipe + same seed -> identical fingerprint across runs."""
    rows = _two_class_rows(n=30)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    recipe = _make_recipe(components=[_component()], seed=7)

    run_curate(recipe, out_a, ctx=_ctx(), sources_override=[_fake(rows)])
    run_curate(recipe, out_b, ctx=_ctx(), sources_override=[_fake(rows)])

    fp_a = (out_a / "fingerprint.txt").read_text(encoding="utf-8").strip()
    fp_b = (out_b / "fingerprint.txt").read_text(encoding="utf-8").strip()
    assert fp_a == fp_b


# ─── composition reference checks ───────────────────────────────────


def test_run_curate_rejects_composition_only_strategy(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    component = _component(
        strategy=StrategyKind.COMPOSITION_ONLY,
        confidence=0.9,
    )
    recipe = _make_recipe(components=[component])
    with pytest.raises(DatasetScoutError, match="composition_only"):
        run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])


def test_run_curate_rejects_unknown_composes_with(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    a = _component(cid="a")
    a = a.model_copy(update={"composes_with": ["b"]})
    recipe = _make_recipe(components=[a])
    with pytest.raises(DatasetScoutError, match="references unknown id"):
        run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])


# ─── unmapped labels are silently dropped ───────────────────────────


def test_unmapped_label_rows_dropped(tmp_path: Path):
    """Rows whose source label isn't in label_value_map are skipped, not crashed."""
    rows: list[dict[str, object]] = [
        {"id": 0, "text": "a", "label": 1},  # mapped
        {"id": 1, "text": "b", "label": 999},  # unmapped — dropped
        {"id": 2, "text": "c", "label": 0},  # mapped
    ]
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    total = 0
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        total += len((tmp_path / split).read_text(encoding="utf-8").splitlines())
    assert total == 2


# ─── multimodal coercion ────────────────────────────────────────────


def test_extras_coerce_bytes_values(tmp_path: Path):
    rows: list[dict[str, object]] = [
        {"id": 0, "text": "hello", "label": 1, "image_bytes": b"\\x89PNG..."},
    ]
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        for line in (tmp_path / split).read_text(encoding="utf-8").splitlines():
            rec = NormalizedRecord.model_validate_json(line)
            assert rec.extras_coercion is True
            assert isinstance(rec.extras["image_bytes"], str)


# ─── M4c: per-component soft failure ────────────────────────────────


def _multi_fake(
    samples_by_id: dict[str, list[dict[str, object]]],
) -> FakeSource:
    candidates = [
        Candidate(source="fake", id=cid, revision="r1", metadata=CandidateMetadata())
        for cid in samples_by_id
    ]
    return FakeSource(candidates, samples=samples_by_id)


def _component_for(source_id: str, *, cid: str | None = None) -> RecipeComponent:
    return RecipeComponent(
        id=cid or f"fake_{source_id.replace('/', '_')}",
        source="fake",
        source_id=source_id,
        revision="r1",
        source_split="train",
        strategy=StrategyKind.DIRECT_USE,
        strategy_confidence=0.9,
        transform=RecipeTransform(
            text_column="text",
            label_column="label",
            label_value_map={"1": "positive", "0": "benign"},
            label_kind_map={},
            filter=None,
            take="all",
        ),
    )


def test_curate_skips_failing_component_and_continues(tmp_path: Path):
    """One component crashes mid-stream — the other still produces a corpus."""
    good_rows = _two_class_rows(n=20)
    bad_sentinel = [
        {
            "_raise": ValueError(
                "Config name is missing.\nPlease pick one among the available "
                "configs: ['jailbreak_2023_05_07', 'jailbreak_2023_12_25']"
            )
        }
    ]
    fake = _multi_fake({"org/good": good_rows, "org/bad": bad_sentinel})
    recipe = _make_recipe(components=[_component_for("org/good"), _component_for("org/bad")])

    result = run_curate(recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake])

    assert result.total_rows == 20
    assert result.components_kept == 1
    assert result.components_failed == 1
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure["category"] == "missing_config"
    assert "source_config" in failure["hint"]


def test_curate_classifies_gated_dataset(tmp_path: Path):
    rows = _two_class_rows(n=10)
    bad = [
        {
            "_raise": Exception(
                "Dataset 'org/bad' is a gated dataset on the Hub. You must be "
                "authenticated to access it."
            )
        }
    ]
    fake = _multi_fake({"org/good": rows, "org/bad": bad})
    recipe = _make_recipe(components=[_component_for("org/good"), _component_for("org/bad")])

    result = run_curate(recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake])

    assert result.failures[0]["category"] == "gated_dataset"
    assert "HF_TOKEN" in result.failures[0]["hint"]


def test_curate_classifies_bad_split(tmp_path: Path):
    rows = _two_class_rows(n=10)
    bad = [{"_raise": ValueError("Bad split: train. Available splits: ['test']")}]
    fake = _multi_fake({"org/good": rows, "org/bad": bad})
    recipe = _make_recipe(components=[_component_for("org/good"), _component_for("org/bad")])

    result = run_curate(recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake])

    assert result.failures[0]["category"] == "bad_split"
    assert "source_split" in result.failures[0]["hint"]


def test_curate_records_failures_in_lockfile(tmp_path: Path):
    rows = _two_class_rows(n=10)
    bad = [
        {
            "_raise": Exception(
                "Dataset 'org/bad' is a gated dataset on the Hub. You must be authenticated."
            )
        }
    ]
    fake = _multi_fake({"org/good": rows, "org/bad": bad})
    recipe = _make_recipe(components=[_component_for("org/good"), _component_for("org/bad")])

    out = tmp_path / "corpus"
    run_curate(recipe, out, ctx=_ctx(), sources_override=[fake])

    lock = yaml.safe_load((out / "recipe.lock.yaml").read_text(encoding="utf-8"))
    assert "failed_components" in lock
    assert len(lock["failed_components"]) == 1
    entry = lock["failed_components"][0]
    assert entry["id"] == "fake_org_bad"
    assert entry["category"] == "gated_dataset"
    assert "exception_type" in entry
    assert "hint" in entry

    # The failed component must NOT show up in the realized components list.
    component_ids = {c["id"] for c in lock["components"]}
    assert "fake_org_good" in component_ids
    assert "fake_org_bad" not in component_ids


def test_curate_records_failures_in_report(tmp_path: Path):
    rows = _two_class_rows(n=10)
    bad = [{"_raise": ValueError("Bad split: train. Available splits: ['test']")}]
    fake = _multi_fake({"org/good": rows, "org/bad": bad})
    recipe = _make_recipe(components=[_component_for("org/good"), _component_for("org/bad")])

    out = tmp_path / "corpus"
    run_curate(recipe, out, ctx=_ctx(), sources_override=[fake])

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "skipped due to upstream errors" in report
    assert "fake_org_bad" in report
    assert "bad_split" in report


def test_curate_raises_when_all_components_fail(tmp_path: Path):
    bad = [{"_raise": Exception("Dataset 'org/bad' is a gated dataset on the Hub.")}]
    fake = _multi_fake({"org/bad": bad})
    recipe = _make_recipe(components=[_component_for("org/bad")])

    with pytest.raises(DatasetScoutError, match="All components failed"):
        run_curate(recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake])


def test_curate_classifies_unknown_failure(tmp_path: Path):
    rows = _two_class_rows(n=10)
    bad = [{"_raise": RuntimeError("totally unexpected error")}]
    fake = _multi_fake({"org/good": rows, "org/bad": bad})
    recipe = _make_recipe(components=[_component_for("org/good"), _component_for("org/bad")])

    result = run_curate(recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake])

    assert result.failures[0]["category"] == "unknown"
    assert result.failures[0]["exception_type"] == "RuntimeError"


def test_curate_classifies_rate_limit_failure(tmp_path: Path):
    """M5 edge #7 partial: HTTP 429 / rate-limit errors should be a
    distinct category with a hint pointing at HF_TOKEN and concurrency."""
    rows = _two_class_rows(n=10)
    bad = [{"_raise": Exception("HTTP 429: Too Many Requests")}]
    fake = _multi_fake({"org/good": rows, "org/bad": bad})
    recipe = _make_recipe(components=[_component_for("org/good"), _component_for("org/bad")])

    result = run_curate(recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake])

    assert result.failures[0]["category"] == "rate_limited"
    assert "HF_TOKEN" in result.failures[0]["hint"]
    assert "max-concurrency" in result.failures[0]["hint"]


# ─── M5: parallel materialisation determinism ───────────────────────


def test_curate_parallel_run_is_deterministic(tmp_path: Path):
    """Same recipe + same seed must produce same fingerprint regardless
    of how completion order interleaves across the worker pool. This is
    the contract that lets a defensible reviewer reproduce a corpus."""
    samples = {f"org/c{i}": _two_class_rows(n=20) for i in range(8)}

    def fake_for_run() -> FakeSource:
        candidates = [
            Candidate(source="fake", id=cid, revision="r1", metadata=CandidateMetadata())
            for cid in samples
        ]
        return FakeSource(candidates, samples=samples)

    recipe = _make_recipe(
        components=[_component_for(cid) for cid in samples],
        seed=42,
    )

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    res_a = run_curate(
        recipe, out_a, ctx=_ctx(), sources_override=[fake_for_run()], max_concurrency=4
    )
    res_b = run_curate(
        recipe, out_b, ctx=_ctx(), sources_override=[fake_for_run()], max_concurrency=4
    )

    assert res_a.fingerprint == res_b.fingerprint
    assert res_a.rows_per_split == res_b.rows_per_split

    # Lockfile component lists also should match in order.
    lock_a = yaml.safe_load((out_a / "recipe.lock.yaml").read_text(encoding="utf-8"))
    lock_b = yaml.safe_load((out_b / "recipe.lock.yaml").read_text(encoding="utf-8"))
    assert [c["id"] for c in lock_a["components"]] == [c["id"] for c in lock_b["components"]]


def test_curate_parallel_matches_sequential(tmp_path: Path):
    """Parallel mode (workers > 1) must produce the same fingerprint
    as sequential mode (workers = 1) for the same inputs."""
    samples = {f"org/c{i}": _two_class_rows(n=15) for i in range(5)}

    def fake_for_run() -> FakeSource:
        candidates = [
            Candidate(source="fake", id=cid, revision="r1", metadata=CandidateMetadata())
            for cid in samples
        ]
        return FakeSource(candidates, samples=samples)

    recipe = _make_recipe(
        components=[_component_for(cid) for cid in samples],
        seed=7,
    )

    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"
    res_seq = run_curate(
        recipe, out_seq, ctx=_ctx(), sources_override=[fake_for_run()], max_concurrency=1
    )
    res_par = run_curate(
        recipe, out_par, ctx=_ctx(), sources_override=[fake_for_run()], max_concurrency=4
    )

    assert res_seq.fingerprint == res_par.fingerprint
    assert res_seq.rows_per_split == res_par.rows_per_split
    assert res_seq.total_rows == res_par.total_rows


def test_curate_parallel_handles_mixed_success_and_failure(tmp_path: Path):
    """Workers may complete in any order, and some may raise. The final
    materialised list and failures list must still be in original
    `kept` order, not completion order."""
    good_rows = _two_class_rows(n=10)
    bad_429 = [{"_raise": Exception("HTTP 429: Too Many Requests")}]
    bad_gated = [{"_raise": Exception("Dataset 'org/x' is a gated dataset on the Hub.")}]

    samples = {
        "org/good_a": good_rows,
        "org/bad_429": bad_429,
        "org/good_b": good_rows,
        "org/bad_gated": bad_gated,
        "org/good_c": good_rows,
    }
    candidates = [
        Candidate(source="fake", id=cid, revision="r1", metadata=CandidateMetadata())
        for cid in samples
    ]
    fake = FakeSource(candidates, samples=samples)

    recipe = _make_recipe(components=[_component_for(cid) for cid in samples], seed=42)

    result = run_curate(
        recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake], max_concurrency=4
    )

    assert result.components_kept == 3
    assert result.components_failed == 2

    # Failures list must be in the original `kept` order: bad_429 (idx 1)
    # before bad_gated (idx 3).
    assert [f["id"] for f in result.failures] == ["fake_org_bad_429", "fake_org_bad_gated"]
    assert result.failures[0]["category"] == "rate_limited"
    assert result.failures[1]["category"] == "gated_dataset"


# ─── M5: max_rows_per_component override ────────────────────────────


def test_curate_max_rows_per_component_caps_take_all(tmp_path: Path):
    """``--max-rows-per-component`` lowers a recipe's ``take: all`` for
    a single run without editing the recipe file."""
    rows = _two_class_rows(n=200)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component(take="all")])

    result = run_curate(
        recipe,
        tmp_path / "corpus",
        ctx=_ctx(),
        sources_override=[fake],
        max_rows_per_component=50,
    )

    assert result.total_rows == 50
    assert result.components_kept == 1


def test_curate_max_rows_per_component_lowers_explicit_take(tmp_path: Path):
    """Override is the lower of (recipe take, override) — never raises
    the recipe's cap."""
    rows = _two_class_rows(n=200)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component(take=120)])

    result = run_curate(
        recipe,
        tmp_path / "corpus",
        ctx=_ctx(),
        sources_override=[fake],
        max_rows_per_component=50,
    )

    assert result.total_rows == 50


def test_curate_max_rows_per_component_does_not_raise_recipe_cap(tmp_path: Path):
    """Override of 500 doesn't expand a recipe's take=120 to 500."""
    rows = _two_class_rows(n=200)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component(take=120)])

    result = run_curate(
        recipe,
        tmp_path / "corpus",
        ctx=_ctx(),
        sources_override=[fake],
        max_rows_per_component=500,
    )

    # Recipe says 120; override is 500; the lower wins.
    assert result.total_rows == 120


# ─── FM1: schema validation at curate time ──────────────────────────


def test_curate_classifies_schema_mismatch(tmp_path: Path):
    """Component with wrong column names fails with schema_mismatch category."""
    rows = _two_class_rows(n=10)
    fake = _fake(rows)
    component = _component(text_column="nonexistent_column")
    recipe = _make_recipe(components=[_component(cid="good"), component])
    # good component uses "text" which exists; bad uses "nonexistent_column"
    result = run_curate(recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake])
    assert result.components_failed == 1
    assert result.failures[0]["category"] == "schema_mismatch"
    assert "nonexistent_column" in result.failures[0]["message"]


def test_curate_schema_mismatch_label_column(tmp_path: Path):
    """Component with wrong label column fails with schema_mismatch category."""
    rows = _two_class_rows(n=10)
    fake = _fake(rows)
    component = _component(cid="bad_label", label_column="fake_label")
    recipe = _make_recipe(components=[_component(cid="good"), component])
    result = run_curate(recipe, tmp_path / "corpus", ctx=_ctx(), sources_override=[fake])
    assert result.components_failed == 1
    assert result.failures[0]["category"] == "schema_mismatch"


# ─── FM5: label distribution warnings ───────────────────────────────


def test_curate_lockfile_has_label_distribution_warnings(tmp_path: Path):
    """Lockfile includes label_distribution_warnings field."""
    rows = _two_class_rows(n=20)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    assert "label_distribution_warnings" in lock
    assert isinstance(lock["label_distribution_warnings"], list)


def test_curate_lockfile_has_dedup_impact(tmp_path: Path):
    """Lockfile includes dedup_impact section."""
    rows = _two_class_rows(n=20)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    assert "dedup_impact" in lock
    assert "rows_total_before_split" in lock["dedup_impact"]
    assert "rows_in_dup_clusters" in lock["dedup_impact"]


def test_curate_warns_on_all_proxy_corpus(tmp_path: Path):
    """Warning emitted when majority of rows are proxy-labeled."""
    rows = _two_class_rows(n=20)
    fake = _fake(rows)
    component = _component(
        strategy=StrategyKind.SIGNAL_PROXY,
        label_kind_map={"1": "proxy", "0": "proxy"},
    )
    recipe = _make_recipe(components=[component])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    warnings = lock["label_distribution_warnings"]
    assert any("proxy" in w.lower() for w in warnings)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "label distribution warnings" in report.lower()
