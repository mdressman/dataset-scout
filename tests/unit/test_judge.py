"""Unit tests for the LLM-as-judge module.

Covers single-judge core (cache + strict promotion + soft failures),
checkpoint resumability, and multi-judge agreement aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataset_scout import (
    JudgeBlock,
    LabelKind,
    NormalizedRecord,
    ScoutContext,
    StrategyKind,
)
from dataset_scout.errors import LLMError
from dataset_scout.judge import (
    _CHECKPOINT_NAME,
    ContentFilterError,
    _cache_key,
    _ChatClient,
    _promote,
    judge_cache_dir,
    render_judge_prompt,
    run_judge,
)

pytestmark = pytest.mark.unit


# ─── helpers ────────────────────────────────────────────────────────


def _ctx(tmp_path: Path) -> ScoutContext:
    return ScoutContext(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")


def _row(
    *,
    text: str,
    label: str = "benign",
    label_kind: LabelKind = LabelKind.PROXY,
    rid: str | None = None,
) -> NormalizedRecord:
    return NormalizedRecord(
        text=text,
        label=label,  # type: ignore[arg-type]
        label_kind=label_kind,
        strategy=StrategyKind.SIGNAL_PROXY,
        strategy_confidence=0.7,
        source="huggingface:fake/ds",
        source_row_id=rid or text[:8],
    )


def _write_corpus(dir_path: Path, rows: list[NormalizedRecord]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    target = dir_path / "train.jsonl"
    target.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    return dir_path


class _ScriptedChat(_ChatClient):
    """A stub chat client returning canned content per call.

    Pass either:
      - ``responses``: a list cycled through one entry per call, or
      - ``by_text``: a dict keyed by the row's text containing the
        response payload(s) to emit for that row.
    """

    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        by_text: dict[str, Any] | None = None,
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.by_text = dict(by_text or {})
        self.raise_on = dict(raise_on or {})
        self.calls: list[list[dict[str, str]]] = []
        self._call_idx = 0
        self._per_text_idx: dict[str, int] = {}

    def call(self, *, messages: list[dict[str, str]], timeout_s: float) -> str:
        self.calls.append(list(messages))
        text = messages[0]["content"]
        for needle, exc in self.raise_on.items():
            if needle in text:
                raise exc
        if self.by_text:
            for needle, payload in self.by_text.items():
                if needle in text:
                    if isinstance(payload, list):
                        idx = self._per_text_idx.get(needle, 0)
                        self._per_text_idx[needle] = idx + 1
                        chosen = payload[idx % len(payload)]
                    else:
                        chosen = payload
                    return chosen if isinstance(chosen, str) else json.dumps(chosen)
        if self.responses:
            chosen = self.responses[self._call_idx % len(self.responses)]
            self._call_idx += 1
            return chosen if isinstance(chosen, str) else json.dumps(chosen)
        return json.dumps(
            {
                "verdict": "ambiguous",
                "subcategory": "default",
                "confidence": 0.5,
                "rationale": "default",
            }
        )


def _payload(
    verdict: str = "positive",
    confidence: float = 0.9,
    subcategory: str = "test-cat",
    rationale: str = "ok",
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "subcategory": subcategory,
        "confidence": confidence,
        "rationale": rationale,
    }


# ─── prompt rendering ───────────────────────────────────────────────


def test_render_includes_axis_and_text() -> None:
    p = render_judge_prompt(axis="psych_harm", rubric="be strict", text="hello")
    assert "psych_harm" in p
    assert "be strict" in p
    assert "hello" in p
    assert "STRICT JSON" in p


def test_render_uses_placeholder_when_no_rubric() -> None:
    p = render_judge_prompt(axis="psych_harm", rubric=None, text="hi")
    assert "(no rubric supplied" in p


_SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "judge_prompt.txt"


def test_render_snapshot_stable() -> None:
    """Prompt drift surfaces as a PR diff. Mirrors the decompose /
    strategy-assessor snapshot scaffold; if intentional, delete the
    snapshot file and re-run the test."""
    rendered = render_judge_prompt(
        axis="psych_harm",
        rubric=(
            "Score the row 'positive' if it depicts or seeks help with "
            "psychological self-harm, suicidal ideation, or severe "
            "emotional distress directed at the self. 'Negative' is "
            "ordinary emotion talk, fiction, or third-person reportage. "
            "'Ambiguous' is borderline distress without a clear self-harm cue."
        ),
        text="I just don't see the point of any of this anymore.",
    )
    if not _SNAPSHOT_PATH.exists():
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")
    expected = _SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert rendered == expected, (
        "Judge prompt drifted from snapshot. If intentional, delete "
        f"{_SNAPSHOT_PATH} and re-run the test."
    )


def test_cache_key_changes_with_template_version() -> None:
    a = _cache_key(prompt="x", axis="y", model="m", template_version="1")
    b = _cache_key(prompt="x", axis="y", model="m", template_version="2")
    assert a != b


# ─── core run_judge ─────────────────────────────────────────────────


def test_run_judge_promotes_high_confidence_positive(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="self-harm content here", rid="r1")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="psych_harm", chat_client=chat, threshold=0.8)
    assert result.stats.n_input == 1
    assert result.stats.n_judged == 1
    assert result.stats.n_promoted_positive == 1
    assert result.stats.n_left_unknown == 0
    out_file = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out_file.read_text().splitlines()[0]))
    assert rec.label == "positive"
    assert rec.label_kind == LabelKind.JUDGED
    assert rec.label_confidence == pytest.approx(0.95)
    assert rec.judge is not None
    assert rec.judge.axis == "psych_harm"
    assert rec.judge.verdict == "positive"


def test_run_judge_does_not_promote_below_threshold(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="borderline", rid="r1")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.6)])
    result = run_judge(ctx, target, axis="x", chat_client=chat, threshold=0.8)
    assert result.stats.n_promoted_positive == 0
    assert result.stats.n_left_unknown == 1
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    # Original label preserved, but JudgeBlock attached.
    assert rec.label == "benign"
    assert rec.label_kind == LabelKind.PROXY
    assert rec.judge is not None
    assert rec.judge.verdict == "positive"


def test_run_judge_skips_ground_truth_by_default(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [
            _row(text="t1", label_kind=LabelKind.GROUND_TRUTH, rid="r1"),
            _row(text="t2", label_kind=LabelKind.PROXY, rid="r2"),
        ],
    )
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_judged == 1
    assert len(chat.calls) == 1


def test_run_judge_re_judge_all_overrides_skip(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="t1", label_kind=LabelKind.GROUND_TRUTH, rid="r1")],
    )
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat, re_judge_all=True)
    assert result.stats.n_judged == 1


def test_run_judge_uses_disk_cache_on_second_run(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="hi", rid="r1")])
    chat1 = _ScriptedChat(responses=[_payload("positive", 0.95)])
    run_judge(ctx, target, axis="x", chat_client=chat1)
    assert len(chat1.calls) == 1

    # Second run with a different out_dir but same workspace cache.
    chat2 = _ScriptedChat(raise_on={"hi": LLMError("must not be called — cache should hit")})
    result = run_judge(ctx, target, axis="x", chat_client=chat2, out_dir=tmp_path / "out2")
    assert result.stats.n_cache_hits == 1
    assert result.stats.n_judged == 1
    assert len(chat2.calls) == 0


def test_run_judge_recovers_from_corrupt_cache_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="hi", rid="r1")])
    # Plant a corrupt cache entry.
    cache_dir = judge_cache_dir(ctx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # We don't know the key without rendering the prompt; brute-force a
    # corrupt file in every two-char shard the run might use.
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_judged == 1
    # Now corrupt the resulting cache file and re-run.
    corrupted = 0
    for f in cache_dir.rglob("*.json"):
        f.write_text("{not json", encoding="utf-8")
        corrupted += 1
    assert corrupted >= 1
    chat2 = _ScriptedChat(responses=[_payload("positive", 0.9)])
    res2 = run_judge(ctx, target, axis="x", chat_client=chat2, out_dir=tmp_path / "out2")
    # Re-judged because the cache file was corrupt and got deleted.
    assert res2.stats.n_judged == 1
    assert len(chat2.calls) == 1


def test_run_judge_soft_fails_on_api_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="bad", rid="r1"), _row(text="good", rid="r2")],
    )
    chat = _ScriptedChat(
        by_text={"good": _payload("positive", 0.95)},
        raise_on={"bad": LLMError("transient")},
    )
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_input == 2
    assert result.stats.n_judged == 1
    assert result.stats.n_skipped == 1
    assert result.stats.n_api_errors == 1
    assert len(result.failures) == 1


def test_run_judge_soft_fails_on_content_filter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    chat = _ScriptedChat(raise_on={"x": ContentFilterError("blocked")})
    result = run_judge(ctx, target, axis="psych_harm", chat_client=chat)
    assert result.stats.n_content_filter_blocked == 1
    assert result.stats.n_skipped == 1


def test_run_judge_retries_once_on_invalid_json(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="hi", rid="r1")])
    chat = _ScriptedChat(responses=["not json", _payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_judged == 1
    assert len(chat.calls) == 2
    # Retry message is sent as a user follow-up.
    assert any("STRICT JSON ONLY" in m["content"] for m in chat.calls[1])


def test_run_judge_soft_fails_after_two_invalid_json(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="hi", rid="r1")])
    chat = _ScriptedChat(responses=["not json", "still bad"])
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_parse_errors == 1
    assert result.stats.n_skipped == 1


def test_run_judge_dry_run_makes_no_calls(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="a", rid="r1"), _row(text="b", rid="r2")],
    )
    chat = _ScriptedChat(raise_on={"a": LLMError("nope"), "b": LLMError("nope")})
    result = run_judge(
        ctx, target, axis="x", chat_client=chat, dry_run=True, judges=3, agreement="majority"
    )
    assert result.dry_run is True
    assert result.estimated_calls == 6
    assert len(chat.calls) == 0
    # No output JSONL files written on dry-run.
    assert not (result.out_dir / "train.jsonl").exists()


# ─── checkpoint / resumability ──────────────────────────────────────


def test_run_judge_writes_checkpoint(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text=f"t{i}", rid=f"r{i}") for i in range(3)])
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    out = tmp_path / "out"
    run_judge(ctx, target, axis="x", chat_client=chat, out_dir=out, batch_size=1)
    cp = json.loads((out / _CHECKPOINT_NAME).read_text(encoding="utf-8"))
    assert cp["axis"] == "x"
    assert len(cp["completed_row_ids"]) == 3


def test_run_judge_resume_skips_completed_rows(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="t1", rid="r1"), _row(text="t2", rid="r2")],
    )
    out = tmp_path / "out"
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    run_judge(ctx, target, axis="x", chat_client=chat, out_dir=out)
    # Second run: cache hits keep call count at zero; checkpoint
    # records the rows as already completed.
    chat2 = _ScriptedChat(
        raise_on={"t1": LLMError("must not be called"), "t2": LLMError("must not be called")}
    )
    result = run_judge(ctx, target, axis="x", chat_client=chat2, out_dir=out)
    assert result.stats.n_resumed == 2
    assert len(chat2.calls) == 0


def test_run_judge_corrupt_checkpoint_starts_fresh(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / _CHECKPOINT_NAME).write_text("{not json", encoding="utf-8")
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat, out_dir=out)
    assert result.stats.n_resumed == 0
    assert result.stats.n_judged == 1


# ─── multi-judge agreement ──────────────────────────────────────────


def test_majority_agreement_picks_majority_verdict(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    # 3 judges: positive (0.9), positive (0.8), negative (0.7).
    chat = _ScriptedChat(
        responses=[
            _payload("positive", 0.9, subcategory="cat-a"),
            _payload("positive", 0.8, subcategory="cat-a"),
            _payload("negative", 0.7),
        ]
    )
    result = run_judge(
        ctx,
        target,
        axis="x",
        chat_client=chat,
        judges=3,
        agreement="majority",
        threshold=0.5,
    )
    assert len(chat.calls) == 3
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    assert rec.judge is not None
    assert rec.judge.verdict == "positive"
    assert rec.judge.n_judges == 3
    assert rec.judge.agreement == "majority"
    # Derived label_confidence = (2/3) * mean(0.9, 0.8) = 0.5666...
    assert rec.label_confidence == pytest.approx(0.566667, rel=1e-3)
    assert rec.label_kind == LabelKind.JUDGED


def test_majority_no_majority_marks_ambiguous(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    chat = _ScriptedChat(
        responses=[
            _payload("positive", 0.9),
            _payload("negative", 0.9),
            _payload("ambiguous", 0.9),
        ]
    )
    result = run_judge(ctx, target, axis="x", chat_client=chat, judges=3, agreement="majority")
    assert result.stats.n_left_unknown == 1
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    assert rec.judge is not None
    assert rec.judge.verdict == "ambiguous"


def test_unanimous_requires_all_to_agree(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    # 5 judges, one disagrees → ambiguous.
    chat = _ScriptedChat(
        responses=[
            _payload("positive", 0.9),
            _payload("positive", 0.9),
            _payload("positive", 0.9),
            _payload("positive", 0.9),
            _payload("negative", 0.9),
        ]
    )
    result = run_judge(ctx, target, axis="x", chat_client=chat, judges=5, agreement="unanimous")
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    assert rec.judge is not None
    assert rec.judge.verdict == "ambiguous"


def test_unanimous_promotes_when_all_agree(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.85)] * 5)
    result = run_judge(ctx, target, axis="x", chat_client=chat, judges=5, agreement="unanimous")
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    assert rec.judge is not None
    assert rec.judge.verdict == "positive"
    assert rec.label_confidence == pytest.approx(0.85)
    assert rec.label_kind == LabelKind.JUDGED


def test_majority_requires_judges_ge_3(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    with pytest.raises(Exception, match="majority"):
        run_judge(
            ctx,
            target,
            axis="x",
            judges=2,
            agreement="majority",
            chat_client=_ScriptedChat(),
        )


def test_unanimous_requires_judges_ge_3(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    with pytest.raises(Exception, match="unanimous"):
        run_judge(
            ctx,
            target,
            axis="x",
            judges=2,
            agreement="unanimous",
            chat_client=_ScriptedChat(),
        )


def test_single_with_judges_gt_1_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    with pytest.raises(Exception, match="agreement"):
        run_judge(
            ctx,
            target,
            axis="x",
            judges=3,
            agreement="single",
            chat_client=_ScriptedChat(),
        )


# ─── promotion rule directly ────────────────────────────────────────


def test_promote_negative_above_threshold_writes_benign() -> None:
    rec = _row(text="t", label="positive", label_kind=LabelKind.PROXY, rid="r1")
    block = JudgeBlock(
        axis="x",
        verdict="negative",
        subcategory="benign-talk",
        confidence=0.95,
        rationale="ok",
        model="m",
        template_version="1",
        n_judges=1,
        agreement="single",
    )
    out = _promote(rec, block, 0.95, threshold=0.8)
    assert out.label == "benign"
    assert out.label_kind == LabelKind.JUDGED
    assert out.label_confidence == pytest.approx(0.95)


def test_promote_ambiguous_keeps_label() -> None:
    rec = _row(text="t", rid="r1")
    block = JudgeBlock(
        axis="x",
        verdict="ambiguous",
        subcategory="unclear",
        confidence=0.99,
        rationale="ok",
        model="m",
        template_version="1",
        n_judges=1,
        agreement="single",
    )
    out = _promote(rec, block, 0.99, threshold=0.5)
    assert out.label_kind == LabelKind.PROXY
    assert out.judge is not None
    assert out.label_confidence == pytest.approx(0.99)


def test_run_judge_writes_lockfile_and_report(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="psych_harm", chat_client=chat)
    lock_path = result.out_dir / "judge.lock.yaml"
    report_path = result.out_dir / "judge.report.md"
    assert lock_path.is_file()
    assert report_path.is_file()
    import yaml as _yaml

    payload = _yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert payload["judge"]["axis"] == "psych_harm"
    assert payload["judge"]["model"]
    assert payload["judge"]["template_version"] == "2"
    assert payload["judge"]["n_judges"] == 1
    assert payload["judge"]["agreement"] == "single"
    assert payload["judge"]["threshold"] == 0.8
    assert payload["judge"]["stats"]["n_judged"] == 1
    assert "psych_harm" in report_path.read_text(encoding="utf-8")


# ─── calibration ────────────────────────────────────────────────────


def test_calibration_records_metrics(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    gold_rows = [
        _row(text="pos1", label="positive", label_kind=LabelKind.GROUND_TRUTH, rid="g1"),
        _row(text="pos2", label="positive", label_kind=LabelKind.GROUND_TRUTH, rid="g2"),
        _row(text="neg1", label="benign", label_kind=LabelKind.GROUND_TRUTH, rid="g3"),
        _row(text="neg2", label="benign", label_kind=LabelKind.GROUND_TRUTH, rid="g4"),
    ]
    gold_dir = _write_corpus(tmp_path / "gold", gold_rows)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="rx")])
    chat = _ScriptedChat(
        by_text={
            "pos1": _payload("positive", 0.95),
            "pos2": _payload("positive", 0.95),
            "neg1": _payload("negative", 0.95),
            "neg2": _payload("negative", 0.95),
            "x": _payload("ambiguous", 0.5),
        }
    )
    result = run_judge(
        ctx,
        target,
        axis="x",
        chat_client=chat,
        calibrate_against=gold_dir,
        calibration_seed_n=10,
        threshold=0.8,
    )
    assert result.calibration is not None
    assert result.calibration["n_sampled"] == 4
    assert result.calibration["precision"] == pytest.approx(1.0)
    assert result.calibration["recall"] == pytest.approx(1.0)
    assert result.calibration["f1"] == pytest.approx(1.0)


def test_calibration_floor_blocks_run_without_proceed(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    gold_rows = [
        _row(text="g1", label="positive", label_kind=LabelKind.GROUND_TRUTH, rid="g1"),
        _row(text="g2", label="benign", label_kind=LabelKind.GROUND_TRUTH, rid="g2"),
    ]
    gold_dir = _write_corpus(tmp_path / "gold", gold_rows)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="rx")])
    # Judge calls everything positive → precision = 0.5.
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    with pytest.raises(Exception, match="floor"):
        run_judge(
            ctx,
            target,
            axis="x",
            chat_client=chat,
            calibrate_against=gold_dir,
            calibration_floor=0.9,
        )


def test_calibration_floor_proceed_overrides(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    gold_rows = [
        _row(text="g1", label="positive", label_kind=LabelKind.GROUND_TRUTH, rid="g1"),
        _row(text="g2", label="benign", label_kind=LabelKind.GROUND_TRUTH, rid="g2"),
    ]
    gold_dir = _write_corpus(tmp_path / "gold", gold_rows)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="rx")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(
        ctx,
        target,
        axis="x",
        chat_client=chat,
        calibrate_against=gold_dir,
        calibration_floor=0.9,
        proceed=True,
    )
    assert result.calibration is not None
    assert result.calibration["precision"] == pytest.approx(0.5)


# ─── M10 polish: report sample rows + resume-detect log ─────────────


def test_judge_report_includes_sample_rows(tmp_path: Path) -> None:
    """The judge.report.md surfaces top-N rows per bucket so reviewers
    can eyeball the run without rg-ing the JSONL."""
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [
            _row(text="clear positive row", rid="rp"),
            _row(text="clear negative row", rid="rn"),
            _row(text="genuinely ambiguous row", rid="ra"),
        ],
    )
    chat = _ScriptedChat(
        by_text={
            "clear positive row": _payload("positive", 0.96, subcategory="strong-pos"),
            "clear negative row": _payload("negative", 0.94, subcategory="strong-neg"),
            "genuinely ambiguous row": _payload("ambiguous", 0.85, subcategory="ambiguous-cat"),
        }
    )
    result = run_judge(ctx, target, axis="psych_harm", chat_client=chat, threshold=0.8)

    report = (result.out_dir / "judge.report.md").read_text(encoding="utf-8")
    assert "## Sample rows" in report
    assert "Highest-confidence promoted POSITIVES" in report
    assert "Highest-confidence promoted NEGATIVES" in report
    assert "Highest-confidence AMBIGUOUS (not promoted)" in report
    # Subcategories from our scripted payloads should appear in the report.
    assert "strong-pos" in report
    assert "strong-neg" in report
    assert "ambiguous-cat" in report


def test_judge_report_resume_tip_when_state_file_present(tmp_path: Path) -> None:
    """When the per-batch checkpoint is on disk after a run, the report
    surfaces the resume tip so users discover it without reading the
    code."""
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat, threshold=0.8)

    report = (result.out_dir / "judge.report.md").read_text(encoding="utf-8")
    assert _CHECKPOINT_NAME in report
    assert "resume" in report.lower()


def test_judge_resume_emits_log_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Re-running with an existing checkpoint logs an explicit resume
    line on entry so users see it in their terminal, not just in the
    post-run panel."""
    import logging

    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="row1", rid="r1"), _row(text="row2", rid="r2")],
    )
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)] * 4)

    # First run: produces the checkpoint.
    run_judge(ctx, target, axis="x", chat_client=chat, threshold=0.8)
    assert (target / "judged" / _CHECKPOINT_NAME).exists()

    # Second run: should emit the resume log line on entry.
    with caplog.at_level(logging.INFO, logger="dataset_scout.judge"):
        run_judge(ctx, target, axis="x", chat_client=chat, threshold=0.8)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("resuming axis=" in m and "already completed" in m for m in msgs), msgs


# ─── M10 polish: curate panel surfaces zero-row components ─────────


def test_curate_result_components_zero_row_default_and_set() -> None:
    """The new components_zero_row field defaults to 0 (back-compat) and
    accepts a positive count."""
    from dataset_scout.curate import CurateResult

    r = CurateResult(
        out_dir=Path("."),
        components_kept=10,
        components_skipped=2,
        components_failed=1,
        rows_per_split={"train": 100, "val": 20, "test": 20},
        fingerprint="abc",
        elapsed_seconds=1.0,
    )
    assert r.components_zero_row == 0

    r2 = CurateResult(
        out_dir=Path("."),
        components_kept=10,
        components_skipped=2,
        components_failed=1,
        rows_per_split={"train": 100},
        fingerprint="abc",
        elapsed_seconds=1.0,
        components_zero_row=3,
    )
    assert r2.components_zero_row == 3


# ─── regression: judge --model override actually flows to the call ──


def test_litellm_chat_client_forwards_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_judge(model=...)`` must override ``ctx.model`` at the
    actual completion call site, not only in the cache-key / audit
    trail. Pre-fix regression: ``_resolve_model_name`` recorded the
    override, but ``_LiteLLMChatClient.call()`` invoked
    ``build_completion_kwargs(self.ctx, ...)`` with no model arg, so
    the call ran under ctx.model while the cache + lockfile said
    otherwise. Asserts that the model arrives unchanged at litellm.
    """
    from unittest.mock import MagicMock

    from dataset_scout.judge import _LiteLLMChatClient

    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"verdict":"positive","subcategory":"x","confidence":0.9,"rationale":"y"}'
        return resp

    fake_litellm = MagicMock()
    fake_litellm.completion.side_effect = fake_completion

    monkeypatch.setattr("dataset_scout.judge.import_litellm", lambda: fake_litellm)

    # ctx points at AOAI; override should win.
    ctx = ScoutContext(
        aoai_endpoint="https://example.openai.azure.com",
        aoai_deployment="azure-deployment-name",
    )
    monkeypatch.setattr(
        "dataset_scout.llm_client.make_token_provider",
        lambda: (lambda: "tok"),
    )
    chat = _LiteLLMChatClient(ctx=ctx, model_override="github_copilot/gpt-5-mini")
    chat.call(messages=[{"role": "user", "content": "hi"}], timeout_s=5.0)

    assert captured["model"] == "github_copilot/gpt-5-mini"
    # And the github_copilot/ branch must NOT have attached the Azure
    # token provider — otherwise litellm sees a malformed config.
    assert "azure_ad_token_provider" not in captured
    assert "api_base" not in captured
