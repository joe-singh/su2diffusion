from pathlib import Path

import pytest

from su2diffusion import (
    PaperBenchmarkSuiteResult,
    ThreeQubitTokenRepeatabilityResult,
    ThreeQubitTokenRepeatabilityRunRow,
    get_paper_benchmark_config,
    get_three_qubit_cz_template,
    paper_benchmark_rows,
    paper_benchmark_summary_rows,
    save_paper_benchmark_artifacts,
)


def _fake_suite() -> PaperBenchmarkSuiteResult:
    repeatability = ThreeQubitTokenRepeatabilityResult(
        runs=[],
        rows=[
            ThreeQubitTokenRepeatabilityRunRow(
                run=0,
                source="token",
                n_targets=4,
                proposal_mean=0.9,
                refined_mean=0.99,
                refinement_success=1.0,
                median_steps=4.0,
                mean_movement=0.1,
                max_movement=0.2,
            ),
            ThreeQubitTokenRepeatabilityRunRow(
                run=1,
                source="token",
                n_targets=4,
                proposal_mean=0.8,
                refined_mean=0.97,
                refinement_success=0.5,
                median_steps=6.0,
                mean_movement=0.2,
                max_movement=0.3,
            ),
            ThreeQubitTokenRepeatabilityRunRow(
                run=0,
                source="haar",
                n_targets=4,
                proposal_mean=0.1,
                refined_mean=0.95,
                refinement_success=0.25,
                median_steps=10.0,
                mean_movement=0.5,
                max_movement=0.7,
            ),
        ],
        threshold=0.99,
        template=get_three_qubit_cz_template("line-4cz"),
    )
    return PaperBenchmarkSuiteResult(
        config=get_paper_benchmark_config("smoke"),
        repeatability=repeatability,
    )


def test_named_paper_benchmark_configs() -> None:
    smoke = get_paper_benchmark_config("smoke")
    quick = get_paper_benchmark_config("quick")
    full = get_paper_benchmark_config("level3")

    assert smoke.train_steps < quick.train_steps < full.train_steps
    assert smoke.template == quick.template == full.template == "line-4cz"
    assert full.n_heldout_targets == 48
    with pytest.raises(ValueError):
        get_paper_benchmark_config("not-a-config")


def test_paper_benchmark_rows_and_summary() -> None:
    suite = _fake_suite()

    rows = paper_benchmark_rows(suite)
    assert len(rows) == 3
    assert rows[0]["source"] == "token"

    summary = paper_benchmark_summary_rows(suite)
    assert [row.source for row in summary] == ["token", "haar"]
    token = summary[0]
    assert token.runs == 2
    assert token.n_per_run == 4
    assert token.proposal_mean == pytest.approx(0.85)
    assert token.refined_mean == pytest.approx(0.98)
    assert token.success_mean == pytest.approx(0.75)
    assert token.max_movement == pytest.approx(0.3)


def test_save_paper_benchmark_artifacts(tmp_path: Path) -> None:
    suite = _fake_suite()

    paths = save_paper_benchmark_artifacts(suite, tmp_path)

    assert set(paths) == {"config", "rows", "summary", "figure"}
    for path in paths.values():
        assert path.exists()
    assert "token" in paths["rows"].read_text()
    assert "proposal_mean" in paths["summary"].read_text()

