"""Paper-oriented benchmark wrappers for Hamiltonian-to-circuit synthesis."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .circuit import CircuitExperimentConfig, get_circuit_experiment_config
from .hamiltonian import (
    ThreeQubitTokenRepeatabilityResult,
    ThreeQubitTokenRepeatabilityRunRow,
    plot_three_qubit_token_repeatability,
    print_three_qubit_token_repeatability,
    print_three_qubit_token_repeatability_summary,
    run_three_qubit_hamiltonian_token_repeatability_benchmark,
    summarize_three_qubit_token_repeatability,
)


@dataclass(frozen=True)
class PaperBenchmarkConfig:
    """Canonical three-qubit Hamiltonian benchmark settings."""

    name: str = "level3-line4cz-repeatability"
    run_seeds: tuple[int, ...] = (0, 1, 2)
    n_heldout_targets: int = 48
    train_target_count: int = 32
    train_steps: int = 500
    template: str = "line-4cz"
    terms: tuple[str, ...] = ("XII", "IZI", "IIZ", "XXI", "IZZ", "ZXZ")
    coefficient_scale: float = 0.25
    time: float = 0.6
    n_random_candidates: int = 10_000
    top_k: int = 1
    solution_refinement_steps: int = 80
    basin_refinement_steps: int = 80
    refinement_lr: float = 0.05
    fidelity_threshold: float = 0.0
    threshold: float = 0.99
    solutions_per_target: int = 1
    solution_selection: str = "top"
    selection_pool_size: int | None = None
    keep_models: bool = False
    clear_cuda_cache: bool = True


@dataclass
class PaperBenchmarkSuiteResult:
    """Result object for the paper-style benchmark wrapper."""

    config: PaperBenchmarkConfig
    repeatability: ThreeQubitTokenRepeatabilityResult


@dataclass(frozen=True)
class PaperBenchmarkSummaryRow:
    source: str
    runs: int
    n_per_run: int
    proposal_mean: float
    proposal_std: float
    refined_mean: float
    refined_std: float
    success_mean: float
    success_std: float
    median_steps: float
    mean_movement: float
    max_movement: float


def get_paper_benchmark_config(name: str = "level3") -> PaperBenchmarkConfig:
    """Return a named paper-benchmark preset.

    `smoke` is for local syntax/CPU checks, `quick` is for a short Colab sanity run,
    and `level3` is the current paper-style three-qubit repeatability benchmark.
    """

    normalized = name.lower().replace("_", "-")
    if normalized in {"smoke", "test"}:
        return PaperBenchmarkConfig(
            name="smoke-paper-benchmark",
            run_seeds=(0,),
            n_heldout_targets=3,
            train_target_count=2,
            train_steps=5,
            n_random_candidates=64,
            solution_refinement_steps=2,
            basin_refinement_steps=2,
            keep_models=False,
        )
    if normalized in {"quick", "colab-quick"}:
        return PaperBenchmarkConfig(
            name="quick-paper-benchmark",
            run_seeds=(0,),
            n_heldout_targets=12,
            train_target_count=16,
            train_steps=250,
            n_random_candidates=5_000,
            keep_models=False,
        )
    if normalized in {"level3", "paper", "full"}:
        return PaperBenchmarkConfig()
    raise ValueError(f"Unknown paper benchmark config {name!r}")


def run_paper_benchmark_suite(
    generated_gates: torch.Tensor,
    generated_labels: list[str],
    circuit_config: CircuitExperimentConfig | str,
    benchmark_config: PaperBenchmarkConfig | str | None = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> PaperBenchmarkSuiteResult:
    """Run the canonical paper-style three-qubit Hamiltonian benchmark."""

    if isinstance(circuit_config, str):
        circuit_config = get_circuit_experiment_config(circuit_config)
    if benchmark_config is None:
        benchmark_config = get_paper_benchmark_config("level3")
    elif isinstance(benchmark_config, str):
        benchmark_config = get_paper_benchmark_config(benchmark_config)

    repeatability = run_three_qubit_hamiltonian_token_repeatability_benchmark(
        generated_gates=generated_gates,
        generated_labels=generated_labels,
        config=circuit_config,
        run_seeds=benchmark_config.run_seeds,
        n_heldout_targets=benchmark_config.n_heldout_targets,
        train_target_count=benchmark_config.train_target_count,
        train_steps=benchmark_config.train_steps,
        template=benchmark_config.template,
        terms=benchmark_config.terms,
        coefficient_scale=benchmark_config.coefficient_scale,
        time=benchmark_config.time,
        n_random_candidates=benchmark_config.n_random_candidates,
        top_k=benchmark_config.top_k,
        solution_refinement_steps=benchmark_config.solution_refinement_steps,
        basin_refinement_steps=benchmark_config.basin_refinement_steps,
        refinement_lr=benchmark_config.refinement_lr,
        fidelity_threshold=benchmark_config.fidelity_threshold,
        threshold=benchmark_config.threshold,
        solutions_per_target=benchmark_config.solutions_per_target,
        solution_selection=benchmark_config.solution_selection,
        selection_pool_size=benchmark_config.selection_pool_size,
        keep_models=benchmark_config.keep_models,
        clear_cuda_cache=benchmark_config.clear_cuda_cache,
        device=device,
        show_progress=show_progress,
    )
    return PaperBenchmarkSuiteResult(config=benchmark_config, repeatability=repeatability)


def paper_benchmark_rows(result: PaperBenchmarkSuiteResult) -> list[dict[str, Any]]:
    """Return per-run rows as plain dictionaries for CSV/JSON export."""

    return [asdict(row) for row in summarize_three_qubit_token_repeatability(result.repeatability)]


def _mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() == 0:
        return float("nan"), float("nan")
    std = tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0
    return float(tensor.mean().item()), float(std)


def paper_benchmark_summary_rows(result: PaperBenchmarkSuiteResult) -> list[PaperBenchmarkSummaryRow]:
    """Aggregate repeatability rows by proposal source."""

    rows = summarize_three_qubit_token_repeatability(result.repeatability)
    sources: list[str] = []
    for row in rows:
        if row.source not in sources:
            sources.append(row.source)

    summary: list[PaperBenchmarkSummaryRow] = []
    for source in sources:
        group = [row for row in rows if row.source == source]
        proposal_mean, proposal_std = _mean_std([row.proposal_mean for row in group])
        refined_mean, refined_std = _mean_std([row.refined_mean for row in group])
        success_mean, success_std = _mean_std([row.refinement_success for row in group])
        n_per_run = int(round(sum(row.n_targets for row in group) / len(group)))
        median_steps = torch.tensor([row.median_steps for row in group], dtype=torch.float32)
        mean_move = torch.tensor([row.mean_movement for row in group], dtype=torch.float32)
        max_move = torch.tensor([row.max_movement for row in group], dtype=torch.float32)
        summary.append(
            PaperBenchmarkSummaryRow(
                source=source,
                runs=len(group),
                n_per_run=n_per_run,
                proposal_mean=proposal_mean,
                proposal_std=proposal_std,
                refined_mean=refined_mean,
                refined_std=refined_std,
                success_mean=success_mean,
                success_std=success_std,
                median_steps=float(median_steps.median().item()),
                mean_movement=float(mean_move.mean().item()),
                max_movement=float(max_move.max().item()),
            )
        )
    return summary


def print_paper_benchmark_suite(result: PaperBenchmarkSuiteResult, include_runs: bool = True) -> None:
    """Print the paper benchmark config and headline tables."""

    cfg = result.config
    print(f"paper benchmark: {cfg.name}")
    print(f"template:        {cfg.template}")
    print(f"target family:   {', '.join(cfg.terms)}")
    print(f"time:            {cfg.time:g}")
    print(f"runs:            {', '.join(str(seed) for seed in cfg.run_seeds)}")
    print(f"heldout targets: {cfg.n_heldout_targets}")
    print(f"train targets:   {cfg.train_target_count}")
    print(f"train steps:     {cfg.train_steps}")
    print()
    if include_runs:
        print_three_qubit_token_repeatability(result.repeatability)
        print()
    print_three_qubit_token_repeatability_summary(result.repeatability)


def plot_paper_benchmark_suite(result: PaperBenchmarkSuiteResult) -> None:
    """Plot the canonical repeatability figure."""

    plot_three_qubit_token_repeatability(result.repeatability)


def save_paper_benchmark_artifacts(
    result: PaperBenchmarkSuiteResult,
    output_dir: str | Path,
    figure_name: str = "paper_benchmark_repeatability.png",
) -> dict[str, Path]:
    """Save config, row CSVs, and the repeatability figure."""

    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config_path = output / "paper_benchmark_config.json"
    config_path.write_text(json.dumps(asdict(result.config), indent=2) + "\n")

    rows = paper_benchmark_rows(result)
    rows_path = output / "paper_benchmark_rows.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = [asdict(row) for row in paper_benchmark_summary_rows(result)]
    summary_path = output / "paper_benchmark_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)

    plot_paper_benchmark_suite(result)
    figure_path = output / figure_name
    plt.gcf().savefig(figure_path, dpi=200, bbox_inches="tight")

    return {
        "config": config_path,
        "rows": rows_path,
        "summary": summary_path,
        "figure": figure_path,
    }

