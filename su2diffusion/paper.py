"""Paper-oriented benchmark wrappers for Hamiltonian-to-circuit synthesis."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
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
from .pareto import (
    CircuitDiversityCoverageResult,
    CircuitDiversityMultiTargetPropertyResult,
    CircuitDiversityPropertyResult,
    CircuitDiversityResult,
    CircuitUnitaryCrossFidelityResult,
    ParetoCircuitResult,
    compare_circuit_diversity_properties,
    pareto_frontier_rows,
    plot_circuit_diversity_coverage,
    plot_circuit_diversity_properties,
    plot_circuit_unitary_cross_fidelity,
    plot_multitarget_circuit_diversity_properties,
    plot_pareto_circuit_sampling,
    summarize_circuit_diversity,
    summarize_circuit_diversity_coverage,
    summarize_circuit_diversity_properties,
    summarize_circuit_unitary_cross_fidelity,
    test_circuit_diversity_properties,
    top_pareto_rows,
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
    sample_count: int | None = 256
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
            n_heldout_targets=1,
            train_target_count=1,
            train_steps=2,
            sample_count=4,
            n_random_candidates=16,
            solution_refinement_steps=1,
            basin_refinement_steps=1,
            keep_models=False,
        )
    if normalized in {"quick", "colab-quick"}:
        return PaperBenchmarkConfig(
            name="quick-paper-benchmark",
            run_seeds=(0,),
            n_heldout_targets=4,
            train_target_count=4,
            train_steps=50,
            sample_count=32,
            n_random_candidates=512,
            solution_refinement_steps=10,
            basin_refinement_steps=10,
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
    if benchmark_config.sample_count is not None:
        circuit_config = CircuitExperimentConfig(
            name=circuit_config.name,
            schedule=circuit_config.schedule,
            train=circuit_config.train,
            data=circuit_config.data,
            sample_count=benchmark_config.sample_count,
            eta=circuit_config.eta,
            deterministic_eta=circuit_config.deterministic_eta,
            n_slots=circuit_config.n_slots,
        )

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


def paper_plot_rc() -> dict[str, Any]:
    """Return a compact Matplotlib style for paper draft figures."""

    return {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return path


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_table_value(value: Any) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def _write_latex_table(
    path: Path,
    rows: list[dict[str, Any]],
    caption: str | None = None,
    label: str | None = None,
) -> Path:
    columns = list(rows[0].keys()) if rows else []
    lines = []
    if caption is not None or label is not None:
        lines.append(r"\begin{table}")
        lines.append(r"\centering")
    colspec = "l" * max(1, len(columns))
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    if columns:
        lines.append(" & ".join(_latex_escape(column) for column in columns) + r" \\")
        lines.append(r"\hline")
        for row in rows:
            lines.append(
                " & ".join(_latex_escape(_format_table_value(row[column])) for column in columns)
                + r" \\"
            )
    lines.append(r"\end{tabular}")
    if caption is not None:
        lines.append(rf"\caption{{{_latex_escape(caption)}}}")
    if label is not None:
        lines.append(rf"\label{{{_latex_escape(label)}}}")
    if caption is not None or label is not None:
        lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _save_current_figure(output_dir: Path, stem: str) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig = plt.gcf()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {f"{stem}_png": png_path, f"{stem}_pdf": pdf_path}


def _plot_and_save(output_dir: Path, stem: str, plotter: Any) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    with plt.rc_context(paper_plot_rc()):
        original_show = plt.show
        try:
            # Notebook plotting helpers call ``plt.show()`` for interactive use.
            # Some inline backends clear or replace the current figure after
            # ``show()``, so suppress it while exporting and save the figure
            # object while it is still live.
            plt.show = lambda *args, **kwargs: None
            plotter()
            return _save_current_figure(output_dir, stem)
        finally:
            plt.show = original_show


def _asdict_rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def _artifact_metadata(
    kind: str,
    metadata: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {"artifact_kind": kind, **extra}
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def save_level3f_diversity_artifacts(
    coverage: CircuitDiversityCoverageResult,
    unitary_cross_fidelity: CircuitUnitaryCrossFidelityResult,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Export Level 3F diversity and unitary cross-fidelity results."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    coverage_rows = _asdict_rows(summarize_circuit_diversity_coverage(coverage))
    cross_rows = _asdict_rows(summarize_circuit_unitary_cross_fidelity(unitary_cross_fidelity))
    paths = {
        "metadata": _write_json(
            output / "metadata.json",
            _artifact_metadata(
                "level3f_diversity",
                metadata,
                target=coverage.reference.target.name,
                template=coverage.reference.template.name,
                reference_source=coverage.reference.source,
                cluster_radius=coverage.cluster_radius,
                success_threshold=coverage.success_threshold,
                reference_cluster_count=coverage.reference_cluster_count,
            ),
        ),
        "coverage_csv": _write_csv(output / "coverage_summary.csv", coverage_rows),
        "coverage_tex": _write_latex_table(
            output / "coverage_summary.tex",
            coverage_rows,
            caption="Level 3F circuit-diversity coverage summary.",
            label="tab:level3f-coverage",
        ),
        "unitary_cross_fidelity_csv": _write_csv(
            output / "unitary_cross_fidelity_summary.csv",
            cross_rows,
        ),
        "unitary_cross_fidelity_tex": _write_latex_table(
            output / "unitary_cross_fidelity_summary.tex",
            cross_rows,
            caption="Level 3F unitary cross-fidelity summary.",
            label="tab:level3f-cross-fidelity",
        ),
    }
    paths.update(
        _plot_and_save(
            output,
            "coverage",
            lambda: plot_circuit_diversity_coverage(coverage),
        )
    )
    paths.update(
        _plot_and_save(
            output,
            "unitary_cross_fidelity",
            lambda: plot_circuit_unitary_cross_fidelity(unitary_cross_fidelity),
        )
    )
    return paths


def save_level3g_property_artifacts(
    properties: CircuitDiversityPropertyResult,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Export Level 3G cluster/property comparison tables and figures."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cluster_summary = _asdict_rows(summarize_circuit_diversity_properties(properties, scope="cluster"))
    circuit_summary = _asdict_rows(summarize_circuit_diversity_properties(properties, scope="circuit"))
    cluster_tests = _asdict_rows(test_circuit_diversity_properties(properties, scope="cluster"))
    paths = {
        "metadata": _write_json(
            output / "metadata.json",
            _artifact_metadata(
                "level3g_properties",
                metadata,
                cluster_radius=properties.cluster_radius,
                success_threshold=properties.success_threshold,
                sources=list(properties.results),
            ),
        ),
        "cluster_summary_csv": _write_csv(output / "cluster_summary.csv", cluster_summary),
        "cluster_summary_tex": _write_latex_table(
            output / "cluster_summary.tex",
            cluster_summary,
            caption="Level 3G cluster-level circuit-property summary.",
            label="tab:level3g-cluster-summary",
        ),
        "cluster_tests_csv": _write_csv(output / "cluster_tests.csv", cluster_tests),
        "cluster_tests_tex": _write_latex_table(
            output / "cluster_tests.tex",
            cluster_tests,
            caption="Level 3G non-parametric property tests.",
            label="tab:level3g-property-tests",
        ),
        "circuit_summary_csv": _write_csv(output / "circuit_summary.csv", circuit_summary),
        "circuit_summary_tex": _write_latex_table(
            output / "circuit_summary.tex",
            circuit_summary,
            caption="Level 3G per-circuit property robustness summary.",
            label="tab:level3g-circuit-summary",
        ),
    }
    paths.update(
        _plot_and_save(
            output,
            "cluster_properties",
            lambda: plot_circuit_diversity_properties(properties, scope="cluster"),
        )
    )
    paths.update(
        _plot_and_save(
            output,
            "circuit_properties",
            lambda: plot_circuit_diversity_properties(properties, scope="circuit"),
        )
    )
    return paths


def save_level3h_multitarget_artifacts(
    result: CircuitDiversityMultiTargetPropertyResult,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Export Level 3H multi-target diversity/property robustness results."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = _asdict_rows(result.rows)
    paths = {
        "metadata": _write_json(
            output / "metadata.json",
            _artifact_metadata(
                "level3h_multitarget",
                metadata,
                cluster_radius=result.cluster_radius,
                success_threshold=result.success_threshold,
                primary_source=result.primary_source,
                reference_source=result.reference_source,
                targets=list(result.diagnostics),
                regimes=result.regimes,
            ),
        ),
        "summary_csv": _write_csv(output / "summary.csv", rows),
        "summary_tex": _write_latex_table(
            output / "summary.tex",
            rows,
            caption="Level 3H multi-target circuit-diversity and property summary.",
            label="tab:level3h-summary",
        ),
    }
    paths.update(
        _plot_and_save(
            output,
            "multitarget_properties",
            lambda: plot_multitarget_circuit_diversity_properties(result),
        )
    )
    return paths


def summarize_angle_steering_artifacts(
    diagnostics: dict[str, dict[str, CircuitDiversityResult]],
    train_angle_by_source: dict[str, float],
    cluster_radius: float = 0.15,
    success_threshold: float = 0.99,
    source_order: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Summarize Level 3I low-/high-angle training-policy diagnostics."""

    rows: list[dict[str, Any]] = []
    for target_name, result_map in diagnostics.items():
        if source_order is None:
            sources = tuple(result_map)
        else:
            sources = source_order
        properties = compare_circuit_diversity_properties(
            result_map,
            cluster_radius=cluster_radius,
            success_threshold=success_threshold,
        )
        property_by_source = {
            row.source: row
            for row in summarize_circuit_diversity_properties(properties, scope="cluster")
        }
        for source in sources:
            diversity_summary = summarize_circuit_diversity(result_map[source])
            property_summary = property_by_source[source]
            rows.append(
                {
                    "target": target_name,
                    "source": source,
                    "train_angle_mean": train_angle_by_source.get(source, float("nan")),
                    "success_fraction": diversity_summary.success_fraction,
                    "cluster_count": property_summary.n,
                    "output_angle_mean": property_summary.angle_mean,
                    "output_angle_std": property_summary.angle_std,
                    "refined_mean": property_summary.refined_mean,
                    "refined_std": property_summary.refined_std,
                }
            )
    return rows


def _plot_angle_steering_rows(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    targets = list(dict.fromkeys(row["target"] for row in rows))
    sources = list(dict.fromkeys(row["source"] for row in rows))
    xs = torch.arange(len(targets), dtype=torch.float32).numpy()
    width = 0.8 / max(1, len(sources))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for index, source in enumerate(sources):
        group = {row["target"]: row for row in rows if row["source"] == source}
        offset = (index - 0.5 * (len(sources) - 1)) * width
        axes[0].bar(
            xs + offset,
            [group[target]["output_angle_mean"] for target in targets],
            width,
            label=source,
        )
        axes[1].bar(
            xs + offset,
            [group[target]["success_fraction"] for target in targets],
            width,
            label=source,
        )
    axes[0].set_title("output total local angle")
    axes[0].set_ylabel("radians")
    axes[1].set_title("refined success")
    axes[1].set_ylabel("fraction with F >= threshold")
    axes[1].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.set_xticks(xs)
        axis.set_xticklabels(targets, rotation=20, ha="right")
        axis.legend()
    fig.suptitle("Angle-steering ablation")
    fig.tight_layout()


def save_level3i_angle_steering_artifacts(
    diagnostics: dict[str, dict[str, CircuitDiversityResult]],
    train_angle_by_source: dict[str, float],
    output_dir: str | Path,
    cluster_radius: float = 0.15,
    success_threshold: float = 0.99,
    source_order: tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Export Level 3I angle-steering ablation summaries and figures."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = summarize_angle_steering_artifacts(
        diagnostics,
        train_angle_by_source,
        cluster_radius=cluster_radius,
        success_threshold=success_threshold,
        source_order=source_order,
    )
    paths = {
        "metadata": _write_json(
            output / "metadata.json",
            _artifact_metadata(
                "level3i_angle_steering",
                metadata,
                cluster_radius=cluster_radius,
                success_threshold=success_threshold,
                train_angle_by_source=train_angle_by_source,
                targets=list(diagnostics),
            ),
        ),
        "summary_csv": _write_csv(output / "summary.csv", rows),
        "summary_tex": _write_latex_table(
            output / "summary.tex",
            rows,
            caption="Level 3I low-/high-angle training-policy ablation.",
            label="tab:level3i-angle-steering",
        ),
    }
    paths.update(
        _plot_and_save(
            output,
            "angle_steering",
            lambda: _plot_angle_steering_rows(rows),
        )
    )
    return paths


def _pareto_candidate_export_rows(
    result: ParetoCircuitResult,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for row in top_pareto_rows(result, max_rows=max_rows):
        rows.append(
            {
                "is_pareto": row.is_pareto,
                "template": row.template,
                "candidate_rank": row.candidate_rank,
                "n_cz": row.n_cz,
                "n_local_gates": row.n_local_gates,
                "proposal_fidelity": row.proposal_fidelity,
                "refined_fidelity": row.refined_fidelity,
                "steps_to_threshold": row.steps_to_threshold,
                "movement_mean": row.movement_mean,
                "movement_max": row.movement_max,
                "local_angle_sum": row.local_angle_sum,
                "hardware_cost": row.hardware_cost,
                "regularized_score": row.regularized_score,
                "slot_labels": ", ".join(label or "" for label in row.slot_labels),
            }
        )
    return rows


def _pareto_template_summary_rows(result: ParetoCircuitResult) -> list[dict[str, Any]]:
    rows = []
    for template in [item.name for item in result.templates]:
        group = [row for row in result.rows if row.template == template]
        if not group:
            continue
        refined = torch.tensor([row.refined_fidelity for row in group], dtype=torch.float32)
        proposal = torch.tensor([row.proposal_fidelity for row in group], dtype=torch.float32)
        cost = torch.tensor([row.hardware_cost for row in group], dtype=torch.float32)
        score = torch.tensor([row.regularized_score for row in group], dtype=torch.float32)
        frontier = [row for row in group if row.is_pareto]
        rows.append(
            {
                "template": template,
                "n": len(group),
                "n_cz": group[0].n_cz,
                "n_local_gates": group[0].n_local_gates,
                "proposal_mean": float(proposal.mean().item()),
                "refined_best": float(refined.max().item()),
                "refined_mean": float(refined.mean().item()),
                "hardware_cost_min": float(cost.min().item()),
                "hardware_cost_mean": float(cost.mean().item()),
                "regularized_score_best": float(score.max().item()),
                "frontier_count": len(frontier),
            }
        )
    return rows


def save_level3j_pareto_artifacts(
    result: ParetoCircuitResult,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    max_candidate_rows: int | None = None,
) -> dict[str, Path]:
    """Export Level 3J Pareto candidate-cloud tables and figures."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_rows = _pareto_candidate_export_rows(result, max_rows=max_candidate_rows)
    summary_rows = _pareto_template_summary_rows(result)
    frontier_rows = _pareto_candidate_export_rows(
        replace(result, rows=pareto_frontier_rows(result)),
        max_rows=None,
    )

    paths = {
        "metadata": _write_json(
            output / "metadata.json",
            _artifact_metadata(
                "level3j_pareto",
                metadata,
                target=result.target.name,
                source=result.source,
                threshold=result.threshold,
                templates=[template.name for template in result.templates],
                scoring=asdict(result.scoring),
            ),
        ),
        "candidates_csv": _write_csv(output / "candidates.csv", candidate_rows),
        "candidates_tex": _write_latex_table(
            output / "candidates.tex",
            candidate_rows,
            caption="Level 3J top Pareto candidate rows.",
            label="tab:level3j-pareto-candidates",
        ),
        "template_summary_csv": _write_csv(output / "template_summary.csv", summary_rows),
        "template_summary_tex": _write_latex_table(
            output / "template_summary.tex",
            summary_rows,
            caption="Level 3J Pareto template summary.",
            label="tab:level3j-pareto-summary",
        ),
        "frontier_csv": _write_csv(output / "frontier.csv", frontier_rows),
        "frontier_tex": _write_latex_table(
            output / "frontier.tex",
            frontier_rows,
            caption="Level 3J Pareto frontier rows.",
            label="tab:level3j-pareto-frontier",
        ),
    }
    paths.update(
        _plot_and_save(
            output,
            "pareto_candidate_cloud",
            lambda: plot_pareto_circuit_sampling(result),
        )
    )
    return paths
