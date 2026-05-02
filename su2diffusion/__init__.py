"""Tools for toy diffusion experiments on SU(2)."""

from .data import (
    BlobConfig,
    DataConfig,
    center_names_for_config,
    clifford_centers,
    gate_centers,
    sample_balanced_labels,
    sample_clean,
    sample_clean_blobs,
)
from .diagnostics import (
    ConditionalLabelDiagnostics,
    SampleDiagnostics,
    diagnose_conditional_labels,
    diagnose_samples,
    print_center_mass_table,
    print_conditional_label_table,
    print_diagnostics_table,
    print_per_center_table,
)
from .diffusion import DiffusionSchedule, brownian_forward_heat_target, heat_epsilon_target
from .experiment import ExperimentConfig, ExperimentResult, ResampleResult, get_experiment_config, resample_experiment, run_experiment
from .forward_checks import ForwardProcessDiagnostics, diagnose_forward_process, print_forward_diagnostics
from .model import SU2Denoiser
from .quaternion import q_exp, q_inv, q_log, q_mul, q_normalize, sample_haar, su2_distance
from .sampling import sample_reverse, sample_reverse_trajectory
from .train import TrainConfig, train_heat_kernel_model
from .viz import plot_experiment_report

__all__ = [
    "BlobConfig",
    "DataConfig",
    "DiffusionSchedule",
    "ExperimentConfig",
    "ExperimentResult",
    "ForwardProcessDiagnostics",
    "ConditionalLabelDiagnostics",
    "ResampleResult",
    "SampleDiagnostics",
    "SU2Denoiser",
    "TrainConfig",
    "brownian_forward_heat_target",
    "center_names_for_config",
    "clifford_centers",
    "diagnose_conditional_labels",
    "diagnose_samples",
    "diagnose_forward_process",
    "gate_centers",
    "get_experiment_config",
    "heat_epsilon_target",
    "print_center_mass_table",
    "print_conditional_label_table",
    "print_diagnostics_table",
    "print_per_center_table",
    "print_forward_diagnostics",
    "plot_experiment_report",
    "q_exp",
    "q_inv",
    "q_log",
    "q_mul",
    "q_normalize",
    "resample_experiment",
    "run_experiment",
    "sample_clean",
    "sample_balanced_labels",
    "sample_clean_blobs",
    "sample_haar",
    "sample_reverse",
    "sample_reverse_trajectory",
    "su2_distance",
    "train_heat_kernel_model",
]
