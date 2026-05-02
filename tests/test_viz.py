from types import SimpleNamespace

import torch

from su2diffusion.diagnostics import DistanceSummary, SampleDiagnostics
from su2diffusion.experiment import ExperimentConfig
from su2diffusion.diffusion import DiffusionSchedule
from su2diffusion.data import DataConfig
from su2diffusion.train import TrainConfig
from su2diffusion import viz


def _diagnostics():
    summary = DistanceSummary(mean=0.5, std=0.1, q05=0.2, q50=0.5, q95=0.8)
    return {
        "deterministic": SampleDiagnostics(
            norm_error_mean=0.0,
            distance_to_clean_w1=0.1,
            distance_to_haar_w1=0.4,
            nearest_center_distance=summary,
            nearest_center_mass=[0.25, 0.25, 0.25, 0.25],
            clean_center_mass=[0.25, 0.25, 0.25, 0.25],
            center_mass_l1=0.0,
        ),
        "stochastic": SampleDiagnostics(
            norm_error_mean=0.0,
            distance_to_clean_w1=0.2,
            distance_to_haar_w1=0.3,
            nearest_center_distance=summary,
            nearest_center_mass=[0.2, 0.3, 0.3, 0.2],
            clean_center_mass=[0.25, 0.25, 0.25, 0.25],
            center_mass_l1=0.2,
        ),
    }


def test_plot_experiment_report_calls_standard_plots(monkeypatch):
    calls = []
    config = ExperimentConfig(
        name="test",
        schedule=DiffusionSchedule(T=3),
        train=TrainConfig(num_steps=1),
        data=DataConfig(kind="gates"),
        eta=0.7,
    )
    result = SimpleNamespace(
        config=config,
        losses=[1.0],
        clean_reference=torch.zeros(1, 4),
        haar_reference=torch.zeros(1, 4),
        generated_deterministic=torch.zeros(1, 4),
        generated_stochastic=torch.zeros(1, 4),
        diagnostics=_diagnostics(),
    )

    monkeypatch.setattr(viz, "plot_loss", lambda *args, **kwargs: calls.append("loss"))
    def fake_hist(*args, **kwargs):
        assert kwargs["centers"].shape == (7, 4)
        assert kwargs["clean_label"] == "Clean gates"
        calls.append("hist")

    monkeypatch.setattr(viz, "plot_nearest_center_histogram", fake_hist)
    monkeypatch.setattr(viz, "plot_diagnostics_bars", lambda *args, **kwargs: calls.append("bars"))
    monkeypatch.setattr(viz, "plot_center_mass", lambda *args, **kwargs: calls.append("mass"))

    viz.plot_experiment_report(result)

    assert calls == ["loss", "hist", "bars", "mass"]
