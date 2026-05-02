from su2diffusion.diffusion import DiffusionSchedule
from su2diffusion.forward_checks import diagnose_forward_process


def test_forward_process_diagnostics_are_finite():
    diagnostics = diagnose_forward_process(
        schedule=DiffusionSchedule(T=6, beta_start=1e-4, beta_end=0.005),
        batch_size=16,
        timesteps=[1, 3, 6],
        n_terms=8,
        device="cpu",
    )

    assert diagnostics.timesteps == [1, 3, 6]
    assert diagnostics.norm_error_mean < 1e-5
    assert diagnostics.final_distance_to_haar_w1 >= 0.0
    assert diagnostics.target_abs_mean >= 0.0
    assert diagnostics.target_abs_max >= diagnostics.target_abs_mean
    assert diagnostics.target_finite


def test_forward_process_distances_generally_increase():
    diagnostics = diagnose_forward_process(
        schedule=DiffusionSchedule(T=20, beta_start=1e-4, beta_end=0.02),
        batch_size=128,
        timesteps=[1, 10, 20],
        n_terms=8,
        device="cpu",
    )

    first, _, last = diagnostics.mean_nearest_center_distance

    assert last > first
