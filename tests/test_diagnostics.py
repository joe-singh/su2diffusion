import torch

from su2diffusion.data import default_centers, sample_clean_blobs
from su2diffusion.diagnostics import diagnose_samples, nearest_center_mass, wasserstein_1d
from su2diffusion.quaternion import sample_haar


def test_wasserstein_1d_identical_samples_is_zero():
    x = torch.tensor([0.0, 1.0, 2.0])

    assert wasserstein_1d(x, x) == 0.0


def test_nearest_center_mass_sums_to_one():
    centers = default_centers(device="cpu")
    q, _ = sample_clean_blobs(32, centers=centers)

    mass = nearest_center_mass(q, centers=centers)

    assert len(mass) == centers.shape[0]
    assert abs(sum(mass) - 1.0) < 1e-6


def test_diagnose_samples_returns_finite_metrics():
    centers = default_centers(device="cpu")
    q_clean, _ = sample_clean_blobs(16, centers=centers)
    q_haar = sample_haar(16, device="cpu")

    diagnostics = diagnose_samples(q_clean, q_clean, q_haar, centers=centers)

    assert diagnostics.norm_error_mean < 1e-5
    assert diagnostics.distance_to_clean_w1 >= 0.0
    assert diagnostics.distance_to_haar_w1 >= 0.0
    assert diagnostics.nearest_center_distance.mean >= 0.0
    assert len(diagnostics.nearest_center_mass) == centers.shape[0]
