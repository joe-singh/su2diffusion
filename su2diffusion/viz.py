import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from typing import TYPE_CHECKING

from .data import centers_for_config, default_centers, sample_clean_blobs
from .quaternion import sample_haar, su2_distance

if TYPE_CHECKING:
    from .diagnostics import SampleDiagnostics


@torch.no_grad()
def plot_quaternions(q: torch.Tensor, title: str, labels: torch.Tensor | None = None, n_plot: int = 3000) -> None:
    q = q.detach().cpu()[:n_plot]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    if labels is None:
        ax.scatter(q[:, 1], q[:, 2], q[:, 3], s=4, alpha=0.6)
    else:
        labels_cpu = labels[:n_plot].detach().cpu()
        ax.scatter(q[:, 1], q[:, 2], q[:, 3], c=labels_cpu, s=4, alpha=0.6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    plt.show()


@torch.no_grad()
def nearest_center_dist(q: torch.Tensor, centers: torch.Tensor | None = None) -> torch.Tensor:
    device = q.device
    centers = centers if centers is not None else default_centers(device=device)
    dists = []

    for k in range(centers.shape[0]):
        c = centers[k].view(1, 4).expand(q.shape[0], 4)
        dists.append(su2_distance(c, q))

    dists = torch.stack(dists, dim=1)
    return dists.min(dim=1).values


@torch.no_grad()
def plot_nearest_center_histogram(
    q_clean: torch.Tensor,
    q_haar: torch.Tensor,
    generated: dict[str, torch.Tensor],
    centers: torch.Tensor | None = None,
    clean_label: str = "Clean reference",
    center_label: str = "nearest center",
) -> None:
    centers = centers if centers is not None else default_centers(device=q_clean.device)

    plt.figure(figsize=(7, 4))
    plt.hist(nearest_center_dist(q_haar, centers).cpu(), bins=60, alpha=0.4, density=True, label="Haar")
    plt.hist(nearest_center_dist(q_clean, centers).cpu(), bins=60, alpha=0.4, density=True, label=clean_label)

    for label, q in generated.items():
        plt.hist(nearest_center_dist(q, centers).cpu(), bins=60, alpha=0.4, density=True, label=label)

    plt.xlabel(f"distance to {center_label}")
    plt.ylabel("density")
    plt.legend()
    plt.title(f"Generated samples should match {center_label} distances")
    plt.show()


def plot_loss(losses: list[float], title: str = "Training loss") -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.show()


def plot_center_mass(
    diagnostics: dict[str, "SampleDiagnostics"],
    title: str = "Nearest-center mass",
) -> None:
    names = list(diagnostics)
    if not names:
        raise ValueError("plot_center_mass needs at least one diagnostics entry")

    masses = torch.tensor([diagnostics[name].nearest_center_mass for name in names])
    x = torch.arange(masses.shape[1]).float()
    width = min(0.8 / len(names), 0.35)

    plt.figure(figsize=(7, 4))
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        plt.bar((x + offset).tolist(), masses[i].tolist(), width=width, label=name)

    plt.xlabel("nearest center")
    plt.ylabel("sample fraction")
    plt.ylim(0.0, max(1.0, float(masses.max().item()) * 1.15))
    plt.title(title)
    plt.legend()
    plt.show()


def plot_diagnostics_bars(
    diagnostics: dict[str, "SampleDiagnostics"],
    title: str = "Sample diagnostics",
) -> None:
    names = list(diagnostics)
    if not names:
        raise ValueError("plot_diagnostics_bars needs at least one diagnostics entry")

    metrics = {
        "W1 clean": [diagnostics[name].distance_to_clean_w1 for name in names],
        "W1 Haar": [diagnostics[name].distance_to_haar_w1 for name in names],
        "mass L1": [diagnostics[name].center_mass_l1 for name in names],
        "dist mean": [diagnostics[name].nearest_center_distance.mean for name in names],
    }

    x = torch.arange(len(metrics)).float()
    width = min(0.8 / len(names), 0.35)

    plt.figure(figsize=(7, 4))
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * width
        values = [metrics[metric][i] for metric in metrics]
        plt.bar((x + offset).tolist(), values, width=width, label=name)

    plt.xticks(x.tolist(), list(metrics))
    plt.ylabel("metric value")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_experiment_report(result) -> None:
    """Render standard plots for an ExperimentResult."""
    centers = centers_for_config(result.config.data, device=result.clean_reference.device)
    center_label = f"nearest {result.config.data.kind} center"
    plot_loss(result.losses, title=f"{result.config.name} training loss")
    plot_nearest_center_histogram(
        result.clean_reference,
        result.haar_reference,
        {
            "Generated deterministic": result.generated_deterministic,
            f"Generated stochastic eta={result.config.eta}": result.generated_stochastic,
        },
        centers=centers,
        clean_label=f"Clean {result.config.data.kind}",
        center_label=center_label,
    )
    plot_diagnostics_bars(result.diagnostics)
    plot_center_mass(result.diagnostics)


def animate_reverse_histogram(
    frames: list[torch.Tensor],
    t_values: list[int],
    clean_reference: torch.Tensor | None = None,
    haar_reference: torch.Tensor | None = None,
    bins: int = 50,
    device: torch.device | str | None = None,
):
    device = torch.device(device) if device is not None else torch.device("cpu")
    centers = default_centers(device=device)
    fig, ax = plt.subplots(figsize=(7, 4))

    if clean_reference is None:
        clean_reference, _ = sample_clean_blobs(5000, centers=centers)
    if haar_reference is None:
        haar_reference = sample_haar(5000, device=device)

    d_clean = nearest_center_dist(clean_reference.to(device), centers).detach().cpu().numpy()
    d_haar = nearest_center_dist(haar_reference.to(device), centers).detach().cpu().numpy()

    def update(i):
        ax.cla()
        d_frame = nearest_center_dist(frames[i].to(device), centers).detach().cpu().numpy()
        ax.hist(d_haar, bins=bins, density=True, alpha=0.35, label="Haar final distribution")
        ax.hist(d_clean, bins=bins, density=True, alpha=0.35, label="Starting distribution")
        ax.hist(d_frame, bins=bins, density=True, alpha=0.45, label="Current reverse samples")
        ax.set_xlabel("distance to nearest blob center")
        ax.set_ylabel("density")
        ax.set_title(f"Heat-target reverse process, t={t_values[i]}")
        ax.legend()

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=False)
    plt.close(fig)
    return anim
