"""Plots for model analysis and low-dimensional dynamics."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import numpy.typing as npt

from low_rank_rnn.plotting.style import (
    COLORS,
    COVARIANCE_CMAP,
    COVARIANCE_COLORS,
    REFERENCE_LINE_STYLE,
    SIGNED_VALUE_CMAP,
)


def _plot_performance_comparison(
    trained_value: float,
    sampled_values: npt.ArrayLike,
    *,
    ylabel: str,
    title: str,
    baseline: float | None = None,
    baseline_label: str | None = None,
    value_format: str = "{:.1%}",
    axis: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    sampled_values = np.asarray(sampled_values, dtype=float)
    sample_positions = np.linspace(0.86, 1.14, len(sampled_values))
    sampled_mean = float(sampled_values.mean())
    owns_figure = axis is None
    fig, axis = (
        plt.subplots(figsize=(6.4, 4.2))
        if owns_figure
        else (axis.figure, axis)
    )

    axis.scatter(
        0,
        trained_value,
        color=COLORS["green"],
        marker="D",
        s=80,
        zorder=3,
    )
    axis.scatter(
        sample_positions,
        sampled_values,
        color=COLORS["purple"],
        edgecolor="white",
        linewidth=0.6,
        s=65,
        zorder=3,
    )
    axis.axhline(
        trained_value,
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.2,
        label=f"trained network: {value_format.format(trained_value)}",
    )
    axis.hlines(
        sampled_mean,
        0.82,
        1.18,
        color=COLORS["purple"],
        linewidth=2.2,
        label=f"sample mean: {value_format.format(sampled_mean)}",
    )
    if baseline is not None:
        axis.axhline(
            baseline,
            color=COLORS["gray"],
            linestyle=":",
            linewidth=1,
            label=baseline_label,
        )

    axis.set_xticks((0, 1), labels=("Trained network", "Gaussian samples"))
    axis.set_xlim(-0.35, 1.35)
    axis.set(ylabel=ylabel, title=title)
    axis.legend(loc="lower left")
    if owns_figure:
        fig.tight_layout()
    return fig, axis


def plot_accuracy_comparison(
    trained_accuracy: float,
    sampled_accuracies: npt.ArrayLike,
    *,
    title: str = "Held-out perceptual decision performance",
) -> tuple[plt.Figure, plt.Axes]:
    """Compare a trained network with Gaussian-sampled networks."""
    fig, axis = _plot_performance_comparison(
        trained_accuracy,
        sampled_accuracies,
        ylabel="Decision accuracy",
        title=title,
        baseline=0.5,
        baseline_label="chance: 50%",
    )
    axis.set_ylim(0, 1.03)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    return fig, axis


def plot_reduced_system_accuracy(
    trained_accuracy: float,
    reduced_accuracy: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare the trained RNN and its reduced-system accuracy."""
    fig, axis = plt.subplots(figsize=(5.2, 4.2))
    axis.bar(
        ("Trained RNN", "1D system"),
        (trained_accuracy, reduced_accuracy),
        color=(COLORS["green"], COLORS["purple"]),
        width=0.6,
    )
    axis.axhline(
        0.5,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1,
        label="chance: 50%",
    )
    axis.set_ylim(0, 1.03)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set(
        ylabel="Decision accuracy",
        title="Held-out perceptual decision performance",
    )
    axis.legend(loc="lower left")
    fig.tight_layout()
    return fig, axis


def _add_covariance_ellipse(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    covariance = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 4 * np.sqrt(eigenvalues)
    color = COVARIANCE_COLORS[
        "positive" if covariance[0, 1] >= 0 else "negative"
    ]
    axis.add_patch(
        Ellipse(
            (x.mean(), y.mean()),
            width,
            height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.15,
            linewidth=2,
        )
    )
    return float(covariance[0, 1])


def plot_connectivity_pairs(
    vectors: Mapping[str, npt.ArrayLike],
    *,
    row_names: Sequence[str],
    column_names: Sequence[str],
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the requested unique pairs of connectivity vectors."""
    row_names = tuple(row_names)
    column_names = tuple(column_names)
    fig, axes = plt.subplots(
        len(row_names),
        len(column_names),
        figsize=(3 * len(column_names), 3 * len(row_names)),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.set_visible(False)

    plotted_pairs: set[frozenset[str]] = set()
    for row, y_name in enumerate(row_names):
        for column, x_name in enumerate(column_names):
            pair = frozenset((x_name, y_name))
            if len(pair) < 2 or pair in plotted_pairs:
                continue
            plotted_pairs.add(pair)

            x = np.asarray(vectors[x_name])
            y = np.asarray(vectors[y_name])
            axis = axes[row, column]
            axis.set_visible(True)
            axis.scatter(
                x,
                y,
                color=COLORS["gray"],
                s=22,
                alpha=0.65,
                edgecolors="none",
            )
            covariance = _add_covariance_ellipse(axis, x, y)
            axis.set_xlim(-1.05 * np.max(np.abs(x)), 1.05 * np.max(np.abs(x)))
            axis.set_ylim(-1.05 * np.max(np.abs(y)), 1.05 * np.max(np.abs(y)))
            axis.spines["left"].set_position(("data", 0))
            axis.spines["bottom"].set_position(("data", 0))
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            axis.set_xlabel(x_name, loc="right")
            axis.set_ylabel(y_name, loc="top", rotation=0)
            axis.set_title(f"{x_name}–{y_name}\nCov = {covariance:.3f}")

    fig.suptitle("Connectivity-space covariance")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_connectivity_covariance(
    names: Sequence[str],
    covariance: npt.ArrayLike,
    *,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot unique cross-covariances as a triangular heat map."""
    names = tuple(names)
    covariance = np.asarray(covariance)
    pair_covariances = covariance[:-1, 1:]
    mask = np.tril(np.ones_like(pair_covariances, dtype=bool), k=-1)
    visible = pair_covariances[~mask]
    color_limit = max(float(np.max(np.abs(visible))), 1e-12)
    n_pairs = len(names) - 1
    fig, axis = plt.subplots(
        figsize=(1.35 * n_pairs + 1.4, 1.0 * n_pairs + 1.0)
    )
    image = axis.imshow(
        np.ma.masked_array(pair_covariances, mask=mask),
        cmap=COVARIANCE_CMAP,
        vmin=-color_limit,
        vmax=color_limit,
    )
    axis.set_xticks(
        range(len(names) - 1),
        labels=[rf"${name}$" for name in names[1:]],
    )
    axis.set_yticks(
        range(len(names) - 1),
        labels=[rf"${name}$" for name in names[:-1]],
    )
    axis.xaxis.tick_top()
    axis.yaxis.tick_right()
    axis.tick_params(length=0, labelsize=12)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row, first_name in enumerate(names[:-1]):
        for column in range(row, len(names) - 1):
            second_name = names[column + 1]
            value = pair_covariances[row, column]
            text_color = "white" if abs(value) > 0.55 * color_limit else "#27313B"
            axis.text(
                column,
                row,
                rf"$\mathrm{{Cov}}({first_name}, {second_name})$",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    fig.colorbar(image, ax=axis, shrink=0.78, pad=0.12, label="Covariance")
    if title is not None:
        axis.set_title(title, pad=18)
    fig.tight_layout()
    return fig, axis


def _trajectory_limits(
    trajectories: npt.ArrayLike,
    *,
    padding: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    trajectories = np.asarray(trajectories)
    limits = np.max(
        np.abs(trajectories),
        axis=tuple(range(trajectories.ndim - 1)),
    )
    limits = np.maximum(limits * (1 + padding), 1e-6)
    return (-float(limits[0]), float(limits[0])), (
        -float(limits[1]),
        float(limits[1]),
    )


def _style_trajectory_axis(
    axis: plt.Axes,
    trajectories: npt.ArrayLike,
) -> None:
    x_limits, y_limits = _trajectory_limits(trajectories)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.spines["left"].set_position(("data", 0))
    axis.spines["bottom"].set_position(("data", 0))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.tick_params(labelsize=8)
    axis.grid(False)


def plot_activity_trajectories_by_stimulus(
    trajectories: npt.ArrayLike,
    mean_stimuli: npt.ArrayLike,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot two-dimensional activity trajectories by mean stimulus."""
    trajectories = np.asarray(trajectories)
    mean_stimuli = np.asarray(mean_stimuli)
    stimulus_limit = max(float(np.max(np.abs(mean_stimuli))), 1e-12)
    norm = plt.Normalize(-stimulus_limit, stimulus_limit)

    fig, axis = plt.subplots(figsize=(8, 7))
    for trajectory, mean_stimulus in zip(trajectories, mean_stimuli):
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=SIGNED_VALUE_CMAP(norm(mean_stimulus)),
            alpha=0.4,
        )

    _style_trajectory_axis(axis, trajectories)
    axis.set(
        xlabel=r"activity along $m$",
        ylabel=r"activity along $I_\perp$",
        title="Activity trajectories colored by mean stimulus",
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=SIGNED_VALUE_CMAP),
        ax=axis,
    )
    colorbar.set_label(r"mean stimulus, $\bar{u}$")
    fig.tight_layout()
    return fig, axis


def plot_reduced_system_trajectories(
    trajectories: npt.ArrayLike,
    mean_stimuli: npt.ArrayLike,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot reduced-system trajectories in the kappa-v plane."""
    fig, axis = plot_activity_trajectories_by_stimulus(
        trajectories,
        mean_stimuli,
    )
    axis.set(
        xlabel=r"latent state, $\kappa$",
        ylabel=r"filtered input, $v$",
        title="Equivalent one-dimensional system trajectories",
    )
    return fig, axis


def plot_loading_distributions(
    trained_vectors: Mapping[str, npt.ArrayLike],
    sampled_vectors: Mapping[str, npt.ArrayLike],
) -> tuple[plt.Figure, np.ndarray]:
    """Compare trained and Gaussian-sampled loading marginals."""
    names = tuple(trained_vectors)
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.4 * columns, 2.8 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, name in zip(axes.flat, names):
        axis.hist(
            trained_vectors[name],
            bins=18,
            density=True,
            alpha=0.55,
            color=COLORS["gray"],
            label="trained",
        )
        axis.hist(
            sampled_vectors[name],
            bins=18,
            density=True,
            alpha=0.45,
            color=COLORS["purple"],
            label="Gaussian sample",
        )
        axis.set_title(name)
    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("Loading marginals: empirical and Gaussian sample")
    return fig, axes


def plot_explained_variance(
    explained_variance: npt.ArrayLike,
    *,
    num_components: int = 6,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot PCA explained variance for the leading components."""
    values = np.asarray(explained_variance)[:num_components]
    colors = (COLORS["blue"], COLORS["red"], *([COLORS["gray"]] * (len(values) - 2)))
    fig, axis = plt.subplots(figsize=(7, 3.8))
    axis.bar(np.arange(1, len(values) + 1), 100 * values, color=colors)
    axis.set(
        xlabel="principal component",
        ylabel="explained variance (%)",
        title="PCA rediscovers the low-dimensional state space",
    )
    fig.tight_layout()
    return fig, axis


def plot_fixed_points(
    grid: npt.ArrayLike,
    flow: npt.ArrayLike,
    fixed_points: npt.ArrayLike,
    slopes: npt.ArrayLike,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a one-dimensional flow, energy, and classified fixed points."""
    grid = np.asarray(grid)
    flow = np.asarray(flow)
    fixed_points = np.asarray(fixed_points)
    slopes = np.asarray(slopes)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(grid, flow, color=COLORS["blue"])
    axes[0].axhline(0, **REFERENCE_LINE_STYLE)
    axes[0].set(xlabel=r"$\kappa$", ylabel=r"$f(\kappa)$", title="Zero-input flow")
    axes[1].semilogy(grid, 0.5 * flow**2 + 1e-12, color=COLORS["purple"])
    axes[1].set(
        xlabel=r"$\kappa$",
        ylabel=r"$q(\kappa)$",
        title="Fixed-point energy",
    )
    for point, slope in zip(fixed_points, slopes):
        stable = slope < 0
        color = COLORS["green"] if stable else COLORS["red"]
        marker = "o" if stable else "X"
        axes[0].scatter(point, 0, color=color, marker=marker, s=70, zorder=3)
        axes[1].scatter(point, 1e-12, color=color, marker=marker, s=70, zorder=3)
    return fig, axes


def plot_training_loss(
    losses: npt.ArrayLike,
    *,
    title: str,
    stage_ends: Sequence[int] = (),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot training loss with optional curriculum boundaries."""
    losses = np.asarray(losses)
    fig, axis = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    axis.semilogy(np.arange(1, len(losses) + 1), losses)
    for stage_end in stage_ends:
        axis.axvline(stage_end + 0.5, **REFERENCE_LINE_STYLE)
    axis.set(xlabel="epoch", ylabel="decision MSE", title=title)
    return fig, axis


def plot_covariance_comparison(
    names: Sequence[str],
    first: npt.ArrayLike,
    second: npt.ArrayLike,
    *,
    titles: tuple[str, str],
) -> tuple[plt.Figure, np.ndarray]:
    """Compare two loading covariance matrices on one color scale."""
    first, second = np.asarray(first), np.asarray(second)
    limit = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, covariance, title in zip(axes, (first, second), titles):
        image = axis.imshow(covariance, cmap=COVARIANCE_CMAP, norm=norm)
        axis.set_xticks(range(len(names)), labels=names, rotation=35, ha="right")
        axis.set_yticks(range(len(names)), labels=names)
        axis.set_title(title)
    fig.colorbar(image, ax=axes, label="covariance", shrink=0.78)
    fig.suptitle("Loading covariance")
    return fig, axes
