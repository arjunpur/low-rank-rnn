"""Plots for low-rank RNN outputs and dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import numpy.typing as npt

from low_rank_rnn.constants import STIMULUS_WINDOW
from low_rank_rnn.plotting.style import (
    CHOICE_COLORS,
    COLORS,
    COVARIANCE_CMAP,
    COVARIANCE_COLORS,
    MEAN_CHOICE_COLORS,
)


def plot_accuracy_comparison(
    trained_accuracy: float,
    sampled_accuracies: npt.ArrayLike,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare one trained network with Gaussian-sampled networks."""
    sampled_accuracies = np.asarray(sampled_accuracies, dtype=float)
    sample_positions = np.linspace(0.86, 1.14, len(sampled_accuracies))
    sampled_mean = float(sampled_accuracies.mean())

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.scatter(
        0,
        trained_accuracy,
        color=COLORS["green"],
        marker="D",
        s=80,
        zorder=3,
    )
    axis.scatter(
        sample_positions,
        sampled_accuracies,
        color=COLORS["purple"],
        edgecolor="white",
        linewidth=0.6,
        s=65,
        zorder=3,
    )
    axis.axhline(
        trained_accuracy,
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.2,
        label=f"trained network: {trained_accuracy:.1%}",
    )
    axis.hlines(
        sampled_mean,
        0.82,
        1.18,
        color=COLORS["purple"],
        linewidth=2.2,
        label=f"sample mean: {sampled_mean:.1%}",
    )
    axis.axhline(
        0.5,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.0,
        label="chance: 50%",
    )

    axis.set_xticks((0, 1), labels=("Trained network", "Gaussian samples"))
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(0, 1.03)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Decision accuracy")
    axis.set_title("Held-out perceptual decision performance")
    axis.legend(loc="lower left")
    fig.tight_layout()
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
        linewidth=1.0,
        label="chance: 50%",
    )
    axis.set_ylim(0, 1.03)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Decision accuracy")
    axis.set_title("Held-out perceptual decision performance")
    axis.legend(loc="lower left")
    fig.tight_layout()
    return fig, axis


def plot_trial_outputs(
    inputs: npt.ArrayLike,
    labels: npt.ArrayLike,
    outputs: npt.ArrayLike,
    *,
    stimulus_window: Sequence[int] = STIMULUS_WINDOW,
    decision_steps: int = 15,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot task inputs and model outputs for each supplied trial."""
    inputs = np.asarray(inputs)
    labels = np.asarray(labels)
    outputs = np.asarray(outputs)
    n_trials, n_steps = inputs.shape
    time_steps = np.arange(n_steps)
    stimulus_start, stimulus_end = stimulus_window
    decision_start = n_steps - decision_steps

    fig, axes = plt.subplots(
        n_trials,
        2,
        figsize=(12, 1.8 * n_trials),
        sharex="col",
        squeeze=False,
    )
    for trial, (input_axis, output_axis) in enumerate(axes):
        choice_color = CHOICE_COLORS[int(np.sign(labels[trial]))]
        input_axis.plot(time_steps, inputs[trial], color=choice_color, lw=0.8)
        input_axis.axvspan(stimulus_start, stimulus_end, color="gray", alpha=0.12)
        direction = "right (+1)" if labels[trial] > 0 else "left (-1)"
        input_axis.set_ylabel(f"trial {trial}\n{direction}")

        output_axis.plot(time_steps, outputs[trial], color=choice_color, lw=0.8)
        output_axis.axhline(
            labels[trial],
            color=choice_color,
            ls="--",
            lw=0.8,
            alpha=0.7,
        )
        output_axis.axvspan(stimulus_start, stimulus_end, color="gray", alpha=0.12)
        output_axis.axvspan(decision_start, n_steps - 1, color="C2", alpha=0.15)
        output_axis.set_ylim(-1.5, 1.5)

    axes[0, 0].set_title("input")
    axes[0, 1].set_title("output")
    axes[-1, 0].set_xlabel("time step")
    axes[-1, 1].set_xlabel("time step")
    fig.suptitle("Input and model output")
    fig.tight_layout()
    return fig, axes


def _add_covariance_ellipse(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_std: float = 2.0,
) -> float:
    covariance = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    covariance_color = COVARIANCE_COLORS[
        "positive" if covariance[0, 1] >= 0 else "negative"
    ]
    axis.add_patch(
        Ellipse(
            (x.mean(), y.mean()),
            width,
            height,
            angle=angle,
            facecolor=covariance_color,
            edgecolor=covariance_color,
            alpha=0.15,
            linewidth=2,
        )
    )
    return covariance[0, 1]


def plot_connectivity_pairs(
    vectors: Mapping[str, npt.ArrayLike],
    *,
    row_names: Sequence[str],
    column_names: Sequence[str],
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the requested unique pairs of connectivity vectors."""
    row_names = tuple(row_names)
    column_names = tuple(column_names)
    if not row_names or not column_names:
        raise ValueError("row_names and column_names must not be empty")

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
) -> tuple[plt.Figure, plt.Axes]:
    """Plot unique cross-covariances as a triangular heat map."""
    names = tuple(names)
    covariance = np.asarray(covariance)
    pair_covariances = covariance[:-1, 1:]
    mask = np.tril(np.ones_like(pair_covariances, dtype=bool), k=-1)
    visible = pair_covariances[~mask]
    color_limit = max(float(np.max(np.abs(visible))), 1e-12)

    fig, axis = plt.subplots(figsize=(5.2, 4.0))
    image = axis.imshow(
        np.ma.masked_array(pair_covariances, mask=mask),
        cmap=COVARIANCE_CMAP,
        vmin=-color_limit,
        vmax=color_limit,
    )
    axis.set_xticks(range(len(names) - 1), labels=[rf"${name}$" for name in names[1:]])
    axis.set_yticks(range(len(names) - 1), labels=[rf"${name}$" for name in names[:-1]])
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

    colorbar = fig.colorbar(image, ax=axis, shrink=0.78, pad=0.12)
    colorbar.set_label("Covariance", rotation=270, labelpad=18)
    fig.tight_layout()
    return fig, axis


def activity_trajectory_limits(
    projected_states: npt.ArrayLike,
    *,
    padding: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return symmetric, independently scaled limits for trajectory axes."""
    projected_states = np.asarray(projected_states)
    if projected_states.shape[-1] != 2:
        raise ValueError("projected_states must end with two trajectory coordinates")

    coordinate_limits = np.max(
        np.abs(projected_states),
        axis=tuple(range(projected_states.ndim - 1)),
    )
    coordinate_limits = np.maximum(coordinate_limits * (1 + padding), 1e-6)
    x_limit, y_limit = coordinate_limits
    return (-float(x_limit), float(x_limit)), (-float(y_limit), float(y_limit))


def _style_activity_trajectory_axis(
    axis: plt.Axes,
    projected_states: npt.ArrayLike,
) -> None:
    """Style a two-dimensional activity trajectory axis."""
    x_limits, y_limits = activity_trajectory_limits(projected_states)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.spines["left"].set_position(("data", 0))
    axis.spines["bottom"].set_position(("data", 0))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.tick_params(labelsize=8)
    axis.grid(False)
    axis.set_xlabel(r"activity along $m$")
    axis.xaxis.set_label_coords(1, -0.04)
    axis.xaxis.label.set_horizontalalignment("right")
    axis.set_ylabel(r"activity along $I_\perp$")
    axis.yaxis.set_label_coords(-0.04, 1)
    axis.yaxis.label.set_horizontalalignment("right")


def plot_activity_trajectories(
    projected_states: npt.ArrayLike,
    labels: npt.ArrayLike,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot individual and condition-averaged activity trajectories."""
    projected_states = np.asarray(projected_states)
    labels = np.asarray(labels)
    fig, axis = plt.subplots(figsize=(8, 7))

    for choice, color in CHOICE_COLORS.items():
        trajectories = projected_states[labels == choice]
        if len(trajectories) == 0:
            continue

        for trajectory in trajectories:
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.25)

        mean_trajectory = trajectories.mean(axis=0)
        mean_color = MEAN_CHOICE_COLORS[choice]
        axis.plot(
            mean_trajectory[:, 0],
            mean_trajectory[:, 1],
            color=mean_color,
            linewidth=3,
            label=f"mean choice {choice:+d}",
        )
        arrow_times = np.arange(0, len(mean_trajectory) - 1, 8)
        steps = np.diff(mean_trajectory, axis=0)
        axis.quiver(
            mean_trajectory[arrow_times, 0],
            mean_trajectory[arrow_times, 1],
            steps[arrow_times, 0],
            steps[arrow_times, 1],
            color=mean_color,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
        )
        axis.scatter(*mean_trajectory[0], color=mean_color, marker="o", s=70)
        axis.scatter(*mean_trajectory[-1], color=mean_color, marker="X", s=90)

    _style_activity_trajectory_axis(axis, projected_states)
    axis.set_title("Population activity trajectories in the $m$–$I$ plane")
    axis.legend(title="circle: start, X: end")
    axis.set_aspect("auto")
    fig.tight_layout()
    return fig, axis


def plot_activity_trajectories_by_stimulus(
    projected_states: npt.ArrayLike,
    mean_stimuli: npt.ArrayLike,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot activity trajectories colored by mean stimulus."""
    projected_states = np.asarray(projected_states)
    mean_stimuli = np.asarray(mean_stimuli)
    stimulus_limit = max(float(np.max(np.abs(mean_stimuli))), 1e-12)
    stimulus_norm = plt.Normalize(-stimulus_limit, stimulus_limit)
    stimulus_cmap = plt.get_cmap("coolwarm")

    fig, axis = plt.subplots(figsize=(8, 7))
    for trajectory, mean_stimulus in zip(projected_states, mean_stimuli):
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=stimulus_cmap(stimulus_norm(mean_stimulus)),
            alpha=0.4,
        )

    _style_activity_trajectory_axis(axis, projected_states)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=stimulus_norm, cmap=stimulus_cmap),
        ax=axis,
    )
    colorbar.set_label(r"mean stimulus, $\bar{u}$")
    axis.set_title("Activity trajectories colored by mean stimulus")
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
    axis.set_xlabel(r"latent state, $\kappa$")
    axis.set_ylabel(r"filtered input, $v$")
    axis.set_title("Equivalent one-dimensional system trajectories")
    return fig, axis
