"""Plots for rank-one RNN outputs and dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import numpy.typing as npt

from low_rank_rnn.constants import STIMULUS_WINDOW
from low_rank_rnn.plotting.style import (
    CHOICE_COLORS,
    COLORS,
    COVARIANCE_CMAP,
    COVARIANCE_COLORS,
)


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
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the six unique pairs of rank-one connectivity vectors."""
    row_names = ("I", "n", "m")
    column_names = ("n", "m", "w")
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for axis in axes.flat:
        axis.set_visible(False)

    for row, y_name in enumerate(row_names):
        for column in range(row, len(column_names)):
            x_name = column_names[column]
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

    fig.suptitle("Upper-triangular connectivity-space covariance")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_connectivity_covariance(
    names: Sequence[str],
    covariance: npt.ArrayLike,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the six unique cross-covariances as a triangular heat map."""
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
                rf"$\sigma_{{{first_name}{second_name}}}$",
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
        axis.plot(
            mean_trajectory[:, 0],
            mean_trajectory[:, 1],
            color=color,
            linewidth=3,
            label=f"choice {choice:+d}",
        )
        arrow_times = np.arange(0, len(mean_trajectory) - 1, 8)
        steps = np.diff(mean_trajectory, axis=0)
        axis.quiver(
            mean_trajectory[arrow_times, 0],
            mean_trajectory[arrow_times, 1],
            steps[arrow_times, 0],
            steps[arrow_times, 1],
            color=color,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
        )
        axis.scatter(*mean_trajectory[0], color=color, marker="o", s=70)
        axis.scatter(*mean_trajectory[-1], color=color, marker="X", s=90)

    axis.axhline(0, color="0.85", linewidth=0.8)
    axis.axvline(0, color="0.85", linewidth=0.8)
    axis.set_xlabel(r"activity along $m$")
    axis.set_ylabel(r"activity along $I_\perp$")
    axis.set_title("Population activity trajectories in the $m$–$I$ plane")
    axis.legend(title="circle: start, X: end")
    x_limits, y_limits = activity_trajectory_limits(projected_states)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("auto")
    fig.tight_layout()
    return fig, axis
