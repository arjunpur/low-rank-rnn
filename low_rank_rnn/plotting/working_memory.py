"""Plots for parametric working-memory experiments."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import numpy.typing as npt

from low_rank_rnn.plotting.style import (
    COLORS,
    DECISION_WINDOW_STYLE,
    FREQUENCY_CMAP,
    REFERENCE_LINE_STYLE,
    RESIDUAL_CMAP,
    SIGNED_VALUE_CMAP,
    STIMULUS_WINDOW_STYLE,
)


def _label_frequency_axes(
    axis: plt.Axes,
    frequencies: npt.ArrayLike,
) -> None:
    frequencies = np.asarray(frequencies)
    axis.set_xticks(range(len(frequencies)), labels=frequencies.astype(int))
    axis.set_yticks(range(len(frequencies)), labels=frequencies.astype(int))
    axis.set(xlabel=r"$f_2$ (Hz)", ylabel=r"$f_1$ (Hz)")


def plot_memory_trials(
    frequency_pairs: npt.ArrayLike,
    inputs: npt.ArrayLike,
    targets: npt.ArrayLike,
    target_matrix: npt.ArrayLike,
    frequencies: npt.ArrayLike,
    *,
    first_window: tuple[int, int],
    second_window: tuple[int, int],
    decision_steps: int,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot representative two-pulse inputs and the full target grid."""
    frequency_pairs = np.asarray(frequency_pairs)
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    offsets = 1.35 * np.arange(len(inputs))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for pair, target, trial, offset in zip(
        frequency_pairs,
        targets,
        inputs,
        offsets,
    ):
        axes[0].plot(
            trial + offset,
            label=rf"$f_1={pair[0]:g}, f_2={pair[1]:g}, y={target:+.2f}$",
        )
    axes[0].axvspan(first_window[0], first_window[1] - 1, **STIMULUS_WINDOW_STYLE)
    axes[0].axvspan(second_window[0], second_window[1] - 1, **STIMULUS_WINDOW_STYLE)
    axes[0].axvspan(
        inputs.shape[1] - decision_steps,
        inputs.shape[1] - 1,
        **DECISION_WINDOW_STYLE,
    )
    axes[0].set(xlabel="time step", yticks=[], title="Representative two-pulse inputs")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    image = axes[1].imshow(
        target_matrix,
        origin="lower",
        cmap=SIGNED_VALUE_CMAP,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
    )
    _label_frequency_axes(axes[1], frequencies)
    axes[1].set_title(r"Continuous target $(f_1-f_2)/24$")
    fig.colorbar(image, ax=axes[1], label="normalized target $y$")
    return fig, axes


def plot_memory_behavior(
    rank_two_losses: npt.ArrayLike,
    rank_one_losses: npt.ArrayLike,
    target_matrix: npt.ArrayLike,
    predictions: npt.ArrayLike,
    frequencies: npt.ArrayLike,
    *,
    mse: float,
    r_squared: float,
    loss_threshold: float,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot training and the complete fixed-delay condition grid."""
    target_matrix = np.asarray(target_matrix)
    prediction_matrix = np.asarray(predictions).reshape(target_matrix.shape)
    residual_matrix = prediction_matrix - target_matrix
    residual_limit = max(float(np.max(np.abs(residual_matrix))), 1e-3)
    value_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit,
        vcenter=0,
        vmax=residual_limit,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].semilogy(rank_two_losses, color=COLORS["blue"], label="rank 2")
    axes[0, 0].semilogy(
        rank_one_losses,
        color=COLORS["gold"],
        label="rank 1 control",
    )
    axes[0, 0].axhline(
        loss_threshold,
        **REFERENCE_LINE_STYLE,
        label=f"target = {loss_threshold:g}",
    )
    axes[0, 0].set(
        xlabel="training epoch",
        ylabel="decision-window MSE",
        title="Optimization",
    )
    axes[0, 0].legend()

    for axis, matrix, title in (
        (axes[0, 1], target_matrix, "Target"),
        (axes[1, 0], prediction_matrix, "Rank-two prediction"),
    ):
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap=SIGNED_VALUE_CMAP,
            norm=value_norm,
        )
        _label_frequency_axes(axis, frequencies)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.78, label="normalized value")

    residual_image = axes[1, 1].imshow(
        residual_matrix,
        origin="lower",
        cmap=RESIDUAL_CMAP,
        norm=residual_norm,
    )
    _label_frequency_axes(axes[1, 1], frequencies)
    axes[1, 1].set_title("Prediction residual")
    fig.colorbar(
        residual_image,
        ax=axes[1, 1],
        shrink=0.78,
        label="prediction − target (separate scale)",
    )
    fig.suptitle(
        rf"Working-memory behavior  |  MSE={mse:.2e}, $R^2$={r_squared:.3f}"
    )
    return fig, axes


def plot_latent_sweeps(
    first_coordinates: npt.ArrayLike,
    second_coordinates: npt.ArrayLike,
    frequencies: npt.ArrayLike,
    *,
    time_values: npt.ArrayLike,
    window_spans: Sequence[tuple[float, float]],
    column_titles: tuple[str, str],
    row_labels: tuple[str, str],
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot two latent modes while sweeping each task frequency."""
    first_coordinates = np.asarray(first_coordinates)
    second_coordinates = np.asarray(second_coordinates)
    frequencies = np.asarray(frequencies)
    time_values = np.asarray(time_values)
    norm = plt.Normalize(frequencies.min(), frequencies.max())
    plot_steps = len(time_values)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 6.5),
        sharex=True,
        constrained_layout=True,
    )
    for column, (coordinates, column_title) in enumerate(
        zip((first_coordinates, second_coordinates), column_titles)
    ):
        for frequency, trajectory in zip(frequencies, coordinates):
            axes[0, column].plot(
                time_values,
                trajectory[:plot_steps, 0],
                color=FREQUENCY_CMAP(norm(frequency)),
                linewidth=1.8,
            )
            axes[1, column].plot(
                time_values,
                trajectory[:plot_steps, 1],
                color=FREQUENCY_CMAP(norm(frequency)),
                linewidth=1.8,
            )
        axes[0, column].set_title(column_title)
        time_label = "time (ms)" if time_values[-1] > plot_steps else "time step"
        axes[1, column].set_xlabel(time_label)
        for axis in axes[:, column]:
            for start, stop in window_spans:
                axis.axvspan(start, stop, **STIMULUS_WINDOW_STYLE)

    for row in range(2):
        limit = 1.05 * max(
            np.abs(first_coordinates[:, :plot_steps, row]).max(),
            np.abs(second_coordinates[:, :plot_steps, row]).max(),
        )
        for axis in axes[row]:
            axis.set_ylim(-limit, limit)
    axes[0, 0].set_ylabel(row_labels[0])
    axes[1, 0].set_ylabel(row_labels[1])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=FREQUENCY_CMAP),
        ax=axes,
        pad=0.02,
    )
    colorbar.set_label("swept frequency (Hz)")
    fig.suptitle(title)
    return fig, axes


def plot_latent_plane(
    coordinates: npt.ArrayLike,
    frequencies: npt.ArrayLike,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot rank-two trajectories in the recurrent plane."""
    coordinates = np.asarray(coordinates)
    frequencies = np.asarray(frequencies)
    norm = plt.Normalize(frequencies.min(), frequencies.max())
    fig, axis = plt.subplots(figsize=(6, 5.5))
    for frequency, trajectory in zip(frequencies, coordinates):
        color = FREQUENCY_CMAP(norm(frequency))
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color)
        axis.scatter(*trajectory[-1, :2], color=color, s=32)
    axis.set(
        xlabel=r"$\kappa_1$",
        ylabel=r"$\kappa_2$",
        title=r"Population trajectories in the $m_1$–$m_2$ plane",
    )
    fig.tight_layout()
    return fig, axis


def plot_memory_reductions(
    trained_mse: float,
    rank_one_mse: float,
    fitted_mse: float,
    paper_mse: float,
    sampled_mses: npt.ArrayLike,
    targets: npt.ArrayLike,
    trained_decisions: npt.ArrayLike,
    fitted_decisions: npt.ArrayLike,
    paper_decisions: npt.ArrayLike,
    *,
    threshold: float,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare trained, sampled, and reduced working-memory systems."""
    sampled_mses = np.asarray(sampled_mses)
    categories = (
        "rank 1\ncontrol",
        "rank 2\ntrained",
        "fitted Gaussian\ncircuit",
        "paper\ncircuit",
        "Gaussian\nsamples",
    )
    values = (rank_one_mse, trained_mse, fitted_mse, paper_mse)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].scatter(
        np.arange(len(values)),
        values,
        color=COLORS["blue"],
        s=70,
        zorder=3,
    )
    axes[0].scatter(
        4 + np.linspace(-0.18, 0.18, len(sampled_mses)),
        sampled_mses,
        color=COLORS["purple"],
        edgecolor="white",
        linewidth=0.5,
        s=48,
        zorder=3,
    )
    axes[0].axhline(
        threshold,
        color=COLORS["gold"],
        linestyle="--",
        label=f"target = {threshold:g}",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(categories)), labels=categories)
    axes[0].set(ylabel="condition-balanced MSE", title="Behavioral comparison")
    axes[0].legend()

    for decisions, color, marker, label in (
        (trained_decisions, COLORS["blue"], "o", "trained RNN"),
        (fitted_decisions, COLORS["purple"], "s", "fitted Gaussian circuit"),
        (paper_decisions, COLORS["gold"], "^", "paper circuit"),
    ):
        axes[1].scatter(
            targets,
            decisions,
            color=color,
            marker=marker,
            s=34,
            alpha=0.75,
            label=label,
        )
    limit = 1.05 * max(
        1.0,
        np.max(np.abs(fitted_decisions)),
        np.max(np.abs(paper_decisions)),
    )
    axes[1].plot(
        (-limit, limit),
        (-limit, limit),
        color=COLORS["gray"],
        linestyle="--",
    )
    axes[1].set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        xlabel="target",
        ylabel="decision",
        title="Full and reduced readouts",
    )
    axes[1].legend()
    return fig, axes
