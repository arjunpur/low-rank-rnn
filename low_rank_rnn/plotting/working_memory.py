"""Plots for parametric working-memory experiments."""

from collections.abc import Sequence
from numbers import Real as RealNumber

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from jaxtyping import Real

from low_rank_rnn._typing import typechecked
from low_rank_rnn.plotting.style import (
    COLORS,
    DECISION_WINDOW_STYLE,
    FREQUENCY_CMAP,
    REFERENCE_LINE_STYLE,
    SIGNED_VALUE_CMAP,
    STIMULUS_WINDOW_STYLE,
)


@typechecked
def _label_frequency_axes(
    axis: plt.Axes,
    frequencies: Real[np.ndarray, "frequency"],
) -> None:
    frequencies = np.asarray(frequencies)
    axis.set_xticks(range(len(frequencies)), labels=frequencies.astype(int))
    axis.set_yticks(range(len(frequencies)), labels=frequencies.astype(int))
    axis.set(xlabel=r"$f_2$ (Hz)", ylabel=r"$f_1$ (Hz)")


@typechecked
def plot_memory_trials(
    frequency_pairs: Real[np.ndarray, "trial 2"],
    inputs: Real[np.ndarray, "trial time"],
    targets: Real[np.ndarray, "trial"],
    outputs: Real[np.ndarray, "trial time"],
    *,
    first_window: tuple[int, int],
    second_window: tuple[int, int],
    decision_steps: int,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot representative working-memory inputs and model outputs."""
    frequency_pairs = np.asarray(frequency_pairs)
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    outputs = np.asarray(outputs)
    num_trials, num_steps = inputs.shape
    decision_start = num_steps - decision_steps
    time_steps = np.arange(num_steps)

    fig, axes = plt.subplots(
        num_trials,
        2,
        figsize=(12, 1.8 * num_trials),
        sharex=True,
        squeeze=False,
    )
    for pair, target, trial_input, trial_output, row_axes in zip(
        frequency_pairs,
        targets,
        inputs,
        outputs,
        axes,
    ):
        input_axis, output_axis = row_axes
        input_axis.plot(time_steps, trial_input, color=COLORS["blue"])
        input_axis.axvspan(
            first_window[0],
            first_window[1] - 1,
            **STIMULUS_WINDOW_STYLE,
        )
        input_axis.axvspan(
            second_window[0],
            second_window[1] - 1,
            **STIMULUS_WINDOW_STYLE,
        )
        input_axis.set_ylabel(f"{pair[0]:g} − {pair[1]:g} Hz")

        decision = float(np.mean(trial_output[-decision_steps:]))
        output_axis.plot(
            time_steps,
            trial_output,
            color=COLORS["blue"],
            label="output",
        )
        output_axis.axhline(
            target,
            **REFERENCE_LINE_STYLE,
            label="target",
        )
        output_axis.hlines(
            decision,
            decision_start,
            num_steps - 1,
            color=COLORS["gold"],
            linewidth=2.5,
            label="final decision",
        )
        output_axis.axvspan(
            decision_start,
            num_steps - 1,
            **DECISION_WINDOW_STYLE,
        )
        output_axis.set_title(
            f"target = {target:.3f}, decision = {decision:.3f}",
            loc="right",
        )

    axes[0, 0].set_title("input")
    axes[0, 1].legend(loc="upper left", ncols=3)
    axes[-1, 0].set_xlabel("time step")
    axes[-1, 1].set_xlabel("time step")
    fig.suptitle("Representative fixed-delay trials")
    fig.tight_layout()
    return fig, axes


@typechecked
def plot_memory_behavior(
    rank_two_losses: Real[np.ndarray, "rank_two_epoch"] | Sequence[float],
    rank_one_losses: Real[np.ndarray, "rank_one_epoch"] | Sequence[float],
    target_matrix: Real[np.ndarray, "frequency frequency"],
    predictions: Real[np.ndarray, "prediction"],
    frequencies: Real[np.ndarray, "frequency"],
    *,
    mse: float,
    loss_threshold: float,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot training losses and predictions across the frequency grid."""
    target_matrix = np.asarray(target_matrix)
    prediction_matrix = np.asarray(predictions).reshape(target_matrix.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].semilogy(rank_two_losses, color=COLORS["blue"], label="rank 2")
    axes[0].semilogy(
        rank_one_losses,
        color=COLORS["gold"],
        label="rank 1 control",
    )
    axes[0].axhline(
        loss_threshold,
        **REFERENCE_LINE_STYLE,
        label=f"target = {loss_threshold:g}",
    )
    axes[0].set(
        xlabel="training epoch",
        ylabel="decision-window MSE",
        title="Optimization",
    )
    axes[0].legend()

    prediction_image = axes[1].imshow(
        prediction_matrix,
        origin="lower",
        cmap=SIGNED_VALUE_CMAP,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
    )
    _label_frequency_axes(axes[1], frequencies)
    axes[1].set_title(f"Predicted $y$  |  MSE={mse:.2e}")
    fig.colorbar(
        prediction_image,
        ax=axes[1],
        label="predicted normalized value $y$",
    )
    fig.suptitle("Working-memory behavior")
    return fig, axes


@typechecked
def plot_latent_sweeps(
    first_coordinates: Real[np.ndarray, "frequency time 2"],
    second_coordinates: Real[np.ndarray, "frequency time 2"],
    frequencies: Real[np.ndarray, "frequency"],
    *,
    time_values: Real[np.ndarray, "plot_time"],
    window_spans: Sequence[tuple[RealNumber, RealNumber]],
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


@typechecked
def plot_latent_plane(
    coordinates: Real[np.ndarray, "frequency time 2"],
    frequencies: Real[np.ndarray, "frequency"],
    *,
    colorbar_label: str = "swept frequency (Hz)",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot rank-two trajectories in the recurrent plane."""
    coordinates = np.asarray(coordinates)
    frequencies = np.asarray(frequencies)
    norm = plt.Normalize(frequencies.min(), frequencies.max())
    fig, axis = plt.subplots(figsize=(6, 5.5))
    for frequency, trajectory in zip(frequencies, coordinates):
        color = FREQUENCY_CMAP(norm(frequency))
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color)
        arrow_step = 2 * len(trajectory) // 3
        arrow_start = max(0, arrow_step - 3)
        axis.annotate(
            "",
            xy=trajectory[arrow_step, :2],
            xytext=trajectory[arrow_start, :2],
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 1.2,
                "mutation_scale": 8,
            },
        )
    axis.set(
        xlabel=r"$\kappa_1$",
        ylabel=r"$\kappa_2$",
        title="RNN latent trajectories under isolated stimuli",
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=FREQUENCY_CMAP),
        ax=axis,
        pad=0.02,
    )
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig, axis


@typechecked
def plot_circuit_trajectories(
    trajectories: Real[np.ndarray, "system frequency time 2"],
    frequencies: Real[np.ndarray, "frequency"],
    system_labels: Sequence[str],
    *,
    colorbar_label: str,
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare rank-two trajectories from several equivalent circuits."""
    trajectories = np.asarray(trajectories)
    frequencies = np.asarray(frequencies)
    if len(system_labels) != len(trajectories):
        raise ValueError("system_labels must match trajectories")

    norm = plt.Normalize(frequencies.min(), frequencies.max())
    fig, axes = plt.subplots(
        1,
        len(trajectories),
        figsize=(5.5 * len(trajectories), 4.8),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for axis, system_trajectories, label in zip(
        axes,
        trajectories,
        system_labels,
        strict=True,
    ):
        for frequency, trajectory in zip(
            frequencies,
            system_trajectories,
            strict=True,
        ):
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=FREQUENCY_CMAP(norm(frequency)),
                linewidth=1.8,
            )
        axis.scatter(0, 0, color="black", marker="x", label="initial state")
        axis.set(
            xlabel=r"$\kappa_1$",
            ylabel=r"$\kappa_2$",
            title=label,
        )
        axis.set_aspect("equal", adjustable="datalim")

    axes[0].legend(loc="best")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=FREQUENCY_CMAP),
        ax=axes,
        pad=0.02,
    )
    colorbar.set_label(colorbar_label)
    fig.suptitle(title)
    return fig, axes


@typechecked
def plot_regression_comparison(
    targets: Real[np.ndarray, "trial"],
    predictions: Real[np.ndarray, "system trial"],
    system_labels: Sequence[str],
    *,
    mean_squared_errors: Real[np.ndarray, "system"],
    r_squared_values: Real[np.ndarray, "system"],
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare predicted and target values for several systems."""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if len(system_labels) != len(predictions):
        raise ValueError("system_labels must match predictions")

    limit = 1.05 * max(
        np.abs(targets).max(),
        np.abs(predictions).max(),
    )
    fig, axes = plt.subplots(
        1,
        len(predictions),
        figsize=(5.25 * len(predictions), 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for axis, system_predictions, mse, r_squared, label in zip(
        axes,
        predictions,
        mean_squared_errors,
        r_squared_values,
        system_labels,
        strict=True,
    ):
        axis.scatter(targets, system_predictions, alpha=0.75)
        axis.plot(
            (-limit, limit),
            (-limit, limit),
            color=COLORS["gray"],
            linestyle="--",
        )
        axis.set(
            xlim=(-limit, limit),
            ylim=(-limit, limit),
            xlabel="target",
            ylabel="circuit decision",
            title=f"{label}\nMSE={mse:.4f}, $R^2$={r_squared:.3f}",
        )
        axis.set_aspect("equal", adjustable="box")

    fig.suptitle(title)
    return fig, axes


@typechecked
def plot_delay_mse(
    delay_steps: Real[np.ndarray, "delay"],
    mean_squared_errors: Real[np.ndarray, "delay"],
    *,
    title: str,
    reference_delay: RealNumber | None = None,
    reference_label: str = "reference delay",
    annotation_delays: Sequence[RealNumber] = (),
    annotation_labels: Sequence[str] = (),
    x_ticks: Sequence[RealNumber] = (),
    y_limit: RealNumber | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot task prediction error across blank delays."""
    delay_steps = np.asarray(delay_steps)
    mean_squared_errors = np.asarray(mean_squared_errors)
    if annotation_labels and len(annotation_labels) != len(annotation_delays):
        raise ValueError(
            "annotation_labels must match annotation_delays",
        )

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(
        delay_steps,
        mean_squared_errors,
        color=COLORS["blue"],
        linewidth=2.2,
    )

    upper_limit = (
        float(y_limit)
        if y_limit is not None
        else 1.08 * float(mean_squared_errors.max())
    )
    upper_limit = max(upper_limit, 1e-3)
    delay_span = float(delay_steps[-1] - delay_steps[0])

    if reference_delay is not None:
        reference_index = int(
            np.argmin(np.abs(delay_steps - reference_delay))
        )
        reference_x = float(delay_steps[reference_index])
        reference_y = float(mean_squared_errors[reference_index])
        axis.axvline(
            reference_x,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1.2,
        )
        axis.scatter(
            reference_x,
            reference_y,
            s=42,
            color=COLORS["blue"],
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        axis.annotate(
            f"{reference_label}\n{reference_x:g} steps",
            xy=(reference_x, reference_y),
            xytext=(
                reference_x + 0.03 * delay_span,
                0.16 * upper_limit,
            ),
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["gray"],
                "linewidth": 1,
            },
            color="#27313B",
            fontsize=9,
        )

    labels = (
        annotation_labels
        if annotation_labels
        else tuple("selected delay" for _ in annotation_delays)
    )
    for delay, label in zip(annotation_delays, labels, strict=True):
        index = int(np.argmin(np.abs(delay_steps - delay)))
        annotation_x = float(delay_steps[index])
        annotation_y = float(mean_squared_errors[index])
        label_on_right = (
            annotation_x
            < float(delay_steps[0]) + 0.7 * delay_span
        )
        text_x = annotation_x + (
            0.025 * delay_span if label_on_right else -0.18 * delay_span
        )
        axis.scatter(
            annotation_x,
            annotation_y,
            s=38,
            facecolor="white",
            edgecolor=COLORS["blue"],
            linewidth=1.5,
            zorder=3,
        )
        axis.annotate(
            f"{label}\n{annotation_x:g} steps",
            xy=(annotation_x, annotation_y),
            xytext=(
                text_x,
                min(annotation_y + 0.16 * upper_limit, 0.85 * upper_limit),
            ),
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["gray"],
                "linewidth": 1,
            },
            color="#27313B",
            fontsize=9,
        )

    axis.set(
        xlabel="blank delay between stimuli (time steps)",
        ylabel="mean squared error",
        xlim=(delay_steps[0], delay_steps[-1]),
        ylim=(0, upper_limit),
    )
    if len(x_ticks) > 0:
        axis.set_xticks(x_ticks)
    axis.set_title(title, loc="left")
    return fig, axis
