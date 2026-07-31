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
        title=r"Population trajectories in the $m_1$–$m_2$ plane",
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
def plot_readout_coefficients(
    delay_steps: Real[np.ndarray, "delay"],
    coefficients: Real[np.ndarray, "delay 2"],
    *,
    trained_delay: RealNumber,
    annotation_delays: Sequence[RealNumber] = (),
) -> tuple[plt.Figure, plt.Axes]:
    """Show how decision-time readout weights change with the delay."""
    delay_steps = np.asarray(delay_steps)
    coefficients = np.asarray(coefficients)
    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    marker_interval = max(1, len(delay_steps) // 10)

    axis.plot(
        delay_steps,
        coefficients[:, 0],
        color=COLORS["blue"],
        marker="o",
        markevery=marker_interval,
        label=r"first stimulus: $\beta_1$",
    )
    axis.plot(
        delay_steps,
        coefficients[:, 1],
        color=COLORS["gold"],
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markevery=marker_interval,
        label=r"second stimulus: $\beta_2$",
    )
    for ideal_value in (-1, 1):
        axis.axhline(
            ideal_value,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1,
            zorder=0,
        )

    axis.axvline(
        trained_delay,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.2,
    )
    axis.text(
        trained_delay,
        0.48,
        f"trained delay\n{trained_delay:g} steps",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="center",
        color=COLORS["gray"],
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 1.5,
        },
    )

    for delay in annotation_delays:
        index = int(np.argmin(np.abs(delay_steps - delay)))
        axis.scatter(
            delay_steps[index],
            coefficients[index, 0],
            color=COLORS["blue"],
            edgecolor="white",
            linewidth=0.8,
            s=48,
            zorder=4,
        )
        axis.annotate(
            f"{coefficients[index, 0]:+.2f}",
            xy=(delay_steps[index], coefficients[index, 0]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=COLORS["blue"],
            fontsize=9,
        )

    axis.set(
        xlabel="blank delay between stimuli (time steps)",
        ylabel=r"fitted decision coefficient, $\beta_i$",
        title=(
            "Oscillatory RNN stimulus contributions by delay\n"
            r"Fit over all 49 conditions: "
            r"$\bar{z}=a+\beta_1\tilde{f}_1+\beta_2\tilde{f}_2$"
        ),
    )
    axis.legend(loc="upper right")
    return fig, axis


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


@typechecked
def plot_gaussian_pipeline(
    trained_mse: float,
    full_covariance_mse: float,
    sampled_mses: Real[np.ndarray, "sample"],
    targets: Real[np.ndarray, "trial"],
    trained_decisions: Real[np.ndarray, "trial"],
    full_covariance_decisions: Real[np.ndarray, "trial"],
    *,
    threshold: float,
    title: str,
    score_label: str,
    readout_title: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare a trained RNN, its Gaussian circuit, and finite samples."""
    sampled_mses = np.asarray(sampled_mses)
    categories = (
        "source\ntrained RNN",
        "full-covariance\nGaussian circuit",
        "finite Gaussian-\nsampled RNNs",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for position, score, color, marker, label in (
        (0, trained_mse, COLORS["blue"], "D", "source trained RNN"),
        (
            1,
            full_covariance_mse,
            COLORS["purple"],
            "s",
            "full-covariance Gaussian circuit",
        ),
    ):
        axes[0].scatter(
            position,
            score,
            color=color,
            marker=marker,
            s=70,
            zorder=3,
            label=label,
        )
    sample_positions = 2 + np.linspace(-0.16, 0.16, len(sampled_mses))
    axes[0].scatter(
        sample_positions,
        sampled_mses,
        color=COLORS["gray"],
        edgecolor="white",
        linewidth=0.5,
        s=48,
        zorder=3,
    )
    axes[0].plot(
        (sample_positions.min(), sample_positions.max()),
        (np.median(sampled_mses),) * 2,
        color=COLORS["gray"],
        linewidth=2.5,
        label=f"finite-sample median (n={len(sampled_mses)})",
    )
    axes[0].axhline(
        threshold,
        **REFERENCE_LINE_STYLE,
        label=f"target = {threshold:g}",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(categories)), labels=categories)
    axes[0].set(ylabel=score_label, title="Task performance")
    axes[0].legend()

    for decisions, color, marker, label in (
        (trained_decisions, COLORS["blue"], "o", "source trained RNN"),
        (
            full_covariance_decisions,
            COLORS["purple"],
            "s",
            "full-covariance Gaussian circuit",
        ),
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
        np.max(np.abs(targets)),
        np.max(np.abs(trained_decisions)),
        np.max(np.abs(full_covariance_decisions)),
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
        title=readout_title,
    )
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].legend()
    fig.suptitle(title)
    return fig, axes
