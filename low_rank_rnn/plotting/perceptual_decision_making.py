"""Plots for the perceptual decision-making task."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from jaxtyping import Integer, Real

from low_rank_rnn._typing import typechecked
from low_rank_rnn.data.perceptual_decision_making import STIMULUS_WINDOW
from low_rank_rnn.plotting.style import (
    CHOICE_COLORS,
    COLORS,
    DECISION_WINDOW_STYLE,
    REFERENCE_LINE_STYLE,
    STIMULUS_WINDOW_STYLE,
)


@typechecked
def plot_decision_trials(
    inputs: Real[np.ndarray, "trial time"],
    labels: Real[np.ndarray, "trial"],
    *,
    mean_stimuli: Real[np.ndarray, "trial"] | None = None,
    decision_steps: int = 15,
    stimulus_window: Sequence[int] = STIMULUS_WINDOW,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot representative task trials with stimulus and decision windows."""
    inputs = np.asarray(inputs)
    labels = np.asarray(labels)
    time_steps = np.arange(inputs.shape[1])
    fig, axes = plt.subplots(
        len(inputs),
        1,
        figsize=(10, 1.8 * len(inputs)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for trial, axis in enumerate(axes):
        color = CHOICE_COLORS[int(np.sign(labels[trial]))]
        axis.plot(time_steps, inputs[trial], color=color, linewidth=0.8)
        axis.axvspan(*stimulus_window, **STIMULUS_WINDOW_STYLE)
        axis.axvspan(
            inputs.shape[1] - decision_steps,
            inputs.shape[1] - 1,
            **DECISION_WINDOW_STYLE,
        )
        direction = "right (+1)" if labels[trial] > 0 else "left (-1)"
        axis.set_ylabel(f"trial {trial}\n{direction}")
        if mean_stimuli is not None:
            axis.text(
                0.99,
                0.83,
                rf"$\bar{{\mu}}\approx {np.asarray(mean_stimuli)[trial]:+.3f}$",
                transform=axis.transAxes,
                ha="right",
                fontsize=9,
            )

    axes[0].legend(
        handles=(
            Patch(label="stimulus", **STIMULUS_WINDOW_STYLE),
            Patch(label="decision", **DECISION_WINDOW_STYLE),
        ),
        loc="upper left",
    )
    axes[-1].set_xlabel("time step")
    fig.suptitle("Noisy decision trials: weak/strong and left/right examples")
    fig.tight_layout()
    return fig, axes


@typechecked
def plot_decision_summary(
    losses: Real[np.ndarray, "epoch"] | Sequence[float],
    stimulus_estimates: Real[np.ndarray, "trial"],
    decisions: Real[np.ndarray, "trial"],
    labels: Real[np.ndarray, "trial"],
    *,
    accuracy: float,
    loss_threshold: float,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot rank-one optimization and held-out decisions."""
    losses = np.asarray(losses)
    stimulus_estimates = np.asarray(stimulus_estimates)
    decisions = np.asarray(decisions)
    labels = np.asarray(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    axes[0].semilogy(np.arange(1, len(losses) + 1), losses)
    axes[0].axhline(
        loss_threshold,
        **REFERENCE_LINE_STYLE,
        label=f"target = {loss_threshold:g}",
    )
    axes[0].set(
        xlabel="training epoch",
        ylabel="decision-window MSE",
        title="Rank-one optimization",
    )
    axes[0].legend()

    for choice, color in CHOICE_COLORS.items():
        mask = labels == choice
        axes[1].scatter(
            stimulus_estimates[mask],
            decisions[mask],
            s=22,
            alpha=0.55,
            color=color,
            label=f"target {choice:+d}",
        )
    axes[1].axhline(0, color=COLORS["gray"], linewidth=0.9)
    axes[1].set(
        xlabel=r"mean evidence $\bar{\mu}$",
        ylabel="mean final readout",
        title=f"Held-out decisions  |  accuracy = {accuracy:.1%}",
    )
    axes[1].legend()
    return fig, axes


@typechecked
def plot_trial_outputs(
    inputs: Real[np.ndarray, "trial time"],
    labels: Real[np.ndarray, "trial"],
    outputs: Real[np.ndarray, "trial time"],
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
    fig, axes = plt.subplots(
        n_trials,
        2,
        figsize=(12, 1.8 * n_trials),
        sharex="col",
        squeeze=False,
    )
    for trial, (input_axis, output_axis) in enumerate(axes):
        color = CHOICE_COLORS[int(np.sign(labels[trial]))]
        input_axis.plot(time_steps, inputs[trial], color=color, linewidth=0.8)
        input_axis.axvspan(*stimulus_window, **STIMULUS_WINDOW_STYLE)
        direction = "right (+1)" if labels[trial] > 0 else "left (-1)"
        input_axis.set_ylabel(f"trial {trial}\n{direction}")

        output_axis.plot(time_steps, outputs[trial], color=color, linewidth=0.8)
        output_axis.axhline(
            labels[trial],
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )
        output_axis.axvspan(*stimulus_window, **STIMULUS_WINDOW_STYLE)
        output_axis.axvspan(
            n_steps - decision_steps,
            n_steps - 1,
            **DECISION_WINDOW_STYLE,
        )
        output_axis.set_ylim(-1.5, 1.5)

    axes[0, 0].set_title("input")
    axes[0, 1].set_title("output")
    axes[-1, 0].set_xlabel("time step")
    axes[-1, 1].set_xlabel("time step")
    fig.suptitle("Input and model output")
    fig.tight_layout()
    return fig, axes


@typechecked
def plot_output_comparison(
    full_outputs: Real[np.ndarray, "trial time"],
    reduced_outputs: Real[np.ndarray, "trial time"],
    labels: Real[np.ndarray, "trial"],
    trial_indices: Integer[np.ndarray, "selection"],
    *,
    decision_steps: int,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare full-RNN and Gaussian-circuit readouts."""
    full_outputs = np.asarray(full_outputs)
    reduced_outputs = np.asarray(reduced_outputs)
    labels = np.asarray(labels)
    trial_indices = np.asarray(trial_indices)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 6.5),
        sharex=True,
        constrained_layout=True,
    )
    for axis, trial in zip(axes.flat, trial_indices):
        color = CHOICE_COLORS[int(labels[trial])]
        axis.plot(full_outputs[trial], color=color, label="full RNN")
        axis.plot(
            reduced_outputs[trial],
            color=COLORS["purple"],
            linestyle="--",
            label="Gaussian circuit",
        )
        axis.axhline(
            labels[trial],
            color=COLORS["gray"],
            linestyle=":",
            label="target",
        )
        axis.axvspan(
            full_outputs.shape[1] - decision_steps,
            full_outputs.shape[1] - 1,
            **DECISION_WINDOW_STYLE,
        )
        axis.set_title(f"trial {trial}: target {labels[trial]:+d}")
    axes[0, 0].legend()
    fig.supxlabel("time step")
    fig.supylabel("readout")
    fig.suptitle("Full rank-one RNN and its Gaussian circuit")
    return fig, axes
