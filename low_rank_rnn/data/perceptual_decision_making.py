"""Perceptual decision-making task data."""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float, Integer, Real

from low_rank_rnn._typing import typechecked


TRIAL_STEPS = 75
STIMULUS_WINDOW = (5, 45)
STIMULUS_STRENGTHS = np.concatenate(
    (
        2 ** np.arange(5) * (3.2 / 100),
        -(2 ** np.arange(5) * (3.2 / 100)),
    )
)
NOISE_MEAN = 0.0
NOISE_STD = 0.03


@typechecked
def generate_trials(
    num_trials: int,
    trial_steps: int = TRIAL_STEPS,
    *,
    stimulus_strengths: Real[np.ndarray, "strength"] = STIMULUS_STRENGTHS,
    stimulus_window: Sequence[int] = STIMULUS_WINDOW,
    noise_mean: float = NOISE_MEAN,
    noise_std: float = NOISE_STD,
    rng: np.random.Generator | None = None,
) -> tuple[
    Float[np.ndarray, "trial time"],
    Integer[np.ndarray, "trial"],
]:
    """Generate noisy input trials and signed choice labels.

    Each trial is Gaussian noise with a constant signed stimulus added during
    ``stimulus_window`` (inclusive start and end time steps). Labels are the
    sign of the sampled stimulus strength: ``+1`` for right and ``-1`` for left.
    """
    window_start, window_end = stimulus_window

    generator = rng if rng is not None else np.random.default_rng()
    strengths = np.asarray(stimulus_strengths, dtype=float)

    data = generator.normal(noise_mean, noise_std, (num_trials, trial_steps))
    trial_strengths = generator.choice(strengths, size=num_trials)
    data[:, window_start : window_end + 1] += trial_strengths[:, np.newaxis]

    labels = np.sign(trial_strengths).astype(int)
    return data, labels


@typechecked
def mean_stimulus(
    inputs: Real[np.ndarray, "trial time"],
    stimulus_window: Sequence[int] = STIMULUS_WINDOW,
) -> Float[np.ndarray, "trial"]:
    """Average each trial's input over the inclusive stimulus window."""
    start, end = stimulus_window
    return np.asarray(inputs)[:, start : end + 1].mean(axis=1)


@typechecked
def representative_trial_indices(
    inputs: Real[np.ndarray, "trial time"],
    labels: Real[np.ndarray, "trial"],
) -> Integer[np.ndarray, "selection"]:
    """Select weak and strong examples for both choices."""
    means = mean_stimulus(inputs)
    labels = np.asarray(labels)
    selected = []
    for choice in (-1, 1):
        indices = np.flatnonzero(labels == choice)
        selected.extend(
            (
                indices[np.argmin(means[indices])],
                indices[np.argmax(means[indices])],
            )
        )
    return np.asarray(selected)
