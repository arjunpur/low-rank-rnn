"""Fixed-delay parametric working-memory task."""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float, Integer, Real

from low_rank_rnn._typing import typechecked

FREQUENCIES = np.array([10, 14, 18, 22, 26, 30, 34], dtype=float)
MIN_FREQUENCY = float(FREQUENCIES.min())
MAX_FREQUENCY = float(FREQUENCIES.max())
FREQUENCY_CENTER = (MIN_FREQUENCY + MAX_FREQUENCY) / 2
FREQUENCY_RANGE = MAX_FREQUENCY - MIN_FREQUENCY

FIXED_FIRST_WINDOW = (5, 11)
FIXED_SECOND_WINDOW = (60, 71)
FIXED_DECISION_STEPS = 5
FIXED_TRIAL_STEPS = 80

@typechecked
def stimulus_amplitudes(
    frequencies: Real[np.ndarray, "..."],
) -> Float[np.ndarray, "..."]:
    """Center and scale frequencies onto the network's scalar input."""
    values = np.asarray(frequencies, dtype=float)
    return (values - FREQUENCY_CENTER) / FREQUENCY_RANGE


@typechecked
def make_fixed_delay_trials(
    frequency_pairs: Real[np.ndarray, "trial 2"],
    *,
    second_start: int = FIXED_SECOND_WINDOW[0],
) -> tuple[
    Float[np.ndarray, "trial time"],
    Float[np.ndarray, "trial"],
]:
    """Build the fixed-window task used in the final analysis."""
    frequency_pairs = np.asarray(frequency_pairs, dtype=float)
    second_steps = FIXED_SECOND_WINDOW[1] - FIXED_SECOND_WINDOW[0]
    tail_steps = FIXED_TRIAL_STEPS - FIXED_SECOND_WINDOW[1]
    trial_steps = second_start + second_steps + tail_steps

    amplitudes = stimulus_amplitudes(frequency_pairs)
    inputs = np.zeros((len(frequency_pairs), trial_steps), dtype=np.float32)
    inputs[:, FIXED_FIRST_WINDOW[0] : FIXED_FIRST_WINDOW[1]] = amplitudes[
        :, 0, None
    ]
    inputs[:, second_start : second_start + second_steps] = amplitudes[:, 1, None]
    targets = (amplitudes[:, 0] - amplitudes[:, 1]).astype(np.float32)
    return inputs, targets


@typechecked
def make_delay_sweep_inputs(
    frequency_pairs: Real[np.ndarray, "trial 2"],
    delay_steps: Integer[np.ndarray, "delay"] | Sequence[int],
) -> list[np.ndarray]:
    """Build one trial batch for each blank delay between the stimuli."""
    return [
        make_fixed_delay_trials(
            frequency_pairs,
            second_start=FIXED_FIRST_WINDOW[1] + int(delay),
        )[0]
        for delay in delay_steps
    ]


@typechecked
def sample_fixed_delay_trials(
    num_trials: int,
    *,
    rng: np.random.Generator,
    frequencies: Real[np.ndarray, "frequency"] = FREQUENCIES,
) -> tuple[
    Float[np.ndarray, "trial 2"],
    Float[np.ndarray, "trial time"],
    Float[np.ndarray, "trial"],
]:
    """Sample frequency pairs and build fixed-delay trials."""
    pairs = rng.choice(
        np.asarray(frequencies, dtype=float),
        size=(num_trials, 2),
    )
    inputs, targets = make_fixed_delay_trials(pairs)
    return pairs, inputs, targets


@typechecked
def frequency_pair_grid(
    frequencies: Real[np.ndarray, "frequency"] = FREQUENCIES,
) -> Float[np.ndarray, "pair 2"]:
    """Return every ordered pair of task frequencies."""
    values = np.asarray(frequencies, dtype=float)
    grid = np.meshgrid(values, values, indexing="ij")
    return np.array(grid).reshape(2, -1).T


@typechecked
def frequency_sweeps(
    frequencies: Real[np.ndarray, "frequency"] = FREQUENCIES,
    *,
    neutral_frequency: float = FREQUENCY_CENTER,
) -> tuple[
    Float[np.ndarray, "frequency 2"],
    Float[np.ndarray, "frequency 2"],
]:
    """Sweep each stimulus while holding the other neutral."""
    values = np.asarray(frequencies, dtype=float)
    neutral = np.full(len(values), neutral_frequency)
    return np.column_stack((values, neutral)), np.column_stack((neutral, values))
