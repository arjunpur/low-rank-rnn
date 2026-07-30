"""Fixed- and variable-delay parametric working-memory tasks.

The paper samples the blank delay from 500-2000 ms. With the model's 20 ms time
step that is 25-100 integer steps. The first stimulus stays fixed in time, while
the second stimulus and the decision window move together on each trial, so
trials are padded to a common length and carry a per-trial decision mask.
"""

import numpy as np
import torch
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

VARIABLE_FIXATION_STEPS = 5
VARIABLE_STIMULUS_STEPS = 5
VARIABLE_DECISION_STEPS = 5
DELAYS = np.arange(25, 101)
VARIABLE_TRIAL_STEPS = (
    VARIABLE_FIXATION_STEPS
    + VARIABLE_STIMULUS_STEPS
    + DELAYS.max()
    + VARIABLE_STIMULUS_STEPS
    + VARIABLE_DECISION_STEPS
)


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
def make_variable_delay_trials(
    frequencies: Real[np.ndarray, "trial 2"],
    delays: Integer[np.ndarray, "trial"],
) -> tuple[
    Float[torch.Tensor, "trial time"],
    Float[torch.Tensor, "trial"],
    Float[torch.Tensor, "trial time"],
]:
    """Build padded trials and mask each trial's own decision window.

    ``frequencies`` holds the ``(f_1, f_2)`` pair for each trial and ``delays``
    the blank interval between them, in time steps.
    """
    amplitudes = stimulus_amplitudes(frequencies)
    delays = np.asarray(delays, dtype=int)

    inputs = np.zeros((len(amplitudes), VARIABLE_TRIAL_STEPS), dtype=np.float32)
    decision_mask = np.zeros_like(inputs)
    first_start = VARIABLE_FIXATION_STEPS
    first_stop = first_start + VARIABLE_STIMULUS_STEPS
    inputs[:, first_start:first_stop] = amplitudes[:, 0, None]

    for trial, delay in enumerate(delays):
        second_start = first_stop + delay
        second_stop = second_start + VARIABLE_STIMULUS_STEPS
        decision_stop = second_stop + VARIABLE_DECISION_STEPS
        inputs[trial, second_start:second_stop] = amplitudes[trial, 1]
        decision_mask[trial, second_stop:decision_stop] = 1

    targets = amplitudes[:, 0] - amplitudes[:, 1]
    return tuple(
        torch.as_tensor(values, dtype=torch.float32)
        for values in (inputs, targets, decision_mask)
    )


@typechecked
def sample_variable_delay_trials(
    num_trials: int,
    delays: Integer[np.ndarray, "delay"] = DELAYS,
    *,
    rng: np.random.Generator,
    frequencies: Real[np.ndarray, "frequency"] = FREQUENCIES,
) -> tuple[
    Float[torch.Tensor, "trial time"],
    Float[torch.Tensor, "trial"],
    Float[torch.Tensor, "trial time"],
]:
    """Draw frequency pairs and delays uniformly, then build the trials."""
    sampled_frequencies = rng.choice(
        np.asarray(frequencies),
        size=(num_trials, 2),
    )
    sampled_delays = rng.choice(np.asarray(delays), size=num_trials)
    return make_variable_delay_trials(sampled_frequencies, sampled_delays)


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
