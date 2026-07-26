"""Parametric working memory with a uniformly random delay.

The paper samples the blank delay from 500-2000 ms. With the model's 20 ms time
step that is 25-100 integer steps. The first stimulus stays fixed in time, while
the second stimulus and the decision window move together on each trial, so
trials are padded to a common length and carry a per-trial decision mask.
"""

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float

from low_rank_rnn._typing import typechecked

FREQUENCIES = np.array([10, 14, 18, 22, 26, 30, 34], dtype=float)
MIN_FREQUENCY = float(FREQUENCIES.min())
MAX_FREQUENCY = float(FREQUENCIES.max())

FIXATION_STEPS = 5
STIMULUS_STEPS = 5
DECISION_STEPS = 5
DELAYS = np.arange(25, 101)
TRIAL_LENGTH = (
    FIXATION_STEPS + STIMULUS_STEPS + DELAYS.max() + STIMULUS_STEPS + DECISION_STEPS
)


@typechecked
def stimulus_amplitudes(frequencies: npt.ArrayLike) -> Float[np.ndarray, "..."]:
    """Center and scale frequencies onto the network's scalar input."""
    values = np.asarray(frequencies, dtype=float)
    midpoint = (MAX_FREQUENCY + MIN_FREQUENCY) / 2
    return (values - midpoint) / (MAX_FREQUENCY - MIN_FREQUENCY)


@typechecked
def make_trials(
    frequencies: npt.ArrayLike,
    delays: npt.ArrayLike,
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

    inputs = np.zeros((len(amplitudes), TRIAL_LENGTH), dtype=np.float32)
    decision_mask = np.zeros_like(inputs)
    first_start = FIXATION_STEPS
    first_stop = first_start + STIMULUS_STEPS
    inputs[:, first_start:first_stop] = amplitudes[:, 0, None]

    for trial, delay in enumerate(delays):
        second_start = first_stop + delay
        second_stop = second_start + STIMULUS_STEPS
        decision_stop = second_stop + DECISION_STEPS
        inputs[trial, second_start:second_stop] = amplitudes[trial, 1]
        decision_mask[trial, second_stop:decision_stop] = 1

    targets = amplitudes[:, 0] - amplitudes[:, 1]
    return tuple(
        torch.as_tensor(values, dtype=torch.float32)
        for values in (inputs, targets, decision_mask)
    )


@typechecked
def sample_trials(
    num_trials: int,
    delays: npt.ArrayLike = DELAYS,
    *,
    rng: np.random.Generator,
) -> tuple[
    Float[torch.Tensor, "trial time"],
    Float[torch.Tensor, "trial"],
    Float[torch.Tensor, "trial time"],
]:
    """Draw frequency pairs and delays uniformly, then build the trials."""
    frequencies = rng.choice(FREQUENCIES, size=(num_trials, 2))
    sampled_delays = rng.choice(np.asarray(delays), size=num_trials)
    return make_trials(frequencies, sampled_delays)


@typechecked
def frequency_pair_grid() -> Float[np.ndarray, "pair 2"]:
    """Return every ordered pair of task frequencies."""
    grid = np.meshgrid(FREQUENCIES, FREQUENCIES, indexing="ij")
    return np.array(grid).reshape(2, -1).T
