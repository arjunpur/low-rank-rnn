"""Generate trials for a perceptual decision making task."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from jaxtyping import Float, Integer

from low_rank_rnn._typing import typechecked
from low_rank_rnn.constants import (
    NOISE_MEAN,
    NOISE_STD,
    STIMULUS_STRENGTHS,
    STIMULUS_WINDOW,
    TRIAL_LENGTH,
)


@typechecked
def generate_perceptual_decision_making_trials(
    num_trials: int,
    trial_length: int = TRIAL_LENGTH,
    *,
    stimulus_strengths: npt.ArrayLike = STIMULUS_STRENGTHS,
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

    data = generator.normal(noise_mean, noise_std, (num_trials, trial_length))
    trial_strengths = generator.choice(strengths, size=num_trials)
    data[:, window_start : window_end + 1] += trial_strengths[:, np.newaxis]

    labels = np.sign(trial_strengths).astype(int)
    return data, labels
