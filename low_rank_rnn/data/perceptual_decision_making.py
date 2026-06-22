"""Generate trials for a perceptual decision making task."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

DT_MS = 20
DEFAULT_TRIAL_LENGTH = 50_000 // DT_MS

STIMULUS_STRENGTH_PREFAC = 3.2 / 100
STIMULUS_STRENGTHS = np.concatenate(
    [
        2 ** np.arange(5) * STIMULUS_STRENGTH_PREFAC,
        -2 ** np.arange(5) * STIMULUS_STRENGTH_PREFAC,
    ]
)
STIMULUS_WINDOW_MS = (5_000, 45_000)

NOISE_MEAN = 0.0
NOISE_STD = 0.03


def generate_perceptual_decision_making_trials(
    num_trials: int,
    trial_length: int = DEFAULT_TRIAL_LENGTH,
    *,
    dt_ms: int = DT_MS,
    stimulus_strengths: npt.ArrayLike = STIMULUS_STRENGTHS,
    stimulus_window_ms: Sequence[int] = STIMULUS_WINDOW_MS,
    noise_mean: float = NOISE_MEAN,
    noise_std: float = NOISE_STD,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy input trials and signed choice labels.

    Each trial is Gaussian noise with a constant signed stimulus added during
    ``stimulus_window_ms``. Labels are the sign of the sampled stimulus
    strength: ``+1`` for right and ``-1`` for left.
    """
    window_start = int(stimulus_window_ms[0] / dt_ms)
    window_end = int(stimulus_window_ms[1] / dt_ms)

    generator = rng if rng is not None else np.random.default_rng()
    strengths = np.asarray(stimulus_strengths, dtype=float)

    data = generator.normal(noise_mean, noise_std, (num_trials, trial_length))
    trial_strengths = generator.choice(strengths, size=num_trials)
    data[:, window_start : window_end + 1] += trial_strengths[:, np.newaxis]

    labels = np.sign(trial_strengths).astype(int)
    return data, labels
