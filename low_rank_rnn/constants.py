"""Project-wide constants."""

from __future__ import annotations

import numpy as np

TRIAL_LENGTH = 75
STIMULUS_WINDOW = (5, 45)

STIMULUS_STRENGTH_PREFAC = 3.2 / 100
STIMULUS_STRENGTHS = np.concatenate(
    [
        2 ** np.arange(5) * STIMULUS_STRENGTH_PREFAC,
        -2 ** np.arange(5) * STIMULUS_STRENGTH_PREFAC,
    ]
)

NOISE_MEAN = 0.0
NOISE_STD = 0.03
