"""Data generation utilities."""

from low_rank_rnn.data.perceptual_decision_making import (
    generate_perceptual_decision_making_trials,
)
from low_rank_rnn.constants import (
    NOISE_MEAN,
    NOISE_STD,
    STIMULUS_STRENGTHS,
    STIMULUS_STRENGTH_PREFAC,
    STIMULUS_WINDOW,
    TRIAL_LENGTH,
)

__all__ = [
    "NOISE_MEAN",
    "NOISE_STD",
    "STIMULUS_STRENGTHS",
    "STIMULUS_STRENGTH_PREFAC",
    "STIMULUS_WINDOW",
    "TRIAL_LENGTH",
    "generate_perceptual_decision_making_trials",
]
