"""Data generation utilities."""

from low_rank_rnn.data.perceptual_decision_making import (
    DEFAULT_TRIAL_LENGTH,
    DT_MS,
    NOISE_MEAN,
    NOISE_STD,
    STIMULUS_STRENGTHS,
    STIMULUS_STRENGTH_PREFAC,
    STIMULUS_WINDOW_MS,
    generate_perceptual_decision_making_trials,
)

__all__ = [
    "DEFAULT_TRIAL_LENGTH",
    "DT_MS",
    "NOISE_MEAN",
    "NOISE_STD",
    "STIMULUS_STRENGTHS",
    "STIMULUS_STRENGTH_PREFAC",
    "STIMULUS_WINDOW_MS",
    "generate_perceptual_decision_making_trials",
]

