"""Plotting utilities."""

from low_rank_rnn.plotting.perceptual_decision_making import (
    plot_first_perceptual_decision_making_trials,
)
from low_rank_rnn.plotting.model import (
    activity_trajectory_limits,
    plot_activity_trajectories,
    plot_connectivity_covariance,
    plot_connectivity_pairs,
    plot_trial_outputs,
)
from low_rank_rnn.plotting.style import (
    CHOICE_COLORS,
    COLORS,
    COVARIANCE_CMAP,
    COVARIANCE_COLORS,
    PALETTE,
    plot_style,
    set_plot_style,
)

__all__ = [
    "CHOICE_COLORS",
    "COLORS",
    "COVARIANCE_CMAP",
    "COVARIANCE_COLORS",
    "PALETTE",
    "activity_trajectory_limits",
    "plot_activity_trajectories",
    "plot_connectivity_covariance",
    "plot_connectivity_pairs",
    "plot_first_perceptual_decision_making_trials",
    "plot_style",
    "plot_trial_outputs",
    "set_plot_style",
]
