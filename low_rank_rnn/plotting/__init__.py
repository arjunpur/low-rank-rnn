"""Plotting utilities."""

from low_rank_rnn.plotting.perceptual_decision_making import (
    plot_first_perceptual_decision_making_trials,
)
from low_rank_rnn.plotting.model import (
    activity_trajectory_limits,
    plot_accuracy_comparison,
    plot_activity_trajectories,
    plot_activity_trajectories_by_stimulus,
    plot_connectivity_covariance,
    plot_connectivity_pairs,
    plot_performance_comparison,
    plot_reduced_system_accuracy,
    plot_reduced_system_trajectories,
    plot_trial_outputs,
)
from low_rank_rnn.plotting.style import (
    CHOICE_COLORS,
    COLORS,
    COVARIANCE_CMAP,
    COVARIANCE_COLORS,
    MEAN_CHOICE_COLORS,
    PALETTE,
    RESIDUAL_CMAP,
    SIGNED_VALUE_CMAP,
    plot_style,
    set_plot_style,
)

__all__ = [
    "CHOICE_COLORS",
    "COLORS",
    "COVARIANCE_CMAP",
    "COVARIANCE_COLORS",
    "MEAN_CHOICE_COLORS",
    "PALETTE",
    "RESIDUAL_CMAP",
    "SIGNED_VALUE_CMAP",
    "activity_trajectory_limits",
    "plot_accuracy_comparison",
    "plot_activity_trajectories",
    "plot_activity_trajectories_by_stimulus",
    "plot_connectivity_covariance",
    "plot_connectivity_pairs",
    "plot_first_perceptual_decision_making_trials",
    "plot_performance_comparison",
    "plot_reduced_system_accuracy",
    "plot_reduced_system_trajectories",
    "plot_style",
    "plot_trial_outputs",
    "set_plot_style",
]
