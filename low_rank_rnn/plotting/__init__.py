"""Reader-facing plotting helpers."""

from low_rank_rnn.plotting.model import (
    plot_accuracy_comparison,
    plot_activity_trajectories_by_stimulus,
    plot_connectivity_covariance,
    plot_connectivity_pairs,
    plot_connectivity_summary,
    plot_covariance_comparison,
    plot_explained_variance,
    plot_fixed_points,
    plot_gaussian_sampling_summary,
    plot_loading_distributions,
    plot_pca_summary,
    plot_reduced_system_accuracy,
    plot_reduced_system_trajectories,
    plot_training_loss,
)
from low_rank_rnn.plotting.perceptual_decision_making import (
    plot_decision_summary,
    plot_decision_trials,
    plot_output_comparison,
    plot_trial_outputs,
)
from low_rank_rnn.plotting.style import set_plot_style
from low_rank_rnn.plotting.working_memory import (
    plot_latent_plane,
    plot_latent_sweeps,
    plot_memory_behavior,
    plot_gaussian_pipeline,
    plot_memory_trials,
)

__all__ = [
    "plot_accuracy_comparison",
    "plot_activity_trajectories_by_stimulus",
    "plot_connectivity_covariance",
    "plot_connectivity_pairs",
    "plot_connectivity_summary",
    "plot_covariance_comparison",
    "plot_decision_summary",
    "plot_decision_trials",
    "plot_explained_variance",
    "plot_fixed_points",
    "plot_gaussian_sampling_summary",
    "plot_latent_plane",
    "plot_latent_sweeps",
    "plot_loading_distributions",
    "plot_memory_behavior",
    "plot_gaussian_pipeline",
    "plot_memory_trials",
    "plot_output_comparison",
    "plot_pca_summary",
    "plot_reduced_system_accuracy",
    "plot_reduced_system_trajectories",
    "plot_training_loss",
    "plot_trial_outputs",
    "set_plot_style",
]
