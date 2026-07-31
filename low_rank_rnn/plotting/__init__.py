"""Reader-facing plotting helpers."""

from low_rank_rnn.plotting.model import (
    plot_accuracy_comparison,
    plot_activity_trajectories_by_stimulus,
    plot_circuit_mse_comparison,
    plot_connectivity_covariance,
    plot_connectivity_pairs,
    plot_connectivity_summary,
    plot_fixed_points,
    plot_gaussian_sampling_summary,
    plot_loading_distributions,
    plot_mse_comparison,
    plot_pca_summary,
    plot_reduced_system_accuracy,
    plot_reduced_system_trajectories,
)
from low_rank_rnn.plotting.perceptual_decision_making import (
    plot_decision_summary,
    plot_decision_trials,
    plot_output_comparison,
    plot_trial_outputs,
)
from low_rank_rnn.plotting.style import save_report_figure, set_plot_style
from low_rank_rnn.plotting.working_memory import (
    plot_circuit_trajectories,
    plot_delay_mse,
    plot_latent_plane,
    plot_latent_sweeps,
    plot_memory_behavior,
    plot_memory_trials,
    plot_regression_comparison,
)

__all__ = [
    "plot_accuracy_comparison",
    "plot_activity_trajectories_by_stimulus",
    "plot_circuit_mse_comparison",
    "plot_circuit_trajectories",
    "plot_connectivity_covariance",
    "plot_connectivity_pairs",
    "plot_connectivity_summary",
    "plot_decision_summary",
    "plot_decision_trials",
    "plot_delay_mse",
    "plot_fixed_points",
    "plot_gaussian_sampling_summary",
    "plot_latent_plane",
    "plot_latent_sweeps",
    "plot_loading_distributions",
    "plot_memory_behavior",
    "plot_memory_trials",
    "plot_mse_comparison",
    "plot_output_comparison",
    "plot_pca_summary",
    "plot_regression_comparison",
    "plot_reduced_system_accuracy",
    "plot_reduced_system_trajectories",
    "plot_trial_outputs",
    "save_report_figure",
    "set_plot_style",
]
