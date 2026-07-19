"""Semantic color contracts for project plots."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np

from low_rank_rnn.plotting import (
    CHOICE_COLORS,
    COVARIANCE_CMAP,
    COVARIANCE_COLORS,
    activity_trajectory_limits,
    plot_activity_trajectories,
    plot_connectivity_covariance,
    plot_connectivity_pairs,
    plot_first_perceptual_decision_making_trials,
    plot_trial_outputs,
)


class PlottingColorTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_covariance_heatmap_uses_shared_colormap(self) -> None:
        covariance = np.array(
            [
                [1.0, -0.5, 0.2, 0.1],
                [-0.5, 1.0, 0.7, -0.4],
                [0.2, 0.7, 1.0, 0.8],
                [0.1, -0.4, 0.8, 1.0],
            ]
        )

        _, axis = plot_connectivity_covariance(("I", "n", "m", "w"), covariance)

        self.assertEqual(axis.images[0].get_cmap().name, COVARIANCE_CMAP.name)

    def test_covariance_ellipses_use_sign_colors(self) -> None:
        increasing = np.arange(4, dtype=float)
        vectors = {
            "I": increasing,
            "n": -increasing,
            "m": increasing,
            "w": increasing,
        }

        _, axes = plot_connectivity_pairs(vectors)

        self.assertEqual(
            to_hex(axes[0, 0].patches[0].get_edgecolor()),
            COVARIANCE_COLORS["negative"].lower(),
        )
        self.assertEqual(
            to_hex(axes[0, 1].patches[0].get_edgecolor()),
            COVARIANCE_COLORS["positive"].lower(),
        )

    def test_choice_conditioned_plots_use_shared_colors(self) -> None:
        labels = np.array([-1, 1])
        trials = np.zeros((2, 3))
        outputs = np.zeros((2, 3))
        trajectories = np.array(
            [
                [[0.0, 0.0], [-1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]
        )

        _, trial_axes = plot_first_perceptual_decision_making_trials(trials, labels)
        _, output_axes = plot_trial_outputs(trials, labels, outputs, decision_steps=1)
        _, trajectory_axis = plot_activity_trajectories(trajectories, labels)

        for trial, choice in enumerate(labels):
            expected_color = CHOICE_COLORS[int(choice)]
            self.assertEqual(trial_axes[trial].lines[0].get_color(), expected_color)
            self.assertEqual(output_axes[trial, 0].lines[0].get_color(), expected_color)
            self.assertEqual(output_axes[trial, 1].lines[0].get_color(), expected_color)

        mean_lines = {
            line.get_label(): line
            for line in trajectory_axis.lines
            if line.get_label().startswith("choice")
        }
        self.assertEqual(mean_lines["choice -1"].get_color(), CHOICE_COLORS[-1])
        self.assertEqual(mean_lines["choice +1"].get_color(), CHOICE_COLORS[1])

    def test_trajectory_axes_scale_each_dimension_independently(self) -> None:
        trajectories = np.array(
            [
                [[-20.0, -1.0], [24.0, 0.5]],
                [[-18.0, 0.0], [20.0, 1.5]],
            ]
        )

        x_limits, y_limits = activity_trajectory_limits(trajectories)

        np.testing.assert_allclose(x_limits, (-25.2, 25.2))
        np.testing.assert_allclose(y_limits, (-1.575, 1.575))


if __name__ == "__main__":
    unittest.main()
