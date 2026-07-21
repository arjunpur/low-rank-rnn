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
    MEAN_CHOICE_COLORS,
    activity_trajectory_limits,
    plot_accuracy_comparison,
    plot_activity_trajectories,
    plot_activity_trajectories_by_stimulus,
    plot_connectivity_covariance,
    plot_connectivity_pairs,
    plot_first_perceptual_decision_making_trials,
    plot_reduced_system_accuracy,
    plot_reduced_system_trajectories,
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

        _, axes = plot_connectivity_pairs(
            vectors,
            row_names=("I", "n", "m"),
            column_names=("n", "m", "w"),
        )

        self.assertEqual(
            to_hex(axes[0, 0].patches[0].get_edgecolor()),
            COVARIANCE_COLORS["negative"].lower(),
        )
        self.assertEqual(
            to_hex(axes[0, 1].patches[0].get_edgecolor()),
            COVARIANCE_COLORS["positive"].lower(),
        )

    def test_rank_two_connectivity_plots_include_every_pair(self) -> None:
        rng = np.random.default_rng(0)
        names = ("I", "n_1", "n_2", "m_1", "m_2", "w")
        vectors = {name: rng.normal(size=8) for name in names}

        pair_figure, axes = plot_connectivity_pairs(
            vectors,
            row_names=names[:-1],
            column_names=names[1:],
        )
        covariance_figure, _ = plot_connectivity_covariance(
            names,
            np.cov(np.stack(tuple(vectors.values()))),
        )

        self.assertEqual(axes.shape, (5, 5))
        self.assertEqual(sum(axis.get_visible() for axis in axes.flat), 15)
        pair_figure.canvas.draw()
        covariance_figure.canvas.draw()

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
            if line.get_label().startswith("mean choice")
        }
        self.assertEqual(
            mean_lines["mean choice -1"].get_color(),
            MEAN_CHOICE_COLORS[-1],
        )
        self.assertEqual(
            mean_lines["mean choice +1"].get_color(),
            MEAN_CHOICE_COLORS[1],
        )

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

    def test_stimulus_trajectory_colorbar_labels_mean_stimulus(self) -> None:
        trajectories = np.array(
            [
                [[0.0, 0.0], [-1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]
        )

        figure, _ = plot_activity_trajectories_by_stimulus(
            trajectories,
            np.array([-0.5, 0.5]),
        )

        self.assertEqual(figure.axes[1].get_ylabel(), r"mean stimulus, $\bar{u}$")

    def test_reduced_system_trajectory_axes_label_state_variables(self) -> None:
        trajectories = np.array(
            [
                [[0.0, 0.0], [-1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]
        )

        _, axis = plot_reduced_system_trajectories(
            trajectories,
            np.array([-0.5, 0.5]),
        )

        self.assertEqual(axis.get_xlabel(), r"latent state, $\kappa$")
        self.assertEqual(axis.get_ylabel(), r"filtered input, $v$")

    def test_accuracy_comparison_shows_every_network(self) -> None:
        sampled_accuracies = np.array([0.5, 0.75, 1.0])

        _, axis = plot_accuracy_comparison(0.9, sampled_accuracies)

        trained_points, sampled_points = axis.collections[:2]
        np.testing.assert_allclose(trained_points.get_offsets()[:, 1], [0.9])
        np.testing.assert_allclose(
            sampled_points.get_offsets()[:, 1],
            sampled_accuracies,
        )

    def test_reduced_system_accuracy_compares_both_models(self) -> None:
        _, axis = plot_reduced_system_accuracy(0.9, 0.8)

        np.testing.assert_allclose(
            [patch.get_height() for patch in axis.patches],
            [0.9, 0.8],
        )


if __name__ == "__main__":
    unittest.main()
