"""Semantic color and labeling contracts for project plots."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np

from low_rank_rnn import plotting
from low_rank_rnn.plotting.style import (
    CHOICE_COLORS,
    COVARIANCE_CMAP,
    COVARIANCE_COLORS,
    SIGNED_VALUE_CMAP,
)


class PlottingTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_covariance_plots_use_shared_semantic_colors(self) -> None:
        increasing = np.arange(4, dtype=float)
        vectors = {
            "I": increasing,
            "n": -increasing,
            "m": increasing,
            "w": increasing,
        }
        covariance = np.cov(np.stack(tuple(vectors.values())))

        _, heatmap_axis = plotting.plot_connectivity_covariance(
            tuple(vectors),
            covariance,
        )
        _, pair_axes = plotting.plot_connectivity_pairs(
            vectors,
            row_names=("I", "n", "m"),
            column_names=("n", "m", "w"),
        )

        self.assertEqual(heatmap_axis.images[0].get_cmap().name, COVARIANCE_CMAP.name)
        self.assertEqual(
            to_hex(pair_axes[0, 0].patches[0].get_edgecolor()),
            COVARIANCE_COLORS["negative"].lower(),
        )
        self.assertEqual(
            to_hex(pair_axes[0, 1].patches[0].get_edgecolor()),
            COVARIANCE_COLORS["positive"].lower(),
        )

    def test_rank_two_connectivity_plots_include_every_pair(self) -> None:
        rng = np.random.default_rng(0)
        names = ("I", "n_1", "n_2", "m_1", "m_2", "w")
        vectors = {name: rng.normal(size=8) for name in names}

        figure, axes = plotting.plot_connectivity_pairs(
            vectors,
            row_names=names[:-1],
            column_names=names[1:],
        )

        self.assertEqual(axes.shape, (5, 5))
        self.assertEqual(sum(axis.get_visible() for axis in axes.flat), 15)
        figure.canvas.draw()

    def test_decision_trials_use_choice_colors(self) -> None:
        labels = np.array([-1, 1])
        trials = np.zeros((2, 3))

        _, axes = plotting.plot_decision_trials(
            trials,
            labels,
            decision_steps=1,
        )

        for trial, choice in enumerate(labels):
            self.assertEqual(
                axes[trial].lines[0].get_color(),
                CHOICE_COLORS[int(choice)],
            )

    def test_hidden_rates_use_shared_signed_colormap(self) -> None:
        _, axis = plotting.plot_hidden_rates(
            np.zeros((3, 2)),
            decision_steps=1,
        )

        self.assertEqual(axis.images[0].get_cmap().name, SIGNED_VALUE_CMAP.name)

    def test_stimulus_trajectory_colorbar_labels_mean_stimulus(self) -> None:
        trajectories = np.array(
            (
                ((0.0, 0.0), (-1.0, 1.0)),
                ((0.0, 0.0), (1.0, 1.0)),
            )
        )

        figure, _ = plotting.plot_activity_trajectories_by_stimulus(
            trajectories,
            np.array([-0.5, 0.5]),
        )

        self.assertEqual(figure.axes[1].get_ylabel(), r"mean stimulus, $\bar{u}$")

    def test_reduced_system_plot_labels_state_variables(self) -> None:
        trajectories = np.array(
            (
                ((0.0, 0.0), (-1.0, 1.0)),
                ((0.0, 0.0), (1.0, 1.0)),
            )
        )

        _, axis = plotting.plot_reduced_system_trajectories(
            trajectories,
            np.array([-0.5, 0.5]),
        )

        self.assertEqual(axis.get_xlabel(), r"latent state, $\kappa$")
        self.assertEqual(axis.get_ylabel(), r"filtered input, $v$")

    def test_accuracy_comparison_shows_every_network(self) -> None:
        sampled_accuracies = np.array([0.5, 0.75, 1.0])

        _, axis = plotting.plot_accuracy_comparison(0.9, sampled_accuracies)

        trained_points, sampled_points = axis.collections[:2]
        np.testing.assert_allclose(trained_points.get_offsets()[:, 1], [0.9])
        np.testing.assert_allclose(
            sampled_points.get_offsets()[:, 1],
            sampled_accuracies,
        )

    def test_memory_behavior_uses_separate_residual_scale(self) -> None:
        targets = np.array(((-1.0, 0.0), (0.0, 1.0)))
        predictions = targets.ravel() + 0.1

        _, axes = plotting.plot_memory_behavior(
            (1.0, 0.1),
            (1.0, 0.5),
            targets,
            predictions,
            np.array((10, 34)),
            mse=0.01,
            r_squared=0.9,
            loss_threshold=0.005,
        )

        self.assertEqual(
            axes[1, 1].images[0].get_cmap().name,
            "residual",
        )


if __name__ == "__main__":
    unittest.main()
