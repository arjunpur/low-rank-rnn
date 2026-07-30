"""Semantic color and labeling contracts for project plots."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
from jaxtyping import TypeCheckError

from low_rank_rnn import plotting
from low_rank_rnn.plotting.style import (
    CHOICE_COLORS,
    COVARIANCE_CMAP,
    COVARIANCE_COLORS,
)


class PlottingTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_latent_plane_requires_one_frequency_per_trajectory(self) -> None:
        with self.assertRaises(TypeCheckError):
            plotting.plot_latent_plane(
                np.zeros((2, 4, 2)),
                np.zeros(3),
            )

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

    def test_connectivity_summary_places_both_plots_in_one_figure(self) -> None:
        rng = np.random.default_rng(0)
        names = ("I", "n", "m", "w")
        vectors = {name: rng.normal(size=8) for name in names}
        covariance = np.cov(np.stack(tuple(vectors.values())))

        figure, pair_axes, covariance_axis = plotting.plot_connectivity_summary(
            vectors,
            names,
            covariance,
            row_names=("I", "n", "m"),
            column_names=("n", "m", "w"),
            covariance_title="Rank-one loading covariance",
        )

        self.assertEqual(pair_axes.shape, (3, 3))
        self.assertIs(pair_axes[0, 0].get_figure(root=True), figure)
        self.assertIs(covariance_axis.get_figure(root=True), figure)
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

    def test_trajectory_labels_sit_at_positive_axis_ends(self) -> None:
        trajectories = np.array(
            (
                ((0.0, 0.0), (-1.0, 1.0)),
                ((0.0, 0.0), (1.0, 1.0)),
            )
        )

        _, axis = plotting.plot_activity_trajectories_by_stimulus(
            trajectories,
            np.array([-0.5, 0.5]),
        )

        self.assertEqual(axis.xaxis.label.get_position()[0], 1)
        self.assertEqual(axis.yaxis.label.get_position()[1], 1)

    def test_pca_summary_labels_variance_and_trial_colors(self) -> None:
        trajectories = np.array(
            (
                ((0.0, 0.0), (-1.0, 1.0)),
                ((0.0, 0.0), (1.0, 1.0)),
            )
        )

        figure, axes = plotting.plot_pca_summary(
            np.array((0.75, 0.25)),
            trajectories,
            np.array((-0.5, 0.5)),
            num_components=2,
        )

        self.assertEqual(axes[1].get_xlabel(), "PC1 (75.0% variance)")
        self.assertEqual(axes[1].get_ylabel(), "PC2 (25.0% variance)")
        self.assertEqual(figure.axes[2].get_ylabel(), r"mean stimulus, $\bar{u}$")
        self.assertNotEqual(
            axes[1].lines[0].get_color(),
            axes[1].lines[1].get_color(),
        )

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

    def test_fixed_point_plot_shows_energy_and_classified_minima(self) -> None:
        grid = np.linspace(-2, 2, 101)
        flow = grid * (1 - grid**2)

        _, axis = plotting.plot_fixed_points(
            grid,
            flow,
            np.array((-1.0, 0.0, 1.0)),
            np.array((-2.0, 1.0, -2.0)),
        )

        self.assertEqual(
            axis.get_legend_handles_labels()[1],
            [
                r"energy $q(\kappa)=\frac{1}{2}f(\kappa)^2$",
                "Stable fixed point",
                "Unstable fixed point",
            ],
        )
        self.assertEqual(axis.get_title(), "Fixed-point energy")
        self.assertEqual(axis.get_yscale(), "log")

    def test_output_comparison_accepts_subfigure_axes(self) -> None:
        figure = plt.figure()
        subfigure = figure.subfigures()
        axes = subfigure.subplots(2, 2)

        returned_figure, returned_axes = plotting.plot_output_comparison(
            np.zeros((4, 10)),
            np.zeros((4, 10)),
            np.array((-1, -1, 1, 1)),
            np.arange(4),
            decision_steps=2,
            axes=axes,
        )

        self.assertIs(returned_figure, subfigure)
        np.testing.assert_array_equal(returned_axes, axes)

    def test_accuracy_comparison_shows_every_network(self) -> None:
        sampled_accuracies = np.array([0.5, 0.75, 1.0])

        _, axis = plotting.plot_accuracy_comparison(0.9, sampled_accuracies)

        trained_points, sampled_points = axis.collections[:2]
        np.testing.assert_allclose(trained_points.get_offsets()[:, 1], [0.9])
        np.testing.assert_allclose(
            sampled_points.get_offsets()[:, 1],
            sampled_accuracies,
        )

    def test_gaussian_sampling_summary_uses_separate_performance_figure(
        self,
    ) -> None:
        rng = np.random.default_rng(0)
        names = ("I", "n", "m", "w")
        trained = {name: rng.normal(size=16) for name in names}
        sampled = {name: rng.normal(size=16) for name in names}
        covariance = np.cov(np.stack(tuple(sampled.values())))

        (
            sampling_figure,
            performance_figure,
            loading_axes,
            covariance_axis,
            accuracy_axis,
        ) = plotting.plot_gaussian_sampling_summary(
            trained,
            sampled,
            names,
            covariance,
            1.0,
            np.ones(4),
            covariance_title="Sampled covariance",
        )

        self.assertEqual(loading_axes.shape, (2, 2))
        self.assertIs(
            loading_axes[0, 0].get_figure(root=True),
            sampling_figure,
        )
        self.assertIs(
            covariance_axis.get_figure(root=True),
            sampling_figure,
        )
        self.assertIs(
            accuracy_axis.get_figure(root=True),
            performance_figure,
        )
        self.assertIsNot(sampling_figure, performance_figure)
        self.assertNotIn(
            "chance: 50%",
            accuracy_axis.get_legend_handles_labels()[1],
        )
        sampling_figure.canvas.draw()
        performance_figure.canvas.draw()

    def test_memory_behavior_shows_losses_and_residual(self) -> None:
        targets = np.array(((-1.0, 0.0), (0.0, 1.0)))
        predictions = targets.ravel() + 0.1

        _, axes = plotting.plot_memory_behavior(
            (1.0, 0.1),
            (1.0, 0.5),
            targets,
            predictions,
            np.array((10, 34)),
            mse=0.01,
            loss_threshold=0.005,
        )

        self.assertEqual(axes.shape, (2,))
        self.assertEqual(
            axes[1].images[0].get_cmap().name,
            "residual",
        )
        self.assertEqual(
            axes[1].get_title(),
            "Prediction residual  |  MSE=1.00e-02",
        )

    def test_latent_plane_uses_arrows_and_labeled_frequency_colorbar(self) -> None:
        trajectories = np.array(
            (
                ((0.0, 0.0), (0.5, 0.2), (1.0, 0.5), (0.8, 0.7)),
                ((0.0, 0.0), (-0.5, -0.2), (-1.0, -0.5), (-0.8, -0.7)),
            )
        )

        figure, axis = plotting.plot_latent_plane(
            trajectories,
            np.array((10.0, 34.0)),
            colorbar_label=r"first stimulus frequency, $f_1$ (Hz)",
        )

        self.assertEqual(len(axis.lines), 2)
        self.assertEqual(len(axis.texts), 2)
        self.assertEqual(len(axis.collections), 0)
        self.assertEqual(
            figure.axes[1].get_ylabel(),
            r"first stimulus frequency, $f_1$ (Hz)",
        )

    def test_gaussian_pipeline_shows_every_sampled_network(self) -> None:
        sampled_mses = np.array((0.1, 0.2, 0.3))

        _, axes = plotting.plot_gaussian_pipeline(
            0.001,
            0.01,
            0.02,
            sampled_mses,
            np.array((-1.0, 0.0, 1.0)),
            np.array((-1.0, 0.0, 1.0)),
            np.array((-0.8, 0.0, 0.8)),
            np.array((-0.6, 0.0, 0.6)),
            threshold=0.005,
            title="Covariance pipeline",
            score_label="MSE",
            readout_title="Readouts",
        )

        sampled_points = axes[0].collections[3]
        np.testing.assert_allclose(
            sampled_points.get_offsets()[:, 1],
            sampled_mses,
        )
        self.assertEqual(
            tuple(label.get_text() for label in axes[0].get_xticklabels()),
            (
                "source\ntrained RNN",
                "full-covariance\nGaussian circuit",
                "handout diagonal\ncircuit",
                "finite Gaussian-\nsampled RNNs",
            ),
        )
        np.testing.assert_allclose(
            [
                collection.get_offsets()[0, 1]
                for collection in axes[0].collections[:3]
            ],
            (0.001, 0.01, 0.02),
        )
        self.assertEqual(
            axes[1].get_legend_handles_labels()[1],
            [
                "source trained RNN",
                "full-covariance Gaussian circuit",
                "handout diagonal circuit",
            ],
        )

    def test_matched_network_comparison_plots_full_delay_battery(self) -> None:
        delays_ms = np.arange(500, 1300, 100)
        network_a_mses = np.linspace(0.001, 0.2, len(delays_ms))
        network_b_mses = np.linspace(0.003, 0.005, len(delays_ms))

        _, axis = plotting.plot_matched_network_comparison(
            delays_ms,
            network_a_mses,
            network_b_mses,
            threshold=0.005,
        )

        np.testing.assert_allclose(
            axis.lines[0].get_ydata(),
            network_a_mses,
        )
        np.testing.assert_allclose(
            axis.lines[1].get_ydata(),
            network_b_mses,
        )
        np.testing.assert_allclose(axis.lines[0].get_xdata(), delays_ms)
        self.assertEqual(axis.get_yscale(), "log")

    def test_training_loss_labels_curriculum_stages(self) -> None:
        _, axis = plotting.plot_training_loss(
            np.linspace(0.01, 0.001, 6),
            title="Curriculum",
            stage_ends=(2, 4),
            stage_labels=("short delays", "medium delays", "long delays"),
        )

        self.assertEqual(
            tuple(label.get_text() for label in axis.texts),
            ("short delays", "medium delays", "long delays"),
        )
        self.assertEqual(len(axis.lines), 3)


if __name__ == "__main__":
    unittest.main()
