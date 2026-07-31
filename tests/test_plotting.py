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

        figure, axes = plotting.plot_decision_trials(
            trials,
            labels,
            decision_steps=1,
        )

        for trial, choice in enumerate(labels):
            self.assertEqual(
                axes[trial].lines[0].get_color(),
                CHOICE_COLORS[int(choice)],
            )
        self.assertEqual(figure.get_suptitle(), "Example decision trials")
        self.assertEqual(
            [text.get_text() for text in axes[0].get_legend().get_texts()],
            [
                "left choice (-1)",
                "right choice (+1)",
                "stimulus window",
                "decision window",
            ],
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

        self.assertEqual(axes[1].get_xlabel(), "PC1")
        self.assertEqual(axes[1].get_ylabel(), "PC2")
        self.assertEqual(axes[1].get_title(), "Projection in PCA space")
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
        self.assertEqual(axis.get_title(), "Equivalent circuit trajectory")

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
        self.assertEqual(axis.get_title(), "Fixed point minimization")
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
        self.assertEqual(
            returned_figure.get_suptitle(),
            "Sample trajectories of RNN vs equivalent circuit",
        )

    def test_reduced_system_accuracy_title(self) -> None:
        _, axis = plotting.plot_reduced_system_accuracy(0.9, 0.85)

        self.assertEqual(axis.get_title(), "Performance comparison")

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

    def test_memory_behavior_shows_losses_and_prediction_grid(self) -> None:
        targets = np.array(((-1.0, 0.0), (0.0, 1.0)))
        predictions = targets.ravel() + 0.1

        figure, axes = plotting.plot_memory_behavior(
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
            "signed_value",
        )
        self.assertEqual(
            axes[1].get_title(),
            "Predicted $y$  |  MSE=1.00e-02",
        )
        np.testing.assert_allclose(
            axes[1].images[0].get_array(),
            predictions.reshape(targets.shape),
        )
        self.assertEqual(axes[1].images[0].norm.vmin, -1)
        self.assertEqual(axes[1].images[0].norm.vmax, 1)
        self.assertEqual(
            figure.axes[-1].get_ylabel(),
            "predicted normalized value $y$",
        )

    def test_memory_trials_pair_each_input_with_its_output(self) -> None:
        inputs = np.zeros((2, 10))
        inputs[:, 2:4] = ((-0.5,), (0.5,))
        inputs[:, 6:8] = ((0.5,), (-0.5,))
        outputs = np.vstack((np.linspace(0, -1, 10), np.linspace(0, 1, 10)))

        _, axes = plotting.plot_memory_trials(
            np.array(((10, 34), (34, 10))),
            inputs,
            np.array((-1.0, 1.0)),
            outputs,
            first_window=(2, 4),
            second_window=(6, 8),
            decision_steps=2,
        )

        self.assertEqual(axes.shape, (2, 2))
        np.testing.assert_allclose(axes[0, 0].lines[0].get_ydata(), inputs[0])
        np.testing.assert_allclose(axes[0, 1].lines[0].get_ydata(), outputs[0])
        self.assertIn("decision =", axes[0, 1].get_title(loc="right"))
        self.assertEqual(
            axes[0, 1].get_legend_handles_labels()[1],
            ["output", "target", "final decision"],
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

    def test_circuit_trajectories_compare_each_system(self) -> None:
        trajectories = np.zeros((2, 3, 4, 2))
        trajectories[0, :, :, 0] = 1
        trajectories[1, :, :, 1] = 1

        figure, axes = plotting.plot_circuit_trajectories(
            trajectories,
            np.array((10.0, 22.0, 34.0)),
            ("Fitted covariance", "Paper covariance"),
            colorbar_label="first frequency (Hz)",
            title="Circuit trajectories",
        )

        self.assertEqual(axes.shape, (2,))
        self.assertEqual(len(axes[0].lines), 3)
        self.assertEqual(axes[1].get_title(), "Paper covariance")
        self.assertEqual(figure.axes[-1].get_ylabel(), "first frequency (Hz)")

    def test_regression_comparison_labels_metrics(self) -> None:
        targets = np.array((-1.0, 0.0, 1.0))
        predictions = np.vstack((targets, 0.5 * targets))

        _, axes = plotting.plot_regression_comparison(
            targets,
            predictions,
            ("Fitted covariance", "Paper covariance"),
            mean_squared_errors=np.array((0.0, 0.1)),
            r_squared_values=np.array((1.0, 0.5)),
            title="Circuit accuracy",
        )

        self.assertEqual(axes.shape, (2,))
        self.assertEqual(
            axes[1].get_title(),
            "Paper covariance\nMSE=0.1000, $R^2$=0.500",
        )
        np.testing.assert_allclose(
            axes[0].collections[0].get_offsets()[:, 1],
            targets,
        )

    def test_delay_mse_marks_reference_and_recurrence_delays(self) -> None:
        delay_steps = np.arange(25, 351)
        mean_squared_errors = 0.2 * (
            1 - np.cos(2 * np.pi * (delay_steps - 49) / 138)
        )

        _, axis = plotting.plot_delay_mse(
            delay_steps,
            mean_squared_errors,
            title="Prediction error over extended delays",
            reference_delay=49,
            reference_label="trained delay",
            annotation_delays=(187,),
            annotation_labels=("low error again",),
            x_ticks=(25, 49, 100, 150, 200, 250, 300, 350),
            y_limit=0.5,
        )

        np.testing.assert_allclose(
            axis.lines[0].get_ydata(),
            mean_squared_errors,
        )
        self.assertEqual(axis.lines[1].get_xdata(), [49, 49])
        self.assertEqual(axis.get_ylim(), (0, 0.5))
        self.assertEqual(
            axis.get_title(loc="left"),
            "Prediction error over extended delays",
        )
        self.assertEqual(
            tuple(text.get_text() for text in axis.texts),
            (
                "trained delay\n49 steps",
                "low error again\n187 steps",
            ),
        )

    def test_mse_comparison_shows_every_sampled_network(self) -> None:
        sampled_mses = np.array((0.1, 0.2, 0.3))

        _, axis = plotting.plot_mse_comparison(
            0.001,
            sampled_mses,
            threshold=0.005,
            title="Gaussian resampling",
        )

        trained_points, sampled_points = axis.collections[:2]
        np.testing.assert_allclose(
            trained_points.get_offsets()[:, 1],
            [0.001],
        )
        np.testing.assert_allclose(
            sampled_points.get_offsets()[:, 1],
            sampled_mses,
        )
        self.assertEqual(
            tuple(label.get_text() for label in axis.get_xticklabels()),
            ("Trained network", "Gaussian samples"),
        )
        self.assertEqual(axis.get_yscale(), "log")
        self.assertEqual(axis.get_legend()._loc, 2)
        self.assertIn(
            "sample median: 0.2",
            axis.get_legend_handles_labels()[1],
        )

    def test_circuit_mse_comparison_shows_all_three_systems(self) -> None:
        values = np.array((0.001, 0.02, 0.03))

        _, axis = plotting.plot_circuit_mse_comparison(
            *values,
            threshold=0.005,
        )

        np.testing.assert_allclose(
            [collection.get_offsets()[0, 1] for collection in axis.collections],
            values,
        )
        self.assertEqual(
            tuple(label.get_text() for label in axis.get_xticklabels()),
            (
                "Source\ntrained RNN",
                "Circuit with\nsource covariance",
                "Circuit with\npaper parameters",
            ),
        )
        self.assertEqual(axis.get_yscale(), "log")
        self.assertEqual(len(axis.texts), 3)

if __name__ == "__main__":
    unittest.main()
