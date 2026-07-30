"""Core mathematical contracts for low-rank RNNs."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np
import torch
from jaxtyping import TypeCheckError
from torch import nn

from low_rank_rnn import analysis, mean_field
from low_rank_rnn.data import working_memory
from low_rank_rnn.model import LowRankRNN, persistent_transient_rnn
from low_rank_rnn.training import decision_accuracy, decision_loss, train_model


class LowRankRNNTests(unittest.TestCase):
    def test_forward_uses_euler_dynamics_and_rate_readout(self) -> None:
        model = LowRankRNN(2)
        with torch.no_grad():
            model.m.zero_()
            model.n.zero_()
            model.I.copy_(torch.tensor([1.0, 2.0]))
            model.w.fill_(1.0)

        outputs, states = model(torch.tensor([[1.0, 0.0]]))

        expected_states = torch.tensor([[[0.0, 0.0], [0.2, 0.4]]])
        expected_output = torch.tanh(expected_states[:, 1]).mean()
        torch.testing.assert_close(states, expected_states)
        torch.testing.assert_close(outputs[0, 0], torch.tensor(0.0))
        torch.testing.assert_close(outputs[0, 1], expected_output)

    def test_only_recurrent_vectors_are_trainable(self) -> None:
        model = LowRankRNN(4)

        self.assertEqual(dict(model.named_parameters()).keys(), {"m", "n"})
        self.assertEqual(dict(model.named_buffers()).keys(), {"I", "w"})

    def test_forward_rejects_inputs_without_batch_and_time_axes(self) -> None:
        with self.assertRaises(TypeCheckError):
            LowRankRNN(4)(torch.zeros(2, 3, 1))

    def test_persistent_transient_initialization_has_expected_loop_gains(self) -> None:
        model = persistent_transient_rnn(64, seed=4)

        overlap = analysis.connectivity_overlap(
            model.m.detach().numpy(),
            model.n.detach().numpy(),
        )

        np.testing.assert_allclose(overlap, np.diag((1.0, 0.5)), atol=1e-5)


class WorkingMemoryDataTests(unittest.TestCase):
    def test_fixed_delay_trials_place_both_stimuli_and_target(self) -> None:
        pairs = np.array(((10, 34), (30, 14)))

        inputs, targets = working_memory.make_fixed_delay_trials(pairs)

        amplitudes = working_memory.stimulus_amplitudes(pairs)
        self.assertEqual(inputs.shape, (2, working_memory.FIXED_TRIAL_STEPS))
        np.testing.assert_allclose(
            inputs[:, 5:11],
            np.repeat(amplitudes[:, 0, None], 6, axis=1),
        )
        np.testing.assert_allclose(
            inputs[:, 60:71],
            np.repeat(amplitudes[:, 1, None], 11, axis=1),
        )
        np.testing.assert_allclose(targets, amplitudes[:, 0] - amplitudes[:, 1])

    def test_variable_delay_mask_tracks_each_trial(self) -> None:
        pairs = np.array(((10, 34), (30, 14)))
        delays = np.array((25, 100))

        inputs, _, mask = working_memory.make_variable_delay_trials(pairs, delays)

        first_stop = (
            working_memory.VARIABLE_FIXATION_STEPS
            + working_memory.VARIABLE_STIMULUS_STEPS
        )
        for trial, delay in enumerate(delays):
            decision_start = (
                first_stop + delay + working_memory.VARIABLE_STIMULUS_STEPS
            )
            self.assertEqual(mask[trial].sum(), working_memory.VARIABLE_DECISION_STEPS)
            torch.testing.assert_close(
                mask[trial, decision_start : decision_start + 5],
                torch.ones(5),
            )
        self.assertEqual(inputs.shape[1], working_memory.VARIABLE_TRIAL_STEPS)


class TrainingTests(unittest.TestCase):
    def test_decision_loss_uses_only_the_final_window(self) -> None:
        outputs = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        labels = torch.tensor([1.0])

        loss = decision_loss(outputs, labels, decision_steps=2)

        torch.testing.assert_close(loss, torch.tensor(2.5))

    def test_decision_loss_rejects_mismatched_batches(self) -> None:
        with self.assertRaises(TypeCheckError):
            decision_loss(torch.zeros(2, 4), torch.zeros(3), decision_steps=2)

    def test_training_runs_and_keeps_fixed_vectors_fixed(self) -> None:
        torch.manual_seed(0)
        model = LowRankRNN(4)
        fixed_input = model.I.clone()
        fixed_readout = model.w.clone()
        inputs = torch.randn(8, 4)
        labels = torch.tensor([-1.0, 1.0] * 4)

        losses = train_model(
            model,
            inputs,
            labels,
            epochs=2,
            batch_size=4,
            decision_steps=2,
            log_every=None,
        )

        self.assertEqual(len(losses), 2)
        self.assertTrue(np.isfinite(losses).all())
        torch.testing.assert_close(model.I, fixed_input)
        torch.testing.assert_close(model.w, fixed_readout)

    def test_training_logs_at_interval_and_final_epoch(self) -> None:
        torch.manual_seed(0)
        model = LowRankRNN(4)
        inputs = torch.randn(8, 4)
        labels = torch.tensor([-1.0, 1.0] * 4)
        output = io.StringIO()

        with redirect_stdout(output):
            train_model(
                model,
                inputs,
                labels,
                epochs=5,
                batch_size=4,
                decision_steps=2,
                log_every=2,
            )

        self.assertEqual(
            [line.split(":", maxsplit=1)[0] for line in output.getvalue().splitlines()],
            ["Epoch 2", "Epoch 4", "Epoch 5"],
        )

    def test_accuracy_restores_evaluation_mode(self) -> None:
        class OutputModel(nn.Module):
            def forward(self, inputs):
                return inputs, inputs[:, :, None]

        model = OutputModel().eval()
        outputs = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
        labels = torch.tensor([-1.0, 1.0])

        accuracy = decision_accuracy(model, outputs, labels, decision_steps=2)

        self.assertEqual(accuracy, 1.0)
        self.assertFalse(model.training)


class AnalysisTests(unittest.TestCase):
    def test_run_model_returns_numpy_and_restores_mode(self) -> None:
        model = LowRankRNN(4)
        model.train()

        outputs, states = analysis.run_model(model, np.zeros((2, 3)))

        self.assertEqual(outputs.shape, (2, 3))
        self.assertEqual(states.shape, (2, 3, 4))
        self.assertTrue(model.training)

    def test_connectivity_vectors_include_each_rank_component(self) -> None:
        vectors = analysis.connectivity_vectors(LowRankRNN(3, rank=2))

        self.assertEqual(tuple(vectors), ("I", "n_1", "n_2", "m_1", "m_2", "w"))
        self.assertTrue(all(vector.shape == (3,) for vector in vectors.values()))

    def test_loading_gaussian_matches_numpy(self) -> None:
        vectors = {
            "I": np.array([1.0, 2.0, 3.0]),
            "n": np.array([0.0, 1.0, 4.0]),
        }
        samples = np.column_stack(tuple(vectors.values()))

        mean, covariance = analysis.fit_loading_gaussian(vectors)

        np.testing.assert_allclose(mean, samples.mean(axis=0))
        np.testing.assert_allclose(
            covariance,
            np.cov(samples, rowvar=False, bias=True),
        )

    def test_sampled_networks_split_higher_rank_loadings_by_column(self) -> None:
        names = ("I", "n_1", "n_2", "m_1", "m_2", "w")
        mean = np.arange(1.0, 7.0)
        covariance = np.zeros((6, 6))

        (network,) = analysis.sample_low_rank_rnns(
            names,
            mean,
            covariance,
            num_networks=1,
            num_neurons=5,
            rng=np.random.default_rng(0),
        )

        for name, expected_value in zip(names, mean, strict=True):
            vector, _, index = name.partition("_")
            column = getattr(network, vector)
            actual = column if not index else column[:, int(index) - 1]
            torch.testing.assert_close(actual, torch.full_like(actual, expected_value))

    def test_sampled_loading_vectors_are_reproducible(self) -> None:
        names = ("a", "b")
        first, first_covariance = analysis.sample_loading_vectors(
            names,
            np.zeros(2),
            np.eye(2),
            num_samples=10,
            rng=np.random.default_rng(5),
        )
        second, second_covariance = analysis.sample_loading_vectors(
            names,
            np.zeros(2),
            np.eye(2),
            num_samples=10,
            rng=np.random.default_rng(5),
        )

        np.testing.assert_allclose(first["a"], second["a"])
        np.testing.assert_allclose(first_covariance, second_covariance)

    def test_svd_canonical_model_preserves_connectivity(self) -> None:
        torch.manual_seed(4)
        model = LowRankRNN(32, rank=2)

        canonical, error = analysis.svd_canonical_model(model)

        self.assertLess(error, 1e-6)
        self.assertAlmostEqual(
            float(canonical.m[:, 0].detach() @ canonical.m[:, 1].detach()),
            0.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(canonical.n[:, 0].detach() @ canonical.n[:, 1].detach()),
            0.0,
            places=5,
        )

    def test_activity_projection_uses_m_and_orthogonal_input_axes(self) -> None:
        model = LowRankRNN(2)
        with torch.no_grad():
            model.m[:, 0].copy_(torch.tensor([1.0, 0.0]))
            model.I.copy_(torch.tensor([1.0, 1.0]))
        states = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])

        projected = analysis.project_rank_one_activity(states, model)

        np.testing.assert_allclose(projected, states.numpy())

    def test_regression_metrics_and_explained_variance(self) -> None:
        mse, r_squared = analysis.regression_metrics(
            np.array((-1.0, 1.0)),
            np.array((-1.0, 1.0)),
        )
        variance = analysis.explained_variance(
            np.array([[[0.0, 0.0], [1.0, 0.0]]])
        )

        self.assertEqual((mse, r_squared), (0.0, 1.0))
        np.testing.assert_allclose(variance, (1.0, 0.0))

    def test_fixed_point_search_classifies_a_double_well_flow(self) -> None:
        grid, flow, points, slopes = analysis.find_fixed_points_1d(
            lambda values: np.asarray(values) - np.asarray(values) ** 3,
            bounds=(-2, 2),
        )

        self.assertEqual(grid.shape, flow.shape)
        np.testing.assert_allclose(points, (-1.0, 0.0, 1.0), atol=1e-5)
        np.testing.assert_allclose(slopes, (-2.0, 1.0, -2.0), atol=1e-4)


class MeanFieldTests(unittest.TestCase):
    def test_gaussian_circuit_returns_rank_generic_histories(self) -> None:
        names = ("I", "n_1", "n_2", "m_1", "m_2", "w")

        outputs, kappa, filtered = mean_field.simulate_gaussian_circuit(
            np.zeros((3, 4)),
            names,
            np.zeros(6),
            np.eye(6),
            step_size=0.2,
        )

        self.assertEqual(outputs.shape, (3, 4))
        self.assertEqual(kappa.shape, (3, 4, 2))
        self.assertEqual(filtered.shape, (3, 4))
        np.testing.assert_allclose(outputs, 0)


if __name__ == "__main__":
    unittest.main()
