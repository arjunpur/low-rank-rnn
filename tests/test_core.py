"""Core mathematical contracts for low-rank RNNs."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np
import torch
from jaxtyping import TypeCheckError
from torch import nn

from low_rank_rnn.analysis import (
    connectivity_covariance,
    connectivity_vectors,
    fit_loading_gaussian,
    project_rank_one_activity,
    sample_loading_gaussian,
    sample_rank_one_rnns,
)
from low_rank_rnn.model import LowRankRNN
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
        model = LowRankRNN(4)

        with self.assertRaises(TypeCheckError):
            model(torch.zeros(2, 3, 1))


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

        log_lines = output.getvalue().splitlines()
        self.assertEqual(
            [line.split(":", maxsplit=1)[0] for line in log_lines],
            ["Epoch 2", "Epoch 4", "Epoch 5"],
        )
        self.assertTrue(all("loss=" in line for line in log_lines))
        self.assertTrue(all("accuracy=" not in line for line in log_lines))

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
    def test_connectivity_covariance_matches_numpy(self) -> None:
        model = LowRankRNN(3)
        with torch.no_grad():
            model.I.copy_(torch.tensor([1.0, 2.0, 3.0]))
            model.n[:, 0].copy_(torch.tensor([0.0, 1.0, 4.0]))
            model.m[:, 0].copy_(torch.tensor([2.0, 0.0, 1.0]))
            model.w.copy_(torch.tensor([-1.0, 2.0, 1.0]))

        vectors = connectivity_vectors(model)
        names, covariance = connectivity_covariance(vectors)

        self.assertEqual(names, ("I", "n", "m", "w"))
        np.testing.assert_allclose(covariance, np.cov(np.stack(list(vectors.values()))))

    def test_connectivity_vectors_include_each_rank_component(self) -> None:
        model = LowRankRNN(3, rank=2)

        vectors = connectivity_vectors(model)

        self.assertEqual(tuple(vectors), ("I", "n_1", "n_2", "m_1", "m_2", "w"))
        self.assertTrue(all(vector.shape == (3,) for vector in vectors.values()))

    def test_loading_gaussian_matches_numpy(self) -> None:
        vectors = {
            "I": np.array([1.0, 2.0, 3.0]),
            "n": np.array([0.0, 1.0, 4.0]),
        }
        samples = np.column_stack(tuple(vectors.values()))

        mean, covariance = fit_loading_gaussian(vectors)

        np.testing.assert_allclose(mean, samples.mean(axis=0))
        np.testing.assert_allclose(
            covariance,
            np.cov(samples, rowvar=False, bias=True),
        )

    def test_sampled_networks_use_gaussian_loading_coordinates(self) -> None:
        names = ("I", "n", "m", "w")
        mean = np.array([1.0, 2.0, 3.0, 4.0])
        covariance = np.zeros((4, 4))

        networks = sample_rank_one_rnns(
            names,
            mean,
            covariance,
            num_networks=3,
            num_neurons=5,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(len(networks), 3)
        for network in networks:
            self.assertEqual(network.n_units, 5)
            for name, expected_value in zip(names, mean, strict=True):
                expected = torch.full_like(getattr(network, name), expected_value)
                torch.testing.assert_close(getattr(network, name), expected)

    def test_loading_gaussian_sampling_is_reproducible(self) -> None:
        mean = np.array([1.0, 2.0])
        covariance = np.eye(2)

        first = sample_loading_gaussian(
            mean,
            covariance,
            num_samples=4,
            rng=np.random.default_rng(5),
        )
        second = sample_loading_gaussian(
            mean,
            covariance,
            num_samples=4,
            rng=np.random.default_rng(5),
        )

        self.assertEqual(first.shape, (4, 2))
        np.testing.assert_allclose(first, second)

    def test_activity_projection_uses_m_and_orthogonal_input_axes(self) -> None:
        model = LowRankRNN(2)
        with torch.no_grad():
            model.m[:, 0].copy_(torch.tensor([1.0, 0.0]))
            model.I.copy_(torch.tensor([1.0, 1.0]))
        states = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])

        projected = project_rank_one_activity(states, model)

        np.testing.assert_allclose(projected, states.numpy())

    def test_rank_one_activity_projection_rejects_higher_rank_models(self) -> None:
        model = LowRankRNN(2, rank=2)

        with self.assertRaisesRegex(ValueError, "rank 1"):
            project_rank_one_activity(torch.zeros(1, 1, 2), model)


if __name__ == "__main__":
    unittest.main()
