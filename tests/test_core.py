"""Core mathematical contracts for the rank-one RNN."""

import unittest

import numpy as np
import torch
from torch import nn

from low_rank_rnn.analysis import (
    connectivity_covariance,
    connectivity_vectors,
    project_activity,
)
from low_rank_rnn.model import RankOneRNN
from low_rank_rnn.training import decision_accuracy, decision_loss, train_model


class RankOneRNNTests(unittest.TestCase):
    def test_forward_uses_euler_dynamics_and_rate_readout(self) -> None:
        model = RankOneRNN(2)
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
        model = RankOneRNN(4)

        self.assertEqual(dict(model.named_parameters()).keys(), {"m", "n"})
        self.assertEqual(dict(model.named_buffers()).keys(), {"I", "w"})


class TrainingTests(unittest.TestCase):
    def test_decision_loss_uses_only_the_final_window(self) -> None:
        outputs = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        labels = torch.tensor([1.0])

        loss = decision_loss(outputs, labels, decision_steps=2)

        torch.testing.assert_close(loss, torch.tensor(2.5))

    def test_training_runs_and_keeps_fixed_vectors_fixed(self) -> None:
        torch.manual_seed(0)
        model = RankOneRNN(4)
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
        )

        self.assertEqual(len(losses), 2)
        self.assertTrue(np.isfinite(losses).all())
        torch.testing.assert_close(model.I, fixed_input)
        torch.testing.assert_close(model.w, fixed_readout)

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
        model = RankOneRNN(3)
        with torch.no_grad():
            model.I.copy_(torch.tensor([1.0, 2.0, 3.0]))
            model.n.copy_(torch.tensor([0.0, 1.0, 4.0]))
            model.m.copy_(torch.tensor([2.0, 0.0, 1.0]))
            model.w.copy_(torch.tensor([-1.0, 2.0, 1.0]))

        vectors = connectivity_vectors(model)
        names, covariance = connectivity_covariance(vectors)

        self.assertEqual(names, ("I", "n", "m", "w"))
        np.testing.assert_allclose(covariance, np.cov(np.stack(list(vectors.values()))))

    def test_activity_projection_uses_m_and_orthogonal_input_axes(self) -> None:
        model = RankOneRNN(2)
        with torch.no_grad():
            model.m.copy_(torch.tensor([1.0, 0.0]))
            model.I.copy_(torch.tensor([1.0, 1.0]))
        states = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])

        projected = project_activity(states, model)

        np.testing.assert_allclose(projected, states.numpy())


if __name__ == "__main__":
    unittest.main()
