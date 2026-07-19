"""Numerical analysis of trained rank-one RNNs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import torch

from low_rank_rnn.model import RankOneRNN


def connectivity_vectors(model: RankOneRNN) -> dict[str, np.ndarray]:
    """Return each neuron's coordinates in connectivity space."""
    return {
        name: getattr(model, name).detach().cpu().numpy()
        for name in ("I", "n", "m", "w")
    }


def connectivity_covariance(
    vectors: Mapping[str, npt.ArrayLike],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return names and sample covariance of connectivity vectors."""
    names = tuple(vectors)
    matrix = np.stack([vectors[name] for name in names])
    return names, np.cov(matrix)


def project_activity(states: torch.Tensor, model: RankOneRNN) -> np.ndarray:
    """Project population activity onto the plane spanned by ``m`` and ``I``."""
    recurrent_vector = model.m.detach()
    input_vector = model.I.detach()

    recurrent_norm = torch.linalg.vector_norm(recurrent_vector)
    if torch.isclose(recurrent_norm, torch.zeros_like(recurrent_norm)):
        raise ValueError("m must be nonzero")
    recurrent_axis = recurrent_vector / recurrent_norm

    orthogonal_input = input_vector - torch.dot(input_vector, recurrent_axis) * recurrent_axis
    input_norm = torch.linalg.vector_norm(orthogonal_input)
    if torch.isclose(input_norm, torch.zeros_like(input_norm)):
        raise ValueError("I and m must span a plane")

    basis = torch.stack((recurrent_axis, orthogonal_input / input_norm), dim=1)
    return (states.detach() @ basis).cpu().numpy()
