"""Numerical analysis of trained low-rank RNNs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch

from low_rank_rnn.model import LowRankRNN


def connectivity_vectors(model: LowRankRNN) -> dict[str, np.ndarray]:
    """Return each neuron's coordinates in connectivity space."""
    rank = model.m.shape[1]
    suffixes = [""] if rank == 1 else [f"_{index + 1}" for index in range(rank)]
    vectors = {"I": model.I.detach().cpu().numpy()}
    for name in ("n", "m"):
        values = getattr(model, name).detach().cpu().numpy()
        vectors.update(
            {
                f"{name}{suffix}": values[:, index]
                for index, suffix in enumerate(suffixes)
            }
        )
    vectors["w"] = model.w.detach().cpu().numpy()
    return vectors


def connectivity_covariance(
    vectors: Mapping[str, npt.ArrayLike],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return names and sample covariance of connectivity vectors."""
    names = tuple(vectors)
    matrix = np.stack([vectors[name] for name in names])
    return names, np.cov(matrix)


def fit_loading_gaussian(
    vectors: Mapping[str, npt.ArrayLike],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian to neuron coordinates in loading space."""
    samples = np.column_stack(tuple(vectors.values()))
    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False, bias=True)
    return mean, covariance


def sample_loading_gaussian(
    mean: npt.ArrayLike,
    covariance: npt.ArrayLike,
    *,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw loading vectors from a fitted Gaussian."""
    generator = rng if rng is not None else np.random.default_rng()
    return generator.multivariate_normal(
        np.asarray(mean),
        np.asarray(covariance),
        size=num_samples,
    )


def sample_rank_one_rnns(
    names: Sequence[str],
    mean: npt.ArrayLike,
    covariance: npt.ArrayLike,
    *,
    num_networks: int,
    num_neurons: int,
    rng: np.random.Generator | None = None,
) -> list[LowRankRNN]:
    """Create rank-one RNNs from samples of a fitted loading Gaussian."""
    names = tuple(names)
    expected_names = ("I", "n", "m", "w")
    if len(names) != len(expected_names) or set(names) != set(expected_names):
        raise ValueError(f"names must contain exactly {expected_names}")

    generator = rng if rng is not None else np.random.default_rng()
    networks = []
    for _ in range(num_networks):
        network_loadings = sample_loading_gaussian(
            mean,
            covariance,
            num_samples=num_neurons,
            rng=generator,
        )
        network = LowRankRNN(n_units=num_neurons, rank=1)
        with torch.no_grad():
            for name, values in zip(names, network_loadings.T, strict=True):
                target = getattr(network, name)
                target.copy_(
                    torch.as_tensor(values, dtype=target.dtype).reshape_as(target)
                )
        networks.append(network)
    return networks


def project_rank_one_activity(
    states: torch.Tensor,
    model: LowRankRNN,
) -> np.ndarray:
    """Project rank-one activity onto orthonormal ``m`` and ``I`` axes."""
    if model.m.shape[1] != 1:
        raise ValueError("model must have rank 1")

    recurrent_vector = model.m[:, 0].detach()
    input_vector = model.I.detach()

    recurrent_norm = torch.linalg.vector_norm(recurrent_vector)
    if torch.isclose(recurrent_norm, torch.zeros_like(recurrent_norm)):
        raise ValueError("m must be nonzero")
    recurrent_axis = recurrent_vector / recurrent_norm

    orthogonal_input = input_vector - torch.dot(input_vector, recurrent_axis) * recurrent_axis
    input_norm = torch.linalg.vector_norm(orthogonal_input)
    if torch.isclose(input_norm, torch.zeros_like(input_norm)):
        raise ValueError("I and m must span a plane")

    input_axis = orthogonal_input / input_norm
    basis = torch.stack((recurrent_axis, input_axis), dim=1)
    return (states.detach() @ basis).cpu().numpy()
