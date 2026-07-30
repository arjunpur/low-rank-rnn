"""Gaussian mean-field circuits for low-rank RNNs."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.polynomial.hermite import hermgauss

from low_rank_rnn.analysis import find_fixed_points_1d


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = hermgauss(40)
_NORMAL_NODES = np.sqrt(2) * _QUADRATURE_NODES
_NORMAL_WEIGHTS = _QUADRATURE_WEIGHTS / np.sqrt(np.pi)


def _loading_indices(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = max((len(names) - 2) // 2, 1)
    n_indices = np.arange(1, 1 + rank)
    m_indices = np.arange(1 + rank, 1 + 2 * rank)
    basis_indices = np.array((0, *m_indices))
    return n_indices, m_indices, basis_indices


def _rate_moments(
    coefficients: npt.ArrayLike,
    mean: npt.ArrayLike,
    covariance: npt.ArrayLike,
    basis_indices: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=float))
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    basis_indices = np.asarray(basis_indices)
    current_mean = coefficients @ mean[basis_indices]
    current_variance = np.einsum(
        "bi,ij,bj->b",
        coefficients,
        covariance[np.ix_(basis_indices, basis_indices)],
        coefficients,
    )
    currents = current_mean[:, None] + np.sqrt(
        np.maximum(current_variance, 0)
    )[:, None] * _NORMAL_NODES
    rates = np.tanh(currents)
    return (
        rates @ _NORMAL_WEIGHTS,
        (1 - rates**2) @ _NORMAL_WEIGHTS,
    )


def gaussian_latent_flow(
    kappa: npt.ArrayLike,
    filtered_input: npt.ArrayLike,
    names: Sequence[str],
    mean: npt.ArrayLike,
    covariance: npt.ArrayLike,
) -> np.ndarray:
    """Evaluate the fitted Gaussian circuit's recurrent flow."""
    kappa = np.atleast_2d(np.asarray(kappa, dtype=float))
    filtered_input = np.broadcast_to(
        np.asarray(filtered_input, dtype=float),
        len(kappa),
    )
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    n_indices, _, basis_indices = _loading_indices(names)
    coefficients = np.column_stack((filtered_input, kappa))
    mean_rate, mean_gain = _rate_moments(
        coefficients,
        mean,
        covariance,
        basis_indices,
    )
    covariance_drive = (
        covariance[np.ix_(n_indices, basis_indices)] @ coefficients.T
    ).T
    recurrent_drive = (
        mean[n_indices][None, :] * mean_rate[:, None]
        + covariance_drive * mean_gain[:, None]
    )
    return -kappa + recurrent_drive


def simulate_gaussian_circuit(
    inputs: npt.ArrayLike,
    names: Sequence[str],
    mean: npt.ArrayLike,
    covariance: npt.ArrayLike,
    *,
    step_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the Gaussian equivalent circuit for any low rank."""
    inputs = np.asarray(inputs, dtype=float)
    names = tuple(names)
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    n_indices, _, basis_indices = _loading_indices(names)
    readout_index = len(names) - 1
    rank = len(n_indices)

    batch_size, num_steps = inputs.shape
    kappa = np.zeros((batch_size, rank))
    filtered_input = np.zeros(batch_size)
    outputs = np.empty((batch_size, num_steps))
    kappa_history = np.empty((batch_size, num_steps, rank))
    filtered_history = np.empty((batch_size, num_steps))

    for time in range(num_steps):
        kappa_history[:, time] = kappa
        filtered_history[:, time] = filtered_input
        coefficients = np.column_stack((filtered_input, kappa))
        mean_rate, mean_gain = _rate_moments(
            coefficients,
            mean,
            covariance,
            basis_indices,
        )
        kappa += step_size * gaussian_latent_flow(
            kappa,
            filtered_input,
            names,
            mean,
            covariance,
        )

        readout_covariance = (
            covariance[readout_index, basis_indices] * coefficients
        ).sum(axis=1)
        outputs[:, time] = (
            mean[readout_index] * mean_rate
            + readout_covariance * mean_gain
        )
        filtered_input += step_size * (-filtered_input + inputs[:, time])

    return outputs, kappa_history, filtered_history


def rank_one_fixed_points(
    names: Sequence[str],
    mean: npt.ArrayLike,
    covariance: npt.ArrayLike,
    *,
    bounds: tuple[float, float] = (-3.0, 3.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find fixed points of a fitted rank-one Gaussian circuit."""

    def flow(kappa: npt.ArrayLike) -> np.ndarray:
        values = np.atleast_1d(np.asarray(kappa, dtype=float))
        return gaussian_latent_flow(
            values[:, None],
            np.zeros_like(values),
            names,
            mean,
            covariance,
        )[:, 0]

    return find_fixed_points_1d(flow, bounds=bounds)


def _standard_gaussian_gain(delta: npt.ArrayLike) -> np.ndarray:
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    rates = np.tanh(delta[:, None] * _NORMAL_NODES)
    return (1 - rates**2) @ _NORMAL_WEIGHTS


def simulate_diagonal_circuit(
    inputs: npt.ArrayLike,
    *,
    recurrent_gain: npt.ArrayLike,
    input_gain: npt.ArrayLike,
    readout_gain: npt.ArrayLike,
    step_size: float,
) -> np.ndarray:
    """Simulate a zero-mean Gaussian circuit with independent unit modes."""
    inputs = np.asarray(inputs, dtype=float)
    recurrent_gain = np.asarray(recurrent_gain, dtype=float)
    input_gain = np.asarray(input_gain, dtype=float)
    readout_gain = np.asarray(readout_gain, dtype=float)
    batch_size, num_steps = inputs.shape
    kappa = np.zeros((batch_size, len(recurrent_gain)))
    filtered_input = np.zeros(batch_size)
    outputs = np.empty((batch_size, num_steps))

    for time in range(num_steps):
        delta = np.sqrt(np.sum(kappa**2, axis=1) + filtered_input**2)
        gain = _standard_gaussian_gain(delta)
        outputs[:, time] = gain * (kappa @ readout_gain)
        recurrent_drive = gain[:, None] * (
            kappa * recurrent_gain
            + filtered_input[:, None] * input_gain
        )
        kappa += step_size * (-kappa + recurrent_drive)
        filtered_input += step_size * (-filtered_input + inputs[:, time])
    return outputs
