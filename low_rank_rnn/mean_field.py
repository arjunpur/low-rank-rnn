"""Gaussian mean-field circuits for low-rank RNNs."""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Float, Integer, Real
from numpy.polynomial.hermite import hermgauss

from low_rank_rnn._typing import typechecked
from low_rank_rnn.analysis import find_fixed_points_1d


_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = hermgauss(40)
_NORMAL_NODES = np.sqrt(2) * _QUADRATURE_NODES
_NORMAL_WEIGHTS = _QUADRATURE_WEIGHTS / np.sqrt(np.pi)

PAPER_WORKING_MEMORY_NAMES = ("I", "n_1", "n_2", "m_1", "m_2", "w")


def _loading_indices(
    names: Sequence[str],
) -> tuple[
    Integer[np.ndarray, "rank"],
    Integer[np.ndarray, "rank"],
    Integer[np.ndarray, "basis"],
]:
    rank = max((len(names) - 2) // 2, 1)
    n_indices = np.arange(1, 1 + rank)
    m_indices = np.arange(1 + rank, 1 + 2 * rank)
    basis_indices = np.array((0, *m_indices))
    return n_indices, m_indices, basis_indices


def _rate_moments(
    coefficients: Real[np.ndarray, "batch basis"],
    mean: Real[np.ndarray, "basis"],
    covariance: Real[np.ndarray, "basis basis"],
) -> tuple[Float[np.ndarray, "batch"], Float[np.ndarray, "batch"]]:
    current_mean = coefficients @ mean
    current_variance = ((coefficients @ covariance) * coefficients).sum(axis=1)
    currents = current_mean[:, None] + np.sqrt(
        np.maximum(current_variance, 0)
    )[:, None] * _NORMAL_NODES
    rates = np.tanh(currents)
    return (
        rates @ _NORMAL_WEIGHTS,
        (1 - rates**2) @ _NORMAL_WEIGHTS,
    )


@typechecked
def gaussian_latent_flow(
    kappa: Real[np.ndarray, "batch rank"],
    filtered_input: Real[np.ndarray, "batch"] | int | float,
    names: Sequence[str],
    mean: Real[np.ndarray, "coordinate"],
    covariance: Real[np.ndarray, "coordinate coordinate"],
) -> Float[np.ndarray, "batch rank"]:
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
        mean[basis_indices],
        covariance[np.ix_(basis_indices, basis_indices)],
    )
    covariance_drive = (
        coefficients @ covariance[np.ix_(n_indices, basis_indices)].T
    )
    return (
        -kappa
        + mean_rate[:, None] * mean[n_indices]
        + mean_gain[:, None] * covariance_drive
    )


@typechecked
def simulate_gaussian_circuit(
    inputs: Real[np.ndarray, "batch time"],
    names: Sequence[str],
    mean: Real[np.ndarray, "coordinate"],
    covariance: Real[np.ndarray, "coordinate coordinate"],
    *,
    step_size: float,
) -> tuple[
    Float[np.ndarray, "batch time"],
    Float[np.ndarray, "batch time rank"],
    Float[np.ndarray, "batch time"],
]:
    """Simulate the Gaussian equivalent circuit for any low rank."""
    inputs = np.asarray(inputs, dtype=float)
    mean = np.asarray(mean)
    covariance = np.asarray(covariance)
    n_indices, _, basis_indices = _loading_indices(names)
    rank = len(n_indices)
    basis_mean = mean[basis_indices]
    basis_covariance = covariance[np.ix_(basis_indices, basis_indices)]
    recurrent_mean = mean[n_indices]
    recurrent_covariance = covariance[np.ix_(n_indices, basis_indices)]
    readout_mean = mean[-1]
    readout_covariance = covariance[-1, basis_indices]

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
            basis_mean,
            basis_covariance,
        )
        covariance_drive = coefficients @ recurrent_covariance.T
        recurrent_drive = (
            mean_rate[:, None] * recurrent_mean
            + mean_gain[:, None] * covariance_drive
        )
        outputs[:, time] = (
            readout_mean * mean_rate
            + mean_gain * (coefficients @ readout_covariance)
        )
        kappa += step_size * (-kappa + recurrent_drive)
        filtered_input += step_size * (-filtered_input + inputs[:, time])

    return outputs, kappa_history, filtered_history


def paper_working_memory_gaussian(
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return the loading Gaussian for the paper's rank-two memory circuit."""
    readout_residual = np.sqrt(16 - 2.8**2 - 2.2**2)
    loading_transform = np.array(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.5, 1.0, 0.0, 0.0),
            (1.9, 0.0, 0.5, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 2.8, -2.2, readout_residual),
        )
    )
    mean = np.zeros(len(PAPER_WORKING_MEMORY_NAMES))
    covariance = loading_transform @ loading_transform.T
    return PAPER_WORKING_MEMORY_NAMES, mean, covariance


@typechecked
def rank_one_fixed_points(
    names: Sequence[str],
    mean: Real[np.ndarray, "coordinate"],
    covariance: Real[np.ndarray, "coordinate coordinate"],
    *,
    bounds: tuple[int | float, int | float] = (-3.0, 3.0),
) -> tuple[
    Float[np.ndarray, "grid"],
    Float[np.ndarray, "grid"],
    Float[np.ndarray, "fixed_point"],
    Float[np.ndarray, "fixed_point"],
]:
    """Find fixed points of a fitted rank-one Gaussian circuit."""

    def flow(
        kappa: Real[np.ndarray, "point"],
    ) -> Float[np.ndarray, "point"]:
        values = np.atleast_1d(np.asarray(kappa, dtype=float))
        return gaussian_latent_flow(
            values[:, None],
            np.zeros_like(values),
            names,
            mean,
            covariance,
        )[:, 0]

    return find_fixed_points_1d(flow, bounds=bounds)
