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


@typechecked
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


@typechecked
def _rate_moments(
    coefficients: Real[np.ndarray, "batch basis"],
    mean: Real[np.ndarray, "coordinate"],
    covariance: Real[np.ndarray, "coordinate coordinate"],
    basis_indices: Integer[np.ndarray, "basis"],
) -> tuple[Float[np.ndarray, "batch"], Float[np.ndarray, "batch"]]:
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

    @typechecked
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


@typechecked
def _standard_gaussian_gain(
    delta: Real[np.ndarray, "batch"],
) -> Float[np.ndarray, "batch"]:
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    rates = np.tanh(delta[:, None] * _NORMAL_NODES)
    return (1 - rates**2) @ _NORMAL_WEIGHTS


@typechecked
def simulate_diagonal_circuit(
    inputs: Real[np.ndarray, "batch time"],
    names: Sequence[str],
    covariance: Real[np.ndarray, "coordinate coordinate"],
    *,
    step_size: float,
) -> tuple[
    Float[np.ndarray, "batch time"],
    Float[np.ndarray, "batch time rank"],
    Float[np.ndarray, "batch time"],
]:
    """Simulate the handout's zero-mean diagonal rank-two circuit.

    The approximation keeps only ``Cov(n_r, m_r)``, ``Cov(n_r, I)``, and
    ``Cov(w, m_r)``. Its nonlinear gain uses the diagonal variance
    ``Var(I) v² + Σ_r Var(m_r) κ_r²``; means and every omitted covariance
    are deliberately ignored.
    """
    inputs = np.asarray(inputs, dtype=float)
    names = tuple(names)
    covariance = np.asarray(covariance, dtype=float)
    n_indices, m_indices, _ = _loading_indices(names)
    if len(n_indices) != 2:
        raise ValueError("the diagonal handout circuit requires rank two")

    input_index = 0
    readout_index = len(names) - 1
    recurrent_covariance = covariance[n_indices, m_indices]
    input_covariance = covariance[n_indices, input_index]
    readout_covariance = covariance[readout_index, m_indices]
    mode_variance = covariance[m_indices, m_indices]
    input_variance = covariance[input_index, input_index]

    batch_size, num_steps = inputs.shape
    kappa = np.zeros((batch_size, len(n_indices)))
    filtered_input = np.zeros(batch_size)
    outputs = np.empty((batch_size, num_steps))
    kappa_history = np.empty((batch_size, num_steps, len(n_indices)))
    filtered_history = np.empty((batch_size, num_steps))

    for time in range(num_steps):
        kappa_history[:, time] = kappa
        filtered_history[:, time] = filtered_input
        delta_squared = (
            (kappa**2 * mode_variance).sum(axis=1)
            + filtered_input**2 * input_variance
        )
        delta = np.sqrt(np.maximum(delta_squared, 0))
        gain = _standard_gaussian_gain(delta)
        outputs[:, time] = gain * (kappa @ readout_covariance)
        recurrent_drive = gain[:, None] * (
            kappa * recurrent_covariance
            + filtered_input[:, None] * input_covariance
        )
        kappa += step_size * (-kappa + recurrent_drive)
        filtered_input += step_size * (-filtered_input + inputs[:, time])
    return outputs, kappa_history, filtered_history
