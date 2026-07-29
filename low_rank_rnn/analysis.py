"""Numerical analysis of trained low-rank RNNs."""

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Complex, Float

from low_rank_rnn._typing import typechecked
from low_rank_rnn.data.variable_delay import (
    DELAYS,
    FIXATION_STEPS,
    FREQUENCIES,
    MAX_FREQUENCY,
    MIN_FREQUENCY,
    STIMULUS_STEPS,
    frequency_pair_grid,
    make_trials,
)
from low_rank_rnn.model import LowRankRNN
from low_rank_rnn.training import masked_decision_loss


@typechecked
def connectivity_vectors(
    model: LowRankRNN,
) -> dict[str, Float[np.ndarray, "unit"]]:
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


@typechecked
def connectivity_covariance(
    vectors: Mapping[str, npt.ArrayLike],
) -> tuple[tuple[str, ...], Float[np.ndarray, "vector vector"]]:
    """Return names and sample covariance of connectivity vectors."""
    names = tuple(vectors)
    matrix = np.stack([vectors[name] for name in names])
    return names, np.cov(matrix)


@typechecked
def svd_connectivity_basis(
    model: LowRankRNN,
) -> tuple[Float[np.ndarray, "unit rank"], Float[np.ndarray, "unit rank"]]:
    """Return ``m`` and ``n`` in the canonical basis of Dubreuil et al.

    Replacing ``m`` by ``mA`` and ``n`` by ``nA^-T`` leaves the connectivity, and
    therefore the network, completely unchanged while altering every covariance
    between the two sets. Covariance structure is only comparable once that
    freedom is fixed. The paper fixes it by requiring the output patterns to be
    mutually orthogonal, and likewise the input-selection patterns, which the
    singular-value decomposition of the connectivity determines uniquely.
    """
    m = model.m.detach().cpu().numpy().astype(float)
    n = model.n.detach().cpu().numpy().astype(float)
    m_basis, m_factor = np.linalg.qr(m)
    n_basis, n_factor = np.linalg.qr(n)
    left, values, right = np.linalg.svd(m_factor @ n_factor.T)
    scale = np.sqrt(values)
    return m_basis @ left * scale, n_basis @ right.T * scale


@typechecked
def connectivity_overlap(
    m: npt.ArrayLike,
    n: npt.ArrayLike,
) -> Float[np.ndarray, "rank rank"]:
    """Return C = n^T m / N, the gain of the recurrent loop between modes.

    Entry ``(a, b)`` is how much of mode ``b``, once written into the activity by
    ``m_b``, comes back as mode ``a`` when read by ``n_a``. Its eigenvalues set
    the timescales of the latent dynamics: one at 1 is a line attractor.
    """
    m, n = np.asarray(m, dtype=float), np.asarray(n, dtype=float)
    return n.T @ m / len(m)


@typechecked
def connectivity_non_normality(m: npt.ArrayLike, n: npt.ArrayLike) -> float:
    """Return ||JJ^T - J^TJ|| / ||J||^2 for the low-rank connectivity J.

    Zero when each ``n_r`` pairs with its own ``m_r``, as in the paper's reduced
    model. Large when the modes are chained, one mode's output being read by
    another mode's selection vector. Unlike the individual covariances this does
    not depend on the choice of basis.
    """
    m, n = np.asarray(m, dtype=float), np.asarray(n, dtype=float)
    connectivity = m @ n.T / len(m)
    commutator = connectivity @ connectivity.T - connectivity.T @ connectivity
    return float(
        np.linalg.norm(commutator) / np.linalg.norm(connectivity) ** 2
    )


@typechecked
def fit_loading_gaussian(
    vectors: Mapping[str, npt.ArrayLike],
) -> tuple[
    Float[np.ndarray, "coordinate"],
    Float[np.ndarray, "coordinate coordinate"],
]:
    """Fit a Gaussian to neuron coordinates in loading space."""
    samples = np.column_stack(tuple(vectors.values()))
    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False, bias=True)
    return mean, covariance


@typechecked
def sample_loading_gaussian(
    mean: npt.ArrayLike,
    covariance: npt.ArrayLike,
    *,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> Float[np.ndarray, "sample coordinate"]:
    """Draw loading vectors from a fitted Gaussian."""
    generator = rng if rng is not None else np.random.default_rng()
    return generator.multivariate_normal(
        np.asarray(mean),
        np.asarray(covariance),
        size=num_samples,
    )


def sample_low_rank_rnns(
    names: Sequence[str],
    mean: npt.ArrayLike,
    covariance: npt.ArrayLike,
    *,
    num_networks: int,
    num_neurons: int,
    rng: np.random.Generator | None = None,
) -> list[LowRankRNN]:
    """Create RNNs from samples of a fitted loading Gaussian.

    ``names`` is the loading order :func:`connectivity_vectors` produces, which
    also fixes the rank: ``("I", "n", "m", "w")`` for rank one, and
    ``("I", "n_1", "n_2", "m_1", "m_2", "w")`` for rank two.
    """
    names = tuple(names)
    rank = max((len(names) - 2) // 2, 1)
    suffixes = [""] if rank == 1 else [f"_{index + 1}" for index in range(rank)]
    expected_names = (
        "I",
        *(f"n{suffix}" for suffix in suffixes),
        *(f"m{suffix}" for suffix in suffixes),
        "w",
    )
    if names != expected_names:
        raise ValueError(f"names must be exactly {expected_names}")

    generator = rng if rng is not None else np.random.default_rng()
    networks = []
    for _ in range(num_networks):
        network_loadings = sample_loading_gaussian(
            mean,
            covariance,
            num_samples=num_neurons,
            rng=generator,
        )
        columns = dict(zip(names, network_loadings.T, strict=True))
        network = LowRankRNN(n_units=num_neurons, rank=rank)
        with torch.no_grad():
            # I and w are single vectors; n and m take one column per rank index.
            for name, loadings in (
                ("I", [columns["I"]]),
                ("n", [columns[f"n{suffix}"] for suffix in suffixes]),
                ("m", [columns[f"m{suffix}"] for suffix in suffixes]),
                ("w", [columns["w"]]),
            ):
                target = getattr(network, name)
                target.copy_(
                    torch.as_tensor(
                        np.column_stack(loadings), dtype=target.dtype
                    ).reshape_as(target)
                )
        networks.append(network)
    return networks


@torch.no_grad()
@typechecked
def delay_eigenvalues(
    model: LowRankRNN,
    coordinates: Float[torch.Tensor, "state rank"],
) -> Complex[np.ndarray, "state rank"]:
    """Linearize the autonomous latent flow at each memory state.

    Eigenvalues are in units of ``1 / tau`` and sorted by decreasing real part,
    so the leading mode comes first: a line attractor holds it near zero, while
    a nonzero imaginary part means the memory rotates.
    """
    eigenvalues = []
    identity = torch.eye(model.m.shape[1])
    for coordinate in coordinates:
        gain = 1 - torch.tanh(model.m @ coordinate).square()
        jacobian = model.n.T @ (gain[:, None] * model.m) / model.n_units - identity
        values = torch.linalg.eigvals(jacobian)
        eigenvalues.append(values[torch.argsort(-values.real)])
    return torch.stack(eigenvalues).cpu().numpy()


@torch.no_grad()
@typechecked
def delay_diagnostics(
    model: LowRankRNN,
    *,
    probe_delays: npt.ArrayLike = DELAYS[::5],
) -> dict[str, np.ndarray]:
    """Summarize how a trained network holds its memory across the delay.

    Sweeps ``f_1`` with a neutral ``f_2`` to read the memory states off the
    delay period, then reports their trajectory, the task error over a grid of
    frequency pairs at each probe delay, how far the states drift relative to
    the span of the memory manifold, and the local spectrum.
    """
    probe_delays = np.asarray(probe_delays)
    neutral_frequency = (MIN_FREQUENCY + MAX_FREQUENCY) / 2
    sweep_frequencies = np.column_stack(
        (FREQUENCIES, np.full_like(FREQUENCIES, neutral_frequency))
    )
    sweep_inputs, _, _ = make_trials(
        sweep_frequencies,
        np.full(len(FREQUENCIES), DELAYS.max()),
    )

    _, states = model(sweep_inputs)
    coordinates = states @ torch.linalg.pinv(model.m).T
    delay_start = FIXATION_STEPS + STIMULUS_STEPS
    memory_coordinates = coordinates[:, delay_start : delay_start + DELAYS.max() + 1]

    pairs = frequency_pair_grid()
    probe_mse = []
    for delay in probe_delays:
        inputs, targets, decision_mask = make_trials(pairs, np.full(len(pairs), delay))
        outputs, _ = model(inputs)
        probe_mse.append(masked_decision_loss(outputs, targets, decision_mask).item())

    reference = DELAYS.min()
    manifold_span = torch.linalg.vector_norm(
        memory_coordinates[-1, reference] - memory_coordinates[0, reference]
    )
    drift = torch.linalg.vector_norm(
        memory_coordinates[:, probe_delays] - memory_coordinates[:, reference, None],
        dim=-1,
    ).mean(dim=0) / manifold_span

    # How much of the initial f_1 separation survives, in units of its value at
    # delay onset. Latent coordinates carry no absolute scale, so this is the
    # comparable measure of whether a network still holds the stimulus.
    probed = memory_coordinates[:, probe_delays]
    onset_span = torch.linalg.vector_norm(
        memory_coordinates[-1, 0] - memory_coordinates[0, 0]
    )
    retained_span = (
        torch.linalg.vector_norm(probed[-1] - probed[0], dim=-1) / onset_span
    )

    return {
        "memory_coordinates": memory_coordinates.cpu().numpy(),
        "probe_mse": np.asarray(probe_mse),
        "fractional_drift": drift.cpu().numpy(),
        "retained_span": retained_span.cpu().numpy(),
        "eigenvalues": delay_eigenvalues(model, memory_coordinates[:, DELAYS.max()]),
        "overlap": (model.n.T @ model.m / model.n_units).cpu().numpy(),
    }


@typechecked
def project_rank_one_activity(
    states: Float[torch.Tensor, "batch time unit"],
    model: LowRankRNN,
) -> Float[np.ndarray, "batch time 2"]:
    """Project rank-one activity onto orthonormal ``m`` and ``I`` axes."""
    if model.m.shape[1] != 1:
        raise ValueError("model must have rank 1")

    m_vector = model.m[:, 0].detach()
    input_vector = model.I.detach()

    m_norm = torch.linalg.vector_norm(m_vector)
    if torch.isclose(m_norm, torch.zeros_like(m_norm)):
        raise ValueError("m must be nonzero")
    m_axis = m_vector / m_norm

    orthogonal_input = input_vector - torch.dot(input_vector, m_axis) * m_axis
    input_norm = torch.linalg.vector_norm(orthogonal_input)
    if torch.isclose(input_norm, torch.zeros_like(input_norm)):
        raise ValueError("I and m must span a plane")

    input_axis = orthogonal_input / input_norm
    basis = torch.stack((m_axis, input_axis), dim=1)
    return (states.detach() @ basis).cpu().numpy()
