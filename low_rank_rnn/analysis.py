"""Analysis helpers for trained low-rank RNNs."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize_scalar
import torch
from jaxtyping import Complex, Float, Real

from low_rank_rnn._typing import typechecked
from low_rank_rnn.model import LowRankRNN


@torch.no_grad()
@typechecked
def run_model(
    model: LowRankRNN,
    inputs: Real[np.ndarray, "batch time"] | Real[torch.Tensor, "batch time"],
) -> tuple[
    Float[np.ndarray, "batch time"],
    Float[np.ndarray, "batch time unit"],
]:
    """Run a model in evaluation mode and return NumPy outputs and states."""
    was_training = model.training
    model.eval()
    tensor = torch.as_tensor(inputs, dtype=torch.float32)
    outputs, states = model(tensor)
    model.train(was_training)
    return outputs.cpu().numpy(), states.cpu().numpy()


@typechecked
def decision_values(
    outputs: Real[np.ndarray, "batch time"],
    decision_steps: int,
) -> Float[np.ndarray, "batch"]:
    """Average scalar readouts over the final decision window."""
    return np.asarray(outputs)[:, -decision_steps:].mean(axis=1)


@typechecked
def regression_metrics(
    predictions: Real[np.ndarray, "sample"],
    targets: Real[np.ndarray, "sample"],
) -> tuple[float, float]:
    """Return mean squared error and coefficient of determination."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    residuals = predictions - targets
    mse = float(np.mean(residuals**2))
    total_variation = np.sum((targets - targets.mean()) ** 2)
    r_squared = 1 - float(np.sum(residuals**2) / total_variation)
    return mse, r_squared


@typechecked
def network_regression_mses(
    models: Sequence[LowRankRNN],
    inputs: Real[np.ndarray, "batch time"],
    targets: Real[np.ndarray, "batch"],
    *,
    decision_steps: int,
) -> Float[np.ndarray, "model"]:
    """Score several models on one regression task."""
    scores = []
    for model in models:
        outputs, _ = run_model(model, inputs)
        decisions = decision_values(outputs, decision_steps)
        scores.append(regression_metrics(decisions, targets)[0])
    return np.asarray(scores)


@typechecked
def principal_component_analysis(
    states: Real[np.ndarray, "batch time unit"],
) -> tuple[
    Float[np.ndarray, "unit"],
    Float[np.ndarray, "unit unit"],
    Float[np.ndarray, "batch time unit"],
]:
    """Return PCA variances, neuron-space axes, and projected trajectories."""
    states = np.asarray(states)
    activity = states.reshape(-1, states.shape[-1])
    mean_activity = activity.mean(axis=0, keepdims=True)
    centered = activity - mean_activity
    covariance = centered.T @ centered / (len(centered) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    components = eigenvectors[:, order]
    projected = (centered @ components).reshape(states.shape)
    return eigenvalues / eigenvalues.sum(), components, projected


@typechecked
def explained_variance(
    states: Real[np.ndarray, "batch time unit"],
) -> Float[np.ndarray, "unit"]:
    """Return covariance-PCA explained-variance fractions."""
    variance, _, _ = principal_component_analysis(states)
    return variance


@typechecked
def connectivity_vectors(
    model: LowRankRNN,
    *,
    m: Real[np.ndarray, "unit rank"] | None = None,
    n: Real[np.ndarray, "unit rank"] | None = None,
) -> dict[str, Float[np.ndarray, "unit"]]:
    """Return neuron coordinates, optionally in a supplied recurrent basis."""
    m_values = (
        model.m.detach().cpu().numpy()
        if m is None
        else np.asarray(m)
    )
    n_values = (
        model.n.detach().cpu().numpy()
        if n is None
        else np.asarray(n)
    )
    rank = m_values.shape[1]
    suffixes = [""] if rank == 1 else [f"_{index + 1}" for index in range(rank)]
    vectors = {"I": model.I.detach().cpu().numpy()}
    for name, values in (("n", n_values), ("m", m_values)):
        vectors.update(
            {
                f"{name}{suffix}": values[:, index]
                for index, suffix in enumerate(suffixes)
            }
        )
    vectors["w"] = model.w.detach().cpu().numpy()
    return vectors


@typechecked
def fit_loading_gaussian(
    vectors: Mapping[str, Real[np.ndarray, "unit"]],
) -> tuple[
    Float[np.ndarray, "coordinate"],
    Float[np.ndarray, "coordinate coordinate"],
]:
    """Fit a Gaussian to neuron coordinates in loading space."""
    samples = np.column_stack(tuple(vectors.values()))
    return samples.mean(axis=0), np.cov(samples, rowvar=False, bias=True)


@typechecked
def _sample_loading_gaussian(
    mean: Real[np.ndarray, "coordinate"],
    covariance: Real[np.ndarray, "coordinate coordinate"],
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> Float[np.ndarray, "sample coordinate"]:
    return rng.multivariate_normal(
        np.asarray(mean),
        np.asarray(covariance),
        size=num_samples,
    )


@typechecked
def sample_loading_vectors(
    names: Sequence[str],
    mean: Real[np.ndarray, "coordinate"],
    covariance: Real[np.ndarray, "coordinate coordinate"],
    *,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[
    dict[str, Float[np.ndarray, "sample"]],
    Float[np.ndarray, "coordinate coordinate"],
]:
    """Sample named loading vectors and return their fitted covariance."""
    samples = _sample_loading_gaussian(
        mean,
        covariance,
        num_samples=num_samples,
        rng=rng,
    )
    vectors = {
        name: samples[:, index]
        for index, name in enumerate(names)
    }
    _, sample_covariance = fit_loading_gaussian(vectors)
    return vectors, sample_covariance


def _loading_names(rank: int) -> tuple[str, ...]:
    suffixes = [""] if rank == 1 else [f"_{index + 1}" for index in range(rank)]
    return (
        "I",
        *(f"n{suffix}" for suffix in suffixes),
        *(f"m{suffix}" for suffix in suffixes),
        "w",
    )


@typechecked
def sample_low_rank_rnns(
    names: Sequence[str],
    mean: Real[np.ndarray, "coordinate"],
    covariance: Real[np.ndarray, "coordinate coordinate"],
    *,
    num_networks: int,
    num_neurons: int,
    rng: np.random.Generator,
) -> list[LowRankRNN]:
    """Create RNNs from samples of a fitted loading Gaussian."""
    names = tuple(names)
    rank = max((len(names) - 2) // 2, 1)
    expected_names = _loading_names(rank)
    if names != expected_names:
        raise ValueError(f"names must be exactly {expected_names}")

    suffixes = [""] if rank == 1 else [f"_{index + 1}" for index in range(rank)]
    networks = []
    for _ in range(num_networks):
        samples = _sample_loading_gaussian(
            mean,
            covariance,
            num_samples=num_neurons,
            rng=rng,
        )
        columns = dict(zip(names, samples.T, strict=True))
        network = LowRankRNN(n_units=num_neurons, rank=rank)
        with torch.no_grad():
            for name, loadings in (
                ("I", [columns["I"]]),
                ("n", [columns[f"n{suffix}"] for suffix in suffixes]),
                ("m", [columns[f"m{suffix}"] for suffix in suffixes]),
                ("w", [columns["w"]]),
            ):
                target = getattr(network, name)
                target.copy_(
                    torch.as_tensor(
                        np.column_stack(loadings),
                        dtype=target.dtype,
                    ).reshape_as(target)
                )
        networks.append(network)
    return networks


@typechecked
def named_covariances(
    names: Sequence[str],
    covariance: Real[np.ndarray, "coordinate coordinate"],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, float]:
    """Select covariance entries by loading name."""
    index = {name: position for position, name in enumerate(names)}
    covariance = np.asarray(covariance)
    return {
        f"Cov({first}, {second})": float(
            covariance[index[first], index[second]]
        )
        for first, second in pairs
    }


@typechecked
def connectivity_overlap(
    m: Real[np.ndarray, "unit rank"],
    n: Real[np.ndarray, "unit rank"],
) -> Float[np.ndarray, "rank rank"]:
    """Return the recurrent loop gain ``n.T @ m / N``."""
    m, n = np.asarray(m, dtype=float), np.asarray(n, dtype=float)
    return n.T @ m / len(m)


@typechecked
def _svd_connectivity_basis(
    model: LowRankRNN,
) -> tuple[Float[np.ndarray, "unit rank"], Float[np.ndarray, "unit rank"]]:
    """Return equivalent connectivity factors in the canonical SVD basis."""
    m = model.m.detach().cpu().numpy().astype(float)
    n = model.n.detach().cpu().numpy().astype(float)
    m_basis, m_factor = np.linalg.qr(m)
    n_basis, n_factor = np.linalg.qr(n)
    left, values, right = np.linalg.svd(m_factor @ n_factor.T)
    scale = np.sqrt(values)
    return m_basis @ left * scale, n_basis @ right.T * scale


def svd_canonical_model(model: LowRankRNN) -> tuple[LowRankRNN, float]:
    """Copy a model into its SVD connectivity basis."""
    canonical_m, canonical_n = _svd_connectivity_basis(model)
    canonical = deepcopy(model)
    with torch.no_grad():
        canonical.m.copy_(torch.as_tensor(canonical_m, dtype=canonical.m.dtype))
        canonical.n.copy_(torch.as_tensor(canonical_n, dtype=canonical.n.dtype))
    canonical.eval()

    original = (
        model.m.detach().cpu().numpy()
        @ model.n.detach().cpu().numpy().T
    )
    reconstructed = canonical_m @ canonical_n.T
    error = float(
        np.linalg.norm(reconstructed - original) / np.linalg.norm(original)
    )
    return canonical, error


@typechecked
def _project_states(
    states: Real[np.ndarray, "... unit"],
    basis: Real[np.ndarray, "unit latent"],
) -> Float[np.ndarray, "... latent"]:
    """Recover state coordinates in a supplied, possibly nonorthogonal basis."""
    return np.asarray(states) @ np.linalg.pinv(np.asarray(basis)).T


@typechecked
def simulate_and_project(
    model: LowRankRNN,
    inputs: Real[np.ndarray, "batch time"] | Real[torch.Tensor, "batch time"],
    basis: Real[np.ndarray, "unit latent"],
) -> tuple[
    Float[np.ndarray, "batch time"],
    Float[np.ndarray, "batch time unit"],
    Float[np.ndarray, "batch time latent"],
]:
    """Run a model and project its states into a supplied basis."""
    outputs, states = run_model(model, inputs)
    return outputs, states, _project_states(states, basis)


@torch.no_grad()
@typechecked
def latent_jacobian_eigenvalues(
    model: LowRankRNN,
    states: Real[np.ndarray, "state unit"] | Real[torch.Tensor, "state unit"],
    *,
    m: (
        Real[np.ndarray, "unit rank"]
        | Real[torch.Tensor, "unit rank"]
        | None
    ) = None,
    n: (
        Real[np.ndarray, "unit rank"]
        | Real[torch.Tensor, "unit rank"]
        | None
    ) = None,
) -> Complex[np.ndarray, "state rank"]:
    """Linearize the recurrent latent flow at each supplied network state."""
    state_tensor = torch.as_tensor(states, dtype=model.m.dtype)
    mode_m = (
        model.m.detach()
        if m is None
        else torch.as_tensor(m, dtype=model.m.dtype)
    )
    mode_n = (
        model.n.detach()
        if n is None
        else torch.as_tensor(n, dtype=model.n.dtype)
    )
    identity = torch.eye(mode_m.shape[1], dtype=mode_m.dtype)
    eigenvalues = []
    for state in state_tensor:
        gain = 1 - torch.tanh(state).square()
        jacobian = mode_n.T @ (gain[:, None] * mode_m) / model.n_units - identity
        values = torch.linalg.eigvals(jacobian)
        eigenvalues.append(values[torch.argsort(-values.real)])
    return torch.stack(eigenvalues).cpu().numpy()


@typechecked
def find_fixed_points_1d(
    flow: Callable[
        [Real[np.ndarray, "point"]],
        Real[np.ndarray, "point"],
    ],
    *,
    bounds: tuple[int | float, int | float],
    grid_size: int = 2_401,
) -> tuple[
    Float[np.ndarray, "grid"],
    Float[np.ndarray, "grid"],
    Float[np.ndarray, "fixed_point"],
    Float[np.ndarray, "fixed_point"],
]:
    """Find and classify the roots of a one-dimensional flow."""
    grid = np.linspace(*bounds, grid_size)
    flow_values = np.asarray(flow(grid))
    energy = 0.5 * flow_values**2
    minima = (
        np.flatnonzero(
            (energy[1:-1] <= energy[:-2])
            & (energy[1:-1] <= energy[2:])
        )
        + 1
    )

    fixed_points = []
    for index in minima:
        result = minimize_scalar(
            lambda value: 0.5
            * float(np.asarray(flow(np.asarray([value])))[0]) ** 2,
            bounds=(grid[index - 1], grid[index + 1]),
            method="bounded",
        )
        if result.fun < 1e-10 and not any(
            abs(result.x - existing) < 1e-3
            for existing in fixed_points
        ):
            fixed_points.append(float(result.x))

    fixed_points = np.asarray(sorted(fixed_points))
    step = 1e-4
    slopes = np.asarray(
        [
            (
                np.asarray(flow(np.asarray([point + step])))[0]
                - np.asarray(flow(np.asarray([point - step])))[0]
            )
            / (2 * step)
            for point in fixed_points
        ]
    )
    return grid, flow_values, fixed_points, slopes


@typechecked
def project_rank_one_activity(
    states: Float[torch.Tensor, "batch time unit"],
    model: LowRankRNN,
) -> Float[np.ndarray, "batch time 2"]:
    """Project rank-one activity onto orthonormal ``m`` and ``I`` axes."""
    if model.m.shape[1] != 1:
        raise ValueError("model must have rank 1")

    m_axis = model.m[:, 0].detach()
    m_axis = m_axis / torch.linalg.vector_norm(m_axis)
    input_axis = model.I.detach() - torch.dot(model.I, m_axis) * m_axis
    input_axis = input_axis / torch.linalg.vector_norm(input_axis)
    basis = torch.stack((m_axis, input_axis), dim=1)
    return (states.detach() @ basis).cpu().numpy()
