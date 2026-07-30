"""Compare the exact zero-input fixed points of the two rank-two networks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import root

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from low_rank_rnn.analysis import connectivity_overlap, svd_canonical_model
from low_rank_rnn.data.working_memory import FREQUENCIES, stimulus_amplitudes
from low_rank_rnn.model import LowRankRNN
from low_rank_rnn.plotting import set_plot_style
from low_rank_rnn.plotting.style import COLORS
from low_rank_rnn.training import train_model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "output" / "fixed_point_analysis"
OSCILLATORY_CHECKPOINT = OUTPUT_DIRECTORY / "fixed_delay_oscillatory_seed2026.pt"
STRUCTURED_CHECKPOINT = (
    ROOT / "output" / "mode_experiments" / "paper_init_dense_seed2032.pt"
)
FIGURE_PATH = OUTPUT_DIRECTORY / "rank2_fixed_point_comparison.png"
RESULTS_PATH = OUTPUT_DIRECTORY / "rank2_fixed_point_results.json"

NUM_UNITS = 128
STEP_SIZE = 0.2
BASE_SEED = 2026


def make_fixed_delay_trials(
    frequency_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact fixed-delay trials used in the complete notebook."""
    amplitudes = stimulus_amplitudes(frequency_pairs)
    inputs = np.zeros((len(frequency_pairs), 80), dtype=np.float32)
    inputs[:, 5:11] = amplitudes[:, 0, None]
    inputs[:, 60:71] = amplitudes[:, 1, None]
    targets = ((frequency_pairs[:, 0] - frequency_pairs[:, 1]) / 24).astype(
        np.float32
    )
    return inputs, targets


def load_model(path: Path) -> LowRankRNN:
    model = LowRankRNN(NUM_UNITS, rank=2)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_or_train_oscillatory_model() -> LowRankRNN:
    """Reproduce the 600-epoch oscillator if its checkpoint is absent."""
    if OSCILLATORY_CHECKPOINT.exists():
        return load_model(OSCILLATORY_CHECKPOINT)

    training_pairs = np.random.default_rng(BASE_SEED).choice(
        FREQUENCIES,
        size=(200, 2),
    )
    inputs, targets = make_fixed_delay_trials(training_pairs)

    torch.manual_seed(BASE_SEED)
    model = LowRankRNN(NUM_UNITS, rank=2)
    train_model(
        model,
        torch.as_tensor(inputs),
        torch.as_tensor(targets),
        epochs=600,
        batch_size=32,
        learning_rate=5e-3,
        decision_steps=5,
        log_every=None,
    )
    model.eval()
    torch.save(model.state_dict(), OSCILLATORY_CHECKPOINT)
    return model


def unit_rms_modes(
    model: LowRankRNN,
    *,
    separate_real_modes: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose readable factors while preserving the recurrent connectivity."""
    if separate_real_modes:
        raw_m = model.m.detach().numpy().astype(float)
        raw_n = model.n.detach().numpy().astype(float)
        values, vectors = np.linalg.eig(connectivity_overlap(raw_m, raw_n))
        order = np.argsort(-values.real)
        if np.max(np.abs(values[order].imag)) > 1e-8:
            raise ValueError("separate_real_modes requires real overlap eigenvalues")
        transform = vectors[:, order].real
        mode_m = raw_m @ transform
        mode_n = raw_n @ np.linalg.inv(transform).T
    else:
        canonical, _ = svd_canonical_model(model)
        mode_m = canonical.m.detach().numpy().astype(float)
        mode_n = canonical.n.detach().numpy().astype(float)

    scales = np.linalg.norm(mode_m, axis=0) / np.sqrt(model.n_units)
    mode_m = mode_m / scales
    mode_n = mode_n * scales

    input_couplings = mode_n.T @ model.I.detach().numpy()
    signs = np.sign(input_couplings)
    signs[signs == 0] = 1
    return mode_m * signs, mode_n * signs


def latent_flow(
    kappa: np.ndarray,
    mode_m: np.ndarray,
    mode_n: np.ndarray,
) -> np.ndarray:
    return -kappa + mode_n.T @ np.tanh(mode_m @ kappa) / len(mode_m)


def latent_jacobian(
    kappa: np.ndarray,
    mode_m: np.ndarray,
    mode_n: np.ndarray,
) -> np.ndarray:
    gain = 1 - np.tanh(mode_m @ kappa) ** 2
    return -np.eye(2) + mode_n.T @ (gain[:, None] * mode_m) / len(mode_m)


def find_fixed_points(
    mode_m: np.ndarray,
    mode_n: np.ndarray,
) -> tuple[list[np.ndarray], int, np.ndarray]:
    """Search broadly, accept by residual, and deduplicate in physical state."""
    # At a root, |kappa_j| <= mean_i |n_ij| because |tanh(x_i)| <= 1.
    bounds = np.mean(np.abs(mode_n), axis=0)
    horizontal = np.linspace(-bounds[0], bounds[0], 31)
    vertical = np.linspace(-bounds[1], bounds[1], 31)
    grid = np.array(np.meshgrid(horizontal, vertical)).reshape(2, -1).T
    random_starts = np.random.default_rng(0).uniform(
        -bounds,
        bounds,
        size=(1_000, 2),
    )
    starts = np.vstack((np.zeros((1, 2)), grid, random_starts))

    fixed_points: list[np.ndarray] = []
    for start in starts:
        solution = root(
            latent_flow,
            start,
            args=(mode_m, mode_n),
            jac=latent_jacobian,
            method="hybr",
        )
        candidate = solution.x
        if np.linalg.norm(latent_flow(candidate, mode_m, mode_n)) > 1e-9:
            continue
        is_duplicate = any(
            np.linalg.norm(mode_m @ (candidate - existing)) / np.sqrt(len(mode_m))
            < 1e-6
            for existing in fixed_points
        )
        if not is_duplicate:
            fixed_points.append(candidate)

    fixed_points.sort(key=lambda point: (point[0], point[1]))
    return fixed_points, len(starts), bounds


def classify(eigenvalues: np.ndarray) -> str:
    if np.max(np.abs(eigenvalues.imag)) > 1e-7:
        return "stable focus" if np.all(eigenvalues.real < 0) else "unstable focus"
    if np.all(eigenvalues.real < 0):
        return "stable node"
    if np.all(eigenvalues.real > 0):
        return "unstable node"
    return "saddle"


def complex_records(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {"real": float(value.real), "imag": float(value.imag)}
        for value in values
    ]


def fixed_point_diagnostics(
    model: LowRankRNN,
    mode_m: np.ndarray,
    mode_n: np.ndarray,
    fixed_points: list[np.ndarray],
) -> list[dict[str, object]]:
    diagnostics = []
    readout = model.w.detach().numpy()

    for point in fixed_points:
        flow = latent_flow(point, mode_m, mode_n)
        jacobian = latent_jacobian(point, mode_m, mode_n)
        eigenvalues = np.linalg.eigvals(jacobian)
        state = mode_m @ point
        gain = 1 - np.tanh(state) ** 2
        full_residual = -state + mode_m @ (
            mode_n.T @ np.tanh(state) / model.n_units
        )
        full_jacobian = (
            -np.eye(model.n_units)
            + mode_m @ (mode_n.T * gain[None, :]) / model.n_units
        )
        full_eigenvalues = np.linalg.eigvals(full_jacobian)
        nontrivial = full_eigenvalues[
            np.argsort(np.abs(full_eigenvalues + 1))[-2:]
        ]
        full_spectrum_error = min(
            np.max(np.abs(nontrivial - eigenvalues)),
            np.max(np.abs(nontrivial - eigenvalues[::-1])),
        )

        finite_difference = np.column_stack(
            [
                (
                    latent_flow(point + 1e-6 * np.eye(2)[column], mode_m, mode_n)
                    - latent_flow(point - 1e-6 * np.eye(2)[column], mode_m, mode_n)
                )
                / 2e-6
                for column in range(2)
            ]
        )
        diagnostics.append(
            {
                "coordinates": point.tolist(),
                "state_rms": float(np.linalg.norm(state) / np.sqrt(model.n_units)),
                "readout": float(np.tanh(state) @ readout / model.n_units),
                "classification": classify(eigenvalues),
                "flow_eigenvalues_per_tau": complex_records(eigenvalues),
                "euler_multiplier_magnitudes": np.abs(
                    1 + STEP_SIZE * eigenvalues
                ).tolist(),
                "latent_residual": float(np.linalg.norm(flow)),
                "full_state_residual_rms": float(
                    np.linalg.norm(full_residual) / np.sqrt(model.n_units)
                ),
                "finite_difference_jacobian_error": float(
                    np.max(np.abs(jacobian - finite_difference))
                ),
                "full_spectrum_error": float(full_spectrum_error),
            }
        )
    return diagnostics


def task_mse(model: LowRankRNN) -> float:
    first, second = np.meshgrid(FREQUENCIES, FREQUENCIES, indexing="ij")
    pairs = np.column_stack((first.ravel(), second.ravel()))
    inputs, targets = make_fixed_delay_trials(pairs)
    with torch.no_grad():
        outputs, _ = model(torch.as_tensor(inputs))
    decisions = outputs[:, -5:].mean(dim=1).numpy()
    return float(np.mean((decisions - targets) ** 2))


def slow_channel_speed(
    mode_m: np.ndarray,
    mode_n: np.ndarray,
    fixed_points: list[np.ndarray],
) -> float | None:
    stable_endpoints = [
        point
        for point in fixed_points
        if classify(np.linalg.eigvals(latent_jacobian(point, mode_m, mode_n)))
        == "stable node"
    ]
    if len(stable_endpoints) != 2:
        return None

    line = np.linspace(stable_endpoints[0], stable_endpoints[1], 1_001)
    physical_speeds = [
        np.linalg.norm(mode_m @ latent_flow(point, mode_m, mode_n))
        / np.sqrt(len(mode_m))
        for point in line
    ]
    return float(np.max(physical_speeds))


def simulate_paths(
    mode_m: np.ndarray,
    mode_n: np.ndarray,
) -> list[np.ndarray]:
    starts = np.array(
        (
            (-0.24, -0.18),
            (-0.24, 0.18),
            (0.24, -0.18),
            (0.24, 0.18),
        )
    )
    times = np.linspace(0, 80, 1_201)
    return [
        solve_ivp(
            lambda _, point: latent_flow(point, mode_m, mode_n),
            (times[0], times[-1]),
            start,
            t_eval=times,
            rtol=1e-9,
            atol=1e-11,
        ).y.T
        for start in starts
    ]


def plot_phase_portrait(
    axis: plt.Axes,
    *,
    mode_m: np.ndarray,
    mode_n: np.ndarray,
    fixed_points: list[np.ndarray],
    title: str,
    labels: tuple[str, str],
) -> None:
    values = np.linspace(-0.3, 0.3, 41)
    horizontal, vertical = np.meshgrid(values, values)
    flow = np.array(
        [
            latent_flow(np.array((x, y)), mode_m, mode_n)
            for x, y in zip(horizontal.ravel(), vertical.ravel(), strict=True)
        ]
    )
    horizontal_flow = flow[:, 0].reshape(horizontal.shape)
    vertical_flow = flow[:, 1].reshape(vertical.shape)

    axis.streamplot(
        horizontal,
        vertical,
        horizontal_flow,
        vertical_flow,
        color="#9AA4AD",
        density=1.0,
        linewidth=0.8,
        arrowsize=0.8,
    )
    axis.contour(
        horizontal,
        vertical,
        horizontal_flow,
        levels=(0,),
        colors=(COLORS["blue"],),
        linewidths=1.6,
    )
    axis.contour(
        horizontal,
        vertical,
        vertical_flow,
        levels=(0,),
        colors=(COLORS["gold"],),
        linestyles="--",
        linewidths=1.6,
    )

    for path in simulate_paths(mode_m, mode_n):
        axis.plot(path[:, 0], path[:, 1], color="#26343E", linewidth=1.1)
        axis.scatter(
            *path[0],
            marker="x",
            color="#26343E",
            s=24,
            linewidth=1.1,
            zorder=4,
        )

    for point in fixed_points:
        point_type = classify(
            np.linalg.eigvals(latent_jacobian(point, mode_m, mode_n))
        )
        if point_type == "saddle":
            axis.scatter(
                *point,
                marker="D",
                facecolor="white",
                edgecolor=COLORS["gold"],
                linewidth=2,
                s=62,
                zorder=6,
            )
        else:
            axis.scatter(
                *point,
                marker="o",
                facecolor=COLORS["blue"],
                edgecolor="white",
                linewidth=1.2,
                s=70,
                zorder=6,
            )

    axis.set(
        xlim=(-0.3, 0.3),
        ylim=(-0.3, 0.3),
        xlabel=labels[0],
        ylabel=labels[1],
        title=title,
        aspect="equal",
    )
    axis.axhline(0, color="#D7DCE0", linewidth=0.6, zorder=0)
    axis.axvline(0, color="#D7DCE0", linewidth=0.6, zorder=0)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    oscillatory_model = load_or_train_oscillatory_model()
    structured_model = load_model(STRUCTURED_CHECKPOINT)
    models = {
        "fixed_delay_oscillatory": (
            oscillatory_model,
            *unit_rms_modes(oscillatory_model, separate_real_modes=False),
        ),
        "structured_persistent_transient": (
            structured_model,
            *unit_rms_modes(structured_model, separate_real_modes=True),
        ),
    }

    results: dict[str, object] = {}
    roots_by_model: dict[str, list[np.ndarray]] = {}
    for name, (model, mode_m, mode_n) in models.items():
        fixed_points, num_starts, search_bounds = find_fixed_points(mode_m, mode_n)
        roots_by_model[name] = fixed_points
        results[name] = {
            "root_search_starts": num_starts,
            "proven_root_coordinate_bounds": search_bounds.tolist(),
            "loop_gains": complex_records(
                np.linalg.eigvals(connectivity_overlap(mode_m, mode_n))
            ),
            "fixed_points": fixed_point_diagnostics(
                model,
                mode_m,
                mode_n,
                fixed_points,
            ),
            "fixed_delay_grid_mse": task_mse(model)
            if name == "fixed_delay_oscillatory"
            else None,
            "maximum_physical_speed_between_stable_endpoints_per_tau": (
                slow_channel_speed(mode_m, mode_n, fixed_points)
            ),
        }

    figure, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    plot_phase_portrait(
        axes[0],
        mode_m=models["fixed_delay_oscillatory"][1],
        mode_n=models["fixed_delay_oscillatory"][2],
        fixed_points=roots_by_model["fixed_delay_oscillatory"],
        title="Fixed-delay network",
        labels=(
            r"rotational coordinate $\kappa_1$",
            r"rotational coordinate $\kappa_2$",
        ),
    )
    plot_phase_portrait(
        axes[1],
        mode_m=models["structured_persistent_transient"][1],
        mode_n=models["structured_persistent_transient"][2],
        fixed_points=roots_by_model["structured_persistent_transient"],
        title="Structured persistent/transient network",
        labels=(
            r"persistent coordinate $\kappa_p$",
            r"transient coordinate $\kappa_t$",
        ),
    )
    figure.suptitle("Exact zero-input latent flow and fixed points")
    figure.text(
        0.5,
        -0.015,
        "Solid blue / dashed gold: the two nullclines. "
        "Filled circles: stable fixed points. Open diamond: saddle. "
        "×: autonomous trajectory starts.",
        ha="center",
        fontsize=9,
    )
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved {FIGURE_PATH}")
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
