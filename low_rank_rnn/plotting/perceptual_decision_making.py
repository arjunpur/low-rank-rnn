"""Plots for the perceptual decision making task."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from low_rank_rnn.data import DT_MS, STIMULUS_WINDOW_MS


def plot_first_perceptual_decision_making_trials(
    data: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    num_trials: int = 5,
    dt_ms: int = DT_MS,
    stimulus_window_ms: Sequence[int] = STIMULUS_WINDOW_MS,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the first few generated perceptual decision making trials."""
    data_array = np.asarray(data)
    labels_array = np.asarray(labels)
    n = min(num_trials, data_array.shape[0])

    time_s = np.arange(data_array.shape[1]) * dt_ms / 1000
    t0, t1 = (t / 1000 for t in stimulus_window_ms)

    fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n), sharex=True, squeeze=False)
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        ax.plot(time_s, data_array[i], lw=0.8)
        ax.axvspan(t0, t1, color="gray", alpha=0.12)
        direction = "right (+1)" if labels_array[i] > 0 else "left (-1)"
        ax.set_ylabel(f"trial {i}\n{direction}")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("First perceptual decision making trials")
    fig.tight_layout()
    return fig, axes
