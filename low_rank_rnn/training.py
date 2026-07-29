"""Training and evaluation for the perceptual decision-making task."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Real
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from low_rank_rnn._typing import typechecked
from low_rank_rnn.data.variable_delay import DELAYS, FREQUENCIES, sample_trials


def _decision_window(
    outputs: Float[torch.Tensor, "batch time"],
    decision_steps: int,
) -> Float[torch.Tensor, "batch decision"]:
    if not 0 < decision_steps <= outputs.shape[1]:
        raise ValueError("decision_steps must be within the output sequence")
    return outputs[:, -decision_steps:]


@typechecked
def decision_loss(
    outputs: Float[torch.Tensor, "batch time"],
    labels: Float[torch.Tensor, "batch"],
    *,
    decision_steps: int = 15,
) -> Float[torch.Tensor, ""]:
    """Mean squared error over the final decision window."""
    decision_outputs = _decision_window(outputs, decision_steps)
    targets = labels[:, None].expand_as(decision_outputs)
    return nn.functional.mse_loss(decision_outputs, targets)


@typechecked
def train_model(
    model: nn.Module,
    inputs: Float[torch.Tensor, "batch time"],
    labels: Float[torch.Tensor, "batch"],
    *,
    epochs: int = 1_000,
    batch_size: int = 32,
    learning_rate: float = 5e-3,
    decision_steps: int = 15,
    log_every: int | None = 100,
) -> list[float]:
    """Train a model and return the mean loss from each epoch.

    Set ``log_every`` to ``None`` to disable progress logging.
    """
    if log_every is not None and log_every <= 0:
        raise ValueError("log_every must be positive or None")

    dataset = TensorDataset(inputs, labels)
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for batch_inputs, batch_labels in batches:
            outputs, _ = model(batch_inputs)
            loss = decision_loss(outputs, batch_labels, decision_steps=decision_steps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_inputs)

        epoch_loss = total_loss / len(dataset)
        losses.append(epoch_loss)
        epoch_number = len(losses)
        should_log = log_every is not None and (
            epoch_number % log_every == 0 or epoch_number == epochs
        )
        if should_log:
            print(f"Epoch {epoch_number}: loss={epoch_loss:.6f}")

    return losses


@typechecked
def masked_decision_loss(
    outputs: Float[torch.Tensor, "batch time"],
    targets: Float[torch.Tensor, "batch"],
    decision_mask: Float[torch.Tensor, "batch time"],
) -> Float[torch.Tensor, ""]:
    """Mean squared error over each trial's own decision window."""
    squared_error = (outputs - targets[:, None]).square()
    return (squared_error * decision_mask).sum() / decision_mask.sum()


@typechecked
def train_variable_delay(
    model: nn.Module,
    stages: Sequence[npt.ArrayLike] = (DELAYS,),
    *,
    rng: np.random.Generator,
    frequencies: npt.ArrayLike = FREQUENCIES,
    num_trials: int = 256,
    epochs_per_stage: int = 300,
    learning_rate: float = 5e-3,
    batch_size: int | None = None,
    max_gradient_norm: float | None = None,
) -> list[float]:
    """Train on random delays, one curriculum stage at a time.

    Each stage draws a fresh trial set from its own range of delays; the
    optimizer carries over between stages. ``frequencies`` defines the values
    sampled for each stimulus. A single stage spanning the whole range trains
    the task without a curriculum, and a single stage holding one delay trains
    the fixed-delay version of the task. ``batch_size`` defaults to full batch;
    set it smaller for minibatch SGD. Returns the mean loss per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    model.train()
    for stage_delays in stages:
        inputs, targets, decision_mask = sample_trials(
            num_trials,
            stage_delays,
            rng=rng,
            frequencies=frequencies,
        )
        batches = DataLoader(
            TensorDataset(inputs, targets, decision_mask),
            batch_size=batch_size if batch_size is not None else num_trials,
            shuffle=True,
        )
        for _ in range(epochs_per_stage):
            total_loss = 0.0
            for batch_inputs, batch_targets, batch_mask in batches:
                outputs, _ = model(batch_inputs)
                loss = masked_decision_loss(outputs, batch_targets, batch_mask)

                optimizer.zero_grad()
                loss.backward()
                if max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_gradient_norm,
                    )
                optimizer.step()
                total_loss += loss.item() * len(batch_inputs)
            losses.append(total_loss / num_trials)

    model.eval()
    return losses


@torch.no_grad()
@typechecked
def decision_accuracy(
    model: nn.Module,
    inputs: Float[torch.Tensor, "batch time"],
    labels: Real[torch.Tensor, "batch"],
    *,
    decision_steps: int = 15,
) -> float:
    """Return classification accuracy over the final decision window."""
    was_training = model.training
    model.eval()
    outputs, _ = model(inputs)
    decisions = _decision_window(outputs, decision_steps).mean(dim=1)
    predictions = torch.where(decisions >= 0, 1, -1)
    accuracy = (predictions == labels).float().mean().item()
    model.train(was_training)
    return accuracy
