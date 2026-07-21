"""Training and evaluation for the perceptual decision-making task."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _decision_window(outputs: torch.Tensor, decision_steps: int) -> torch.Tensor:
    if not 0 < decision_steps <= outputs.shape[1]:
        raise ValueError("decision_steps must be within the output sequence")
    return outputs[:, -decision_steps:]


def decision_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    decision_steps: int = 15,
) -> torch.Tensor:
    """Mean squared error over the final decision window."""
    decision_outputs = _decision_window(outputs, decision_steps)
    targets = labels[:, None].expand_as(decision_outputs)
    return nn.functional.mse_loss(decision_outputs, targets)


def train_model(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
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
            accuracy = decision_accuracy(
                model,
                inputs,
                labels,
                decision_steps=decision_steps,
            )
            print(
                f"Epoch {epoch_number}: "
                f"loss={epoch_loss:.6f}, accuracy={accuracy:.1%}"
            )

    return losses


@torch.no_grad()
def decision_accuracy(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
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
