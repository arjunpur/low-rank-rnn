"""Training and evaluation for low-rank RNN tasks."""

import torch
from jaxtyping import Float, Real
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from low_rank_rnn._typing import typechecked


@typechecked
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
