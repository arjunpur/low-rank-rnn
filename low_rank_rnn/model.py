"""Low-rank recurrent neural network."""

from __future__ import annotations

import torch
from torch import nn


class LowRankRNN(nn.Module):
    """Continuous-time low-rank RNN simulated with forward Euler."""

    def __init__(
        self,
        n_units: int,
        *,
        rank: int = 1,
        dt_ms: float = 20.0,
        tau_ms: float = 100.0,
    ) -> None:
        super().__init__()
        self.n_units = n_units
        self.step_size = dt_ms / tau_ms

        self.m = nn.Parameter(torch.randn(n_units, rank))
        self.n = nn.Parameter(torch.randn(n_units, rank))
        self.register_buffer("I", torch.randn(n_units))
        self.register_buffer("w", 4.0 * torch.randn(n_units))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return readouts and states for inputs shaped ``(batch, time)``."""
        batch_size, n_steps = inputs.shape
        state = inputs.new_zeros(batch_size, self.n_units)

        outputs = []
        states = []
        for step in range(n_steps):
            rates = torch.tanh(state)
            outputs.append(rates @ self.w / self.n_units)
            states.append(state)

            latent = rates @ self.n / self.n_units
            recurrent = latent @ self.m.T
            external = inputs[:, step, None] * self.I[None, :]
            state = state + self.step_size * (-state + recurrent + external)

        return torch.stack(outputs, dim=1), torch.stack(states, dim=1)
