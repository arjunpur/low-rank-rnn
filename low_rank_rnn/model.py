"""Low-rank recurrent neural network."""

import torch
from jaxtyping import Float
from torch import nn

from low_rank_rnn._typing import typechecked


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

    @typechecked
    def forward(
        self,
        inputs: Float[torch.Tensor, "batch time"],
    ) -> tuple[
        Float[torch.Tensor, "batch time"],
        Float[torch.Tensor, "batch time unit"],
    ]:
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


def persistent_transient_rnn(n_units: int, *, seed: int) -> LowRankRNN:
    """Sample an i.i.d. Gaussian persistent/transient rank-two network."""
    torch.manual_seed(seed)
    model = LowRankRNN(n_units, rank=2)
    loadings = torch.randn(n_units, 4)
    input_vector, persistent, transient, readout_residual = loadings.T

    with torch.no_grad():
        model.I.copy_(input_vector)
        model.m.copy_(torch.column_stack((persistent, transient)))
        model.n.copy_(
            torch.column_stack(
                (
                    persistent + 0.5 * input_vector,
                    0.5 * transient + 1.9 * input_vector,
                )
            )
        )
        residual_gain = (16 - 2.8**2 - 2.2**2) ** 0.5
        model.w.copy_(
            2.8 * persistent
            - 2.2 * transient
            + residual_gain * readout_residual
        )
    return model
