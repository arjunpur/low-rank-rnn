# Low-rank RNN

This repository studies rank-one and rank-two recurrent neural networks on
perceptual decision-making and parametric working-memory tasks.

`final.ipynb` is the complete, reproducible analysis. Older exploratory
notebooks and scripts live in `archive/`; they are not part of the core library
contract.

## Library layout

- `low_rank_rnn/model.py` — low-rank dynamics and structured initialization
- `low_rank_rnn/training.py` — losses, training loops, and task accuracy
- `low_rank_rnn/data/` — task definitions and trial generation
- `low_rank_rnn/analysis.py` — model evaluation, connectivity, projections, and
  fixed-point analysis
- `low_rank_rnn/mean_field.py` — Gaussian equivalent circuits
- `low_rank_rnn/plotting/` — all figure construction and shared visual style

The modules have deliberately narrow jobs. The notebook sets experiment
parameters and interprets results; reusable computation and presentation live
in the library.

## Usage

```python
import numpy as np
import torch

from low_rank_rnn import analysis, plotting
from low_rank_rnn.data import perceptual_decision_making as task
from low_rank_rnn.model import LowRankRNN
from low_rank_rnn.training import train_model

plotting.set_plot_style()
inputs, labels = task.generate_trials(
    200,
    rng=np.random.default_rng(2026),
)

torch.manual_seed(2026)
model = LowRankRNN(n_units=128, rank=1)
losses = train_model(
    model,
    torch.as_tensor(inputs, dtype=torch.float32),
    torch.as_tensor(labels, dtype=torch.float32),
)
outputs, states = analysis.run_model(model, inputs)
```

Run the tests with:

```bash
uv run python -m unittest discover -s tests -v
```

Execute the final notebook from top to bottom with:

```bash
uv run jupyter nbconvert --execute --to notebook --inplace final.ipynb
```

## Shape-aware development

Public numerical boundaries use `jaxtyping` with `beartype`. The optional
Trickle setup records runtime values and tensor shapes:

```bash
npm install -g trickle-cli@0.1.223
uv sync
source .venv/bin/activate
trickle run python path/to/script.py
```
