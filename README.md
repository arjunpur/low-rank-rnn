# Low Rank RNN

This repository contains a rank-one recurrent neural network for a perceptual
decision-making task and a short note explaining its dynamics.

## Contents

- `output/pdf/low_rank_connectivity_dynamics_explanation.tex` - LaTeX source
- `output/pdf/low_rank_connectivity_dynamics_explanation.pdf` - rendered PDF
- `low_rank_rnn/data/` - perceptual decision making data generation
- `low_rank_rnn/model.py` - rank-one RNN dynamics
- `low_rank_rnn/training.py` - decision loss, training, and evaluation
- `low_rank_rnn/analysis.py` - connectivity and activity analysis
- `low_rank_rnn/plotting/` - reusable figures and plotting style
- `model.ipynb` - complete experiment using the package

## Usage

```python
import matplotlib.pyplot as plt

from low_rank_rnn.data import generate_perceptual_decision_making_trials
from low_rank_rnn.plotting import (
    plot_first_perceptual_decision_making_trials,
    set_plot_style,
)

set_plot_style()

data, labels = generate_perceptual_decision_making_trials(num_trials=10)
fig, axes = plot_first_perceptual_decision_making_trials(data, labels, num_trials=5)
plt.show()
```

Call `set_plot_style()` once near the top of a notebook. For a temporary style,
use `with plot_style():`; named project colors are available through `COLORS`.

## Shape-aware development

Install the compatible Trickle CLI and project environment once:

```bash
npm install -g trickle-cli@0.1.223
uv sync
```

Then use the **Trickle: Run current Python file** task in Cursor. Trickle records
runtime values in `.trickle/`, and its Cursor extension shows tensor shapes and
types inline. The equivalent terminal command is:

```bash
source .venv/bin/activate
trickle run python path/to/script.py
```

For notebooks, run this in the first cell:

```python
%load_ext trickle
```

Public numerical functions use `jaxtyping` with `beartype` to check tensor and
array shapes at module boundaries. Keep those modules free of
`from __future__ import annotations`, which prevents runtime shape checking.
