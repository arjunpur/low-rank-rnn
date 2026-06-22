# Low Rank RNN

This repository contains a short LaTeX note on low-rank recurrent connectivity and code for generating and plotting perceptual decision making trials.

## Contents

- `output/pdf/low_rank_connectivity_dynamics_explanation.tex` - LaTeX source
- `output/pdf/low_rank_connectivity_dynamics_explanation.pdf` - rendered PDF
- `low_rank_rnn/data/` - perceptual decision making data generation
- `low_rank_rnn/plotting/` - plotting utilities for generated trials
- `model.ipynb` - notebook example using the package functions

## Usage

```python
import matplotlib.pyplot as plt

from low_rank_rnn.data import generate_perceptual_decision_making_trials
from low_rank_rnn.plotting import plot_first_perceptual_decision_making_trials

data, labels = generate_perceptual_decision_making_trials(num_trials=10)
fig, axes = plot_first_perceptual_decision_making_trials(data, labels, num_trials=5)
plt.show()
```
