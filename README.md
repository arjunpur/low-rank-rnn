# Low-rank RNNs

This repository explores how low-rank recurrent neural networks solve two
simple cognitive tasks: perceptual decision-making and parametric working
memory. The main goal is to connect what the trained networks do in latent
space to the statistics of their connectivity vectors.

The full analysis lives in [`final.ipynb`](final.ipynb). A written summary and
the main figures are available in [`report/report.pdf`](report/report.pdf).

## Repository layout

- `final.ipynb` contains the end-to-end experiments and analysis.
- `low_rank_rnn/model.py` defines the low-rank RNN.
- `low_rank_rnn/training.py` contains the training loop and metrics.
- `low_rank_rnn/data/` generates the decision-making and working-memory tasks.
- `low_rank_rnn/analysis.py` contains projections, connectivity analysis, PCA,
  and fixed-point tools.
- `low_rank_rnn/mean_field.py` implements the reduced Gaussian circuits.
- `low_rank_rnn/plotting/` contains the plotting code used by the notebook.
- `tests/` contains the unit tests.
- `report/` contains the report source, generated figures, and final PDF.

## Setup

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/) for dependency
management.

```bash
uv sync
```

## Running the project

To reproduce the analysis, execute the main notebook from top to bottom:

```bash
uv run --with nbconvert jupyter nbconvert \
  --execute --to notebook --inplace final.ipynb
```

The notebook trains several 512-unit networks, so a full run can take a while.
It uses fixed random seeds to make the reported run reproducible.

Run the tests with:

```bash
uv run python -m unittest discover -s tests -v
```

If you have a LaTeX installation with `latexmk`, rebuild the report PDF with:

```bash
make -C report pdf
```

## Key results

- The rank-one network solves the perceptual decision-making task with 100%
  held-out sign accuracy. Its latent dynamics form two stable decision
  attractors, separated by an unstable fixed point at the origin.
- Ten networks sampled from the fitted rank-one loading distribution also
  reach 100% accuracy. The corresponding one-dimensional mean-field circuit
  reproduces the trained network's behavior.
- The rank-two network solves the fixed-delay working-memory task with an MSE
  of 0.000373, while the rank-one control cannot solve the task.
- The rank-two solution is oscillatory and tied to the training delay. Its
  performance varies strongly when that delay changes, so it does not learn a
  robust memory mechanism.
- Gaussian resampling is much less reliable for the rank-two network: the ten
  sampled networks have a median MSE of 0.0839, and only one reaches the target
  MSE below 0.005.

In short, population-level connectivity statistics are enough to explain and
reproduce the rank-one decision circuit. The working-memory task is more
sensitive: a low training error does not guarantee robust latent dynamics or
good behavior under resampling.
