# SU(2) Diffusion

Lie-group diffusion experiments for hardware-aware quantum circuit synthesis.

The public paper workflow is contained in one notebook:

- [`SU2GateExperiments.ipynb`](SU2GateExperiments.ipynb)

That notebook contains the local single-qubit gate setup plus the final paper experiments. 

## Setup

Install the package locally with:

```bash
pip install -e ".[dev]"
pytest
```

For Colab, open the notebook and run its first install cell. After merge, the notebook installs from `main`:

```python
BRANCH = "main"
```

To test a pull-request branch, temporarily replace `main` with that branch name.

## Paper Experiment Notebook

Run order for a fresh notebook runtime:

1. Run the install/import setup cells.
2. Train the local single-qubit gate generator to create `local_gates` and `local_labels`.
3. Run Experiment A to train the best-fidelity Willow-cost selector and skeleton-conditioned diffusion model.
4. Run whichever downstream experiment you need.

The notebook experiments are:

| Experiment | Purpose |
| --- | --- |
| A | Best-fidelity auto-skeleton model training |
| B | Held-out synthesis benchmark |
| C | Pipeline ablation table |
| D | Hardware-aware Pareto frontiers |
| E | Diversity and MDS diagnostics |
| F | Angle-steering priors |
| G | Custom Hamiltonian inference / paper example circuit |

Most experiment cells are disabled by default with a `RUN_EXPERIMENT_... = False` flag. Toggle the experiment you want to run.

## Notes

- Full training/evaluation is intended for Colab or another GPU runtime.
- The source package lives in `su2diffusion/`; tests live in `tests/`.
