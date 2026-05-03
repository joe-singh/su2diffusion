# SU(2) Diffusion

Toy diffusion experiments on `SU(2)` using a heat-kernel target.

The repo is set up so the reusable math/model code lives in importable Python
modules, while the notebook stays as a Colab runner for GPU experiments.

## Local Setup

```bash
pip install -e ".[dev]"
pytest
```

On Apple Silicon, PyTorch can use MPS for small sanity checks. Colab is still
the better place for full training runs.

## Colab Branch Workflow

To run a pull-request branch in Colab:

```python
!pip install -q git+https://github.com/joe-singh/su2diffusion.git@branch-name
```

Then import from `su2diffusion` in the notebook.

## Current Demo

The main notebook trains a conditional diffusion model on synthetic
single-qubit Clifford neighborhoods in `SU(2)`. Generated single-qubit gates
are converted to `2x2` unitaries and used as local layers in shallow 2-qubit
templates such as

```text
(A tensor B) CZ (C tensor D)
```

The synthesis demo has two parts:

- named-gate sanity checks for `CZ`, `CNOT`, and Bell-state preparation;
- hidden shallow-circuit benchmarks where target circuits are generated from
  exact Clifford local layers, the local labels are hidden, and generated gates
  are searched/ranked by unitary fidelity.

In the first 50-target Colab benchmark, generated label-grid search recovered
all hidden shallow circuits above `0.98` best unitary fidelity.

The notebook also includes a depth-2 hidden benchmark for circuits of the form

```text
(A tensor B) CZ (C tensor D) CZ (E tensor F)
```

At this depth exhaustive Clifford search is too large for the default workflow,
so the benchmark uses random search over generated six-slot local-gate
candidates and reports aggregate best-fidelity success rates.
