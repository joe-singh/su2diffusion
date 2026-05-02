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
