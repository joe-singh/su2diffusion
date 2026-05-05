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

The depth-2 demo can also run a local refinement pass: after random search picks
the best generated circuit, the six local `SU(2)` gates are optimized with the
template and entanglers fixed. This tests whether the remaining error is mostly
from coarse generated samples or from the circuit ansatz/search itself.

The notebook includes an ablation that reruns the same local optimizer from
random Haar local-gate starts. That comparison is meant to separate the value of
diffusion/search initialization from the value of continuous local refinement.

For a more continuous synthesis benchmark, the notebook can build hidden
near-Clifford circuits by perturbing each ideal Clifford local gate with a small
`SU(2)` exponential update before composing the depth-2 circuit. This makes the
hardcoded Clifford library an imperfect baseline and gives diffusion-generated
continuous gates a more meaningful role.

The near-Clifford benchmark also includes an analytic noisy-Clifford sampler,
which draws `exp(epsilon) * Clifford` directly without training. This baseline
checks whether diffusion adds value beyond a hand-coded local perturbation model.

The first circuit-level diffusion path models full depth-2 local-gate stacks on
`SU(2)^6`. A sample is the six-gate template
`(A tensor B) CZ (C tensor D) CZ (E tensor F)`, so the denoiser can learn
correlations across circuit slots instead of sampling six local gates
independently.

The solution-stack workflow turns this into a synthesis-driven dataset: hidden
near-Clifford targets are solved by search plus local `SU(2)` refinement, and
the refined six-gate stacks train a second joint circuit diffusion model. This
tests whether learning from successful circuits improves proposal quality over
random near-Clifford circuit stacks.

## Hamiltonian Synthesis Workflow

The current Hamiltonian-to-circuit path synthesizes two-qubit targets

```text
U(t) = exp(-i H t)
```

with the fixed depth-2 template

```text
(A tensor B) CZ (C tensor D) CZ (E tensor F)
```

where each local gate is represented on `SU(2)`. The default baseline is:

1. train the conditional single-qubit `SU(2)` generator;
2. sample generated local gates;
3. search uniformly over six-slot generated-gate candidates;
4. refine the best candidate directly on `SU(2)^6`.

The notebook also keeps learned slot-label prior experiments as optional
diagnostics. Those priors can help on nearby/easy Hamiltonian distributions, but
the harder 15-term Pauli stress test showed poor distribution transfer. For now,
uniform generated search plus local `SU(2)` refinement is the main synthesis
baseline; learned priors should be revisited only with a more principled
Hamiltonian family or a continuous circuit-diffusion objective.
