# SU(2) Diffusion

Toy diffusion experiments on `SU(2)` using a heat-kernel target.

The repo is set up so the reusable math/model code lives in importable Python
modules, while the notebook stays as a Colab runner for GPU experiments.

For a checkpoint of the current research state, architecture, benchmark results,
and next steps, see [`docs/project_context.md`](docs/project_context.md).

For the cleanest current demo, open [`SU2HamiltonianDemo.ipynb`](SU2HamiltonianDemo.ipynb).
It runs the local `SU(2)` diffusion sampler, then demonstrates Hamiltonian-to-circuit
synthesis on `XII`, `YII`, and a small three-qubit transverse-Ising target.

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

The first continuous Hamiltonian-conditioned circuit diffusion path is now
available as a proposal-quality check. It trains on refined Hamiltonian solution
stacks and samples directly from

```text
p(q1, ..., q6 | H, t) on SU(2)^6.
```

This is intentionally separate from the old discrete slot-label prior: the
diffusion model predicts six tangent denoising directions and keeps every local
gate on the `SU(2)` manifold during reverse sampling.

The solution dataset can keep multiple refined decompositions for the same
Hamiltonian target. This is important because many different six-gate stacks can
compile the same `U(t)`, and diffusion is better matched to that one-to-many
conditional distribution than to an arbitrary single refined solution.

The notebook also includes an overfit diagnostic for this conditioned diffusion
path. It trains on the multi-solution Hamiltonian dataset and compares generated
proposal fidelity on the training Hamiltonians versus fresh held-out
Hamiltonians. If the training row is poor, the next fix is model/sampler
mechanics; if only the held-out row is poor, the next fix is dataset scale and
coverage.

A one-step denoising diagnostic now reuses the overfit model and measures
whether it predicts the heat-kernel tangent targets on the training stacks. This
separates basic denoiser fit from failures introduced by the full reverse
sampling chain.

The denoising scale ablation runs the same one-step check across current,
longer, wider, and wider-longer Hamiltonian-conditioned denoisers. Its purpose is
to decide whether the near-zero denoiser collapse responds to ordinary
training/capacity before changing the model family.

The target-normalization diagnostic trains the same conditioned denoiser on
scaled tangent targets, then unscales predictions before reporting the standard
one-step metrics. This tests whether output collapse is caused by the raw
tangent-target scale or by a more structural modeling issue.

The active follow-up is a circuit-token Hamiltonian denoising diagnostic. It
compares the original flat circuit MLP against a small attention model with one
Hamiltonian conditioning token and six local-gate tokens. Both models still
denoise on `SU(2)^6`; the diagnostic asks whether a model that explicitly lets
the six local gates communicate can fit the tangent targets before we either
scale architecture further or freeze direct Hamiltonian-conditioned diffusion as
a negative result.

The next generalization check trains that circuit-token denoiser on refined
solution stacks for one Hamiltonian suite, then reverse-samples proposals for
fresh held-out Hamiltonians. The held-out report compares token diffusion
against Clifford, analytic near-Clifford, generated local-gate, and Haar search
baselines on the same target family.

The current scale-up check repeats that held-out evaluation while increasing
the number of Hamiltonian training targets. It uses the same circuit-token
architecture and the same held-out suite for every row, so the resulting curve
asks a direct question: does more refined Hamiltonian solution data improve
fresh-target proposal quality, or was the strong train-target result mostly
memorization?

The follow-up training-budget check fixes the larger Hamiltonian dataset and
instead varies the number of token-denoiser optimization steps. This separates a
data effect from an undertraining effect: if held-out proposal quality recovers
as the step budget grows, the next scale-up should increase data and training
budget together before changing model architecture.

The Level 1 repeatability check then fixes the chosen budget and repeats the
Hamiltonian-to-`SU(2)^6` token-diffusion experiment across independent
train/test seeds. It reports per-run held-out fidelity, success rates, and the
advantage over the generated local-gate search baseline, plus mean/std summary
statistics. This is the first step from exploratory notebook runs toward a
paper-grade benchmark table.

The refinement-basin follow-up reuses those repeatability runs without
retraining. For each held-out Hamiltonian it starts the same local `SU(2)^6`
optimizer from the best circuit-token proposal and from the best generated
local-gate search proposal, then reports before/after fidelity and steps to a
target threshold. This asks whether diffusion is merely producing a higher
initial fidelity, or whether it also places the optimizer in a better basin.
It also reports mean and max per-slot `SU(2)` movement during refinement, which
checks whether the optimizer is only polishing the proposal or moving far enough
to effectively replace it.

The final Level 1 headline table condenses that workflow into one compact
result: proposal quality, repeatability run standard deviation, refined
fidelity, refinement success, median optimizer steps, and movement for token
diffusion versus generated local-gate search.

Level 2A starts the universal two-qubit template progression by comparing the
current two-CZ ansatz against

```text
(A tensor B) CZ (C tensor D) CZ (E tensor F) CZ (G tensor H)
```

The same generated-gate search and local `SU(2)` refinement now support any
even local-gate chain, so the notebook can build solution stacks on `SU(2)^8`.
This branch uses that to check whether adding the third CZ improves proposal
and refinement geometry before training the circuit-token diffusion model on
eight-slot solution stacks.

Level 2B upgrades the Hamiltonian circuit-token diffusion comparison itself:
it trains the same token denoiser workflow on both the two-CZ `SU(2)^6`
solution stacks and the universal three-CZ `SU(2)^8` stacks, then evaluates
held-out Hamiltonian proposal quality against the generated local-gate search
baseline for each template. This is the first direct check of whether the
diffusion model benefits from the larger universal two-qubit circuit ansatz,
not just whether the local optimizer can use it.

Level 2C keeps the three-CZ `SU(2)^8` token architecture fixed and scales the
solution-stack dataset. It compares settings like `64x2`, `128x2`, `256x2`,
and `128x4`, where the first number is the number of Hamiltonian training
targets and the second is refined solution stacks kept per target. This asks
whether the weaker eight-slot proposal quality is mainly a data coverage issue
before increasing model size.

Level 2D adds a short-term canonicalization rule for the `SU(2)^8` solution
data. For each Hamiltonian, the dataset builder can refine a small pool of
candidate stacks, filter for high fidelity, and keep the solution with minimum
total local rotation `sum_i ||log(q_i)||^2`. The scale-sweep helper also shows
progress during search/refinement, chunks reverse sampling into smaller batches,
and moves completed diagnostic models to CPU by default so Colab does not retain
every row's CUDA memory.

Level 2E reuses that canonical `SU(2)^8` setup for a focused training-budget
check. It builds the largest canonical dataset once, trains on prefixes such as
`128x1` and `256x1`, and compares multiple denoising step budgets against the
same held-out Hamiltonians and generated-search baseline.

Level 3A starts the move from two to three qubits. It does not train a new
diffusion model yet; instead it benchmarks fixed 3-qubit CZ templates such as
`CZ01-CZ12-CZ01`, `CZ12-CZ01-CZ12`, a four-CZ line, and an all-pairs three-CZ
pattern. For each 3-qubit Hamiltonian target it searches local `SU(2)` gates,
refines them on the product manifold, and reports which entangler layout gives
the best proposal/refinement geometry. The winning template becomes the first
candidate for a later `SU(2)^n` circuit-token diffusion model.

Level 3B trains that first 3-qubit circuit-token model on the winning four-CZ
line template

```text
L0 CZ01 L1 CZ12 L2 CZ01 L3 CZ12 L4
```

where each `Lk` is a layer of three local `SU(2)` gates. One sample therefore
lives on `SU(2)^15`. The benchmark builds refined Hamiltonian solution stacks,
trains a Hamiltonian-conditioned token denoiser, and compares proposal quality
against the generated local-gate search baseline. The notebook default now uses
48 held-out Hamiltonian targets for this check.

Level 3C reuses the Level 3B model and runs the same local `SU(2)^15`
optimizer from three starting points: the best token-diffusion proposal, the
best generated local-gate search proposal, and a Haar-random local stack. It
reports before/after fidelity, steps to the target threshold, and per-slot
movement. This checks whether the 3-qubit diffusion model gives a better
refinement basin even when its raw proposal fidelity is still modest. The
headline table reports proposal mean, refined mean, threshold success, median
steps, and movement for token, generated-search, and Haar starts.

Level 3D is the paper-grade repeatability version of Level 3C. It reruns the
48-target `line-4cz` train/evaluate/refine workflow across independent seeds
and aggregates the per-run proposal, refined-fidelity, success-rate, optimizer
step, and movement metrics. The notebook keeps this cell off by default because
it intentionally repeats the long 3-qubit run several times.

Level 3E turns the statistical workflow into a single-target demo report. Given
one Hamiltonian target, it prints the `line-4cz` template, proposal/refined
fidelity, steps to threshold, a readable local-layer circuit, refined local
gates in axis-angle form, and per-slot `SU(2)` movement. The plot combines the
refinement trace, the fixed CZ skeleton, and movement by local slot. If the
Level 3B token model has been run, this demo uses a token-diffusion proposal;
otherwise it falls back to generated local-gate search.
