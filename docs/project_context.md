# SU(2) Diffusion Project Context

Last updated: 2026-05-07

This memo is a checkpoint for the `su2diffusion` project. It is meant to let a
new conversation, collaborator, or future version of us recover the state of the
work without replaying the whole chat history.

## One-Sentence Goal

Given a Hamiltonian `H` and evolution time `t`, learn to generate a useful
quantum circuit proposal for

```text
U(t) = exp(-i H t)
```

where the generated local gates live on the correct Lie group manifold, then use
a small continuous `SU(2)` refinement step to polish the proposal into a high
fidelity circuit.

The current target form is:

```text
H, t -> Hamiltonian-conditioned SU(2)^n diffusion -> circuit proposal -> local SU(2)^n refinement
```

## What Makes This Project Different

The reference circuit-diffusion literature denoises circuit encodings: qubit
axis, time/gate-placement axis, and gate-type embeddings. That produces discrete
or tensor-coded circuits.

This project denoises local gates directly on `SU(2)` and its product
manifolds:

```text
SU(2)^6, SU(2)^8, SU(2)^15, ...
```

The discrete entangling skeleton is fixed during a run, while the local gates
are continuous group elements. The core claim we are working toward is:

```text
Lie-group diffusion gives better continuous local-gate proposals for Hamiltonian
synthesis than generic random/search starts, and those proposals sit in better
optimization basins.
```

This is not yet a full universal compiler. It is a structured research path
toward one.

## Core Mathematical Representation

### Single-Qubit Gates

Each local gate is represented as a unit quaternion

```text
q = (w, x, y, z),    ||q|| = 1.
```

Unit quaternions double-cover `SU(2)`. The package converts quaternions to
`2x2` unitary matrices when composing circuits.

### Diffusion State

Diffusion is not predicting a raw matrix entry by entry. The model predicts a
tangent vector in the Lie algebra, roughly the `SU(2)` analogue of predicting
noise in a Euclidean DDPM.

The forward process perturbs a clean group element by multiplying with a small
exponential-map update:

```text
q_t ~= exp(epsilon_t) q_0
```

or slotwise on product manifolds:

```text
(q_1, ..., q_n)_t ~= (exp(epsilon_1) q_1, ..., exp(epsilon_n) q_n).
```

The reverse sampler uses the predicted tangent vector and the exponential map,
so the generated gates remain on `SU(2)` after every step.

### Heat-Kernel Target

The heat-kernel machinery was introduced so the denoising target respects the
geometry of `SU(2)`, rather than pretending the quaternions live in flat
Euclidean space. In practice, the model is trained to predict tangent directions
associated with the Brownian/heat-kernel noising process on the group.

## Circuit Templates

The project currently fixes an entangling template, then learns/proposes the
continuous local `SU(2)` gates.

### Two-Qubit Templates

The early two-qubit Hamiltonian workflow used:

```text
(A tensor B) CZ (C tensor D) CZ (E tensor F)
```

This has six local `SU(2)` gates, so the diffusion state is `SU(2)^6`.

We also tested the universal two-qubit three-CZ form:

```text
(A tensor B) CZ (C tensor D) CZ (E tensor F) CZ (G tensor H)
```

This has eight local `SU(2)` gates, so the diffusion state is `SU(2)^8`.

### Three-Qubit Template

The current three-qubit winner is the four-CZ line template:

```text
L0 CZ01 L1 CZ12 L2 CZ01 L3 CZ12 L4
```

where each `Lk` is a layer of three local `SU(2)` gates. That means:

```text
5 local layers x 3 qubits = 15 local gates
```

so the current 3-qubit diffusion state is:

```text
SU(2)^15.
```

We chose the four-CZ line because the first three-qubit template screen showed
that four CZs refined much better than the tested three-CZ patterns.

## Training Data

There are several data regimes in the repo. The important current one is the
Hamiltonian solution-stack dataset.

### Hamiltonian Targets

Targets are random Pauli Hamiltonians of the form:

```text
H = sum_j c_j P_j
```

where each `P_j` is a tensor product of Pauli operators, such as:

```text
XII, IZI, IIZ, XXI, IZZ, ZXZ
```

The target unitary is:

```text
U(t) = exp(-i H t).
```

### Solution Stacks

For each training Hamiltonian target, the code builds one or more solution
stacks:

1. Search over local gate candidates for the fixed CZ template.
2. Take the best candidate.
3. Refine the local gates directly on `SU(2)^n`.
4. Keep the refined stack as training data.

So the circuit-token diffusion model is trained on successful local-gate stacks,
not on arbitrary random circuits.

This data generation is expensive, but it gives the model examples of circuits
that actually synthesize the target Hamiltonian evolution under the chosen
template.

## Current Model Architecture

The current successful Hamiltonian model is a circuit-token denoiser.

Inputs:

- noisy local gates, as slot tokens;
- diffusion timestep;
- Hamiltonian/time features, as a conditioning token.

Outputs:

- one tangent-vector prediction per local gate slot.

For the 3-qubit four-CZ template, this means the model predicts 15 tangent
vectors, one for each `SU(2)` local gate.

The key architectural discovery so far:

```text
Flat MLP denoisers collapsed toward near-zero tangent predictions.
Circuit-token denoisers fit the denoising target much better.
```

In one diagnostic, the circuit-token model improved final-step relative MSE
from roughly `0.95` for the flat MLP to about `0.085`, with cosine alignment
around `0.95` and correct output scale.

That was the point where the project became meaningfully alive again.

## Refinement

Refinement is separate from diffusion.

Diffusion produces an initial circuit proposal. Then a classical optimizer
adjusts the local `SU(2)` gates while keeping the entangling skeleton fixed.

The optimizer is not changing the CZ layout. It only moves local gates on the
product manifold:

```text
(q_1, ..., q_n) in SU(2)^n.
```

The important diagnostic question is not only:

```text
Does refinement eventually reach high fidelity?
```

It is also:

```text
Does the diffusion proposal start in a better basin, requiring fewer steps and
less movement than generated-search or Haar starts?
```

That is why the tables report:

- proposal fidelity;
- refined fidelity;
- fraction above a threshold such as `F >= 0.99`;
- median steps to threshold;
- mean and max `SU(2)` movement during refinement.

## Headline Results So Far

### Single-Qubit Gate Diffusion

The conditional Clifford/gate model can generate recognizable single-qubit gate
families on `SU(2)`. It was useful infrastructure and a source of generated
local gates, but the current research focus is now circuit-level Hamiltonian
synthesis.

### Two-Qubit Hamiltonian Workflow

For two-qubit Hamiltonian targets, token diffusion over local-gate stacks became
useful as an optimizer seed.

Representative repeatability/refinement result:

```text
source             n    mean before   mean after   >=threshold   median steps   mean move
token              72       0.9396       0.9995        100.0%            4.0      0.1308
generated-search   72       0.9013       0.9985         95.8%           15.0      0.2297
```

Interpretation: the token proposal was closer, refined faster, and moved less.

### Three-Qubit Template Screen

We tested several 3-qubit CZ skeletons. The useful one was:

```text
line-4cz = CZ01, CZ12, CZ01, CZ12
```

In the template screen, this four-CZ line refined to high fidelity, while the
tested three-CZ templates did not.

### Three-Qubit Token Diffusion

For the current `SU(2)^15` 3-qubit workflow, one 48-heldout-target run gave:

```text
source             n    proposal   refined   >=threshold   median steps   mean move
token              48    0.5622    0.9812         83.3%           29.0      0.2047
generated-search   48    0.5130    0.9504         66.7%           32.0      0.2117
haar               48    0.1110    0.9525         77.1%           46.0      0.4063
```

The 3-seed repeatability result was:

```text
source             runs   n/run   proposal mean/std   refined mean/std   success mean/std   median steps   mean move
token              3      48        0.5389/0.0168    0.9851/0.0067     80.6%/7.1%             29.0      0.2167
generated-search   3      48        0.5103/0.0021    0.9450/0.0039     63.2%/6.4%             33.0      0.2233
haar               3      48        0.1089/0.0100    0.9609/0.0071     75.0%/2.9%             46.0      0.4123
```

Interpretation:

- token diffusion has better raw proposal quality than generated-search;
- token diffusion refines to a better final fidelity than generated-search;
- token diffusion reaches `F >= 0.99` more often than generated-search;
- token diffusion needs fewer steps than generated-search and much fewer than
  Haar;
- Haar can refine surprisingly well, but it moves much farther, which means the
  optimizer is doing more replacement than polishing.

This is the current strongest evidence that the diffusion model is learning
useful proposal geometry.

### Simple By-Hand Sanity Checks

The clearest "this is real" checkpoint is the optional Pauli sanity-check cell
in `SU2GateExperiments.ipynb`. It uses deliberately simple Hamiltonians where
the target evolution can be verified by hand, then compares the generated and
refined circuit against that exact answer.

For

```text
H = 0.5 XII,    t = 1
```

the target is

```text
U = exp(-i 0.5 XII).
```

Since `XII |000> = |100>`, the expected action is:

```text
exp(-i 0.5 XII) |000>
  = cos(0.5) |000> - i sin(0.5) |100>.
```

Numerically, the notebook check produced:

```text
manual / obvious:
|000> =  0.8776
|100> = -0.4794 i

generated/refined:
|000> =  0.8768 - 0.0008 i
|100> = -0.0016 - 0.4809 i

fidelity to exp(-i 0.5 XII): 0.9999704
```

The same check with

```text
H = 0.5 YII,    t = 1
```

uses `YII |000> = i |100>`, so

```text
exp(-i 0.5 YII) |000>
  = cos(0.5) |000> + sin(0.5) |100>.
```

The notebook check produced:

```text
manual:
|000> = 0.8776
|100> = 0.4794

generated/refined:
|000> = 0.8770 - 0.0015 i
|100> = 0.4805 - 0.0011 i

fidelity to exp(-i 0.5 YII): 0.9999725
```

These examples are useful because the generated circuit can look non-obvious
internally, but multiplying out all local `SU(2)` gates and CZs recovers the
same unitary as the hand-computed Hamiltonian evolution.

## What Failed Or Was Deprioritized

### Flat Hamiltonian-Conditioned MLP Denoiser

The first Hamiltonian-conditioned denoiser flattened all local gates and
Hamiltonian features into one vector. It mostly predicted near-zero tangent
vectors and did not beat simple baselines.

Scaling the flat MLP a little did not fix this. The token architecture mattered
more than ordinary width/step increases.

### Discrete Label Priors

We tried learned priors over Clifford-like slot labels. They could memorize or
help on small nearby distributions but did not transfer cleanly to harder
Hamiltonian distributions. Uniform generated search remained more principled
until we had a better dataset.

### Pure Lookup Table Concern

A hardcoded table of 24 Clifford gates is a strong baseline for exact Clifford
targets, and we should always compare against it when relevant.

The reason diffusion remains interesting is that Hamiltonian synthesis generally
needs continuous local corrections, not just exact Clifford choices. A slot may
live near a Clifford family but require a small continuous rotation that is not
in a finite table.

## Current Notebook Workflow

Main notebook:

```text
SU2GateExperiments.ipynb
```

Typical Colab setup:

```python
BRANCH = "main"  # or a codex/... branch when testing a PR
!pip install --no-cache-dir --force-reinstall --no-deps git+https://github.com/joe-singh/su2diffusion.git@{BRANCH}
```

For normal 3-qubit work:

1. Run install/import/device cells.
2. Run the local gate generation cells.
3. Run Level 3B for one 3-qubit token model.
4. Run Level 3C for refinement-basin comparison.

For paper-grade repeatability:

```python
RUN_LEVEL3D_THREE_QUBIT_REPEATABILITY = True
```

Level 3D internally reruns the train/evaluate/refine workflow across seeds, so
it is intentionally off by default.

## Key Files

```text
su2diffusion/quaternion.py
```

Quaternion arithmetic, exponential/log maps, Haar sampling, and `SU(2)`
distance.

```text
su2diffusion/diffusion.py
```

Diffusion schedules.

```text
su2diffusion/model.py
```

Denoiser architectures, including the circuit-token denoiser.

```text
su2diffusion/circuit.py
```

Circuit-level configs and product-manifold diffusion helpers.

```text
su2diffusion/synthesis.py
```

Unitary composition, fidelity scoring, candidate reports, and low-level
synthesis helpers.

```text
su2diffusion/hamiltonian.py
```

Hamiltonian targets, solution-stack dataset generation, token diffusion
training/evaluation, 2-qubit and 3-qubit Hamiltonian benchmarks, refinement, and
reporting.

```text
SU2GateExperiments.ipynb
```

Main Colab runner.

## Current Research Interpretation

The project now has a coherent story:

1. Use Lie-group diffusion to sample local `SU(2)` gates, not discrete circuit
   tensors.
2. Compose those gates into fixed entangling skeletons.
3. Train on solution stacks that actually synthesize Hamiltonian evolutions.
4. Use diffusion as a proposal model.
5. Use local `SU(2)` refinement as a polishing step.
6. Measure whether diffusion improves proposal quality, refinement success,
   optimization steps, and movement.

The strongest current claim is not:

```text
We have solved quantum compilation.
```

It is:

```text
For small Hamiltonian synthesis problems, Hamiltonian-conditioned diffusion on
SU(2)^n can learn proposal geometry that improves downstream continuous
refinement compared with generated-search and Haar baselines.
```

## Near-Term Next Steps

### 1. Turn The 3-Qubit Result Into A Clean Demo

The repo now has a first user-facing function/notebook cell:

```python
H = ...
t = ...
demo = run_three_qubit_hamiltonian_demo(...)
print_hamiltonian_demo(demo)
plot_hamiltonian_demo(demo)
```

It displays:

- Hamiltonian terms;
- chosen template;
- local gates before refinement;
- local gates after refinement;
- axis-angle descriptions of the refined local `SU(2)` gates;
- a fixed-CZ circuit skeleton;
- fidelity before/after;
- optimizer steps;
- movement.

### 2. Improve Visualizations

Useful paper/demo plots:

- proposal vs refined fidelity scatter;
- steps-to-threshold histogram;
- movement vs fidelity gain;
- per-target before/after lines;
- richer circuit diagrams with local gate annotations;
- Hamiltonian-to-circuit pipeline diagram.

### 3. Repeatability Beyond Three Seeds

The 3-seed result is useful. For a paper-grade table, eventually run more seeds
or larger heldout suites if compute permits.

### 4. Better Three-Qubit Training Scale

The 3-qubit model is promising but still modest. Potential next upgrades:

- more 3-qubit training Hamiltonians;
- more solution stacks per Hamiltonian;
- larger token model;
- longer training budget;
- better canonicalization/gauge fixing of solution stacks.

### 5. Template And Skeleton Selection

The current skeleton is fixed. The eventual compiler needs either:

- a small library of skeletons and a selector;
- a learned skeleton prior;
- or a search loop over skeleton families.

This is separate from local-gate diffusion.

### 6. Compare More Carefully To Classical Baselines

For credibility, keep strong baselines:

- Clifford table search;
- analytic near-Clifford sampler;
- generated local-gate search;
- Haar random starts;
- local optimizer from each start;
- deterministic formulas where they exist, especially for 2-qubit KAK.

The project should not claim novelty where deterministic 2-qubit synthesis is
already solved. The stronger angle is continuous Lie-group proposal learning,
especially as we move toward 3+ qubits and Hamiltonian families.

## Important Open Questions

1. Does the 3-qubit token advantage persist with more heldout targets and more
   random seeds?
2. Is the model learning Hamiltonian-conditioned structure or mostly learning a
   helpful prior over the chosen template?
3. How much of the improvement comes from better proposal fidelity versus better
   optimizer basin geometry?
4. What is the right canonicalization/gauge convention for many valid solution
   stacks?
5. When does model scaling beat data scaling?
6. What is the cleanest demonstration target for a paper or AI-for-science
   portfolio piece?

## If Restarting From A New Chat

Recommended first actions:

1. Open this memo.
2. Run:

   ```bash
   git -C /Users/joesingh/Desktop/su2diffusion status --short --branch
   git -C /Users/joesingh/Desktop/su2diffusion pull --ff-only
   ```

3. Inspect:

   ```text
   SU2GateExperiments.ipynb
   su2diffusion/hamiltonian.py
   README.md
   ```

4. Keep branches short.
5. Always provide both:

   ```text
   PR link
   Colab notebook link for the branch
   ```

6. Before merging code branches, run:

   ```bash
   .venv/bin/python -m pytest -q
   ```
