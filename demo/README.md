# Independent Synthetic Demo

This directory contains an independent educational reimplementation written
from scratch in 2026 from the public mathematical description in *Moving
Target Monte Carlo*.

It does not contain, import, or depend on source code, datasets, configuration
files, numerical models, or other materials from the original research
environment. All geometry, observations, and noise are generated locally by
the demo.

## Problem

The example uses a four-cell velocity model and 68 synthetic rays crossing the
domain horizontally and vertically. Straight-ray travel times are calculated
with independent midpoint quadrature, and Gaussian noise is added using a
reproducible random seed.

The forward relation is linear in cell slowness, so an analytical Gaussian
posterior is available as a reference. The demo compares that reference with:

- conventional independent-proposal Metropolis-Hastings; and
- nearest-neighbour MTMC with the same proposal and the Voronoi-cell
  approximation described in
  Section 3.2 of the paper.

## Run

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m demo.run_demo
```

A shorter non-plotting run is:

```powershell
python -m demo.run_demo --steps 800 --burn-in 200 --no-plot
```

Generated metrics, samples, and figures are written below `outputs/`, which is
excluded from version control.

The default deterministic reference run uses 12,000 true target evaluations
for conventional Metropolis-Hastings and 1,644 for MTMC. The synthetic target
is inexpensive, so this example treats evaluation count rather than wall-clock
time as the relevant comparison.

## Tests

```powershell
python -m unittest discover -s demo/tests -v
```

The tests check ray-length conservation, full model rank, posterior covariance,
deterministic sampling, and the defining evaluation-count behavior of MTMC.
