# Determinism & Seeding

This page explains how to make runs reproducible across Restoria and the
legacy GFPP CLI, what "deterministic" means per component, and the
practical limits when using CUDA.

## Quick basics

- Seed controls random, NumPy, and Torch RNGs.
- Deterministic CUDA constrains CuDNN to deterministic algorithms and disables
  benchmarking. This can slow down runs and reduce kernel choices.
- Planning and heuristics are designed to be deterministic for the same inputs
  and options.

## Restoria CLI

Restoria's `restoria run` provides explicit seeding and deterministic options:

- `--seed <int>`: seeds Python `random`, NumPy, and Torch (if available).
- `--deterministic`: enables CuDNN deterministic mode and disables benchmarking
  when Torch is present and CUDA is used.

Example:

```bash
restoria run \
  --input samples/portrait.jpg \
  --output out/ \
  --backend gfpgan \
  --device cuda \
  --seed 123 \
  --deterministic
```

Planning is deterministic: given the same image and flags (backend,
experimental, compile, and ORT providers), the same plan is produced. The seed
and deterministic choices are recorded in the run manifest.

Tips for reproducibility with Restoria:

- Prefer CPU for stricter determinism when comparing outputs across machines.
- Keep `--metrics` mode the same between runs; metrics affect what is computed
  but not the restoration itself.

## GFPP CLI (gfpup)

GFPP exposes seeding and deterministic controls directly.

- `--seed <int>`: sets seeds for Python `random`, NumPy, and Torch.
- `--deterministic`: enables deterministic CuDNN and disables benchmarking when
  using CUDA.

Example:

```bash
gfpup run \
  --input samples/portrait.jpg \
  --output out/ \
  --backend gfpgan \
  --device cuda \
  --seed 123 \
  --deterministic
```

Edge cases and notes:

- CPU runs are typically more reproducible across hardware and drivers.
- CUDA determinism may still vary across different GPU architectures, drivers,
  and Torch versions.
- Some backends may rely on operations that are not strictly deterministic
  under all conditions; in those cases, GFPP CLIs aim to constrain the most
  common sources of nondeterminism.

## Legacy GFPGAN CLI

For completeness, the legacy `gfpgan-infer` supports:

- `--seed <int>`
- `--deterministic-cuda` (enables deterministic CuDNN; slower)

Example:

```bash
gfpgan-infer -i samples/portrait.jpg -o results/ \
  --device cuda --seed 123 --deterministic-cuda
```

## Planner determinism

The orchestrator (planner) computes a stable Plan for a given image and
options. Heuristics use fixed thresholds and normalized scores to avoid minor
numeric jitter, e.g.:

- Quality scores rounded/standardized to canonical values (like defaulting
  weight to 0.6 for moderate degradation)
- Face detection and routing rules that do not depend on global randomness

This means repeated runs with the same inputs and flags will yield the same
`plan` object in the outputs (`metrics.json` includes a `plan` block in
Restoria, and manifests record args and runtime environment).

## Recommended workflow

- For rigorous A/B comparisons: use `gfpup run` with `--seed` and
  `--deterministic`, keep versions and hardware fixed, and compare outputs.
- For everyday use where exact bitwise determinism is not required: use
  `restoria run` and rely on deterministic planning and stable defaults.
