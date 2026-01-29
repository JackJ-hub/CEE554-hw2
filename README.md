# CEE554-hw2
## CEE 554 — Homework 2 Summary

### Problem 1: Linear Regression for Classification (Dataset A)
- Trained a linear regression classifier on Dataset A.
- Result: `W = [-2.06799143, 0.17494643, 0.20105202]`, `Ein = 0.0`.
- Used the linear-regression weight as PLA initialization and compared with random initialization:
  - LR init: converged in `0` updates (already perfectly classified).
  - Random init (seed=0): converged in `75` updates.

### Problem 2: Nonlinear Transformations
- Simulated the target function `sign(x1^2 + x2^2 - 0.6)` with 10% label noise on `N=1000` points.
- (a) No feature transform using `(1, x1, x2)`:
  - Average `Ein ≈ 0.503509` over 1000 runs.
- (b) Applied nonlinear transform `(1, x1, x2, x1x2, x1^2, x2^2)`:
  - Average `Ein ≈ 0.118859` over 1000 runs.
  - Average `W ≈ [-1.0026, 0.0015, -0.0019, 0.0005, 1.5766, 1.5757]`.
  - Closest provided hypothesis: **(i)**.

### Problem 3: Hoeffding Inequality Simulation
- Simulated 1000 fair coins, 10 flips each, repeated 100,000 times.
- Computed distributions for:
  - `v1` (first coin), `vrand` (random coin), `vmin` (coin with minimum heads fraction).
- Results:
  - `E[v1] ≈ 0.50007`, `E[vrand] ≈ 0.500048`, `E[vmin] ≈ 0.037377`.
  - Closest choice for average `vmin`: **(ii) 0.01**.
- Hoeffding inequality (single-coin setting) applies to **c1 and crand**, but not **cmin** (selected based on outcomes).
