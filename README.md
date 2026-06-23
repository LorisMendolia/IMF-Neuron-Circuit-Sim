# IMF Neuron Circuit Sim

Julia simulations of a current-mode ("I-mode") subthreshold CMOS neuromorphic neuron circuit, built up from a transistor-level sigmoid block to single-neuron and small-network spiking/bursting dynamics.

This code simulates the neuron circuit presented in:

> L. Mendolia, C. Wen, E. Chicca, G. Indiveri, R. Sepulchre, J.-M. Redouté, A. Franci, *"A Neuromodulable Current-Mode Silicon Neuron for Robust and Adaptive Neuromorphic Systems"*, [arXiv:2512.01133](https://arxiv.org/abs/2512.01133).

## Contents

- **`imode_sigmoid.jl`** — Core module. Models the current-mode sigmoid circuit (7 subthreshold transistors) at the transistor equation level, finds its DC steady states with [BifurcationKit.jl](https://github.com/bifurcationkit/BifurcationKit.jl), and exposes `Imode_sigmoid_eval(Iin, params)` as a fast, memoized/interpolated input→output current characteristic for reuse in larger ODE models.
- **`Neuron.ipynb`** — Single-neuron models built from the sigmoid block:
  - A minimal **spiking** model (one sigmoid, two filters).
  - A full **bursting** model (two sigmoids with gain inactivation across three timescales), matching the on-chip circuit design.
  - Simulated traces and steady-state I-V curves for both.
- **`Network.ipynb`** — A 4-neuron central pattern generator (CPG) built from the bursting neuron, coupled through a synaptic connectivity matrix.
- **`sigmoid_BK.ipynb`** — Standalone, annotated derivation of the sigmoid circuit equations and its bifurcation-based steady-state solution.
- **`cmode_sigmoid_ode.ipynb`** — Earlier exploratory approach: finding the sigmoid's steady-state by direct time-domain ODE integration (kept for reference; superseded by the BifurcationKit approach).

## Requirements

Julia with: `DifferentialEquations`, `BifurcationKit`, `NLsolve`, `Interpolations`, `Parameters`, `Plots`, `LaTeXStrings`.

## Usage

Open `Neuron.ipynb` or `Network.ipynb` in Jupyter (with IJulia) and run the cells — both `include("imode_sigmoid.jl")` to reuse the sigmoid model. `sigmoid_BK.ipynb` and `cmode_sigmoid_ode.ipynb` are self-contained.

## License

MIT — see [LICENSE](LICENSE).
