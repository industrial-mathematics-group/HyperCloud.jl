<div align="center">

<img alt="HyperCloud.jl" src="logos/svg/logo.svg#gh-light-mode-only" width="480">
<img alt="HyperCloud.jl" src="logos/svg/logo-dark.svg#gh-dark-mode-only" width="480">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23266176.svg)](https://doi.org/10.5281/zenodo.23266176)
[![Build Status](https://github.com/industrial-mathematics-group/HyperCloud.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/industrial-mathematics-group/HyperCloud.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/industrial-mathematics-group/HyperCloud.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/industrial-mathematics-group/HyperCloud.jl)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://industrial-mathematics-group.github.io/HyperCloud.jl/dev/)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://industrial-mathematics-group.github.io/HyperCloud.jl/stable/)

[![AI generation (files)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/industrial-mathematics-group/HyperCloud.jl/badges/aidecl-generation.json)](https://github.com/industrial-mathematics-group/aidecl)
[![Verification (files)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/industrial-mathematics-group/HyperCloud.jl/badges/aidecl-verification.json)](https://github.com/industrial-mathematics-group/aidecl)
</div>

**HyperCloud.jl** is an advanced meshfree simulation framework designed to solve arbitrary hyperbolic conservation laws. Powered by a high-performance, physics-agnostic mathematical engine, HyperCloud provides a complete computational environment. It combines dynamic particle topologies with high-order spatial approximations—including moving-least-squares MUSCL reconstruction and Multi-Dimensional Optimal Order Detection (MOOD)—controlled entirely through a streamlined, macro-driven execution workflow.

HyperCloud integrates natively with the **PDEStudio** ecosystem to deliver a frictionless research pipeline from computation to visualization. All simulation outputs are rigorously managed and serialized via `PDEStudioCore.jl`, providing reproducible, cryptographically hashed disk caching and highly optimized statistical integration. For visual analysis, HyperCloud interfaces directly with `PDEStudio.jl`, allowing you to instantly route your meshfree Lagrangian datasets to a high-performance local renderer (`GLMakie`) or serve interactive web visualizations directly from a headless compute node to your browser (`WGLMakie` via Bonito).

## Quick Start Tutorial

To run any of the built-in examples, you only need to load the framework alongside a Makie backend and the `PDEStudio` visualization suite. 

To execute the simulation and instantly visualize the results in an interactive window, use the `@plot` macro with the uncalled example function name:

```julia
using HyperCloud
using PDEStudio
using GLMakie # Or WGLMakie for browser-based rendering

# Launch the interactive plotter for the 1D Sod Shock Tube
@plot euler_1d_sod
```

If you want to run the simulation headlessly without visualization (e.g., for benchmarking or cluster execution), use the `@sim` macro instead:

```julia
using HyperCloud

# Execute the simulation headlessly
@sim euler_1d_sod
```
## License
The code is licensed under the MIT License (see `LICENSE`).
The logos in `logos/` are licensed under CC BY 4.0 (see `logos/LICENSE`).
"HyperCloud" and the HyperCloud logos identify this project; please don't use them in a way that suggests endorsement.

## AI Declaration

Portions of this codebase and documentation were drafted by Claude Opus 5.5 and Fable 5.1 using Claude Code. All code has been human-reviewed, verified, and tested. If you notice any inaccuracies or unexpected behavior, please open an issue.
