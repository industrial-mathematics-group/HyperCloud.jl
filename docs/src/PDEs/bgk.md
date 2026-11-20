# BGK Equation

[`BGKEquation`](@ref) is a discrete-velocity model of the Bhatnagar–Gross–Krook equation

```math
\partial_t f + \vec v\cdot\nabla_x f = \frac{1}{\tau}\big(\mathcal M[f] - f\big),
```

for the distribution ``f(\vec x, \vec v, t)`` of a gas in ``D`` space dimensions. Its moments
``U = (\rho, \rho\vec u, E)`` are the conservative variables of the Euler equations, which the BGK
solution approaches for ``\tau \to 0``; for ``\tau \to \infty`` every velocity moves freely.

## Velocity space as the state vector

The velocity space is discretised by a tensor grid of `Nv` nodes per direction
([`velocity_grid`](@ref)), and the particle state is the vector of distribution values at the
nodes: the number of equations is the number of discrete velocities. Everything HyperCloudCore
offers (MUSCL, limiters, MOOD, IMEX, moving grids, the device backends) therefore runs on the BGK
model unchanged. On a moving grid the Rusanov flux of the frame (`ALEFrame`, `LagrangianFrame`) keeps the
per-velocity upwind dissipation, `|v_k − w|` per component (the generic Rusanov of the frame dissipated every
velocity with the largest relative speed and smeared the contact of the Sod on Lagrangian particles). Moments are quadrature sums,
``U = \sum_k w_k\,(1, \vec v_k, \tfrac12|\vec v_k|^2)\,f_k``.

Since the state is an `SVector`, its length drives compile time and per-particle cost. The model
keeps it small in three ways:

- **Only `D` velocity directions.** Degrees of freedom beyond the resolved velocity directions (the
  remaining directions of a 3-D gas and internal ones, ``K = 2/(\gamma - 1) - D`` in total) are
  carried by a second distribution ``h = \int \tfrac12|\xi|^2 f\,d\xi`` on the same nodes (a Chu
  reduction). The state is ``f`` for ``\gamma = (D + 2)/D`` (``M = N_v^D``), otherwise ``(f, h)``
  (``M = 2 N_v^D``). A 1-D shock tube with ``\gamma = 1.4`` needs a 1-D velocity grid, not ``N_v^3``
  velocities.
- **Gauss–Hermite nodes** (`:velocity_grid => :gauss_hermite`) integrate polynomials times the
  Gaussian of their scale exactly, so near-equilibrium flows need far fewer nodes than the uniform
  midpoint grid (`:uniform`, the robust choice for strongly non-equilibrium flows).
- **No per-velocity arrays in kernels.** The equation is a small bits type (1-D nodes and weights,
  grid centre, ``K``, ``\gamma``, relaxation model); the node of a component is resolved from its index
  at compile time, so the flux and the dissipation unroll into straight-line code.

## Collision term

The equilibrium is the conservative discrete Maxwellian of [`equilibrium`](@ref)
(``f_k = \exp(\alpha\cdot\psi_k)`` with ``\psi = (1, \vec v, \tfrac12|\vec v|^2)``, Mieussens'
discrete entropic equilibrium, ``h_k = \tfrac K2\,\hat\theta f_k``): ``\alpha`` is found by a damped
Newton iteration on the convex dual problem, started from the continuous Maxwellian, so the discrete
moments of the equilibrium equal ``U`` to round-off and the collision term conserves mass, momentum
and energy exactly. A few microseconds per particle; the iteration runs inside the kernels.

The relaxation time comes from a [`RelaxationTimeModel`](@ref): constant ([`ConstantRelaxationTime`](@ref),
`Inf` for free transport) or ``\tau = \mu(\theta)/p`` with a power-law viscosity
([`ViscosityRelaxationTime`](@ref)).

With an IMEX tableau the collision is the implicit source [`BGKRelaxation`](@ref). Because it
conserves the moments, the implicit stage equation has the closed-form solution
``f = (f^* + \lambda\,\mathcal M[f^*])/(1 + \lambda)``, ``\lambda = \Delta t\,a_{ii}/\tau``, for any
stiffness, which makes the scheme asymptotic preserving: at ``\tau = 10^{-5}`` the BGK Sod solution
is as close to the exact Euler solution as the Euler scheme itself, and ``\tau = 0``
(``\lambda = \infty``) gives ``\mathcal M[f^*]``. With an explicit tableau the collision becomes
[`ExplicitBGKRelaxation`](@ref) (stable for ``\Delta t \lesssim \tau`` only; ``\tau = 0`` is
rejected with an `ArgumentError`). Negative relaxation times are rejected when the equation is
built.

## Transport

The flux is ``v_{k,d} f_k``. The transport is linear and diagonal, so the Rusanov flux of every
velocity is its exact upwind flux: `RusanovFlux`, `UpwindFlux` and `AlignedRusanovFlux` use the
dissipation ``|v_{k,d}|`` per component instead of the largest node speed for all components (which
smears the slow velocities; on a smooth collisionless wave it costs a factor 2 to 10 in the error).
`max_eigenvalue` is the largest node speed, which sets the CFL time step.

The PAD of MOOD (`is_admissible`) asks for positive density and temperature; individual ``f_k`` may
dip below zero.

## Configuration

```julia
:PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 1.4,
:PDE_Nv => 24, :PDE_vmax => 6.0, :PDE_velocity_grid => :uniform,
:PDE_tau => 1e-3,                       # or :PDE_viscosity => μ_ref, :PDE_omega => ω
:sim_func_name => :run_bgk_simulation,  # direct and kinetic runners forward BGK runs here
```

Initial conditions give the macroscopic state ``(\rho, \rho\vec u, E)``, exactly as for an Euler
run with the same ``\gamma``; every particle starts in the discrete Maxwellian of that state. The
stored snapshots are the moments, so BGK and Euler runs can be compared directly
(`:save_relax => true` keeps the distributions). Choose `vmax` so that the Maxwellians of the flow
decay inside the grid (``|u| + 5\sqrt\theta`` or more) and enough nodes to resolve them
(``\Delta v \lesssim \sqrt\theta``).

References: the collisionless model has the exact solution of [`analytic_closure`](@ref)
(each velocity transported, initial data in equilibrium); for small ``\tau`` use
[`euler_limit_solution`](@ref) as the `reference` of a `SimulationConfig` (see the example
`bgk_1d_sod`).

## Cost of the velocity space

Measured on 8 threads (ARS222, MUSCL3 with the Venkatakrishnan limiter, regular lattice, `Float64`),
one time step and the first (compiling) step:

| D | M | particles | step / particle | first step | per-particle buffers | neighbour-slot buffers |
|:--|--:|--:|--:|--:|--:|--:|
| 1 | 16 | 4000 | 0.32 µs | 3 s | 1.9 KiB | 10 × 0.27 KiB |
| 1 | 64 | 4000 | 1.6 µs | 5 s | 6.8 KiB | 10 × 1.0 KiB |
| 1 | 128 | 4000 | 3.2 µs | 11 s | 13 KiB | 10 × 2.0 KiB |
| 1 | 256 | 4000 | 7.1 µs | 30 s | 26 KiB | 10 × 4.0 KiB |
| 2 | 16 | 64² | 2.4 µs | 5 s | 3.1 KiB | 36 × 0.41 KiB |
| 2 | 64 | 64² | 5.8 µs | 15 s | 9.1 KiB | 36 × 1.5 KiB |
| 2 | 144 | 64² | 16 µs | 40 s | 19 KiB | 36 × 3.4 KiB |
| 2 | 256 | 64² | 30 µs | 95 s | 33 KiB | 36 × 6.0 KiB |

The cost per particle grows linearly with ``M`` up to a few hundred equations. In 2-D the per-slot
buffers of the scheme (the flux differences of every neighbour, ``(D + 1) M`` values per slot)
dominate the memory, about 6× the per-particle buffers, and the compile time of the first step grows
faster than ``M``; ``16 \times 16`` velocities still work (95 s to compile, 30 µs per particle and
step). About a third of a step is the collision (two equilibria per stage), a third the gradient
pre-pass with the limiter, and the rest the flux pass.

## Walls

`diffuse_wall(θ_w; velocity, accommodation)` is the Maxwell (diffuse) wall, `slip_wall()` the specular one, both
operators of `MirrorBC`; see [Walls and Moving Boundaries](@ref). A gas in equilibrium at the wall temperature is
reflected into itself exactly.

## Block form for large velocity grids

Beyond a few hundred components the `SVector` state is too large (see the cost table and HyperCloudCore's
"Block States" page). [`BGKBlocks`](@ref)`(eq, Val(W))` is the same BGK model as a block equation of
HyperCloudCore's block pipeline: the `M` components of the state of `eq` (the `f` values, then the `h` values
of the Chu reduction) split into `M / W` blocks of `W` consecutive components, each transported on its own
with the per-velocity upwind dissipation of its velocities, and the collision couples them through the
moments. [`BGKBlockRelaxation`](@ref) is the collision term in two passes: the reduction over the blocks
of a particle gives the moments, the relaxation time and the exponential parameters ``\alpha`` of the
conservative discrete Maxwellian ([`equilibrium_parameters`](@ref)); the update of a block then evaluates
the Maxwellian component by component ([`equilibrium_component`](@ref)), `f = (f* + λ M) / (1 + λ)` as
for the single state. Nothing of length `M` is ever built. A block run reproduces the `SVector` run
(`BGKRelaxation`, same scheme and tableau) to round-off.

```julia
eq  = BGKEquation(Val(1), nodes, weights, 1.4, ConstantRelaxationTime(1e-3))    # M = 2 Nv
bk  = BGKBlocks(eq, Val(8))                                                    # blocks of 8 components
pg  = ParticleGrid(geom, (dx,), weight, 8)
bs  = BlockState(bk, pg)
set_block_states!((x, b) -> equilibrium_block(bk, U0(x), b), bs, pg)          # the Maxwellian of the macroscopic IC
ts  = BlockTimeStepper(bk, MUSCL(Float64, 1, 8, 2, VenkatakrishnanLimiter(:hard), RusanovFlux()),
                       (BlockSourceTerm(BGKBlockRelaxation(bk)),), IMEX_ARS222_Tableau(Float64))
ts(bk, pg, bs, t, dt)
U = block_moments(bk, bs, i)                                                   # (ρ, ρu, E) of particle i
```

From a configuration, `:PDE_block_width => W` runs the block form through `run_bgk_simulation` (also for a
mixture, [`MixtureBlocks`](@ref) with [`MixtureBlockRelaxation`](@ref): `W` must divide every species' state
length; `:PDE_blocks_per_launch` sets how many blocks one launch of the scheme's pre-pass and flux pass covers,
HyperCloudCore's `blocks_per_launch`, the scratch of those passes being replicated that many times; MUSCL streams its
divergence there unless `:Scheme_streamed => false`, see [Configuration](../configuration.md)); the snapshots
hold the macroscopic moments; `:PDE_devices` splits the blocks over the GPUs of one node, every device holding the
whole cloud and a range of blocks and the devices exchanging the moments of their blocks per stage (the collision's
parameters, the frame velocity, the admissibility; the kinetic walls' ghost rows), see HyperCloudCore's `BlockDevices`; `:save_relax` is refused, MOOD works with every criterion (the
admissibility across the blocks is the positivity of the moments), a moving
grid runs in the ALE frame with the mean velocity of the moments as the particle velocity (the Lagrangian movers;
the conservative Lagrangian frame of the single state has no block form), and the kinetic walls act in two passes
(the zero-flux density of a diffuse wall from all velocities; the specular reflection as the exact node flip where
it applies and as the reflection off the node grid otherwise, as on the single state). Moving boundary pieces (a
piston) and the particle adaptivity work on every block; the storage follows the backend (the parameter arrays of the
collision included), and `set_memory_limit!` reserves the block stepper and the block state (the state arrays count
`nb` elements per particle). The collision's reduction is
given in the additive form of HyperCloudCore's `block_partial` (the moments of a block range), which a block state split
over devices exchanges.

## Gas mixtures

[`BGKMixture`](@ref) is the discrete-velocity model of a mixture of ``L`` species with the multi-species
BGK operator of Bobylev, Bisi, Groppi, Spiga and Potapenko (BBGSP),

```math
\partial_t f_s + \vec v\cdot\nabla_x f_s
= \frac{1}{\varepsilon}\,\nu_{ss}\,(n_s M_{ss} - f_s)
+ \frac{1}{\kappa}\sum_{r\ne s}\nu_{sr}\,(n_s M_{sr} - f_s), \qquad s = 1,\dots,L,
```

with the number densities ``n_s``, the collision frequencies ``\nu_{sr} = \lambda_{sr} n_r`` (a symmetric
matrix of constants ``\lambda``: Maxwell molecules), the intra-species Knudsen number ``\varepsilon`` and the
inter-species one ``\kappa``. ``M_{sr}`` is the unit-density Maxwellian of species ``s`` (mass ``m_s``) at the
mixture velocity ``\vec U_{sr} = (1 - a_{sr})\vec U_s + a_{sr}\vec U_r`` and temperature
``T_{sr} = (1 - b_{sr}) T_s + b_{sr} T_r + \gamma_{sr}|\vec U_s - \vec U_r|^2/k_B`` with
``a_{sr} = m_r/(m_s + m_r)``, ``b_{sr} = 2 a_{sr} m_s/(m_s + m_r)`` and ``\gamma_{sr} = m_s a_{sr}^2/n_{\mathrm{dof}}``
(``n_{\mathrm{dof}} = D + K``, 3 for a monatomic gas): the choice for which the operator conserves every species'
number density and the total momentum and energy.

- **One state, species blocks.** Every species is a [`BGKEquation`](@ref) on its own velocity grid (a light
  species needs a wider one; the Chu reduction and the node tables are per species), with the thermal speed
  ``\theta_s = k_B T_s/m_s`` as its temperature. The particle state is the concatenation of the species'
  states, ``M = \sum_s M_s``, and the transport stays diagonal, so the flux, the per-velocity Rusanov
  dissipation (static and in the moving frames, which move with the mass-averaged mixture velocity), MOOD and
  the walls (`slip_wall()` species by species, `diffuse_wall(T_w)` at the thermodynamic wall temperature,
  emitting species ``s`` at ``k_B T_w/m_s``) carry over from the single gas. The macroscopic state of
  [`moments`](@ref) is the species' number moments ``(n_s, n_s\vec u_s, e_s)`` one after the other
  (``L(D + 2)`` numbers; [`mixture_state`](@ref) builds it from ``n_s``, ``\vec u_s``, ``T_s``,
  [`species_primitives`](@ref) reads it back, [`mixture_moments`](@ref) gives ``(\rho, \rho\vec U, E)``).
- **Collision step.** [`MixtureRelaxation`](@ref) solves the implicit stage equation in closed form, as the
  relaxation step of the paper: the species densities are invariant, so ``\nu_{sr}`` is known; the species
  velocities and temperatures after the stage come from two ``L \times L`` symmetric positive definite
  systems (the graph Laplacians of ``d_{sr} = \lambda_{sr} n_s n_r m_s m_r/(m_s + m_r)`` and of
  ``c_{sr} = n_{\mathrm{dof}} k_B \lambda_{sr} n_s n_r m_s m_r/(m_s + m_r)^2``, the latter with the exchanged
  kinetic energy on the right-hand side), which conserve the totals exactly; the Maxwellians are then the
  conservative discrete equilibria of each species at those values (discrete moments exact, so the
  moment correction of the paper is not needed), and
  ``f_s = (f_s^* + \Delta t\,a_{ii}\sum_r (\nu_{sr}/\varepsilon_{sr})\,n_s M_{sr}) / (1 + \Delta t\,a_{ii}\sum_r \nu_{sr}/\varepsilon_{sr})``.
  Exact for any stiffness in ``\varepsilon`` and ``\kappa`` (asymptotic preserving); with an explicit tableau
  [`ExplicitMixtureRelaxation`](@ref). Checked: the species densities, total momentum and energy are kept to
  round-off per step and over a periodic run; a one-species mixture reproduces `BGKRelaxation` with
  ``\tau = \varepsilon/(\lambda n)`` to round-off, and two identical species with equal states are the two
  halves of that single gas (also over a run, with ``\tau = \mu/\rho``, ``\mu = \varepsilon/\lambda``).
- **Units.** ``m_s``, ``k_B`` and ``\lambda`` are plain numbers. Dimensionless problems use ``k_B = 1``;
  with the molar masses in g/mol as ``m_s`` and ``k_B = R = 8.3145`` temperatures are in kelvin and velocities
  in units of ``\sqrt{1000}\,\mathrm{m/s} \approx 31.6\,\mathrm{m/s}`` (He–Ar: ``\lambda_{11} = 19.96``,
  ``\lambda_{12} = 1.153``, ``\lambda_{22} = 17.52``).

```julia
:PDE_name => :bgk_mixture, :PDE_D => 1,
:PDE_masses => (4.0, 40.0), :PDE_lambda => [19.96 1.153; 1.153 17.52], :PDE_kB => 8.3145,
:PDE_epsilon => 1e-3, :PDE_kappa => 1e-3,
:PDE_Nv => 30, :PDE_vmax => (125.0, 60.0),       # one velocity grid per species
:sim_func_name => :run_bgk_simulation,
```

Initial conditions give the species' number moments, e.g. `:IC_uL => Tuple(mixture_state(eq, n, u, T))`;
the stored snapshots are the species' number moments. The state length grows with the number of species
and their grids (two species with 30² nodes and the Chu reduction are 3600 components per particle, far
beyond the table above): start with 1-D problems and measure.

## API

```@docs
BGKEquation
HyperCloudCore.is_admissible(::BGKEquation{D, M, T}, ::State{M, T}) where {D, M, T}
velocity_grid
num_velocities
velocity_node
moments
equilibrium
euler_limit
RelaxationTimeModel
ConstantRelaxationTime
ViscosityRelaxationTime
relaxation_time
BGKRelaxation
ExplicitBGKRelaxation
equilibrium_parameters
equilibrium_component
BGKBlocks
equilibrium_block
block_moments
BGKBlockRelaxation
ExplicitBGKBlockRelaxation
MixtureBlocks
species_of_block
HyperCloud.species_block_moments
MixtureBlockRelaxation
ExplicitMixtureBlockRelaxation
BGKMixture
HyperCloudCore.is_admissible(::BGKMixture{D, M, T}, ::State{M, T}) where {D, M, T}
num_species
species_moments
mixture_moments
species_primitives
mixture_state
MixtureRelaxation
ExplicitMixtureRelaxation
run_bgk_simulation
euler_limit_solution
bgk_1d_sod
```
