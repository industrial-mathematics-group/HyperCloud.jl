# Walls and Moving Boundaries

HyperCloudCore treats a boundary through ghost particles: a [`MirrorBC`](@extref HyperCloudCore HyperCloudCore.MirrorBC) evaluates the interior solution at a ghost's mirror point behind the wall and hands that state, with the wall's normal, position and velocity ([`WallPoint`](@extref HyperCloudCore HyperCloudCore.WallPoint)), to an *operator* that turns it into the ghost state. HyperCloud supplies the operators of the physical walls; the geometry (shapes, normals, moving pieces) lives in HyperCloudCore ([Geometric Domains](@extref HyperCloudCore Geometric-Domains), [Moving Boundaries](@extref HyperCloudCore Moving-Boundaries)).

| Condition | Equation | Ghost state |
|:--|:--|:--|
| `slip_wall()` (`MirrorBC(SlipWall())`) | Euler | density and pressure of the mirrored state, velocity reflected relative to the wall: `u' = u − 2 ((u − w)·n) n` |
| `slip_wall()` | BGK | specular reflection `f'(v) = f(v − 2 (v·n) n)` (exact node flip for a wall along an axis) |
| `diffuse_wall(θ_w; velocity, accommodation)` (`MirrorBC(DiffuseWall(…))`) | BGK | velocities out of the wall: the discrete Maxwellian at `θ_w` and the wall velocity, with the density that returns the incident mass flux; velocities into the wall: the mirrored interior values. `accommodation < 1` mixes in the specular reflection |

All of them need a shape-based domain (`:Grid_domain => :rectangular`, `:spherical` or `:shape`); the walls of a box are tagged `2d − 1` / `2d` per axis, so `:Grid_bc => Dict(1 => FixedDirichlet(), 2 => slip_wall())` is a tube with an inflow on the left and a wall on the right. `degree` and `bounded` of the underlying `MirrorBC` are keyword arguments of both constructors.

## Moving walls

A `Moving` piece of the shape carries its ghosts along and the cloud makes way for it (HyperCloudCore places the ghosts at every stage time and sweeps the interior after every step). The wall velocity reaches the operators through the `WallPoint`: the slip wall reflects relative to it, the diffuse wall emits at it. A piston driven into a gas at rest:

```julia
piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-0.5,)))   # face at x = 1 − 0.5 t
tube   = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), piston)                  # the canvas reaches beyond the face
params[:Grid_domain] = :shape
params[:Grid_shape]  = tube
params[:Grid_mins], params[:Grid_maxs] = (0.0,), (1.5,)
params[:Grid_bc]     = Dict(1 => slip_wall(), 2 => slip_wall())
```

The canvas must reach beyond a moving face: a face on the end of the canvas shares its ghosts with the static wall there, and the grid refuses it. Keep the step below a quarter spacing per step for the piece (`|w| Δt ≤ Δx / 4`; the CFL number does not see the wall speed).

## What was measured

- **Euler, shock reflection** (gas at `u = 1` against a slip wall, `MUSCL2 VK`, 200 particles, `t = 0.3`): the plateau behind the reflected shock is `ρ = 2.0774`, `p = 2.9267`, `u = 4·10⁻⁴` against the Rankine–Hugoniot state `2.0792`, `2.9266`, `0`.
- **Euler, piston** (`w = 0.5` into a gas at rest, 300 particles over `[0, 1.5]`, `t = 0.3`; the example [`euler_1d_piston`](@ref)): `ρ = 1.4895`, `u = −0.5000`, `p = 1.7602` against `1.4899`, `−0.5`, `1.7603`, and the first particle stays within one spacing of the face.
- **BGK, equilibrium**: a gas in equilibrium between diffuse walls at its temperature (also specular walls, or an accommodation of 0.5) stays in equilibrium to `2·10⁻¹⁶`: the discrete flux balance of the diffuse wall is exact on the symmetric velocity grid.
- **BGK, hot wall** (`θ_w = 2` on a gas at `θ = 1`, `τ = 10⁻²`, `t = 0.5`): the temperature next to the wall reaches 1.83 / 1.86 / 1.87 for 50 / 100 / 200 particles (a temperature jump remains at this Knudsen number); the mass of the gas drifts by 0.74 / 0.50 / 0.32 %, so the implied wall flux of the collocation scheme is consistent at first order only.
- **BGK, piston** (`τ = 10⁻⁴`, isothermal diffuse face at the gas temperature, 300 particles): the plateau is within 1 % of the Euler Rankine–Hugoniot state (`ρ = 1.482`, `p = 1.747`, `u = −0.493`); the isothermal face adds a thin thermal layer that the Euler plateau does not have.

## API

```@docs
SlipWall
DiffuseWall
slip_wall
diffuse_wall
euler_1d_piston
exact_piston
```
