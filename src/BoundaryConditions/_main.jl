export SlipWall, DiffuseWall, slip_wall, diffuse_wall

# =========================================================================
# WALL OPERATORS for HyperCloudCore's MirrorBC
# =========================================================================
#
# `MirrorBC(op; degree, bounded)` evaluates the interior solution at a ghost particle's mirror point behind the
# wall and stores `op(eq, U, wall::WallPoint)`. The operators below turn that mirrored state into the ghost state
# of a physical wall: `wall.normal` is the outward unit normal (into the solid), `wall.velocity` the wall's
# velocity there (zero for a static wall, the piece's velocity for a `Moving` one). They are bits types and run
# inside the kernel.

# -------------------------------------------------------------------------
# Slip wall: reflection of the velocity relative to the wall (Euler), specular reflection (BGK)
# -------------------------------------------------------------------------
"""
    SlipWall()

Operator of a reflecting (slip, adiabatic) wall for [`MirrorBC`](@extref HyperCloudCore HyperCloudCore.MirrorBC):
[`slip_wall`](@ref) builds the condition.

- **Euler:** the mirrored state keeps its density and pressure; its velocity is reflected relative to the wall,
  `u' = u - 2 ((u - w) ⋅ n) n` with the wall velocity `w` (zero for a static wall), so the normal velocity at the
  wall equals the wall's and the tangential one is kept.
- **BGK:** specular reflection of the distribution, `f'(v) = f(v - 2 ((v - w) ⋅ n) n)`. Exact on the tensor
  velocity grid when the reflection maps the grid onto itself: a wall normal along an axis whose normal wall
  velocity equals the grid centre on that axis (`:PDE_vcenter`; a static wall on a grid centred at 0), where the
  node index is flipped. Otherwise (oblique or curved walls, moving walls, a shifted grid) the reflected velocity
  is not a node: the reflection is split into the discrete equilibrium of the reflected macroscopic state
  (density and internal energy kept, velocity reflected as for Euler), which is exact, and the non-equilibrium
  rest of the distribution, interpolated (multilinearly on the tensor grid) at the reflected velocities. The
  components leaving the wall are then scaled so that the net mass flux through the wall vanishes, as for
  [`DiffuseWall`](@ref). A gas in equilibrium (at rest next to a curved wall, or drifting along an oblique one)
  is reflected exactly; the interpolation error of the non-equilibrium part is second order in the node spacing.
"""
struct SlipWall end

@inline function (::SlipWall)(eq::EulerEquation{D, M, T, <:Conservative}, U::State{M, T}, wall::WallPoint{D, T}) where {D, M, T}
    V  = cons2prim(eq, U)
    u  = Space{D, T}(ntuple(d -> V[1 + d], Val(D)))
    n  = wall.normal
    un = sum((u - wall.velocity) .* n)
    u2 = u - (2 * un) * n
    return prim2cons(eq, State{M, T}(V[1], u2..., V[M]))
end

# The axis a unit normal is aligned with, 0 if none
@inline function _axis_of(n::Space{D, T}) where {D, T}
    ax = 0
    for d in 1:D
        abs(abs(n[d]) - one(T)) <= T(1e-6) && (ax = d)
    end
    return ax
end

# Component whose velocity is that of `k` with the node index along axis `d` flipped (symmetric node sets)
@inline function _flipped_index(eq::BGKEquation{D, M, T, NV1, NV}, k::Int, d::Int) where {D, M, T, NV1, NV}
    idx  = _node_index(eq, k, d)
    return k + (NV1 + 1 - 2 * idx) * NV1^(d - 1)
end

# Bracketing node index `i` (nodes[i] ≤ x ≤ nodes[i+1], clamped to the grid) and the linear weight of node i + 1
@inline function _bracket(nodes::SVector{NV1, T}, x::T) where {NV1, T}
    i = 1
    @inbounds while i < NV1 - 1 && nodes[i + 1] < x
        i += 1
    end
    @inbounds t = clamp((x - nodes[i]) / (nodes[i + 1] - nodes[i]), zero(T), one(T))
    return i, t
end

# f (block starting at `off`: 0 for f, NV for h) interpolated multilinearly at the velocity `v` on the tensor grid
@inline function _interp_f(eq::BGKEquation{D, M, T, NV1, NV}, f::State{M, T}, v::Space{D, T}, off::Int) where {D, M, T, NV1, NV}
    br = ntuple(d -> _bracket(eq.nodes, v[d] - eq.center[d]), Val(D))
    acc = zero(T)
    @inbounds for corner in 0:(2^D - 1)
        k, wt = 1, one(T)
        for d in 1:D
            hi = (corner >> (d - 1)) & 1 == 1
            i, t = br[d]
            k  += (hi ? i : i - 1) * NV1^(d - 1)
            wt *= hi ? t : one(T) - t
        end
        acc += wt * f[off + k]
    end
    return acc
end

@inline function _specular(eq::BGKEquation{D, M, T, NV1, NV}, f::State{M, T}, n::Space{D, T}, w::Space{D, T}) where {D, M, T, NV1, NV}
    ax = _axis_of(n)
    # Exact node flip: along an axis the reflection v_ax -> 2 w_ax - v_ax maps the (symmetric) grid onto itself if
    # the grid is centred at the wall's normal velocity (it used to be taken for any static wall, which reflected
    # about the grid centre: with :PDE_vcenter != 0 the wall was penetrable)
    if ax > 0 && eq.center[ax] == w[ax]
        return State{M, T}(ntuple(k -> (@inbounds f[_flipped_index(eq, k, ax)]), Val(M)))
    end
    # Otherwise the components leaving the wall ((v - w)⋅n < 0) take the reflection of f at their reflected velocity:
    # the discrete equilibrium of the reflected macroscopic state (density and internal energy kept, velocity
    # reflected relative to the wall) plus the non-equilibrium part of f interpolated (multilinearly) at the reflected
    # velocity, then scaled so that the mass flux through the wall vanishes. The equilibrium part is reflected
    # exactly, so a gas in equilibrium (at rest next to a curved wall, or drifting) is reflected into the correct
    # equilibrium; the nearest node, used before, lost or gained mass (a gas at rest next to a disc drifted by 9 % in
    # t = 0.1). If a discrete equilibrium does not exist, the whole of f is interpolated.
    U  = moments(eq, f)
    ρ  = U[1]
    u  = Space{D, T}(ntuple(d -> U[1 + d], Val(D))) / ρ
    ur = u - (2 * sum((u - w) .* n)) * n
    Ur = SVector{D + 2, T}(ρ, (ρ * ur)..., U[D + 2] + T(0.5) * ρ * (sum(abs2, ur) - sum(abs2, u)))
    Mf, ok1 = equilibrium(eq, U)
    Mr, ok2 = equilibrium(eq, Ur)
    use_eq  = ok1 && ok2
    neq = use_eq ? f - Mf : f
    raw = State{M, T}(ntuple(Val(M)) do k
        kk = k > NV ? k - NV : k
        v, _ = velocity_node(eq, kk)
        c = sum((v - w) .* n)
        (use_eq ? (@inbounds Mr[k]) : zero(T)) + _interp_f(eq, neq, v - (2c) * n, k > NV ? NV : 0)
    end)
    # Mass flux through the wall: incident part of the interior (mirrored) state, emitted part of the reflection
    Φin, Φout = zero(T), zero(T)
    @inbounds for k in 1:NV
        v, wk = velocity_node(eq, k)
        c = sum((v - w) .* n)
        c > zero(T) ? (Φin += wk * c * f[k]) : (Φout += wk * c * raw[k])
    end
    s = Φout < zero(T) ? max(Φin, zero(T)) / (-Φout) : one(T)
    return State{M, T}(ntuple(Val(M)) do k
        kk = k > NV ? k - NV : k
        v, _ = velocity_node(eq, kk)
        @inbounds sum((v - w) .* n) < zero(T) ? s * raw[k] : raw[k]
    end)
end

@inline (::SlipWall)(eq::BGKEquation{D, M, T}, f::State{M, T}, wall::WallPoint{D, T}) where {D, M, T} =
    _specular(eq, f, wall.normal, wall.velocity)

# -------------------------------------------------------------------------
# Diffuse (Maxwell) wall for the BGK model
# -------------------------------------------------------------------------
"""
    DiffuseWall(temperature; velocity = nothing, accommodation = 1)

Operator of a diffusely reflecting wall at the temperature `θ_w = p / ρ` for the BGK model, for
[`MirrorBC`](@extref HyperCloudCore HyperCloudCore.MirrorBC) ([`diffuse_wall`](@ref) builds the condition). The
components of the ghost state whose velocity points out of the wall into the fluid (relative to the wall,
`(v - w) ⋅ n < 0`) are the discrete Maxwellian at the wall temperature and velocity, with the density that
makes the mass flux through the wall vanish (the incident flux of the mirrored interior distribution returns);
the components pointing into the wall keep the mirrored interior values. The wall velocity `w` is the piece's
velocity from the `WallPoint` plus `velocity` (a tangential sliding velocity, e.g. the plates of a Couette
flow; a tuple of length `D`). With an `accommodation` coefficient `α < 1` (Maxwell's model) the emitted
distribution is `α` times the diffuse one plus `1 - α` times the specular reflection of [`SlipWall`](@ref).

Isothermal: a wall at the gas temperature reflects an equilibrium gas at rest into itself exactly (the discrete
flux balance is exact on the symmetric velocity grid), a colder or hotter wall drives a thermal boundary layer.
Internal degrees of freedom (`h`) are emitted at the wall temperature as well.
"""
struct DiffuseWall{T, V}
    temperature::T
    velocity::V
    accommodation::T
end
function DiffuseWall(temperature::Real; velocity = nothing, accommodation::Real = 1)
    temperature > 0 || throw(ArgumentError("DiffuseWall: the wall temperature must be positive"))
    0 <= accommodation <= 1 || throw(ArgumentError("DiffuseWall: the accommodation coefficient must be in [0, 1]"))
    T = float(typeof(temperature))
    vel = velocity === nothing ? nothing : SVector{length(velocity), T}(velocity...)
    return DiffuseWall{T, typeof(vel)}(T(temperature), vel, T(accommodation))
end
@inline _slip_velocity(::DiffuseWall{Tw, Nothing}, ::Val{D}, ::Type{T}) where {Tw, D, T} = zero(Space{D, T})
@inline _slip_velocity(w::DiffuseWall{Tw, <:SVector{D}}, ::Val{D}, ::Type{T}) where {Tw, D, T} = Space{D, T}(w.velocity)

@inline function (wd::DiffuseWall)(eq::BGKEquation{D, M, T, NV1, NV}, f::State{M, T}, wall::WallPoint{D, T}) where {D, M, T, NV1, NV}
    n  = wall.normal
    w  = wall.velocity + _slip_velocity(wd, Val(D), T)
    θw = T(wd.temperature)
    α  = T(wd.accommodation)
    # Unit-density discrete Maxwellian of the wall
    Uw = SVector{D + 2, T}(one(T), w..., T(0.5) * sum(abs2, w) + T(0.5) * (T(D) + eq.K) * θw)
    Mw, ok = equilibrium(eq, Uw)
    ok || return f
    # Flux balance of the relative normal velocity: what the fluid sends into the wall comes back
    Φin, Φout = zero(T), zero(T)
    @inbounds for k in 1:NV
        v, wk = velocity_node(eq, k)
        c = sum((v - w) .* n)
        if c > zero(T)
            Φin += wk * c * f[k]
        else
            Φout += wk * c * Mw[k]
        end
    end
    ρw  = Φout < zero(T) ? max(Φin, zero(T)) / (-Φout) : zero(T)
    spec = α < one(T) ? _specular(eq, f, n, w) : f
    return State{M, T}(ntuple(Val(M)) do k
        kk = k > NV ? k - NV : k
        v, _ = velocity_node(eq, kk)
        c = sum((v - w) .* n)
        @inbounds c > zero(T) ? f[k] : α * ρw * Mw[k] + (one(T) - α) * spec[k]
    end)
end

# -------------------------------------------------------------------------
# Conditions
# -------------------------------------------------------------------------
"""
    slip_wall(; degree = 1, bounded = true) -> MirrorBC

A reflecting wall (Euler: slip, adiabatic; BGK: specular): `MirrorBC(SlipWall(); degree, bounded)`, see
[`SlipWall`](@ref) and [`MirrorBC`](@extref HyperCloudCore HyperCloudCore.MirrorBC). Needs a shape-based domain
(wall normals); follows a `Moving` piece.
"""
slip_wall(; degree::Int = 1, bounded::Bool = true) = MirrorBC(SlipWall(); degree = degree, bounded = bounded)

"""
    diffuse_wall(temperature; velocity = nothing, accommodation = 1, degree = 1, bounded = true) -> MirrorBC

A diffusely reflecting kinetic wall: `MirrorBC(DiffuseWall(temperature; velocity, accommodation); degree, bounded)`,
see [`DiffuseWall`](@ref).
"""
diffuse_wall(temperature::Real; velocity = nothing, accommodation::Real = 1, degree::Int = 1, bounded::Bool = true) =
    MirrorBC(DiffuseWall(temperature; velocity = velocity, accommodation = accommodation); degree = degree, bounded = bounded)
