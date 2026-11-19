# =========================================================================
# Kinetic walls in the block pipeline: the two passes of HyperCloudCore's block `MirrorBC`
# =========================================================================
# The mirrored interior state of every block is in the scratch array `Y` (block b of ghost g at Y[(b - 1) N + g]); the
# reduction over the blocks gives what a wall needs of the whole distribution (the zero-flux density of a diffuse wall,
# the specular parameters), the application gives one block's ghost state. The specular reflection is the exact node
# flip where it applies (an axis-aligned wall whose normal velocity is the grid centre: the mirrored value of the flipped
# velocity, read from its block) and otherwise the reflection off the node grid of `_specular` (the equilibrium of the
# reflected moments plus the interpolated non-equilibrium part, flux-scaled on the emitted velocities).

# --- the specular reflection ------------------------------------------------------------------------------------------
# The parameters of one species at a ghost: the exact node flip (an axis-aligned wall whose normal velocity is the grid
# centre), or the reflection off the node grid of `_specular`: the discrete equilibrium of the reflected moments (α_r)
# plus the non-equilibrium part of the mirrored distribution (f − M[α_f]) interpolated at the reflected velocity,
# scaled by `scale` on the emitted velocities so that the mass flux through the wall vanishes
struct SpecularParams{P, T}
    ax::Int
    flip::Bool
    use_eq::Bool
    alpha_f::SVector{P, T}
    alpha_r::SVector{P, T}
    scale::T
end

@inline _node_flip(eq::BGKEquation, n::Space, w::Space) = (ax = _axis_of(n); ax > 0 && (@inbounds eq.center[ax] == w[ax]) ? (ax, true) : (ax, false))

# The mirrored value of the component with the velocity of `k` flipped along `ax`, read from its block (`boff`: the blocks
# before this species, 0 for a single gas)
@inline function _flipped_value(eq::BGKEquation, Y, N::Int, g::Int, k::Int, ax::Int, boff::Int, ::Val{W}) where {W}
    kp = _flipped_index(eq, k, ax)
    return @inbounds Y[(boff + (kp - 1) ÷ W) * N + g][(kp - 1) % W + 1]
end

# The non-equilibrium part of the mirrored distribution (`f` for `off = 0`, `h` for `off = NV`) at the velocity `v`,
# interpolated multilinearly on the node grid, read from the blocks (as `_interp_f` on the single state)
@inline function _interp_neq_block(eq::BGKEquation{D, M, T, NV1, NV}, Y, N::Int, g::Int, boff::Int, use_eq::Bool, α::SVector{P, T},
                                   v::Space{D, T}, off::Int, ::Val{W}) where {D, M, T, NV1, NV, P, W}
    br  = ntuple(d -> _bracket(eq.nodes, v[d] - eq.center[d]), Val(D))
    acc = zero(T)
    @inbounds for corner in 0:(2^D - 1)
        k, wt = 1, one(T)
        for d in 1:D
            hi = (corner >> (d - 1)) & 1 == 1
            i, t = br[d]
            k  += (hi ? i : i - 1) * NV1^(d - 1)
            wt *= hi ? t : one(T) - t
        end
        kk = off + k
        fk = Y[(boff + (kk - 1) ÷ W) * N + g][(kk - 1) % W + 1]
        acc += wt * (use_eq ? fk - equilibrium_component(eq, α, kk) : fk)
    end
    return acc
end

# The reflected value of component `k` before the flux scaling (clipped at zero, as on the single state)
@inline function _specular_raw(eq::BGKEquation{D, M, T, NV1, NV}, Y, N::Int, g::Int, boff::Int, prm::SpecularParams, k::Int,
                               w::Space{D, T}, n::Space{D, T}, ::Val{W}) where {D, M, T, NV1, NV, W}
    kk = k > NV ? k - NV : k
    v, _ = velocity_node(eq, kk)
    c = sum((v - w) .* n)
    base = prm.use_eq ? equilibrium_component(eq, prm.alpha_r, k) : zero(T)
    return max(zero(T), base + _interp_neq_block(eq, Y, N, g, boff, prm.use_eq, prm.alpha_f, v - (2c) * n, k > NV ? NV : 0, Val(W)))
end

# The specular parameters of a species from the moments `U` of its mirrored distribution
@inline function _specular_params(eq::BGKEquation{D, M, T, NV1, NV}, U::SVector{P, T}, Y, N::Int, g::Int, boff::Int,
                                  w::Space{D, T}, n::Space{D, T}, ::Val{W}) where {D, M, T, NV1, NV, P, W}
    ax, flip = _node_flip(eq, n, w)
    flip && return SpecularParams{P, T}(ax, true, false, zero(SVector{P, T}), zero(SVector{P, T}), one(T))
    ρ  = U[1]
    u  = Space{D, T}(ntuple(d -> U[1 + d], Val(D))) / ρ
    ur = u - (2 * sum((u - w) .* n)) * n
    Ur = SVector{D + 2, T}(ρ, (ρ * ur)..., U[D + 2] + T(0.5) * ρ * (sum(abs2, ur) - sum(abs2, u)))
    α_f, ok1 = equilibrium_parameters(eq, U)
    α_r, ok2 = equilibrium_parameters(eq, Ur)
    prm0 = SpecularParams{P, T}(ax, false, ok1 && ok2, α_f, α_r, one(T))
    # Mass flux through the wall: the incident part of the mirrored state, the emitted part of the reflection
    Φin, Φout = zero(T), zero(T)
    @inbounds for k in 1:NV
        v, wk = velocity_node(eq, k)
        c = sum((v - w) .* n)
        if c > zero(T)
            Φin += wk * c * Y[(boff + (k - 1) ÷ W) * N + g][(k - 1) % W + 1]
        else
            Φout += wk * c * _specular_raw(eq, Y, N, g, boff, prm0, k, w, n, Val(W))
        end
    end
    s = Φout < zero(T) ? max(Φin, zero(T)) / (-Φout) : one(T)
    return SpecularParams{P, T}(ax, false, ok1 && ok2, α_f, α_r, s)
end

# The specular ghost value of component `k` of a species
@inline function _specular_component(eq::BGKEquation{D, M, T, NV1, NV}, Y, N::Int, g::Int, boff::Int, prm::SpecularParams, k::Int,
                                     w::Space{D, T}, n::Space{D, T}, ::Val{W}) where {D, M, T, NV1, NV, W}
    prm.flip && return _flipped_value(eq, Y, N, g, k, prm.ax, boff, Val(W))
    kk = k > NV ? k - NV : k
    v, _ = velocity_node(eq, kk)
    raw = _specular_raw(eq, Y, N, g, boff, prm, k, w, n, Val(W))
    return sum((v - w) .* n) < zero(T) ? prm.scale * raw : raw
end

HyperCloudCore.wall_reduction_type(::SlipWall, ::BGKBlocks{D, W, T}) where {D, W, T} = SpecularParams{D + 2, T}
@inline function HyperCloudCore.block_wall_reduce(::SlipWall, bk::BGKBlocks{D, W, T}, Y, N, nb, g, wall) where {D, W, T}
    return _specular_params(bk.eq, block_moments(bk, Y, N, nb, g), Y, N, g, 0, wall.velocity, wall.normal, Val(W))
end
@inline function HyperCloudCore.block_wall_apply(::SlipWall, bk::BGKBlocks{D, W, T}, U::State{W, T}, b::Int, prm::SpecularParams, wall, Y, N, nb, g) where {D, W, T}
    off = (b - 1) * W
    return State{W, T}(ntuple(j -> _specular_component(bk.eq, Y, N, g, 0, prm, off + j, wall.velocity, wall.normal, Val(W)), Val(W)))
end

# The mixture: the parameters of every species
struct MixtureSpecularParams{L, P, T}
    species::SVector{L, SpecularParams{P, T}}
end
HyperCloudCore.wall_reduction_type(::SlipWall, ::MixtureBlocks{D, W, T, NB, L}) where {D, W, T, NB, L} = MixtureSpecularParams{L, D + 2, T}
@inline function _mixture_specular_params(mb::MixtureBlocks{D, W, T, NB, L}, Y, N::Int, g::Int, w::Space{D, T}, n::Space{D, T}) where {D, W, T, NB, L}
    res = ntuple(Val(L)) do s
        boff = @inbounds mb.block_offsets[s]
        _select(sp -> _specular_params(sp, _species_moments_from_blocks(sp, Y, N, g, boff, Val(W)), Y, N, g, boff, w, n, Val(W)), mb.mix.species, s)
    end
    return MixtureSpecularParams{L, D + 2, T}(SVector{L, SpecularParams{D + 2, T}}(res))
end
@inline HyperCloudCore.block_wall_reduce(::SlipWall, mb::MixtureBlocks, Y, N, nb, g, wall) =
    _mixture_specular_params(mb, Y, N, g, wall.velocity, wall.normal)
@inline function HyperCloudCore.block_wall_apply(::SlipWall, mb::MixtureBlocks{D, W, T}, U::State{W, T}, b::Int, prm::MixtureSpecularParams, wall, Y, N, nb, g) where {D, W, T}
    s, bl = species_of_block(mb, b)
    off  = (bl - 1) * W
    boff = @inbounds mb.block_offsets[s]
    ps   = @inbounds prm.species[s]
    return _select(sp -> State{W, T}(ntuple(j -> _specular_component(sp, Y, N, g, boff, ps, off + j, wall.velocity, wall.normal, Val(W)), Val(W))), mb.mix.species, s)
end

# --- the diffuse (Maxwell) wall ---------------------------------------------------------------------------------------
# The parameters of a ghost: the exponential parameters of the unit-density wall Maxwellian of every species, the density
# that returns the incident flux, whether the wall Maxwellian exists, and the node flip for the specular share
struct DiffuseBlockParams{L, P, T}
    alpha::SVector{L, SVector{P, T}}
    rho_w::SVector{L, T}
    ok::Bool
    spec::SVector{L, SpecularParams{P, T}}       # the specular share of an accommodation below one
end

# The flux balance of species `eq` (its blocks start at `boff`): the incident flux of the mirrored distribution and the
# emitted flux of the unit-density wall Maxwellian with the parameters α; the returning density is their ratio
@inline function _block_wall_density(eq::BGKEquation{D, M, T, NV1, NV}, α::SVector{P, T}, Y, N::Int, g::Int, boff::Int, w::Space{D, T},
                                     n::Space{D, T}, ::Val{W}) where {D, M, T, NV1, NV, P, W}
    Φin, Φout = zero(T), zero(T)
    # (the constants of `equilibrium_component`, once instead of a `log` per node: the same values)
    θh = -one(T) / α[P]
    ch = T(0.5) * eq.K
    lognorm = ch * log(T(2π) * θh)
    @inbounds for k in 1:NV
        v, wk = velocity_node(eq, k)
        c = sum((v - w) .* n)
        if c > zero(T)
            Φin += wk * c * Y[(boff + (k - 1) ÷ W) * N + g][(k - 1) % W + 1]
        else
            Φout += wk * c * _equilibrium_value(eq, α, k, θh, ch, lognorm)
        end
    end
    return Φout < zero(T) ? max(Φin, zero(T)) / (-Φout) : zero(T)
end

# One block of the ghost state of species `eq`: incident components keep the mirrored value, the others take the wall
# Maxwellian (and the specular share for an accommodation below one)
@inline function _diffuse_block(eq::BGKEquation{D, M, T, NV1, NV}, U::State{W, T}, off::Int, α::SVector{P, T}, ρw::T, acc::T,
                                spec::SpecularParams, Y, N::Int, g::Int, boff::Int, w::Space{D, T}, n::Space{D, T}) where {D, M, T, NV1, NV, W, P}
    return State{W, T}(ntuple(Val(W)) do j
        k  = off + j
        kk = k > NV ? k - NV : k
        v, _ = velocity_node(eq, kk)
        c = sum((v - w) .* n)
        @inbounds if c > zero(T)
            U[j]
        else
            diffuse = ρw * equilibrium_component(eq, α, k)
            acc < one(T) ? acc * diffuse + (one(T) - acc) * _specular_component(eq, Y, N, g, boff, spec, k, w, n, Val(W)) : diffuse
        end
    end)
end

# The specular share of a diffuse wall with an accommodation below one (nothing to reflect otherwise)
# (`moms()` gives the moments of the incident state, evaluated only for an accommodation below one: a full pass over the
# ghost's blocks that a fully diffuse wall does not need)
@inline function _diffuse_specular(wd::DiffuseWall, eq::BGKEquation{D, M, T}, moms::F, Y, N::Int, g::Int, boff::Int,
                                   w::Space{D, T}, n::Space{D, T}, ::Val{W}) where {D, M, T, F, W}
    P = D + 2
    T(wd.accommodation) < one(T) || return SpecularParams{P, T}(0, false, false, zero(SVector{P, T}), zero(SVector{P, T}), one(T))
    return _specular_params(eq, moms(), Y, N, g, boff, w, n, Val(W))
end

HyperCloudCore.wall_reduction_type(::DiffuseWall, ::BGKBlocks{D, W, T}) where {D, W, T} = DiffuseBlockParams{1, D + 2, T}
# (the seven-argument form balances at the mirror point, as the operator on its own)
@inline HyperCloudCore.block_wall_reduce(wd::DiffuseWall, bk::BGKBlocks, Y, N, nb, g, wall) = HyperCloudCore.block_wall_reduce(wd, bk, Y, N, nb, g, wall, Y)
@inline function HyperCloudCore.block_wall_reduce(wd::DiffuseWall, bk::BGKBlocks{D, W, T}, Y, N, nb, g, wall, Yw) where {D, W, T}
    eq = bk.eq
    n  = wall.normal
    w  = wall.velocity + _slip_velocity(wd, Val(D), T)
    θw = T(wd.temperature)
    Uw = SVector{D + 2, T}(one(T), w..., T(0.5) * sum(abs2, w) + T(0.5) * (T(D) + eq.K) * θw)
    α, ok = equilibrium_parameters(eq, Uw)
    ρw = ok ? _block_wall_density(eq, α, Yw, N, g, 0, w, n, Val(W)) : zero(T)      # (the incident flux at the wall point)
    spec = _diffuse_specular(wd, eq, @inline(() -> block_moments(bk, Y, N, nb, g)), Y, N, g, 0, w, n, Val(W))
    return DiffuseBlockParams{1, D + 2, T}(SVector{1}(α), SVector{1}(ρw), ok, SVector{1}(spec))
end
@inline function HyperCloudCore.block_wall_apply(wd::DiffuseWall, bk::BGKBlocks{D, W, T}, U::State{W, T}, b::Int, prm::DiffuseBlockParams, wall, Y, N, nb, g) where {D, W, T}
    prm.ok || return U
    w = wall.velocity + _slip_velocity(wd, Val(D), T)
    return _diffuse_block(bk.eq, U, (b - 1) * W, prm.alpha[1], prm.rho_w[1], T(wd.accommodation), prm.spec[1], Y, N, g, 0, w, wall.normal)
end

HyperCloudCore.wall_reduction_type(::DiffuseWall, ::MixtureBlocks{D, W, T, NB, L}) where {D, W, T, NB, L} = DiffuseBlockParams{L, D + 2, T}
@inline HyperCloudCore.block_wall_reduce(wd::DiffuseWall, mb::MixtureBlocks, Y, N, nb, g, wall) = HyperCloudCore.block_wall_reduce(wd, mb, Y, N, nb, g, wall, Y)
@inline function HyperCloudCore.block_wall_reduce(wd::DiffuseWall, mb::MixtureBlocks{D, W, T, NB, L}, Y, N, nb, g, wall, Yw) where {D, W, T, NB, L}
    mix = mb.mix
    n   = wall.normal
    w   = wall.velocity + _slip_velocity(wd, Val(D), T)
    Tw  = T(wd.temperature)
    res = ntuple(Val(L)) do s
        boff = @inbounds mb.block_offsets[s]
        θw   = mix.kB * Tw / (@inbounds mix.masses[s])
        _select(mix.species, s) do sp
            Uw = SVector{D + 2, T}(one(T), w..., T(0.5) * sum(abs2, w) + T(0.5) * (T(D) + sp.K) * θw)
            α, ok = equilibrium_parameters(sp, Uw)
            spec = _diffuse_specular(wd, sp, @inline(() -> _species_moments_from_blocks(sp, Y, N, g, boff, Val(W))), Y, N, g, boff, w, n, Val(W))
            (α, ok ? _block_wall_density(sp, α, Yw, N, g, boff, w, n, Val(W)) : zero(T), ok, spec)
        end
    end
    return DiffuseBlockParams{L, D + 2, T}(SVector{L}(map(first, res)), SVector{L}(map(r -> r[2], res)), all(map(r -> r[3], res)),
                                            SVector{L, SpecularParams{D + 2, T}}(map(last, res)))
end
@inline function HyperCloudCore.block_wall_apply(wd::DiffuseWall, mb::MixtureBlocks{D, W, T}, U::State{W, T}, b::Int, prm::DiffuseBlockParams, wall, Y, N, nb, g) where {D, W, T}
    prm.ok || return U
    s, bl = species_of_block(mb, b)
    boff  = @inbounds mb.block_offsets[s]
    w     = wall.velocity + _slip_velocity(wd, Val(D), T)
    α, ρw, spec = (@inbounds prm.alpha[s]), (@inbounds prm.rho_w[s]), (@inbounds prm.spec[s])
    return _select(sp -> _diffuse_block(sp, U, (bl - 1) * W, α, ρw, T(wd.accommodation), spec, Y, N, g, boff, w, wall.normal), mb.mix.species, s)
end
