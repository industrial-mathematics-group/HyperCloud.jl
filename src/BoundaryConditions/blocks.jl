# =========================================================================
# Kinetic walls in the block pipeline: the two passes of HyperCloudCore's block `MirrorBC`
# =========================================================================
# The mirrored interior state of every block is in the scratch array `Y` (block b of ghost g at Y[(b - 1) N + g]); the
# reduction over the blocks gives what a wall needs of the whole distribution (the zero-flux density of a diffuse wall,
# whether the specular node flip applies), the application gives one block's ghost state. The exact node flip of the
# specular reflection (an axis-aligned wall whose normal velocity is the grid centre) reads the mirrored value of the
# flipped velocity from its block; for an oblique or curved wall the block form has no specular reflection yet (the
# runner refuses it), the operators fall back to the even reflection there.

# --- the specular reflection ------------------------------------------------------------------------------------------
struct SlipBlockParams
    ax::Int
    flip::Bool
end

@inline _node_flip(eq::BGKEquation, n::Space, w::Space) = (ax = _axis_of(n); ax > 0 && (@inbounds eq.center[ax] == w[ax]) ? (ax, true) : (ax, false))

# The mirrored value of the component with the velocity of `k` flipped along `ax`, read from its block (`boff`: the blocks
# before this species, 0 for a single gas)
@inline function _flipped_value(eq::BGKEquation, Y, N::Int, g::Int, k::Int, ax::Int, boff::Int, ::Val{W}) where {W}
    kp = _flipped_index(eq, k, ax)
    return @inbounds Y[(boff + (kp - 1) ÷ W) * N + g][(kp - 1) % W + 1]
end

HyperCloudCore.wall_reduction_type(::SlipWall, ::BGKBlocks) = SlipBlockParams
@inline function HyperCloudCore.block_wall_reduce(::SlipWall, bk::BGKBlocks, Y, N, nb, g, wall)
    ax, flip = _node_flip(bk.eq, wall.normal, wall.velocity)
    return SlipBlockParams(ax, flip)
end
@inline function HyperCloudCore.block_wall_apply(::SlipWall, bk::BGKBlocks{D, W, T}, U::State{W, T}, b::Int, prm::SlipBlockParams, wall, Y, N, nb, g) where {D, W, T}
    prm.flip || return U
    off = (b - 1) * W
    return State{W, T}(ntuple(j -> _flipped_value(bk.eq, Y, N, g, off + j, prm.ax, 0, Val(W)), Val(W)))
end

HyperCloudCore.wall_reduction_type(::SlipWall, ::MixtureBlocks) = SlipBlockParams
@inline function HyperCloudCore.block_wall_reduce(::SlipWall, mb::MixtureBlocks{D, W, T, NB, L}, Y, N, nb, g, wall) where {D, W, T, NB, L}
    ax = _axis_of(wall.normal)
    flip = ax > 0 && all(map(sp -> (@inbounds sp.center[ax] == wall.velocity[ax]), mb.mix.species))
    return SlipBlockParams(ax, flip)
end
@inline function HyperCloudCore.block_wall_apply(::SlipWall, mb::MixtureBlocks{D, W, T}, U::State{W, T}, b::Int, prm::SlipBlockParams, wall, Y, N, nb, g) where {D, W, T}
    prm.flip || return U
    s, bl = species_of_block(mb, b)
    off  = (bl - 1) * W
    boff = @inbounds mb.block_offsets[s]
    return _select(sp -> State{W, T}(ntuple(j -> _flipped_value(sp, Y, N, g, off + j, prm.ax, boff, Val(W)), Val(W))), mb.mix.species, s)
end

# --- the diffuse (Maxwell) wall ---------------------------------------------------------------------------------------
# The parameters of a ghost: the exponential parameters of the unit-density wall Maxwellian of every species, the density
# that returns the incident flux, whether the wall Maxwellian exists, and the node flip for the specular share
struct DiffuseBlockParams{L, P, T}
    alpha::SVector{L, SVector{P, T}}
    rho_w::SVector{L, T}
    ok::Bool
    ax::Int
    flip::Bool
end

# The flux balance of species `eq` (its blocks start at `boff`): the incident flux of the mirrored distribution and the
# emitted flux of the unit-density wall Maxwellian with the parameters α; the returning density is their ratio
@inline function _block_wall_density(eq::BGKEquation{D, M, T, NV1, NV}, α::SVector{P, T}, Y, N::Int, g::Int, boff::Int, w::Space{D, T},
                                     n::Space{D, T}, ::Val{W}) where {D, M, T, NV1, NV, P, W}
    Φin, Φout = zero(T), zero(T)
    @inbounds for k in 1:NV
        v, wk = velocity_node(eq, k)
        c = sum((v - w) .* n)
        if c > zero(T)
            Φin += wk * c * Y[(boff + (k - 1) ÷ W) * N + g][(k - 1) % W + 1]
        else
            Φout += wk * c * equilibrium_component(eq, α, k)
        end
    end
    return Φout < zero(T) ? max(Φin, zero(T)) / (-Φout) : zero(T)
end

# One block of the ghost state of species `eq`: incident components keep the mirrored value, the others take the wall
# Maxwellian (and the specular share for an accommodation below one)
@inline function _diffuse_block(eq::BGKEquation{D, M, T, NV1, NV}, U::State{W, T}, off::Int, α::SVector{P, T}, ρw::T, acc::T,
                                flip::Bool, ax::Int, Y, N::Int, g::Int, boff::Int, w::Space{D, T}, n::Space{D, T}) where {D, M, T, NV1, NV, W, P}
    return State{W, T}(ntuple(Val(W)) do j
        k  = off + j
        kk = k > NV ? k - NV : k
        v, _ = velocity_node(eq, kk)
        c = sum((v - w) .* n)
        @inbounds if c > zero(T)
            U[j]
        else
            diffuse = ρw * equilibrium_component(eq, α, k)
            acc < one(T) ? acc * diffuse + (one(T) - acc) * (flip ? _flipped_value(eq, Y, N, g, k, ax, boff, Val(W)) : U[j]) : diffuse
        end
    end)
end

HyperCloudCore.wall_reduction_type(::DiffuseWall, ::BGKBlocks{D, W, T}) where {D, W, T} = DiffuseBlockParams{1, D + 2, T}
@inline function HyperCloudCore.block_wall_reduce(wd::DiffuseWall, bk::BGKBlocks{D, W, T}, Y, N, nb, g, wall) where {D, W, T}
    eq = bk.eq
    n  = wall.normal
    w  = wall.velocity + _slip_velocity(wd, Val(D), T)
    θw = T(wd.temperature)
    Uw = SVector{D + 2, T}(one(T), w..., T(0.5) * sum(abs2, w) + T(0.5) * (T(D) + eq.K) * θw)
    α, ok = equilibrium_parameters(eq, Uw)
    ρw = ok ? _block_wall_density(eq, α, Y, N, g, 0, w, n, Val(W)) : zero(T)
    ax, flip = _node_flip(eq, n, w)
    return DiffuseBlockParams{1, D + 2, T}(SVector{1}(α), SVector{1}(ρw), ok, ax, flip)
end
@inline function HyperCloudCore.block_wall_apply(wd::DiffuseWall, bk::BGKBlocks{D, W, T}, U::State{W, T}, b::Int, prm::DiffuseBlockParams, wall, Y, N, nb, g) where {D, W, T}
    prm.ok || return U
    w = wall.velocity + _slip_velocity(wd, Val(D), T)
    return _diffuse_block(bk.eq, U, (b - 1) * W, prm.alpha[1], prm.rho_w[1], T(wd.accommodation), prm.flip, prm.ax, Y, N, g, 0, w, wall.normal)
end

HyperCloudCore.wall_reduction_type(::DiffuseWall, ::MixtureBlocks{D, W, T, NB, L}) where {D, W, T, NB, L} = DiffuseBlockParams{L, D + 2, T}
@inline function HyperCloudCore.block_wall_reduce(wd::DiffuseWall, mb::MixtureBlocks{D, W, T, NB, L}, Y, N, nb, g, wall) where {D, W, T, NB, L}
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
            (α, ok ? _block_wall_density(sp, α, Y, N, g, boff, w, n, Val(W)) : zero(T), ok)
        end
    end
    ax = _axis_of(n)
    flip = ax > 0 && all(map(sp -> (@inbounds sp.center[ax] == w[ax]), mix.species))
    return DiffuseBlockParams{L, D + 2, T}(SVector{L}(map(first, res)), SVector{L}(map(r -> r[2], res)), all(map(last, res)), ax, flip)
end
@inline function HyperCloudCore.block_wall_apply(wd::DiffuseWall, mb::MixtureBlocks{D, W, T}, U::State{W, T}, b::Int, prm::DiffuseBlockParams, wall, Y, N, nb, g) where {D, W, T}
    prm.ok || return U
    s, bl = species_of_block(mb, b)
    boff  = @inbounds mb.block_offsets[s]
    w     = wall.velocity + _slip_velocity(wd, Val(D), T)
    α, ρw = (@inbounds prm.alpha[s]), (@inbounds prm.rho_w[s])
    return _select(sp -> _diffuse_block(sp, U, (bl - 1) * W, α, ρw, T(wd.accommodation), prm.flip, prm.ax, Y, N, g, boff, w, wall.normal), mb.mix.species, s)
end
