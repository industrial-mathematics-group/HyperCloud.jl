export MixtureBlocks, species_of_block

# =========================================================================
# THE GAS MIXTURE AS A BLOCK EQUATION
# =========================================================================
"""
    MixtureBlocks(mix::BGKMixture, ::Val{W}) <: HyperbolicPDE{D, W, T}

The mixture of [`BGKMixture`](@ref) as a block equation of HyperCloudCore's block pipeline: the state of every
species (`M_s` components, `f` then the Chu `h`) splits into `M_s / W` blocks of `W` consecutive components, the
blocks of species 1 first; `W` must divide every `M_s`, so no block straddles two species.
[`species_of_block`](@ref) maps a block to its species and its position there. Transport is per block with the
per-velocity upwind dissipation of the block's velocities; the collision ([`MixtureBlockRelaxation`](@ref))
couples all blocks through the species' moments. [`block_moments`](@ref) reduces the species' number moments
(the macroscopic state of [`moments`](@ref)) from a `BlockState`.
"""
struct MixtureBlocks{D, W, T, NB, L, MIX <: BGKMixture{D}} <: HyperbolicPDE{D, W, T}
    mix::MIX
    block_offsets::SVector{L, Int}      # blocks before species s
    nblocks::SVector{L, Int}            # blocks of species s
end

function MixtureBlocks(mix::BGKMixture{D, M, T, L, S}, ::Val{W}) where {D, M, T, L, S, W}
    (W isa Int && W >= 1) || throw(ArgumentError("MixtureBlocks: the block width must be a positive Int, got $W"))
    sizes = _block_sizes(S)
    for (s, Ms) in enumerate(sizes)
        Ms % W == 0 || throw(ArgumentError("MixtureBlocks: the block width $W does not divide the state length $Ms of species $s"))
    end
    nbs  = SVector{L, Int}(ntuple(s -> sizes[s] ÷ W, L))
    offs = SVector{L, Int}(ntuple(s -> sum(nbs[1:(s - 1)]; init = 0), L))
    return MixtureBlocks{D, W, T, sum(nbs), L, typeof(mix)}(mix, offs, nbs)
end

HyperCloudCore.num_blocks(::MixtureBlocks{D, W, T, NB}) where {D, W, T, NB} = NB
num_species(mb::MixtureBlocks) = num_species(mb.mix)
num_velocities(mb::MixtureBlocks) = num_velocities(mb.mix)

"""
    species_of_block(mb::MixtureBlocks, b::Int) -> (s, bl)

The species `s` of block `b` and the block's index `bl` among the blocks of that species (its components are
`(bl − 1) W + 1 … bl W` of the species' state).
"""
@inline function species_of_block(mb::MixtureBlocks{D, W, T, NB, L}, b::Int) where {D, W, T, NB, L}
    s = 1
    @inbounds while s < L && b > mb.block_offsets[s + 1]
        s += 1
    end
    return s, b - (@inbounds mb.block_offsets[s])
end

# `f(species[s])` for a run-time `s` without a dynamic lookup in the heterogeneous species tuple: an unrolled chain,
# type stable when `f` returns the same type for every species
@inline _select(f::F, sp::Tuple{Any}, s::Int) where {F} = f(sp[1])
@inline _select(f::F, sp::Tuple, s::Int) where {F} = s == 1 ? f(sp[1]) : _select(f, Base.tail(sp), s - 1)

# Velocities of the components of block `bl` of species `s` along axis `d`
@inline function _block_speeds(mb::MixtureBlocks{D, W, T}, b::Int, d::Int) where {D, W, T}
    s, bl = species_of_block(mb, b)
    off = (bl - 1) * W
    return _select(sp -> State{W, T}(ntuple(j -> _velocity_component(sp, off + j, d), Val(W))), mb.mix.species, s)
end

@inline HyperCloudCore.block_flux(mb::MixtureBlocks{D, W, T}, U::State{W, T}, b::Int) where {D, W, T} =
    Flux{D, W, T}(ntuple(d -> _block_speeds(mb, b, d) .* U, Val(D)))
@inline HyperCloudCore.block_max_eigenvalue(mb::MixtureBlocks, U::State, d::Int, b::Int) = @inbounds mb.mix.max_speed[d]

@inline function (::RusanovFlux)(f_L::Flux{D, W, T}, f_R::Flux{D, W, T}, F_L::Flux{D, W, T}, F_R::Flux{D, W, T},
                                 w::BlockOf{D, W, T, <:MixtureBlocks}) where {D, W, T}
    return Flux{D, W, T}(ntuple(Val(D)) do d
        v = _block_speeds(w.eq, w.b, d)
        T(0.5) * (F_L[d] + F_R[d] - abs.(v) .* (f_R[d] - f_L[d]))
    end)
end

# Number moments of species `s` (the `moments` of its BGKEquation, summed in the same order) from the block-major array
@inline function _species_moments_from_blocks(sp::BGKEquation{D, Ms, T, NV1, NV}, Y, N::Int, p::Int, boff::Int, ::Val{W}) where {D, Ms, T, NV1, NV, W}
    m = zero(SVector{D + 2, T})
    @inbounds for k in 1:NV
        v, w = velocity_node(sp, k)
        fk = Y[(boff + (k - 1) ÷ W) * N + p][(k - 1) % W + 1]
        e  = T(0.5) * sum(abs2, v) * fk
        if _has_h(sp)
            kh = k + NV
            e += Y[(boff + (kh - 1) ÷ W) * N + p][(kh - 1) % W + 1]
        end
        m += w * SVector{D + 2, T}(fk, (fk * v)..., e)
    end
    return m
end

"""
    species_block_moments(mb::MixtureBlocks, Y, N, nb, p) -> NTuple{L, SVector{D + 2}}

Number moments `(n_s, n_s u_s…, e_s)` of every species of particle `p` from the block-major state array `Y`
(the [`species_moments`](@ref) of the single state). Runs inside kernels.
"""
@inline function species_block_moments(mb::MixtureBlocks{D, W, T, NB, L}, Y, N::Int, nb::Int, p::Int) where {D, W, T, NB, L}
    return ntuple(Val(L)) do s
        boff = @inbounds mb.block_offsets[s]
        _select(sp -> _species_moments_from_blocks(sp, Y, N, p, boff, Val(W)), mb.mix.species, s)
    end
end

"""
    block_moments(mb::MixtureBlocks, Y, N, nb, p) -> SVector{L (D + 2)}
    block_moments(mb::MixtureBlocks, bs::BlockState, p) -> SVector{L (D + 2)}

The species' number moments one after the other (the macroscopic state of [`moments`](@ref) of the mixture) of
particle `p`.
"""
@inline block_moments(mb::MixtureBlocks, Y, N::Int, nb::Int, p::Int) = _cat(species_block_moments(mb, Y, N, nb, p))
block_moments(mb::MixtureBlocks, bs::BlockState, p::Int) = block_moments(mb, parent(bs.data), bs.N[], bs.nb, p)

# The PAD of MOOD across the blocks: every species with finite moments, positive density and temperature
@inline function HyperCloudCore.block_admissible(mb::MixtureBlocks{D, W, T, NB, L}, Y, N::Int, nb::Int, p::Int) where {D, W, T, NB, L}
    moms = species_block_moments(mb, Y, N, nb, p)
    ok = true
    for s in 1:L
        m = @inbounds moms[s]
        all(isfinite, m) || return false
        n, _, θ = _select(sp -> _primitives(sp, m), mb.mix.species, s)
        ok &= n > zero(T) && θ > zero(T)
    end
    return ok
end

# Block `bl` of the discrete Maxwellian of species `s` with the parameters α
@inline function _species_equilibrium_block(mb::MixtureBlocks{D, W, T}, α::SVector{P, T}, s::Int, bl::Int) where {D, W, T, P}
    off = (bl - 1) * W
    return _select(sp -> State{W, T}(ntuple(j -> equilibrium_component(sp, α, off + j), Val(W))), mb.mix.species, s)
end

"""
    set_initial_conditions!(bs::BlockState, mb::MixtureBlocks, pg::ParticleGrid, IC::InitialCondition)

Every species of every particle starts in its conservative discrete Maxwellian of the species' number moments
given by `IC(x)` (`L (D + 2)` numbers, see [`mixture_state`](@ref)); one Newton solve per particle and species.
"""
function set_initial_conditions!(bs::BlockState{W, T}, mb::MixtureBlocks{D, W, T, NB, L}, pg::ParticleGrid{D, W, T},
                                 IC::InitialCondition) where {D, W, T, NB, L}
    N   = pg.meta.N
    nb  = bs.nb
    P   = L * (D + 2)
    pos = host_view(pg.core.positions, N)
    buf = Vector{State{W, T}}(undef, nb * N)
    n_bad, worst = 0, zero(T)
    for i in 1:N
        U = IC(pos[i])
        length(U) == P || error("mixture initial conditions give the species' number moments (n_s, n_s u_s..., e_s) of length $P, got $(length(U))")
        Uv = SVector{P, T}(U)
        for s in 1:L
            off = (s - 1) * (D + 2)
            Us  = SVector{D + 2, T}(ntuple(k -> Uv[off + k], Val(D + 2)))
            α, ok = _select(sp -> equilibrium_parameters(sp, Us), mb.mix.species, s)
            ok || error("initial state $Us of species $s at $(pos[i]) is not admissible for the mixture (n > 0, T > 0)")
            n, u, θ = _select(sp -> _primitives(sp, Us), mb.mix.species, s)
            scale = SVector{D + 2, T}(n, ntuple(_ -> n * (sqrt(sum(abs2, u)) + sqrt(θ)), Val(D))..., Us[D + 2])
            m, _ = _select(sp -> _equilibrium_system(sp, α), mb.mix.species, s)
            r = _scaled_residual(m - Us, scale)
            r > sqrt(eps(T)) && (n_bad += 1; worst = max(worst, r))
            for bl in 1:mb.nblocks[s]
                buf[(mb.block_offsets[s] + bl - 1) * N + i] = _species_equilibrium_block(mb, α, s, bl)
            end
        end
    end
    bs.N[] = N
    HyperCloudCore.ensure_capacity!(bs.data, nb * N)
    copyto!(parent(bs.data), 1, buf, 1, nb * N)
    n_bad > 0 && @warn "mixture initial state: a velocity grid cannot represent the macroscopic state of $n_bad " *
                       "species state(s) (relative moment error up to $worst); enlarge `vmax` or refine that grid"
    return nothing
end
