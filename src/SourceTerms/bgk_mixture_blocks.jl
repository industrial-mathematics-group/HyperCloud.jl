export MixtureBlockRelaxation, ExplicitMixtureBlockRelaxation

# =========================================================================
# THE MIXTURE COLLISION TERM IN TWO PASSES (block pipeline)
# =========================================================================
"""
    MixtureBlockRelaxation(mb::MixtureBlocks) <: AbstractImplicitBlockSource
    ExplicitMixtureBlockRelaxation(mb::MixtureBlocks) <: AbstractExplicitBlockSource

The BBGSP collision operator of [`MixtureRelaxation`](@ref) for the block form [`MixtureBlocks`](@ref), as a block
source of HyperCloudCore's block pipeline (wrap it in a `BlockSourceTerm`). The reduction over the blocks of a
particle gives the species' moments, the species' velocities and temperatures after the stage (the two `L × L`
systems, for `dt_coeff = Δt a_ii`; the current ones for an explicit evaluation), the exponential parameters of
the Maxwellians `M_sr` of every species pair ([`equilibrium_parameters`](@ref), `L²` Newton solves per particle
and stage) and the rates `ν_sr / ε_sr`; the update of a block of species `s` evaluates its `M_sr` component by
component and forms `(f* + Δt a_ii Σ_r (ν_sr/ε_sr) n_s M_sr) / (1 + Δt a_ii Σ_r ν_sr/ε_sr)`, summed in the order
of the single-state form. A particle whose moments are not physical, or whose Maxwellians the grids cannot
represent, is left untouched (zero source).
"""
struct MixtureBlockRelaxation{B <: MixtureBlocks} <: AbstractImplicitBlockSource
    blocks::B
end

"The explicit form of the block mixture collision term, see [`MixtureBlockRelaxation`](@ref)."
struct ExplicitMixtureBlockRelaxation{B <: MixtureBlocks} <: AbstractExplicitBlockSource
    blocks::B
end

const _MixtureBlockCollision = Union{MixtureBlockRelaxation, ExplicitMixtureBlockRelaxation}

# The reduced parameters of a particle: α of M_sr at index (s − 1) L + r, the rates ν_sr / ε_sr, and the flag
struct MixtureBlockParams{L2, P, T}
    alpha::SVector{L2, SVector{P, T}}
    rate::SVector{L2, T}
    ok::Bool
end

HyperCloudCore.reduction_type(st::_MixtureBlockCollision) = _mixture_params_type(st.blocks)
_mixture_params_type(::MixtureBlocks{D, W, T, NB, L}) where {D, W, T, NB, L} = MixtureBlockParams{L * L, D + 2, T}

# The additive form: the species' moments of a block range add up over ranges
HyperCloudCore.reduction_partial_type(st::_MixtureBlockCollision) = _mixture_partial_type(st.blocks)
_mixture_partial_type(::MixtureBlocks{D, W, T, NB, L}) where {D, W, T, NB, L} = SVector{L * (D + 2), T}
@inline HyperCloudCore.block_partial(st::_MixtureBlockCollision, p, Y, N, nb, blocks::UnitRange, pg, t) =
    _cat(species_block_moments(st.blocks, Y, N, nb, p, blocks))
@inline function HyperCloudCore.finalize_reduction(st::_MixtureBlockCollision, p, moms_flat::SVector{Q, T}, pg, t, dt_coeff) where {Q, T}
    mb = st.blocks
    moms = _split_species(mb, moms_flat)
    return _mixture_block_params(mb, mb.mix, moms, dt_coeff)
end
@inline function _split_species(::MixtureBlocks{D, W, T, NB, L}, v::SVector{Q, T}) where {D, W, T, NB, L, Q}
    return ntuple(s -> SVector{D + 2, T}(ntuple(i -> (@inbounds v[(s - 1) * (D + 2) + i]), Val(D + 2))), Val(L))
end

@inline function _mixture_block_params(mb::MixtureBlocks{D, W, T, NB, L}, eq::BGKMixture{D, M, T, L}, moms, dt_coeff) where {D, W, T, NB, L, M}
    n, Ut, Tt, ok = _species_state_from_moments(eq, moms)
    Δ = T(dt_coeff)
    U, Tn, spd = ok && Δ > zero(T) ? _relaxed_moments(eq, Δ, n, Ut, Tt) : (Ut, Tt, true)
    good = ok && spd
    pairs = ntuple(Val(L * L)) do i
        s, r = (i - 1) ÷ L + 1, (i - 1) % L + 1
        rate = @inbounds eq.lambda[s, r] * n[r] / (s == r ? eq.epsilon : eq.kappa)
        if good && rate > zero(T)
            α, okm = _select(sp -> equilibrium_parameters(sp, _pair_target(eq, s, r, n, U, Tn)), eq.species, s)
            (α, rate, okm)
        else
            (zero(SVector{D + 2, T}), rate, true)
        end
    end
    return MixtureBlockParams{L * L, D + 2, T}(SVector{L * L, SVector{D + 2, T}}(map(first, pairs)),
                                                 SVector{L * L, T}(map(x -> x[2], pairs)), good && all(map(last, pairs)))
end

# The species of a block as a static value (`Val(s)`, an unrolled chain over the run-time `s`), and the loops over the
# partner species unrolled (`_species_fold`): the pair parameters `prm.alpha[(s − 1) L + r]` are then read at constant
# positions. With run-time positions the parameters of the particle sat in local memory, a stack of 2.5 KB per work-item
# of the implicit step on a GPU.
@inline function HyperCloudCore.block_implicit_solve(st::MixtureBlockRelaxation, f::State{W, T}, b::Int, prm::MixtureBlockParams,
                                                     dt_coeff, p, pg, t) where {W, T}
    prm.ok || return f
    mb = st.blocks
    s, bl = species_of_block(mb, b)
    Δ = T(dt_coeff)
    return _static_species(@inline(vs -> _implicit_species(mb, f, prm, vs, bl, Δ)), s, Val(num_species(mb)))
end
@inline function _implicit_species(mb::MixtureBlocks{D, W, T, NB, L}, f::State{W, T}, prm, ::Val{s}, bl::Int, Δ::T) where {D, W, T, NB, L, s}
    num, den = _species_fold((f, one(T)), Val(L)) do (num, den), r
        @inline
        i = (s - 1) * L + r
        rate = Δ * (@inbounds prm.rate[i])
        rate > zero(T) || return (num, den)
        (num + rate * _species_equilibrium_block(mb, (@inbounds prm.alpha[i]), s, bl), den + rate)
    end
    return num / den
end

@inline function HyperCloudCore.block_source(st::_MixtureBlockCollision, f::State{W, T}, b::Int, prm::MixtureBlockParams, p, pg, t) where {W, T}
    prm.ok || return zero(f)
    mb = st.blocks
    s, bl = species_of_block(mb, b)
    return _static_species(@inline(vs -> _source_species(mb, f, prm, vs, bl)), s, Val(num_species(mb)))
end
@inline function _source_species(mb::MixtureBlocks{D, W, T, NB, L}, f::State{W, T}, prm, ::Val{s}, bl::Int) where {D, W, T, NB, L, s}
    return _species_fold(zero(f), Val(L)) do acc, r
        @inline
        i = (s - 1) * L + r
        rate = @inbounds prm.rate[i]
        rate > zero(T) || return acc
        acc + rate * (_species_equilibrium_block(mb, (@inbounds prm.alpha[i]), s, bl) - f)
    end
end

# `g(Val(s))` for a run-time species `s ∈ 1:L` (an unrolled chain of comparisons)
@inline _static_species(g::G, s::Int, ::Val{L}) where {G, L} = _static_species(g, s, Val(1), Val(L))
@inline _static_species(g::G, s::Int, ::Val{L}, ::Val{L}) where {G, L} = g(Val(L))      # (the last species)
@inline function _static_species(g::G, s::Int, ::Val{k}, ::Val{L}) where {G, k, L}
    s == k && return g(Val(k))
    return _static_species(g, s, Val(k + 1), Val(L))
end
# `op(…op(op(init, 1), 2)…, L)`, unrolled (`r` a constant in every call)
@inline _species_fold(op::F, init, ::Val{L}) where {F, L} = _species_fold(op, init, Val(1), Val(L))
@inline _species_fold(op::F, acc, ::Val{L}, ::Val{L}) where {F, L} = op(acc, L)
@inline _species_fold(op::F, acc, ::Val{r}, ::Val{L}) where {F, r, L} = _species_fold(op, op(acc, r), Val(r + 1), Val(L))
