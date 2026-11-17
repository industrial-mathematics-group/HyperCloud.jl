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

@inline function HyperCloudCore.block_reduce(st::_MixtureBlockCollision, p, Y, N, nb, pg, t, dt_coeff)
    mb = st.blocks
    eq = mb.mix
    moms = species_block_moments(mb, Y, N, nb, p)
    return _mixture_block_params(mb, eq, moms, dt_coeff)
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

@inline function HyperCloudCore.block_implicit_solve(st::MixtureBlockRelaxation, f::State{W, T}, b::Int, prm::MixtureBlockParams,
                                                     dt_coeff, p, pg, t) where {W, T}
    prm.ok || return f
    mb = st.blocks
    L  = num_species(mb)
    s, bl = species_of_block(mb, b)
    Δ   = T(dt_coeff)
    num = f
    den = one(T)
    @inbounds for r in 1:L
        i = (s - 1) * L + r
        rate = Δ * prm.rate[i]
        rate > zero(T) || continue
        num += rate * _species_equilibrium_block(mb, prm.alpha[i], s, bl)
        den += rate
    end
    return num / den
end

@inline function HyperCloudCore.block_source(st::_MixtureBlockCollision, f::State{W, T}, b::Int, prm::MixtureBlockParams, p, pg, t) where {W, T}
    prm.ok || return zero(f)
    mb = st.blocks
    L  = num_species(mb)
    s, bl = species_of_block(mb, b)
    acc = zero(f)
    @inbounds for r in 1:L
        i = (s - 1) * L + r
        rate = prm.rate[i]
        rate > zero(T) || continue
        acc += rate * (_species_equilibrium_block(mb, prm.alpha[i], s, bl) - f)
    end
    return acc
end
