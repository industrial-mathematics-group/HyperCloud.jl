export BGKBlockRelaxation, ExplicitBGKBlockRelaxation

# =========================================================================
# THE BGK COLLISION TERM IN TWO PASSES (block pipeline)
# =========================================================================
"""
    BGKBlockRelaxation(bk::BGKBlocks) <: AbstractImplicitBlockSource
    ExplicitBGKBlockRelaxation(bk::BGKBlocks) <: AbstractExplicitBlockSource

The BGK collision term `(M[f] − f) / τ` of [`BGKRelaxation`](@ref) for the block form [`BGKBlocks`](@ref), as a
block source of HyperCloudCore's block pipeline (wrap it in a `BlockSourceTerm`). The reduction over the blocks
of a particle gives the moments `U`, the relaxation time `τ(U)`, the exponential parameters `α` of the
conservative discrete Maxwellian ([`equilibrium_parameters`](@ref)) and the stiffness `λ = Δt a_ii / τ`; the
update of block `b` evaluates the Maxwellian component by component ([`equilibrium_component`](@ref)):
`f = (f* + λ M) / (1 + λ)` for the implicit stage (`τ = 0` projects onto `M`), `(M − f) / τ` explicitly. The
same arithmetic as for the single state, so a block run reproduces the `SVector` run to round-off. A particle
whose moments are not physical is left untouched (zero source).
"""
struct BGKBlockRelaxation{B <: BGKBlocks} <: AbstractImplicitBlockSource
    blocks::B
end

"The explicit form of the block collision term, see [`BGKBlockRelaxation`](@ref)."
struct ExplicitBGKBlockRelaxation{B <: BGKBlocks} <: AbstractExplicitBlockSource
    blocks::B
end

const _BGKBlockCollision = Union{BGKBlockRelaxation, ExplicitBGKBlockRelaxation}

# The reduced parameters of a particle: α of the Maxwellian, τ, λ = dt_coeff / τ, and whether the state is physical
struct BGKBlockParams{P, T}
    alpha::SVector{P, T}
    tau::T
    lambda::T
    ok::Bool
end

HyperCloudCore.reduction_type(st::_BGKBlockCollision) = _bgk_params_type(st.blocks)
_bgk_params_type(::BGKBlocks{D, W, T}) where {D, W, T} = BGKBlockParams{D + 2, T}

@inline function HyperCloudCore.block_reduce(st::_BGKBlockCollision, p, Y, N, nb, pg, t, dt_coeff)
    bk = st.blocks
    eq = bk.eq
    U  = block_moments(bk, Y, N, nb, p)
    α, ok = equilibrium_parameters(eq, U)
    T  = eltype(U)
    τ  = ok ? relaxation_time(eq, U) : one(T)
    return BGKBlockParams(α, τ, T(dt_coeff) / τ, ok)
end

@inline function HyperCloudCore.block_implicit_solve(st::BGKBlockRelaxation, f::State{W, T}, b::Int, prm::BGKBlockParams,
                                                     dt_coeff, p, pg, t) where {W, T}
    prm.ok || return f
    M = _equilibrium_block(st.blocks, prm.alpha, b)
    λ = prm.lambda
    isinf(λ) && return M                       # τ = 0: projection onto the equilibrium
    return (f + λ * M) / (one(T) + λ)
end

@inline function HyperCloudCore.block_source(st::_BGKBlockCollision, f::State{W, T}, b::Int, prm::BGKBlockParams, p, pg, t) where {W, T}
    prm.ok || return zero(f)
    M = _equilibrium_block(st.blocks, prm.alpha, b)
    return (M - f) / prm.tau
end
