export BGKBlocks, equilibrium_block, block_moments

# =========================================================================
# THE BGK EQUATION AS A BLOCK EQUATION (HyperCloudCore's block pipeline)
# =========================================================================
"""
    BGKBlocks(eq::BGKEquation, ::Val{W}) <: HyperbolicPDE{D, W, T}

The BGK model of `eq` as a block equation of HyperCloudCore's block pipeline (`BlockState`, `BlockTimeStepper`):
the `M` components of the state of `eq` (the `f` values at the velocity nodes, then the `h` values of the Chu
reduction) split into `M / W` blocks of `W` consecutive components; block `b` holds the components
`(b − 1) W + 1 … b W`. Every block is transported on its own (`block_flux`: `v_{k,d} f_k` per component, the
per-velocity upwind dissipation of `RusanovFlux`, `UpwindFlux`, `AlignedRusanovFlux`), the collision
([`BGKBlockRelaxation`](@ref)) couples the blocks through the moments. `W` must divide `M`.

The equation object is `eq` itself (its small node tables); only the state is distributed, so the state
length `M` can be thousands of components without compiling anything of that size. [`block_moments`](@ref)
reduces the moments of a particle from a `BlockState`, [`equilibrium_block`](@ref) gives one block of the
discrete Maxwellian of a macroscopic state (initial conditions).
"""
struct BGKBlocks{D, W, T, NB, EQ <: BGKEquation{D}} <: HyperbolicPDE{D, W, T}
    eq::EQ
end

function BGKBlocks(eq::BGKEquation{D, M, T}, ::Val{W}) where {D, M, T, W}
    (W isa Int && W >= 1) || throw(ArgumentError("BGKBlocks: the block width must be a positive Int, got $W"))
    M % W == 0 || throw(ArgumentError("BGKBlocks: the block width $W does not divide the state length $M of the BGK equation"))
    return BGKBlocks{D, W, T, M ÷ W, typeof(eq)}(eq)
end

HyperCloudCore.num_blocks(::BGKBlocks{D, W, T, NB}) where {D, W, T, NB} = NB
num_velocities(bk::BGKBlocks) = num_velocities(bk.eq)

# The velocities of the components of block `b` along axis `d` (node-table lookups with a run-time offset)
@inline function _block_speeds(bk::BGKBlocks{D, W, T}, b::Int, d::Int) where {D, W, T}
    off = (b - 1) * W
    return State{W, T}(ntuple(j -> _velocity_component(bk.eq, off + j, d), Val(W)))
end

@inline HyperCloudCore.block_flux(bk::BGKBlocks{D, W, T}, U::State{W, T}, b::Int) where {D, W, T} =
    Flux{D, W, T}(ntuple(d -> _block_speeds(bk, b, d) .* U, Val(D)))
@inline HyperCloudCore.block_max_eigenvalue(bk::BGKBlocks, U::State, d::Int, b::Int) = @inbounds bk.eq.max_speed[d]

# The per-velocity upwind flux of a block (UpwindFlux and AlignedRusanovFlux fall back to RusanovFlux)
@inline function (::RusanovFlux)(f_L::Flux{D, W, T}, f_R::Flux{D, W, T}, F_L::Flux{D, W, T}, F_R::Flux{D, W, T},
                                 w::BlockOf{D, W, T, <:BGKBlocks}) where {D, W, T}
    return Flux{D, W, T}(ntuple(Val(D)) do d
        v = _block_speeds(w.eq, w.b, d)
        T(0.5) * (F_L[d] + F_R[d] - abs.(v) .* (f_R[d] - f_L[d]))
    end)
end

"""
    block_moments(bk::BGKBlocks, Y, N, nb, p) -> State{D + 2}
    block_moments(bk::BGKBlocks, bs::BlockState, p) -> State{D + 2}

Macroscopic state `(ρ, ρu..., E)` of particle `p` from a block-major state array `Y` (block `b` at
`Y[(b − 1) N + p]`) or a `BlockState`: the quadrature of [`moments`](@ref) over the blocks, summed in the same
order as for a single state (the `h` value of a velocity is read from its block). Runs inside kernels.
"""
@inline function block_moments(bk::BGKBlocks{D, W, T}, Y, N::Int, nb::Int, p::Int) where {D, W, T}
    eq = bk.eq
    NV = num_velocities(eq)
    m  = zero(SVector{D + 2, T})
    @inbounds for k in 1:NV
        v, w = velocity_node(eq, k)
        fk = Y[((k - 1) ÷ W) * N + p][(k - 1) % W + 1]
        e  = T(0.5) * sum(abs2, v) * fk
        if _has_h(eq)
            kh = k + NV
            e += Y[((kh - 1) ÷ W) * N + p][(kh - 1) % W + 1]
        end
        m += w * SVector{D + 2, T}(fk, (fk * v)..., e)
    end
    return m
end
block_moments(bk::BGKBlocks, bs::BlockState, p::Int) = block_moments(bk, parent(bs.data), bs.N[], bs.nb, p)

"""
    equilibrium_block(bk::BGKBlocks, U::State{D + 2}, b::Int) -> State{W}

Block `b` of the conservative discrete Maxwellian of the macroscopic state `U` (for initial conditions:
`set_block_states!((x, b) -> equilibrium_block(bk, U0(x), b), bs, pg)`). Errors for a non-physical `U`.
"""
function equilibrium_block(bk::BGKBlocks{D, W, T}, U::SVector{P, T}, b::Int) where {D, W, T, P}
    α, ok = equilibrium_parameters(bk.eq, U)
    ok || error("equilibrium_block: the macroscopic state $U is not admissible for the BGK equation (ρ > 0, θ > 0)")
    return _equilibrium_block(bk, α, b)
end
@inline function _equilibrium_block(bk::BGKBlocks{D, W, T}, α::SVector{P, T}, b::Int) where {D, W, T, P}
    off = (b - 1) * W
    return State{W, T}(ntuple(j -> equilibrium_component(bk.eq, α, off + j), Val(W)))
end
