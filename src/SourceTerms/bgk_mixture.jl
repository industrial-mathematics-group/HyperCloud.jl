export MixtureRelaxation, ExplicitMixtureRelaxation

# =========================================================================
# MULTI-SPECIES BGK COLLISION TERM
# =========================================================================
"""
    MixtureRelaxation(eq::BGKMixture) <: AbstractImplicitSourceTerm
    ExplicitMixtureRelaxation(eq::BGKMixture) <: AbstractExplicitSourceTerm

The BBGSP collision operator of a [`BGKMixture`](@ref),
`Q_s = (1/ε) ν_ss (n_s M_ss − f_s) + (1/κ) Σ_{r≠s} ν_sr (n_s M_sr − f_s)`, treated implicitly (IMEX tableaus)
or explicitly (explicit tableaus, stable for `Δt ≲ ε/ν` only).

The implicit stage equation `f = f* + Δt a_ii Q(f)` is solved in closed form, species by species, as in the
paper's relaxation step: the collision conserves the species number densities, so `n_s` and the frequencies
`ν_sr = λ_sr n_r` are known from `f*`; the species velocities and temperatures after the stage follow from
two `L × L` symmetric positive definite systems (one per velocity component, the graph Laplacian of the
weights `d_sr = λ_sr n_s n_r m_s m_r / (m_s + m_r)`, and one for the temperatures with
`c_sr = n_dof k_B λ_sr n_s n_r m_s m_r / (m_s + m_r)²`, whose right-hand side holds the kinetic energy exchanged),
which conserve the total momentum and energy exactly. The Maxwellians `M_ss`, `M_sr` are then built at those
values with the conservative discrete [`equilibrium`](@ref) of each species (discrete moments exact to
round-off, so no moment correction is needed), and
`f_s = (f_s* + Δt a_ii Σ_r (ν_sr/ε_sr) n_s M_sr) / (1 + Δt a_ii Σ_r ν_sr/ε_sr)` with `ε_ss = ε`, `ε_sr = κ`.
Exact for any stiffness in `ε` and `κ` (asymptotic preserving). A particle whose state is not physical
(a species with `n_s ≤ 0` or `T_s ≤ 0`, or an equilibrium the grid cannot represent) is left untouched (zero source).
"""
struct MixtureRelaxation{E <: BGKMixture} <: AbstractImplicitSourceTerm
    eq::E
end

"The explicit form of the mixture collision term, see [`MixtureRelaxation`](@ref)."
struct ExplicitMixtureRelaxation{E <: BGKMixture} <: AbstractExplicitSourceTerm
    eq::E
end

# Species data of a state: number densities n (L), velocities U (L × D, one row per species), thermodynamic
# temperatures Tt (L), and whether every species is physical.
@inline _species_state(eq::BGKMixture, blocks::Tuple) = _species_state_from_moments(eq, map((s, b) -> moments(s, b), eq.species, blocks))
@inline function _species_state_from_moments(eq::BGKMixture{D, M, T, L}, moms::Tuple) where {D, M, T, L}
    prim = map((s, m) -> _primitives(s, m), eq.species, moms)
    n  = SVector{L, T}(ntuple(s -> (@inbounds prim[s][1]), Val(L)))
    U  = SMatrix{L, D, T}(ntuple(i -> (@inbounds prim[(i - 1) % L + 1][2][(i - 1) ÷ L + 1]), Val(L * D)))
    θ  = SVector{L, T}(ntuple(s -> (@inbounds prim[s][3]), Val(L)))
    ok = all(isfinite, n) && all(isfinite, U) && all(isfinite, θ) && all(>(zero(T)), n) && all(>(zero(T)), θ)
    return n, U, θ .* eq.masses ./ eq.kB, ok
end

@inline _row(U::SMatrix{L, D, T}, s::Int) where {L, D, T} = Space{D, T}(ntuple(d -> (@inbounds U[s, d]), Val(D)))

# Number moments (n_s, n_s U_sr, e_sr) of n_s M_sr: the Maxwellian of species s against r at the mixture velocity
# U_sr and temperature T_sr of the BBGSP model (for ν_sr = λ_sr n_r); s = r gives species s' own Maxwellian.
@inline function _pair_target(eq::BGKMixture{D, M, T, L}, s::Int, r::Int, n::SVector{L, T}, U::SMatrix{L, D, T},
                              Tt::SVector{L, T}) where {D, M, T, L}
    @inbounds ms, mr = eq.masses[s], eq.masses[r]
    ndof = T(D) + eq.K
    a = mr / (ms + mr)
    b = 2a * ms / (ms + mr)
    γ = ms * a^2 / ndof
    Us, Ur = _row(U, s), _row(U, r)
    Usr = (one(T) - a) * Us + a * Ur
    @inbounds Tsr = (one(T) - b) * Tt[s] + b * Tt[r] + γ / eq.kB * sum(abs2, Us - Ur)
    θ  = eq.kB * Tsr / ms
    @inbounds ns = n[s]
    return SVector{D + 2, T}(ns, (ns * Usr)..., ns * (T(0.5) * ndof * θ + T(0.5) * sum(abs2, Usr)))
end

# The matrices of the velocity system (D_ρ + μ L_d) and of the temperature system (n_dof/2 k_B D_n + μ L_c): the graph
# Laplacians of the symmetric weights d_sr = λ_sr n_s n_r m_s m_r / (m_s + m_r) and c_sr = n_dof k_B λ_sr n_s n_r m_s m_r /
# (m_s + m_r)², both symmetric positive definite. (A plain function: the matrices are rebuilt entry by entry, which
# a closure capturing them would box.)
@inline function _mixture_systems(eq::BGKMixture{D, M, T, L}, μ::T, n::SVector{L, T}, ρ::SVector{L, T}, ndof::T) where {D, M, T, L}
    A = SMatrix{L, L, T}(ntuple(i -> (i - 1) % L == (i - 1) ÷ L ? (@inbounds ρ[(i - 1) % L + 1]) : zero(T), Val(L * L)))
    B = SMatrix{L, L, T}(ntuple(i -> (i - 1) % L == (i - 1) ÷ L ? (@inbounds T(0.5) * ndof * eq.kB * n[(i - 1) % L + 1]) : zero(T), Val(L * L)))
    @inbounds for s in 1:L, r in 1:L
        s == r && continue
        ms, mr = eq.masses[s], eq.masses[r]
        d = eq.lambda[s, r] * n[s] * n[r] * ms * mr / (ms + mr)
        c = ndof * eq.kB * eq.lambda[s, r] * n[s] * n[r] * ms * mr / (ms + mr)^2
        A = setindex(A, A[s, s] + μ * d, s, s)
        A = setindex(A, A[s, r] - μ * d, s, r)
        B = setindex(B, B[s, s] + μ * c, s, s)
        B = setindex(B, B[s, r] - μ * c, s, r)
    end
    return A, B
end

# Right-hand side of the temperature equation of species s: its thermal energy before the stage, the kinetic energy
# it lost to the velocity relaxation, and the kinetic energy exchanged with the other species at the new velocities.
@inline function _thermal_rhs(eq::BGKMixture{D, M, T, L}, s::Int, μ::T, n::SVector{L, T}, ρ::SVector{L, T}, ndof::T,
                              U::SMatrix{L, D, T}, Un::SMatrix{L, D, T}, Tt::SVector{L, T}) where {D, M, T, L}
    @inbounds begin
        Us = _row(Un, s)
        Es = T(0.5) * ndof * n[s] * eq.kB * Tt[s] + T(0.5) * ρ[s] * (sum(abs2, _row(U, s)) - sum(abs2, Us))
        ex = zero(T)
        for r in 1:L
            r == s && continue
            ms, mr = eq.masses[s], eq.masses[r]
            a   = mr / (ms + mr)
            γ   = ms * a^2 / ndof
            Ur  = _row(Un, r)
            Usr = (one(T) - a) * Us + a * Ur
            ex += eq.lambda[s, r] * n[r] * (T(0.5) * ρ[s] * (sum(abs2, Us) - sum(abs2, Usr)) -
                                             T(0.5) * ndof * n[s] * γ * sum(abs2, Us - Ur))
        end
        return Es - μ * ex
    end
end

# Species velocities and temperatures after the implicit stage (paper, "implicit relaxation step"): the velocity
# system (D_ρ + μ L_d) U = ρ Ũ per component and the temperature system (n_dof/2 k_B D_n + μ L_c) T = E with the
# kinetic energy exchange on the right-hand side, μ = Δt a_ii / κ. The intra-species terms cancel in both.
@inline function _relaxed_moments(eq::BGKMixture{D, M, T, L}, Δ::T, n::SVector{L, T}, U::SMatrix{L, D, T},
                                  Tt::SVector{L, T}) where {D, M, T, L}
    μ = Δ / eq.kappa
    (iszero(μ) || L == 1) && return U, Tt, true
    ρ    = eq.masses .* n
    ndof = T(D) + eq.K
    A, B = _mixture_systems(eq, μ, n, ρ, ndof)
    cols = ntuple(d -> _spd_solve(A, SVector{L, T}(ntuple(s -> (@inbounds ρ[s] * U[s, d]), Val(L)))), Val(D))
    all(c -> c[2], cols) || return U, Tt, false
    Un = SMatrix{L, D, T}(ntuple(i -> (@inbounds cols[(i - 1) ÷ L + 1][1][(i - 1) % L + 1]), Val(L * D)))
    E  = SVector{L, T}(ntuple(s -> _thermal_rhs(eq, s, μ, n, ρ, ndof, U, Un, Tt), Val(L)))
    Tn, spd = _spd_solve(B, E)
    return Un, Tn, spd
end

@inline function implicit_solve(st::MixtureRelaxation{<:BGKMixture{D, M, T, L}}, f::State{M, T}, dt_coeff::Real,
                                p_idx::Int, pg, t::Real) where {D, M, T, L}
    eq = st.eq
    blocks = _blocks(eq, f)
    n, Ut, Tt, ok = _species_state(eq, blocks)
    ok || return f
    Δ = T(dt_coeff)
    U, Tn, spd = _relaxed_moments(eq, Δ, n, Ut, Tt)
    spd || return f
    parts = map(eq.species, blocks, _indices(eq)) do sp, b, s
        num  = b
        den  = one(T)
        good = true
        for r in 1:L
            @inbounds rate = Δ * eq.lambda[s, r] * n[r] / (s == r ? eq.epsilon : eq.kappa)
            rate > zero(T) || continue
            Meq, okm = equilibrium(sp, _pair_target(eq, s, r, n, U, Tn))
            good &= okm
            num += rate * Meq
            den += rate
        end
        (num / den, good)
    end
    all(map(last, parts)) || return f
    return _cat(map(first, parts))
end

const _MixtureCollision = Union{MixtureRelaxation, ExplicitMixtureRelaxation}

# The explicit evaluation Q(f) at the current moments (the explicit term, and the IMEX stages with a_ii = 0 of the
# implicit one, as for the single gas)
@inline function evaluate_source(st::_MixtureCollision, f::State{M, T}, p_idx::Int, pg, t::Real) where {M, T}
    return _mixture_collision(st.eq, f)
end

@inline function _mixture_collision(eq::BGKMixture{D, M, T, L}, f::State{M, T}) where {D, M, T, L}
    blocks = _blocks(eq, f)
    n, U, Tt, ok = _species_state(eq, blocks)
    ok || return zero(f)
    parts = map(eq.species, blocks, _indices(eq)) do sp, b, s
        acc  = zero(b)
        good = true
        for r in 1:L
            @inbounds rate = eq.lambda[s, r] * n[r] / (s == r ? eq.epsilon : eq.kappa)
            rate > zero(T) || continue
            Meq, okm = equilibrium(sp, _pair_target(eq, s, r, n, U, Tt))
            good &= okm
            acc += rate * (Meq - b)
        end
        (acc, good)
    end
    all(map(last, parts)) || return zero(f)
    return _cat(map(first, parts))
end

"""
    bgk_sources(eq::BGKMixture, tableau::IMEXButcherTableau) -> (explicit, implicit)

The source tuples of a mixture run: [`MixtureRelaxation`](@ref) for an IMEX tableau,
[`ExplicitMixtureRelaxation`](@ref) for an explicit one, none if the mixture is collisionless (every
`λ_sr = 0`, or `ε = κ = Inf`).
"""
function bgk_sources(eq::BGKMixture, tableau::IMEXButcherTableau)
    _is_collisionless(eq) && return (), ()
    all(iszero, tableau.a) || return (), (MixtureRelaxation(eq),)
    return (ExplicitMixtureRelaxation(eq),), ()
end

# =========================================================================
# INITIAL CONDITIONS
# =========================================================================

"""
    set_initial_conditions!(pg::ParticleGrid, eq::BGKMixture, IC::InitialCondition)

For a mixture `IC` gives the species' number moments `(n_s, n_s u_s..., e_s)` one after the other
(`L (D + 2)` numbers, see [`mixture_state`](@ref)); every species starts in its conservative discrete
Maxwellian. Warns once, as for the single gas, if a velocity grid cannot represent a state.
"""
function set_initial_conditions!(pg::ParticleGrid, eq::BGKMixture{D, M, T, L}, IC::InitialCondition) where {D, M, T, L}
    n_bad, worst = Ref(0), Ref(zero(T))
    P = L * (D + 2)
    set_states!(pg) do pos
        U = IC(pos)
        length(U) == P || error("mixture initial conditions give the species' number moments (n_s, n_s u_s..., e_s) " *
                                "of length $P, got $(length(U))")
        Uv = SVector{P, T}(U)
        parts = map(eq.species, _indices(eq)) do sp, s
            off = (s - 1) * (D + 2)
            Us  = SVector{D + 2, T}(ntuple(i -> Uv[off + i], Val(D + 2)))
            f, ok = equilibrium(sp, Us)
            ok || error("initial state $Us of species $s at $pos is not admissible for the mixture (n > 0, T > 0)")
            n, u, θ = _primitives(sp, Us)
            scale = SVector{D + 2, T}(n, ntuple(_ -> n * (sqrt(sum(abs2, u)) + sqrt(θ)), Val(D))..., Us[D + 2])
            r = _scaled_residual(moments(sp, f) - Us, scale)
            r > sqrt(eps(T)) && (n_bad[] += 1; worst[] = max(worst[], r))
            f
        end
        _cat(parts)
    end
    n_bad[] > 0 && @warn "mixture initial state: a velocity grid cannot represent the macroscopic state of $(n_bad[]) " *
                         "species state(s) (relative moment error up to $(worst[])); enlarge `vmax` or refine that grid"
    return nothing
end
