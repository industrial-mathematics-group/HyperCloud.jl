export BGKMixture, num_species, species_moments, mixture_moments, species_primitives, mixture_state

# =========================================================================
# GAS MIXTURES: THE EQUATION
# =========================================================================

"""
    BGKMixture{D, M, T, L, S, LL} <: DiscreteVelocityEquation{D, M, T}
    BGKMixture(species::NTuple{L, BGKEquation}, masses, lambda; epsilon = 1, kappa = epsilon, kB = 1)
    BGKMixture(::Val{D}, masses, lambda, grids; gamma = 5/3, epsilon = 1, kappa = epsilon, kB = 1, centers = nothing)

Discrete-velocity model of a gas mixture of `L` species with the multi-species BGK collision operator of
Bobylev, Bisi, Groppi, Spiga and Potapenko (BBGSP),

    ∂_t f_s + v·∇f_s = (1/ε) ν_ss (n_s M_ss − f_s) + (1/κ) Σ_{r≠s} ν_sr (n_s M_sr − f_s),   s = 1 … L,

with the species number densities `n_s = ∫ f_s dv`, the collision frequencies `ν_sr = λ_sr n_r` (`λ` a
symmetric matrix of constants: Maxwell molecules), the intra-species Knudsen number `ε` and the inter-species
one `κ`, and the unit-density Maxwellians `M_sr` of species `s` (mass `m_s`) at the mixture velocity and
temperature

    U_sr = (1 − a_sr) U_s + a_sr U_r,     T_sr = (1 − b_sr) T_s + b_sr T_r + γ_sr / k_B |U_s − U_r|²,
    a_sr = m_r / (m_s + m_r),   b_sr = 2 a_sr m_s / (m_s + m_r),   γ_sr = m_s a_sr² / n_dof,

(`n_dof = D + K` the number of translational degrees of freedom, 3 for a monatomic gas), the choice for which
the operator conserves every species number density and the total momentum and energy
([`MixtureRelaxation`](@ref)). `M_ss` is the Maxwellian of species `s` itself.

Every species `s` is a [`BGKEquation`](@ref) on its own velocity grid (nodes, weights, centre and the Chu
reduction `h` for the unresolved degrees of freedom; its relaxation model is ignored, the mixture's `λ`,
`ε`, `κ` define the collisions), with the thermal speed `θ_s = k_B T_s / m_s` as its temperature. The
state of a particle is the concatenation of the species' states (`M = Σ_s M_s` components); the transport is
diagonal, so the flux, the per-velocity Rusanov dissipation (also in the `ALEFrame` and `LagrangianFrame` of a
moving grid, with the mass-averaged mixture velocity as the fluid velocity), MOOD and the walls (`SlipWall`
per species, `DiffuseWall` at the thermodynamic wall temperature, emitting every species at `k_B T_w / m_s`)
carry over from the single gas.

Units: `m_s`, `k_B` and `λ` are plain numbers. Dimensionless problems use `k_B = 1`; with the molar masses
`M_s` (g/mol) as `m_s` and `k_B = R = 8.3145` (J/(mol K)) temperatures are in kelvin and velocities in units
of `√1000 m/s ≈ 31.6 m/s`. `ε` and `κ` must be positive (`Inf` switches that part of the collision off; every
`λ_sr = 0` or `ε = κ = Inf` is collisionless). All species need the same `D`, element type and `K`.

The convenience constructor takes one `(nodes, weights)` pair per species (from [`velocity_grid`](@ref); the
grids may differ, a light species needs a wider one) and a common `gamma` (default `5/3`, monatomic;
`K = 2/(γ − 1) − D` unresolved degrees of freedom per species), optionally `centers` (one tuple per species).

The macroscopic state of [`moments`](@ref) is the concatenation of the species' number moments
`(n_s, n_s u_s…, e_s)` with `e_s = n_s (n_dof/2 θ_s + |u_s|²/2)` (the Euler state of species `s` per unit
mass), `L (D + 2)` numbers: what an initial condition returns ([`mixture_state`](@ref) builds it from
`n_s`, `u_s`, `T_s`), [`species_primitives`](@ref) turns it back into `(n_s, u_s, T_s)`, and
[`mixture_moments`](@ref) gives the mass-weighted mixture state `(ρ, ρU, E)`.
"""
struct BGKMixture{D, M, T, L, S <: Tuple, LL} <: DiscreteVelocityEquation{D, M, T}
    species::S
    masses::SVector{L, T}
    lambda::SMatrix{L, L, T, LL}
    epsilon::T
    kappa::T
    kB::T
    K::T
    max_speed::SVector{D, T}
end

_dmt(::BGKEquation{D, M, T}) where {D, M, T} = (D, M, T)

function BGKMixture(species::Tuple{Vararg{BGKEquation}}, masses, lambda; epsilon::Real = 1, kappa::Real = epsilon,
                    kB::Real = 1)
    L = length(species)
    L >= 1 || throw(ArgumentError("BGKMixture needs at least one species"))
    D, _, T = _dmt(species[1])
    K = species[1].K
    for s in 2:L
        Ds, _, Ts = _dmt(species[s])
        (Ds == D && Ts == T) || throw(ArgumentError("BGKMixture: species $s is $(Ds)-D with $Ts, species 1 is $(D)-D with $T"))
        abs(species[s].K - K) <= 1e-8 || throw(ArgumentError(
            "BGKMixture: species $s has K = $(species[s].K) unresolved degrees of freedom, species 1 has K = $K " *
            "(the model needs one γ for all species)"))
    end
    length(masses) == L || throw(ArgumentError("BGKMixture: $(length(masses)) masses for $L species"))
    all(m -> isfinite(m) && m > 0, masses) || throw(ArgumentError("BGKMixture: the masses must be positive, got $masses"))
    lam = SMatrix{L, L, T}(lambda)
    all(l -> isfinite(l) && l >= 0, lam) || throw(ArgumentError("BGKMixture: the collision constants λ must be ≥ 0"))
    lam == lam' || throw(ArgumentError("BGKMixture: the collision constants λ_sr must be symmetric"))
    (epsilon > 0 && kappa > 0 && !isnan(epsilon) && !isnan(kappa)) ||
        throw(ArgumentError("BGKMixture: ε and κ must be positive (Inf for no collisions), got ε = $epsilon, κ = $kappa"))
    (isfinite(kB) && kB > 0) || throw(ArgumentError("BGKMixture: k_B must be positive, got $kB"))
    M = sum(s -> _dmt(s)[2], species)
    smax = SVector{D, T}(ntuple(d -> maximum(s -> s.max_speed[d], species), Val(D)))
    return BGKMixture{D, M, T, L, typeof(species), L * L}(species, SVector{L, T}(masses), lam, T(epsilon), T(kappa), T(kB),
                                                        T(K), smax)
end

function BGKMixture(::Val{D}, masses, lambda, grids::Tuple; gamma::Real = 5 / 3, epsilon::Real = 1, kappa::Real = epsilon,
                    kB::Real = 1, centers = nothing) where {D}
    L = length(grids)
    length(masses) == L || throw(ArgumentError("BGKMixture: $(length(masses)) masses for $L velocity grids"))
    centers === nothing || length(centers) == L || throw(ArgumentError("BGKMixture: $(length(centers)) centres for $L species"))
    species = ntuple(L) do s
        nodes, weights = grids[s]
        center = centers === nothing ? ntuple(_ -> 0, D) : centers[s]
        BGKEquation(Val(D), nodes, weights, gamma, ConstantRelaxationTime(convert(eltype(nodes), Inf)); center = center)
    end
    return BGKMixture(species, masses, lambda; epsilon = epsilon, kappa = kappa, kB = kB)
end

"""
    num_species(eq::BGKMixture) -> Int

Number of species `L` of the mixture.
"""
num_species(::BGKMixture{D, M, T, L}) where {D, M, T, L} = L

num_velocities(eq::BGKMixture) = sum(num_velocities, eq.species)

_is_collisionless(eq::BGKMixture) = all(iszero, eq.lambda) || (isinf(eq.epsilon) && isinf(eq.kappa))

# --- the species blocks of a state ---------------------------------------------------------------------------
# Component ranges of the species as compile-time constants: the block of species s is the SUnitRange
# (offset_s + 1 : offset_s + M_s), so a state splits into the species' states without index arithmetic at run time.
_block_sizes(::Type{S}) where {S <: Tuple} = ntuple(i -> S.parameters[i].parameters[2], length(S.parameters))

@generated function _blocks(::BGKMixture{D, M, T, L, S}, U::State{M, T}) where {D, M, T, L, S}
    sizes = _block_sizes(S)
    offs  = cumsum((0, sizes[1:(end - 1)]...))
    parts = [:(U[StaticArrays.SUnitRange{$(offs[s] + 1), $(sizes[s])}()]) for s in 1:L]
    return :(@inbounds ($(parts...),))
end

@inline _cat(parts::Tuple) = vcat(parts...)

# Species indices as a tuple, to `map` over the species together with their blocks: a `map` over tuples is unrolled
# with the concrete type of every element, whereas `species[s]` with a runtime `s` is a dynamic lookup when the
# species have different grids.
@inline _indices(::BGKMixture{D, M, T, L}) where {D, M, T, L} = ntuple(identity, Val(L))

@inline _component_speeds(eq::BGKMixture, ::Val{d}) where {d} = _cat(map(s -> _component_speeds(s, Val(d)), eq.species))

# --- moments -------------------------------------------------------------------------------------------------
"""
    species_moments(eq::BGKMixture, f::State) -> NTuple{L, SVector{D + 2}}

Number moments `(n_s, n_s u_s…, e_s)` of every species (the `moments` of its [`BGKEquation`](@ref) on its block
of `f`).
"""
@inline species_moments(eq::BGKMixture{D, M, T}, f::State{M, T}) where {D, M, T} =
    map((s, b) -> moments(s, b), eq.species, _blocks(eq, f))

"""
    moments(eq::BGKMixture, f::State) -> SVector{L (D + 2)}

The species' number moments `(n_s, n_s u_s…, e_s)` one after the other (see [`BGKMixture`](@ref)).
"""
@inline moments(eq::BGKMixture{D, M, T}, f::State{M, T}) where {D, M, T} = _cat(species_moments(eq, f))

"""
    mixture_moments(eq::BGKMixture, f::State) -> SVector{D + 2}

Mass-weighted mixture state `(ρ, ρU, E) = Σ_s m_s (n_s, n_s u_s, e_s)`: the mass density, momentum and energy
of the mixture (`U` is the mass-averaged velocity, `E` the total energy with the thermal part `n_dof/2 Σ_s n_s k_B T_s`).
"""
@inline function mixture_moments(eq::BGKMixture{D, M, T, L}, f::State{M, T}) where {D, M, T, L}
    sm  = species_moments(eq, f)
    acc = zero(SVector{D + 2, T})
    @inbounds for s in 1:L
        acc += eq.masses[s] * sm[s]
    end
    return acc
end

"""
    species_primitives(eq::BGKMixture, U::SVector{L (D + 2)}) -> NTuple{L, (n_s, u_s::Space, T_s)}

Number density, velocity and thermodynamic temperature `T_s = m_s θ_s / k_B` of every species from the
macroscopic state of [`moments`](@ref).
"""
@inline function species_primitives(eq::BGKMixture{D, M, T, L}, U::SVector{P, T}) where {D, M, T, L, P}
    P == L * (D + 2) || throw(ArgumentError("species_primitives: a state of length $(L * (D + 2)) expected, got $P"))
    return map(eq.species, Tuple(eq.masses), _indices(eq)) do sp, m, s
        off = (s - 1) * (D + 2)
        Us  = SVector{D + 2, T}(ntuple(i -> (@inbounds U[off + i]), Val(D + 2)))
        n, u, θ = _primitives(sp, Us)
        (n, u, m * θ / eq.kB)
    end
end

"""
    mixture_state(eq::BGKMixture, n, u, T) -> SVector{L (D + 2)}

Macroscopic state (the format of [`moments`](@ref) and of the initial conditions) from the species' number
densities `n`, velocities `u` (a tuple of `D`-tuples; plain numbers in 1-D) and thermodynamic temperatures
`T`, one entry per species: `θ_s = k_B T_s / m_s`, `e_s = n_s (n_dof/2 θ_s + |u_s|²/2)`.
"""
function mixture_state(eq::BGKMixture{D, M, T, L}, n, u, temp) where {D, M, T, L}
    (length(n) == L && length(u) == L && length(temp) == L) ||
        throw(ArgumentError("mixture_state: $L species need $L densities, velocities and temperatures"))
    parts = map(Tuple(eq.masses), _indices(eq)) do m, s
        us = u[s] isa Number ? (u[s],) : Tuple(u[s])
        length(us) == D || throw(ArgumentError("mixture_state: the velocity of species $s has $(length(us)) components, the model is $D-D"))
        ns = T(n[s])
        uv = Space{D, T}(us)
        θ  = eq.kB * T(temp[s]) / m
        SVector{D + 2, T}(ns, (ns * uv)..., ns * (T(0.5) * (T(D) + eq.K) * θ + T(0.5) * sum(abs2, uv)))
    end
    return _cat(parts)
end

# =========================================================================
# HYPERCLOUDCORE API
# =========================================================================

@inline max_eigenvalue(eq::BGKMixture, U::State, d::Int) = @inbounds eq.max_speed[d]

@inline max_eigenvalue(eq::BGKMixture{D, M, T}, U::State{M, T}, d::Int, shift::Real) where {D, M, T} =
    maximum(map(s -> _max_relative_speed(s, d, T(shift)), eq.species))

"Admissible mixture state (MOOD PAD): every species finite with positive density and temperature."
@inline is_admissible(eq::BGKMixture{D, M, T}, U::State{M, T}) where {D, M, T} =
    all(map((s, b) -> is_admissible(s, b), eq.species, _blocks(eq, U)))

# The mass-averaged mixture velocity: a Lagrangian cloud follows the mixture
@inline function fluid_velocity(eq::BGKMixture{D, M, T}, U::State{M, T}) where {D, M, T}
    mm = mixture_moments(eq, U)
    return Space{D, T}(ntuple(d -> (@inbounds mm[1 + d]), Val(D))) / mm[1]
end
