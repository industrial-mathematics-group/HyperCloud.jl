export euler_1d_piston, exact_piston

# =========================================================================
# 1-D PISTON: a wall driven into a gas at rest (moving boundary)
# =========================================================================
#
# Gas at rest (ρ = 1, p = 1, γ = 1.4) in the tube [0, 1.5]; the right end is a piston face, a `Moving` half
# space at x = 1 - w t with w = 0.5, treated as a slip wall that moves (`slip_wall()` on a `Moving` piece: the
# mirrored state is reflected relative to the wall velocity). The piston drives a shock into the gas; between
# shock and piston the gas moves with the piston at the Rankine–Hugoniot state (ρ₂, p₂), which `exact_piston`
# gives exactly (the shock runs at S = w ρ₂ / (ρ₂ − ρ₁)).

const PISTON_SPEED = 0.5
const PISTON_FACE  = 1.0

"""
    piston_state(ρ₁, p₁, w, γ) -> (ρ₂, p₂, S)

Rankine–Hugoniot state behind a piston moving at speed `w` into a gas `(ρ₁, p₁)` at rest, and the shock speed `S`
(relative to the gas at rest; the shock moves away from the piston).
"""
function piston_state(ρ₁::Real, p₁::Real, w::Real, γ::Real)
    g(p₂) = (p₂ - p₁) / sqrt(ρ₁ * ((γ + 1) / 2 * p₂ + (γ - 1) / 2 * p₁)) - w
    lo, hi = p₁, 2p₁
    while g(hi) <= 0                      # g grows like √p₂: any piston speed is bracketed
        lo, hi = hi, 2hi
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        g(mid) > 0 ? (hi = mid) : (lo = mid)
    end
    p₂ = (lo + hi) / 2
    ρ₂ = ρ₁ * ((γ + 1) * p₂ + (γ - 1) * p₁) / ((γ - 1) * p₂ + (γ + 1) * p₁)
    return ρ₂, p₂, w * ρ₂ / (ρ₂ - ρ₁)
end

"""
    exact_piston(params) -> (spacetime -> State)

Exact solution of [`euler_1d_piston`](@ref): the gas at rest ahead of the shock, the Rankine–Hugoniot state moving
with the piston behind it (and that state, for reference, beyond the piston face). Reads the gas state from the
`IC` keys, `γ`, the piston speed `:piston_speed` and face position `:piston_face` from `params`.
"""
function exact_piston(params::Dict)
    γ   = Float64(get(params, :PDE_gamma, 1.4))
    uL  = Float64.(collect(params[:IC_uL]))
    ρ₁  = uL[1]
    p₁  = (γ - 1) * (uL[3] - 0.5 * uL[2]^2 / uL[1])
    w   = Float64(get(params, :piston_speed, PISTON_SPEED))
    x_f = Float64(get(params, :piston_face, PISTON_FACE))
    ρ₂, p₂, S = piston_state(ρ₁, p₁, w, γ)
    U₁ = State{3, Float64}(ρ₁, 0.0, p₁ / (γ - 1))
    U₂ = State{3, Float64}(ρ₂, -ρ₂ * w, p₂ / (γ - 1) + 0.5 * ρ₂ * w^2)
    return xt -> xt[1] < x_f - S * xt[2] ? U₁ : U₂
end

# The tube with the piston face at `face` moving at `-speed`, as a string of code in the run's float type `T`: the
# parameter hash sees a string in full (the motion of a `Moving` instance is a closure, which it reduces to
# "Closure", so runs at different piston speeds shared one cache file), and `T` makes it work in Float32
piston_shape(speed::Real, face::Real) =
    "intersect(Tagged(Cuboid((T(0),), (T(1.5),)), 1), Moving(Tagged(HalfSpace((T($face),), (T(1),)), 2), translating((-T($speed),))))"

"""
    euler_1d_piston(; speed = 0.5, face = 1.0) -> SimulationConfig

A piston face starting at `face` and driven at `speed` into a gas at rest in the tube `[0, 1.5]` (the canvas
reaches beyond the face), with the exact solution [`exact_piston`](@ref) as reference. The shape is given as a
string (`:Grid_shape`), so the piston speed is part of the parameter hash and the run follows `:T`; the keys
`:piston_speed` and `:piston_face` feed the reference. Change the speed through the keyword (shape and reference
together), not by overriding `:piston_speed` alone.
"""
function euler_1d_piston(; speed::Real = PISTON_SPEED, face::Real = PISTON_FACE)
    w = speed
    0 < face < 1.5 || throw(ArgumentError("euler_1d_piston: the face must lie inside the canvas (0, 1.5), got $face"))

    shared = Dict{Symbol, Any}(
        :PDE_name => :euler,
        :PDE_D => 1,
        :PDE_gamma => 1.4,

        :sim_func_name => :run_direct_simulation,
        :tmax => 0.3,
        :snapshots => 30,
        :piston_speed => w,
        :piston_face => face,

        :Grid_domain => :shape,
        :Grid_shape => piston_shape(w, face),
        :Grid_mins => (0.0,),
        :Grid_maxs => (1.5,),
        :Grid_Ns => (300,),
        :Grid_periodic => false,
        :Grid_bc => Dict{Int, AbstractBoundaryCondition}(1 => slip_wall(), 2 => slip_wall()),
        :Grid_randomness_factor => (0.0,),
        :Grid_seed => 1,
        :Grid_mover => :none,

        :Weight_name => :exponential,
        :Weight_alpha => 1.0,
        :Weight_range => 2.5,

        :IC_name => :riemann,
        :IC_uL => (1.0, 0.0, 2.5),
        :IC_uR => (1.0, 0.0, 2.5),
        :IC_p0 => (0.5,),
        :IC_n => (1.0,),

        :Time_CFL => 0.2,
    )

    methods = Dict{Symbol, Dict{Symbol, Any}}(
        :exact_piston => Dict(),
        :RK2MUSCL2_VK => Dict(
            :Time_stepper => :RK2,
            :Scheme_name => :MUSCL, :Scheme_order => 2, :Scheme_MLS_order => 0,
            :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard,
            :MOOD_criterion => :none
        ),
        :RK3MUSCL3_U2_EPD1 => Dict(
            :Time_stepper => :RK3,
            :Scheme_name => :MUSCL, :Scheme_order => 3, :Scheme_MLS_order => 0,
            :Flux_name => :Rusanov, :Limiter_name => :none,
            :MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-4
        ),
    )

    active_methods = [:exact_piston, :RK2MUSCL2_VK, :RK3MUSCL3_U2_EPD1]

    return SimulationConfig(shared, methods, active_methods; reference = :exact_piston)
end
