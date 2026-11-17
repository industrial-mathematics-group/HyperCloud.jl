
export run_bgk_simulation

"""
    run_bgk_simulation(params::ParamDict) -> Union{AbstractSimData, Nothing}

Runs the discrete-velocity BGK model (`:PDE_name => :bgk`, see [`BGKEquation`](@ref)) or a gas mixture
(`:PDE_name => :bgk_mixture`, [`BGKMixture`](@ref)). The particle
state is the distribution at the velocity nodes; the `IC` namespace gives the macroscopic state
`(ρ, ρu..., E)` (initial conditions are built with `M = D + 2`, like an Euler run with the same `γ`; for a
mixture the species' number moments `(n_s, n_s u_s..., e_s)`, `M = L (D + 2)`, see [`mixture_state`](@ref)), and
every particle starts in its discrete Maxwellian. The collision term is implicit for an IMEX tableau
([`BGKRelaxation`](@ref)), explicit for an explicit one ([`ExplicitBGKRelaxation`](@ref)) and dropped in
the collisionless limit (`:PDE_tau => Inf`). Schemes, fluxes, limiters, MOOD and the grid are configured
exactly as for the other equations.

The returned snapshots hold the macroscopic moments `(ρ, ρu..., E)`, directly comparable with Euler
runs (for a mixture the species' number moments, see [`species_primitives`](@ref)); with `:save_relax => true` (or `:Kinetic_save_relax => true`, as for `run_kinetic_simulation`) they keep the
full distributions instead.
`run_direct_simulation` and `run_kinetic_simulation` forward BGK configurations here, so the method
dictionaries work unchanged.
"""
function run_bgk_simulation(params::ParamDict)::Union{AbstractSimData, Nothing}
    T = get(params, :T, Float64)

    try
        @info "--- Running BGK Simulation ---"
        context = Dict{Symbol, Any}()
        context[:T] = T
        context[:Storage] = storage_for(get_backend())   # a GPU backend runs on its device
        apply_memory_limit!(context)                      # set_memory_limit!: reserve the buffers once

        # 1. Extract Config Namespaces
        pde_conf    = extract_namespace(params, :PDE)
        domain_conf = extract_namespace(params, :Grid)
        grid_conf   = extract_namespace(params, :Grid)
        weight_conf = extract_namespace(params, :Weight)
        flux_conf   = extract_namespace(params, :Flux)
        lim_conf    = extract_namespace(params, :Limiter)
        mood_conf   = extract_namespace(params, :MOOD)
        scheme_conf = extract_namespace(params, :Scheme)
        ic_conf     = extract_namespace(params, :IC)
        time_conf   = extract_namespace(params, :Time)
        adapt_conf  = extract_namespace(params, :Adapt)

        # 2. Equation; the initial condition lives in the macroscopic (Euler) variables
        eq = build_equation(pde_conf, context)
        eq isa DiscreteVelocityEquation ||
            error("run_bgk_simulation needs a BGK equation (:PDE_name => :bgk or :bgk_mixture), got $(typeof(eq))")
        D = get_D(eq)
        W = Int(get(pde_conf, :block_width, 0))
        W > 0 && return _run_bgk_blocks(params, context, eq, W)

        context[:Equation] = _ic_equation(eq)
        context[:D] = D
        context[:M] = _ic_length(eq)
        IC = build_ic(ic_conf, context)

        context[:Equation] = eq
        context[:M] = get_M(eq)

        # 3. Collision term: implicit for IMEX tableaus, explicit otherwise
        context[:Tableau] = build_tableau(time_conf, context)
        context[:ExplicitSources], context[:ImplicitSources] = bgk_sources(eq, context[:Tableau])
        collision = _is_collisionless(eq) ? "none (collisionless)" :
                    isempty(context[:ImplicitSources]) ? "explicit" : "implicit"
        @info "$(_describe(eq)), collision: $collision"

        # 4. Geometry & Grid
        geom = build_domain(domain_conf, context)
        _check_wall_ops(eq, geom)
        context[:Domain] = geom
        context[:WeightConf] = weight_conf

        pg = build_run_grid(grid_conf, context)           # also sets context[:Grid]
        adaptivity = build_adaptivity(adapt_conf, context)

        # 5. Spatial Schemes
        context[:Flux]    = build_flux(flux_conf, context)
        context[:Limiter] = build_limiter(lim_conf, context)
        context[:MOOD]    = build_mood(mood_conf, context)
        context[:Scheme]  = build_scheme(scheme_conf, context)

        # 6. Time Stepper
        method = build_run_timestepper(context; time_conf = time_conf)
        _log_memory(context)

        # 7. Initialization (discrete Maxwellian of the macroscopic initial state)
        set_initial_conditions!(pg, eq, IC)

        # 8. Time Settings
        is_cfl = haskey(time_conf, :CFL)
        dt = is_cfl ? T(time_conf[:CFL]) : T(time_conf[:dt])
        tmax = T(get(time_conf, :tmax, params[:tmax]))

        # 9. Execution
        xs_svector, us_svector, ts_full, k_step, elapsed_time = solve_equation(
            method, eq, pg, tmax, dt;
            is_cfl = is_cfl, snapshots = params[:snapshots],
            remove_ghosts = get(params, :remove_ghosts, true),
            show_progress = _PROGRESS_BAR[], progress_interval = _PROGRESS_INTERVAL[],
            adaptivity = adaptivity
        )

        @info "BGK Simulation (D=$D) finished in $(round(elapsed_time, digits=2)) seconds."

        # 10. Data Packaging: macroscopic moments unless the distributions are requested
        valid_indices = findall(i -> isassigned(us_svector, i), 1:length(us_svector))
        us_final = if get(params, :save_relax, false) || get(params, :Kinetic_save_relax, false)   # as the kinetic runner
            us_svector[valid_indices]
        else
            [[moments(eq, u) for u in snap] for snap in us_svector[valid_indices]]
        end

        sim_data = create_sim_data(xs_svector[valid_indices], us_final, ts_full[valid_indices], params; xmins=Tuple(geom.mins), xmaxs=Tuple(geom.maxs), tmin=zero(T), tmax=tmax,
                                   metadata = run_metadata(context, k_step, ts_full[last(valid_indices)], tmax;
                                                           final_state = last(us_final)))
        add_stat!(sim_data, :Runtime, elapsed_time, :none)
        add_stat!(sim_data, :k_step, Float64(k_step), :none)
        add_stat!(sim_data, :final_particle_count, Float64(pg.meta.N), :none)
        record_memory!(sim_data, context)

        return sim_data

    catch e
        @error "Error during BGK Simulation!" exception=(e, catch_backtrace())
        return nothing
    end
end

# BGK configurations carry their own collision term, so the direct and kinetic runners hand them over.
_is_bgk(params) = get(params, :PDE_name, nothing) in (:bgk, :bgk_mixture)

"""
    run_bgk_simulation with `:PDE_block_width => W`

The BGK model or the mixture on HyperCloudCore's block pipeline ([`BGKBlocks`](@ref), [`BGKBlockRelaxation`](@ref);
[`MixtureBlocks`](@ref), [`MixtureBlockRelaxation`](@ref)): the state of `W` components per particle and block,
`M / W` blocks (`W` must divide the state length of every species), the collision in two passes. Every particle starts in the
discrete Maxwellian of the `IC` state, and the snapshots hold the macroscopic moments (`:save_relax` is refused:
the full distribution is never assembled). MOOD works with every criterion (the admissibility across the blocks is the
positivity of the moments). Moving grids run in the ALE frame
with the mean velocity of the moments as the particle velocity (the Lagrangian movers; no conservative Lagrangian
frame on blocks), the kinetic walls (`SlipWall`, `DiffuseWall`) act in two passes (the specular reflection as the exact
node flip where it applies, the reflection off the node grid otherwise). Moving boundary pieces are swept with every
block remapped, the particle adaptivity acts on every block (host grids); the storage follows the backend as for the
standard form; the other keys as for the standard form.
"""
function _run_bgk_blocks(params::ParamDict, context::Dict, eq, W::Int)
    T = context[:T]
    pde_conf    = extract_namespace(params, :PDE)
    domain_conf = extract_namespace(params, :Grid)
    grid_conf   = extract_namespace(params, :Grid)
    weight_conf = extract_namespace(params, :Weight)
    flux_conf   = extract_namespace(params, :Flux)
    lim_conf    = extract_namespace(params, :Limiter)
    mood_conf   = extract_namespace(params, :MOOD)
    scheme_conf = extract_namespace(params, :Scheme)
    ic_conf     = extract_namespace(params, :IC)
    time_conf   = extract_namespace(params, :Time)
    adapt_conf  = extract_namespace(params, :Adapt)
    D = get_D(eq)

    haskey(context, :MemoryLimit) && error("the block form does not reserve its buffers under a memory limit yet (set_memory_limit!(nothing))")
    (get(params, :save_relax, false) || get(params, :Kinetic_save_relax, false)) &&
        error("the block form stores the macroscopic moments only (:save_relax is not available)")

    bk = _block_form(eq, W)
    context[:Equation] = _ic_equation(eq)
    context[:D] = D
    context[:M] = _ic_length(eq)
    IC = build_ic(ic_conf, context)
    context[:Equation] = bk
    context[:M] = W

    context[:Tableau] = build_tableau(time_conf, context)
    sources = _block_sources(bk, context[:Tableau])
    @info "$(_describe(eq)), block form: $(HyperCloudCore.num_blocks(bk)) blocks of $W, collision: " *
          (isempty(sources) ? "none (collisionless)" : HyperCloudCore._is_implicit(sources[1]) ? "implicit" : "explicit")

    geom = build_domain(domain_conf, context)
    _check_wall_ops(eq, geom)
    context[:Domain] = geom
    context[:WeightConf] = weight_conf
    pg = build_run_grid(grid_conf, context)
    adaptivity = build_adaptivity(adapt_conf, context)
    context[:Flux]    = build_flux(flux_conf, context)
    context[:Limiter] = build_limiter(lim_conf, context)
    context[:MOOD]    = build_mood(mood_conf, context)
    context[:Scheme]  = build_scheme(scheme_conf, context)
    # the standard stepper for one block carries the options (the volume estimator of a moving grid, the MOOD substeps, …)
    context[:Equation] = BlockOf(bk, 1)
    context[:ExplicitSources], context[:ImplicitSources] = (), ()
    inner  = build_run_timestepper(context; time_conf = time_conf)
    context[:Equation] = bk
    method = BlockTimeStepper(bk, inner, sources)

    bs = BlockState(bk, pg; storage = context[:Storage])
    set_initial_conditions!(bs, bk, pg, IC)

    is_cfl = haskey(time_conf, :CFL)
    dt = is_cfl ? T(time_conf[:CFL]) : T(time_conf[:dt])
    tmax = T(get(time_conf, :tmax, params[:tmax]))
    xs_svector, us_svector, ts_full, k_step, elapsed_time = solve_equation(
        method, bk, pg, bs, tmax, dt;
        reduce = (Y, N, nb, i) -> block_moments(bk, Y, N, nb, i),
        is_cfl = is_cfl, snapshots = params[:snapshots], remove_ghosts = get(params, :remove_ghosts, true),
        show_progress = _PROGRESS_BAR[], progress_interval = _PROGRESS_INTERVAL[], adaptivity = adaptivity)
    @info "BGK Simulation (block form, D=$D) finished in $(round(elapsed_time, digits=2)) seconds."

    valid_indices = findall(i -> isassigned(us_svector, i), 1:length(us_svector))
    us_final = us_svector[valid_indices]
    sim_data = create_sim_data(xs_svector[valid_indices], us_final, ts_full[valid_indices], params; xmins=Tuple(geom.mins), xmaxs=Tuple(geom.maxs), tmin=zero(T), tmax=tmax,
                               metadata = run_metadata(context, k_step, ts_full[last(valid_indices)], tmax;
                                                       final_state = last(us_final)))
    add_stat!(sim_data, :Runtime, elapsed_time, :none)
    add_stat!(sim_data, :k_step, Float64(k_step), :none)
    add_stat!(sim_data, :final_particle_count, Float64(pg.meta.N), :none)
    return sim_data
end

_block_form(eq::BGKEquation, W::Int) = BGKBlocks(eq, Val(W))
_block_form(eq::BGKMixture, W::Int) = MixtureBlocks(eq, Val(W))

# The block sources of a run: implicit for an IMEX tableau, explicit otherwise, none when collisionless
function _block_sources(bk::BGKBlocks, tableau::IMEXButcherTableau)
    eq = bk.eq
    _is_collisionless(eq.relaxation) && return ()
    all(iszero, tableau.a) || return (BlockSourceTerm(BGKBlockRelaxation(bk)),)
    _is_zero_relaxation(eq.relaxation) && throw(ArgumentError(
        "BGK: the relaxation time is zero, which the explicit collision term (M[f] - f) / τ cannot handle; " *
        "use an IMEX time stepper (e.g. :ARS222) for τ = 0"))
    return (BlockSourceTerm(ExplicitBGKBlockRelaxation(bk)),)
end
function _block_sources(mb::MixtureBlocks, tableau::IMEXButcherTableau)
    _is_collisionless(mb.mix) && return ()
    all(iszero, tableau.a) || return (BlockSourceTerm(MixtureBlockRelaxation(mb)),)
    return (BlockSourceTerm(ExplicitMixtureBlockRelaxation(mb)),)
end

# The equation and state length the initial conditions are built for, and the log line of the run
_ic_equation(eq::BGKEquation) = euler_limit(eq)
_ic_equation(eq::BGKMixture) = eq
_ic_length(eq::BGKEquation) = get_D(eq) + 2
_ic_length(eq::BGKMixture) = num_species(eq) * (get_D(eq) + 2)
_describe(eq::BGKEquation) = "BGK: D = $(get_D(eq)), $(num_velocities(eq)) velocities, $(get_M(eq)) equations, " *
                             "γ = $(round(eq.gamma; digits = 4))"
_describe(eq::BGKMixture) = "BGK mixture: D = $(get_D(eq)), $(num_species(eq)) species, $(num_velocities(eq)) velocities, " *
                            "$(get_M(eq)) equations, ε = $(eq.epsilon), κ = $(eq.kappa)"
