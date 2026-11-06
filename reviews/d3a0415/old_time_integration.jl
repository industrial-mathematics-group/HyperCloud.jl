export solve_equation, saveData!

"""
    NoFallbackGrad <: DivergenceInterpolator

Placeholder used by [`solve_equation`](@ref) when the time stepper exposes no
`divergence_interpolator`: the CFL estimate falls back to a first-order MLS fit and the
maximum consistency order is 1.
"""
struct NoFallbackGrad <: DivergenceInterpolator end
HyperCloudCore._extract_order(::NoFallbackGrad) = 1
HyperCloudCore._divergence_degree(::NoFallbackGrad) = 1
HyperCloudCore.update_size!(::NoFallbackGrad, ::Int) = nothing

# Largest absolute component of a stored state (NaN or Inf if any component is not finite).
function _max_abs(states::AbstractVector)
    m = zero(eltype(eltype(states)))
    for u in states, c in u
        isfinite(c) || return oftype(m, Inf)
        m = max(m, abs(c))
    end
    return m
end

"""
    saveData!(xs_storage, us_storage, ts_storage, snap_idx, pg, current_t, remove_ghosts)

Extracts and archives the simulation state at a specific snapshot index.

# Details
- Records the current time into `ts_storage`.
- If `remove_ghosts` is true, it filters out boundary particles and saves only the active domain core. Otherwise, it copies the entire grid.
- Allocates new state and position vectors for the targeted snapshot and copies the corresponding views from the particle grid.
"""
function saveData!(
    xs_storage::AbstractVector, 
    us_storage::AbstractVector, 
    ts_storage::AbstractVector, 
    snap_idx::Int, 
    pg::ParticleGrid{D, M, T}, 
    current_t::Real, 
    remove_ghosts::Bool
) where {D, M, T}
    
    ts_storage[snap_idx] = T(current_t)
    N_active = pg.meta.N 
    
    pos_array = host_view(get_positions(pg), N_active)
    rho_array = host_view(pg.rhos, N_active)
    
    if remove_ghosts
        indices = findall(!, host_view(pg.core.is_boundary, N_active))
        N_save = length(indices)
        
        xs_storage[snap_idx] = Vector{Space{D, T}}(undef, N_save)
        us_storage[snap_idx] = Vector{State{M, T}}(undef, N_save)
        
        copyto!(xs_storage[snap_idx], view(pos_array, indices))
        copyto!(us_storage[snap_idx], view(rho_array, indices))
    else
        N_save = N_active
        
        xs_storage[snap_idx] = Vector{Space{D, T}}(undef, N_save)
        us_storage[snap_idx] = Vector{State{M, T}}(undef, N_save)
        
        copyto!(xs_storage[snap_idx], view(pos_array, 1:N_save))
        copyto!(us_storage[snap_idx], view(rho_array, 1:N_save))
    end
end

"""
    solve_equation(timestepper, eq, pg, tmax, dt; kwargs...)

The primary simulation orchestrator governing the main time-stepping loop. 

# Arguments
- `timestepper::TimeStepper`: The selected Runge-Kutta or IMEX time integrator.
- `eq::HyperbolicPDE`: The physical equation system.
- `pg::ParticleGrid`: The active mesh-free domain configuration.
- `tmax::Real`: The final simulation time.
- `dt::Real`: The baseline time step.

# Keyword Arguments
- `is_cfl::Bool`: If true, `dt` is treated as a CFL number, and the physical time step is dynamically computed at each iteration using the grid and interpolator properties.
- `snapshots::Integer`: The number of output intervals (at least 1); `snapshots + 1` states are stored, including `t = 0`.
- `remove_ghosts::Bool`: Strips boundary/ghost particles from the returned snapshot data if true.
- `show_progress::Bool`: Toggles visual progress tracking.
- `progress_interval::Real`: Sets the refresh rate (in seconds) for logging the simulation's progress and calculating the ETA.
- `adaptivity`: a `ParticleAdaptivity` applied between steps (particle insertion/merging on moving grids), or `nothing`.

# Returns
- A tuple containing: `(position_history, state_history, time_history, total_steps, elapsed_wall_time)`.

The loop stops early with a warning that gives the time reached if the time step collapses (below `1e-12`)
or is NaN, or if a stored snapshot is not finite (a blown-up state; this also covers fixed time steps and
equations with constant wave speeds, whose step never collapses). The last stored state is the one at that
time, and the snapshots that were not reached are left unassigned. A snapshot that grew by more than a
factor `1e8` over the initial state gives a warning (once) but does not stop the run. An infinite step (no
wave moves) advances straight to `tmax`.
"""
function solve_equation(
    timestepper::TimeStepper, 
    eq::HyperbolicPDE{D, M, T}, 
    pg::ParticleGrid{D, M, T},
    tmax::Real,
    dt::Real;
    is_cfl::Bool = false,
    snapshots::Integer = 10,
    remove_ghosts::Bool = false,
    show_progress::Bool = true,
    progress_interval::Real = 1.0,
    adaptivity::Union{Nothing, ParticleAdaptivity} = nothing
) where {D, M, T}
    
    snapshots >= 1 || throw(ArgumentError("snapshots must be ≥ 1 (the number of output intervals), got $snapshots"))
    xs = Vector{Vector{Space{D, T}}}(undef, snapshots + 1)
    us = Vector{Vector{State{M, T}}}(undef, snapshots + 1)
    ts = Vector{T}(undef, snapshots + 1)

    tmax_val = T(tmax)
    dt_inp = T(dt)
    t_snap = range(zero(T), tmax_val, length=snapshots+1)
    snap_counter = 1
    t = zero(T)
    k_step = 0
    
    div_interp = hasproperty(timestepper, :divergence_interpolator) ? timestepper.divergence_interpolator : NoFallbackGrad()

    saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
    snap_counter += 1 
    u0_scale = _max_abs(us[1])
    growth_warned = false
    
    @info "Using $(nameof(typeof(get_backend()))) for parallel runs!"

    p = Progress(10000, desc="Running Simulation...", dt=progress_interval, enabled=show_progress)
    
    # Initialize trackers for the interval-based ETA
    last_log_time = time()
    last_sim_time = t

    elapsed_time = @elapsed while t < tmax_val
        dt_step = is_cfl ? dt_inp * get_time_step(pg, eq, div_interp) : dt_inp
        # A collapsed or NaN step (NaN states, unbounded wave speeds) ends the run early; `!(dt > ε)` also
        # catches NaN, so no NaN time is ever recorded. An infinite step (no wave moves) goes straight to tmax.
        if !(dt_step > T(1e-12))
            @warn "solve_equation stopped at t = $t before tmax = $tmax_val after $k_step steps: the time step " *
                  "is $(isnan(dt_step) ? "not finite" : "collapsed") (dt = $dt_step); the state has probably " *
                  "blown up. The data end at t = $t."
            break
        end
        dt_val = min(dt_step, tmax_val - t)
        if dt_val <= T(1e-12); break; end      # tmax reached up to round-off

        timestepper(eq, pg, t, dt_val)
        adaptivity === nothing || adapt_particles!(pg, adaptivity, _extract_order(div_interp))

        t += dt_val
        k_step += 1
        
        blown_up = false
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            saveData!(xs, us, ts, snap_counter, pg, t, remove_ghosts)
            # The step does not see every blow-up (fixed steps, constant wave speeds, CFL estimates that
            # skip NaN particles), so check the stored state as well.
            scale = _max_abs(us[snap_counter])
            if !isfinite(scale)
                @warn "solve_equation stopped at t = $t before tmax = $tmax_val after $k_step steps: the state " *
                      "is not finite (blown up). The data end at t = $t."
                blown_up = true
            elseif !growth_warned && u0_scale > 0 && scale > T(1e8) * u0_scale
                @warn "solve_equation: at t = $t the state is $(scale / u0_scale) times larger than at t = 0; " *
                      "the run is probably unstable."
                growth_warned = true
            end
            snap_counter += 1
            blown_up && break
        end
        blown_up && break

        if show_progress
            current_progress = ceil(Int, (t / tmax_val) * 10000)
            update!(p, min(current_progress, 10000))
        else
            current_time = time()
            wall_dt = current_time - last_log_time
            
            if wall_dt > progress_interval
                sim_dt = t - last_sim_time
                pct = round((t / tmax_val) * 100, digits=1)
                
                # Calculate ETA only if simulation has advanced
                if sim_dt > 0
                    eta_seconds = (tmax_val - t) * (wall_dt / sim_dt)
                    eta_secs_int = round(Int, eta_seconds)
                    
                    # Format as HH:MM:SS
                    h = eta_secs_int ÷ 3600
                    m = (eta_secs_int % 3600) ÷ 60
                    s = eta_secs_int % 60
                    eta_str = string(lpad(h, 2, '0'), ":", lpad(m, 2, '0'), ":", lpad(s, 2, '0'))
                    
                    @info "Simulation Progress: $pct% | ETA: $eta_str"
                else
                    @info "Simulation Progress: $pct% | ETA: Calculating..."
                end
                
                # Reset interval trackers
                last_log_time = current_time
                last_sim_time = t
            end
        end
    end
    
    show_progress && finish!(p)

    if snap_counter <= snapshots + 1
        saveData!(xs, us, ts, snapshots + 1, pg, t, remove_ghosts)
    end

    return xs, us, ts, k_step, elapsed_time
end