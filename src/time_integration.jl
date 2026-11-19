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

# The step does not see every blow-up (fixed steps, constant wave speeds, CFL estimates that skip NaN
# particles), so every stored state is checked as well. Returns `(blown_up, growth_warned)`.
function _check_snapshot(state, t, tmax, k_step, u0_scale, growth_warned::Bool; final::Bool = false)
    scale = _max_abs(state)
    if !isfinite(scale)
        @warn "solve_equation: the state at t = $t is not finite (blown up)" *
              (final ? "." : "; the run stopped before tmax = $tmax after $k_step steps. The data end at t = $t.")
        return true, growth_warned
    elseif !growth_warned && u0_scale > 0 && scale > 1e8 * u0_scale
        @warn "solve_equation: at t = $t the state is $(scale / u0_scale) times larger than at t = 0; " *
              "the run is probably unstable."
        return false, true
    end
    return false, growth_warned
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

The loop stops early with a warning that gives the time reached if the time step collapses (below `1e-12 tmax`)
or is not finite, or if a stored snapshot is not finite (a blown-up state; this also covers fixed time steps
and equations with constant wave speeds, whose step never collapses). Every stored snapshot is checked,
the final one included. The last stored state is the one at that time, and the snapshots that were not
reached are left unassigned. Each blow-up is reported once (the final slot of a stopped run is not checked again).
A snapshot that grew by more than a factor `1e8` over the initial state gives a warning (once) but does not stop
the run. An infinite CFL step with a finite state and a finite CFL number (no wave moves) advances straight to
`tmax`; an infinite CFL number (or fixed step) stops at once.
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
    div_interp = hasproperty(timestepper, :divergence_interpolator) ? timestepper.divergence_interpolator : NoFallbackGrad()

    step!(t, d) = (timestepper(eq, pg, t, d); adaptivity === nothing || adapt_particles!(pg, adaptivity, _extract_order(div_interp)); nothing)
    save!(k, t) = (saveData!(xs, us, ts, k, pg, t, remove_ghosts); us[k])
    k_step, elapsed_time = _time_loop(step!, () -> get_time_step(pg, eq, div_interp), save!,
                                      () -> isfinite(_max_abs(host_view(pg.rhos, pg.meta.N))),
                                      T(tmax), T(dt), is_cfl, Int(snapshots), show_progress, progress_interval)
    return xs, us, ts, k_step, elapsed_time
end

"""
    solve_equation(timestepper::BlockTimeStepper, eq, pg, bs::BlockState, tmax, dt; reduce, kwargs...)

The time loop of [`solve_equation`](@ref) for the block pipeline: the block state `bs` is advanced, and the
stored snapshots hold `reduce(Y, N, nb, i)` per particle (the macroscopic moments, e.g.
`(Y, N, nb, i) -> block_moments(bk, Y, N, nb, i)`; a kernel function, run on the storage of the state and copied to the
host), never the full distribution; `adaptivity` is applied between steps on every block. The other keywords as for the
standard pipeline.
"""
function solve_equation(
    timestepper::BlockTimeStepper,
    eq::HyperbolicPDE{D, W, T},
    pg::ParticleGrid{D, W, T},
    bs::BlockState{W, T},
    tmax::Real,
    dt::Real;
    reduce,
    is_cfl::Bool = false,
    snapshots::Integer = 10,
    remove_ghosts::Bool = false,
    show_progress::Bool = true,
    progress_interval::Real = 1.0,
    adaptivity::Union{Nothing, ParticleAdaptivity} = nothing
) where {D, W, T}
    snapshots >= 1 || throw(ArgumentError("snapshots must be ≥ 1 (the number of output intervals), got $snapshots"))
    # the type of the reduction from particle 1's blocks (a host copy of nb states, not of the state)
    R  = typeof(reduce(Array(copy(view(parent(bs.data), 1:bs.N[]:(bs.nb * bs.N[])))), 1, bs.nb, 1))
    xs = Vector{Vector{Space{D, T}}}(undef, snapshots + 1)
    us = Vector{Vector{R}}(undef, snapshots + 1)
    ts = Vector{T}(undef, snapshots + 1)
    div_interp = timestepper.inner.divergence_interpolator
    # the reduction of every particle on the storage of the state (a kernel), copied to the host: a snapshot moves
    # (D + 2) N values, not the state
    moments = HyperCloudCore.Growable{R}(typeof(parent(bs.data)).name.wrapper)
    function reduce_all!()
        N = pg.meta.N
        HyperCloudCore.ensure_capacity!(moments, N)
        foreach_particle(_block_reduce_all!, get_backend(), N, (moments, bs.data, bs.N[], bs.nb, reduce))
        return host_view(parent(moments), N)
    end

    step!(t, d) = (timestepper(eq, pg, bs, t, d);
                   adaptivity === nothing || adapt_particles!(pg, adaptivity, _extract_order(div_interp), bs); nothing)
    save!(k, t) = (_save_block_data!(xs, us, ts, k, pg, t, remove_ghosts, reduce_all!()); us[k])
    finite()    = all(m -> all(isfinite, m), reduce_all!())
    k_step, elapsed_time = _time_loop(step!, () -> get_time_step(pg, eq, div_interp, bs), save!, finite,
                                      T(tmax), T(dt), is_cfl, Int(snapshots), show_progress, progress_interval)
    return xs, us, ts, k_step, elapsed_time
end

"""
    solve_equation(devs::BlockDevices, eq, tmax, dt; reduce_partial, partial_type, kwargs...)

The time loop of a block state split over devices: `devs(t, dt)` advances, and the stored snapshots hold the sum
over the parts of `reduce_partial(Y, N, nb, blocks, i)` (a kernel function of a part's blocks, the moments of a block
range: `(Y, N, nb, blocks, i) -> block_moments(bk, Y, N, nb, i, blocks)`), of type `partial_type`, computed on every
device and summed on the host (`host_reduction`). No adaptivity. The other keywords as for the block state.
"""
function solve_equation(devs::BlockDevices{D, W, T}, eq, tmax::Real, dt::Real; reduce_partial, partial_type::Type{P}, is_cfl::Bool = false,
                        snapshots::Integer = 10, remove_ghosts::Bool = false, show_progress::Bool = true, progress_interval::Real = 1.0) where {D, W, T, P}
    snapshots >= 1 || throw(ArgumentError("snapshots must be ≥ 1 (the number of output intervals), got $snapshots"))
    pg = first(devs.parts).pg
    xs = Vector{Vector{Space{D, T}}}(undef, snapshots + 1)
    us = Vector{Vector{P}}(undef, snapshots + 1)
    ts = Vector{T}(undef, snapshots + 1)
    moments()   = host_reduction(reduce_partial, devs, P)
    step!(t, d) = (devs(t, d); nothing)
    save!(k, t) = (_save_block_data!(xs, us, ts, k, pg, t, remove_ghosts, moments()); us[k])
    finite()    = all(m -> all(isfinite, m), moments())
    k_step, elapsed_time = _time_loop(step!, () -> get_time_step(devs), save!, finite,
                                      T(tmax), T(dt), is_cfl, Int(snapshots), show_progress, progress_interval)
    return xs, us, ts, k_step, elapsed_time
end

# Kernel body (see HyperCloudCore's Backend.jl): the reduction of particle `i` over its blocks
@inline function _block_reduce_all!(i, out, Y, N, nb, reduce)
    @inbounds out[i] = reduce(Y, N, nb, i)
    return nothing
end
# The stored data of a block state: positions and the per-particle reductions (`moments`, on the host)
function _save_block_data!(xs, us, ts, k::Int, pg::ParticleGrid{D, W, T}, t, remove_ghosts::Bool, moments) where {D, W, T}
    N  = pg.meta.N
    ts[k] = T(t)
    pos = host_view(get_positions(pg), N)
    idx = remove_ghosts ? findall(!, host_view(pg.core.is_boundary, N)) : collect(1:N)
    xs[k] = [pos[i] for i in idx]
    us[k] = [moments[i] for i in idx]
    return nothing
end
# The block-major state on the host (a view of a host array, a copy of a device array)
_host_blocks(bs::BlockState) = host_view(parent(bs.data), bs.nb * bs.N[])

# The time loop itself: `step!(t, dt)` advances, `cfl_step()` is the CFL step, `save!(k, t)` stores snapshot `k` and
# returns the stored states, `state_finite()` says whether the current state is finite. Returns `(k_step, elapsed)`.
function _time_loop(step!, cfl_step, save!, state_finite, tmax_val::T, dt_inp::T, is_cfl::Bool, snapshots::Int,
                    show_progress::Bool, progress_interval::Real) where {T}
    t_snap = range(zero(T), tmax_val, length=snapshots+1)
    snap_counter = 1
    t = zero(T)
    k_step = 0

    u0_scale = _max_abs(save!(snap_counter, t))
    snap_counter += 1
    growth_warned = false

    @info "Using $(nameof(typeof(get_backend()))) for parallel runs!"

    p = Progress(10000, desc="Running Simulation...", dt=progress_interval, enabled=show_progress)
    
    # Initialize trackers for the interval-based ETA
    last_log_time = time()
    last_sim_time = t
    stopped = false      # stopped early: the stop has been reported, the final slot is not checked again

    elapsed_time = @elapsed while t < tmax_val
        dt_step = is_cfl ? dt_inp * cfl_step() : dt_inp
        # A collapsed or non-finite step (NaN states, unbounded wave speeds) ends the run early, so no NaN
        # time is ever recorded. An infinite CFL step means that no wave moves, and the run goes straight to
        # tmax, unless the state is not finite (the CFL estimate skips particles with NaN wave speeds).
        # An infinite CFL number is a configuration error, not a field at rest: it stops below.
        if is_cfl && isfinite(dt_inp) && dt_step == T(Inf)
            if !state_finite()
                @warn "solve_equation stopped at t = $t before tmax = $tmax_val after $k_step steps: the state " *
                      "is not finite (blown up). The data end at t = $t."
                stopped = true
                break
            end
        elseif !(isfinite(dt_step) && dt_step > T(1e-12) * tmax_val)      # relative: no fixed time scale
            @warn "solve_equation stopped at t = $t before tmax = $tmax_val after $k_step steps: the time step " *
                  "is $(isfinite(dt_step) ? "collapsed" : "not finite") (dt = $dt_step" *
                  (isfinite(dt_inp) && dt_inp > zero(T) ? "); the state has probably blown up." :
                   ", from $(is_cfl ? "the CFL number" : "the fixed step") $dt_inp, which must be positive and finite).") *
                  " The data end at t = $t."
            stopped = true
            break
        end
        dt_val = min(dt_step, tmax_val - t)
        if dt_val <= max(T(1e-12), 4 * eps(T)) * tmax_val; break; end      # tmax reached up to round-off (relative)

        step!(t, dt_val)

        t += dt_val
        k_step += 1
        
        blown_up = false
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            state = save!(snap_counter, t)
            blown_up, growth_warned = _check_snapshot(state, t, tmax_val, k_step, u0_scale, growth_warned)
            snap_counter += 1
            blown_up && break
        end
        blown_up && (stopped = true; break)

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
        state = save!(snapshots + 1, t)
        # The final slot is checked too (with one snapshot it is the only one after t = 0), unless the run stopped:
        # then it holds the state whose stop was just reported
        stopped || _check_snapshot(state, t, tmax_val, k_step, u0_scale, growth_warned; final = true)
    end

    return k_step, elapsed_time
end
