export solve_equation, saveData!, snapshot_field, DEFAULT_SNAPSHOT_FIELDS

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

# =========================================================================
# WHAT A RUN STORES BESIDES THE STATES: PER-PARTICLE ARRAYS OF THE GRID
# =========================================================================
"""
    snapshot_field(::Val{name}, pg::ParticleGrid) -> AbstractVector

The per-particle array of the grid that [`solve_equation`](@ref) stores with every snapshot for `name` in its
`snapshot_fields` (as a statistic that keeps all dimensions, converted to the run's float type; the ghosts are left out
with the positions when `remove_ghosts` is set). The array may live on a device and be longer than the particle count
(the first `pg.meta.N` entries are read). Defined names:

- `:volumes`: the particle volumes (`Σ V u` is the discrete mass of a moving or adapted cloud);
- `:particle_orders`: the consistency order of every particle after the last step (lowered by MOOD);
- `:neighbor_counts`: the number of neighbours;
- `:is_boundary`: 1 for a ghost (boundary) particle, 0 for an interior one;
- `:tags`: the boundary tag (0 for an interior particle);
- `:wall_distances`: the distance of a ghost to its wall.

A method for another `name` makes it available (`HyperCloud.snapshot_field(::Val{:my_field}, pg) = …`).
"""
snapshot_field(::Val{name}, pg::ParticleGrid) where {name} =
    throw(ArgumentError("no snapshot field :$name (defined: :volumes, :particle_orders, :neighbor_counts, :is_boundary, :tags, " *
                        ":wall_distances; add a method of HyperCloud.snapshot_field for another one)"))
snapshot_field(::Val{:volumes}, pg::ParticleGrid) = pg.core.volumes
snapshot_field(::Val{:particle_orders}, pg::ParticleGrid) = pg.core.particle_orders
snapshot_field(::Val{:neighbor_counts}, pg::ParticleGrid) = pg.neighbor.counts
snapshot_field(::Val{:is_boundary}, pg::ParticleGrid) = pg.core.is_boundary
snapshot_field(::Val{:tags}, pg::ParticleGrid) = pg.core.tags
snapshot_field(::Val{:wall_distances}, pg::ParticleGrid) = pg.core.wall_distances

"The per-particle arrays [`solve_equation`](@ref) stores with every snapshot by default: `(:volumes, :particle_orders)`."
const DEFAULT_SNAPSHOT_FIELDS = (:volumes, :particle_orders)

# `:snapshot_fields` of a runner's parameters (a name, or a tuple or vector of names; `default` without the key)
function _snapshot_fields(params::AbstractDict, default = DEFAULT_SNAPSHOT_FIELDS)
    f = get(params, :snapshot_fields, default)
    f isa Symbol && (f = (f,))
    (f isa Union{Tuple, AbstractVector} && all(x -> x isa Symbol, f)) ||
        throw(ArgumentError(":snapshot_fields must be a field name or a tuple or vector of names (Symbols), got $(repr(f))"))
    return Tuple(f)
end

# The storage of the snapshot fields, one slot per snapshot (in the order of `names`; an unknown name fails here, before
# the run)
function _field_storage(names, pg::ParticleGrid{D, M, T}, nslots::Int) where {D, M, T}
    all(n -> n isa Symbol, names) || throw(ArgumentError("snapshot_fields must be field names (Symbols), got $(repr(names))"))
    allunique(names) || throw(ArgumentError("snapshot_fields names a field twice: $(repr(names))"))
    foreach(n -> snapshot_field(Val(n), pg), names)
    return [n => Vector{Vector{T}}(undef, nslots) for n in names]
end
# Stores the snapshot fields of slot `k`: the particles `idx` of every field (the same as the positions)
function _save_fields!(fields, k::Int, pg::ParticleGrid{D, M, T}, idx) where {D, M, T}
    for (name, store) in fields
        a = host_view(snapshot_field(Val(name), pg), pg.meta.N)
        store[k] = T[T(a[i]) for i in idx]
    end
    return nothing
end
# The dataset of a run (PDEStudioCore's `LSimData`): the stored snapshots (the slots that were reached), the run's
# metadata and its statistics with the dimensions each keeps
function _run_result(xs, us, ts, fields, k_step::Int, elapsed::Real, pg::ParticleGrid, tmax::Real, params, checkpoint, xmins, xmaxs)
    taken = findall(k -> isassigned(us, k), eachindex(us))
    xs, us, ts = xs[taken], us[taken], ts[taken]
    T = eltype(ts)
    mins = something(xmins, _domain_bounds(pg)[1])
    maxs = something(xmaxs, _domain_bounds(pg)[2])
    sim = create_sim_data(xs, us, ts, params; xmins = Tuple(mins), xmaxs = Tuple(maxs), tmin = zero(T), tmax = tmax,
                          metadata = run_metadata(T, k_step, last(ts), tmax; final_state = last(us), checkpoint = checkpoint))
    add_stat!(sim, :Runtime, Float64(elapsed), Symbol[])
    add_stat!(sim, :k_step, k_step, Symbol[])
    add_stat!(sim, :final_particle_count, pg.meta.N, Symbol[])
    add_stat!(sim, :particle_count, [length(x) for x in xs], :time)
    for (name, store) in fields
        add_stat!(sim, name, store[taken], :all)
    end
    return sim
end
# The bounds of the domain (the dataset's extent): the geometry's box, else the canvas of the computational domain
_domain_bounds(pg::ParticleGrid) = hasproperty(pg.geometry, :mins) ? (pg.geometry.mins, pg.geometry.maxs) :
                                                                      (pg.domain.canvas_mins, pg.domain.canvas_maxs)

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
    saveData!(xs_storage, us_storage, ts_storage, snap_idx, pg, current_t, remove_ghosts; fields = (), snapshot_state = identity)

Extracts and archives the simulation state at a specific snapshot index.

# Details
- Records the current time into `ts_storage`.
- If `remove_ghosts` is true, it filters out boundary particles and saves only the active domain core. Otherwise, it copies the entire grid.
- Allocates new state and position vectors for the targeted snapshot and copies the corresponding views from the particle grid.
- `fields`: `name => storage` pairs; slot `snap_idx` of each storage gets the grid's [`snapshot_field`](@ref) `name` of the
  same particles.
- `snapshot_state`: the stored value of a particle's state `u` is `snapshot_state(u)` (e.g. the macroscopic moments of
  a kinetic state); `identity` stores the state.
"""
function saveData!(
    xs_storage::AbstractVector, 
    us_storage::AbstractVector, 
    ts_storage::AbstractVector, 
    snap_idx::Int, 
    pg::ParticleGrid{D, M, T}, 
    current_t::Real, 
    remove_ghosts::Bool;
    fields = (),
    snapshot_state::F = identity
) where {D, M, T, F}
    
    ts_storage[snap_idx] = T(current_t)
    N_active = pg.meta.N 
    
    pos_array = host_view(get_positions(pg), N_active)
    rho_array = host_view(pg.rhos, N_active)
    
    if remove_ghosts
        indices = findall(!, host_view(pg.core.is_boundary, N_active))
        N_save = length(indices)
        
        xs_storage[snap_idx] = Vector{Space{D, T}}(undef, N_save)
        copyto!(xs_storage[snap_idx], view(pos_array, indices))
        us_storage[snap_idx] = _stored_states(snapshot_state, view(rho_array, indices))
        _save_fields!(fields, snap_idx, pg, indices)
    else
        N_save = N_active
        
        xs_storage[snap_idx] = Vector{Space{D, T}}(undef, N_save)
        copyto!(xs_storage[snap_idx], view(pos_array, 1:N_save))
        us_storage[snap_idx] = _stored_states(snapshot_state, view(rho_array, 1:N_save))
        _save_fields!(fields, snap_idx, pg, 1:N_save)
    end
    return nothing
end

# The stored values of a snapshot's states: a copy of the states, or `f` of each
_stored_states(::typeof(identity), states) = collect(states)
_stored_states(f::F, states) where {F} = [f(u) for u in states]

"""
    solve_equation(timestepper, eq, pg, tmax, dt; kwargs...) -> LSimData

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
- `checkpoint`: a [`Checkpointing`](@ref) (or `nothing`): the state of the run is saved to its file every `every` seconds
  of wall time, and a run that finds the file resumes from it (the grid, the states and the snapshots stored so far are
  restored before the first step; the initial conditions set before are overwritten) and continues bit for bit as the
  run that wrote it would have. The file stays after the run (`remove_checkpoint`).
- `snapshot_fields`: the per-particle arrays of the grid stored with every snapshot ([`snapshot_field`](@ref) names,
  default [`DEFAULT_SNAPSHOT_FIELDS`](@ref) `= (:volumes, :particle_orders)`; `()` stores none).
- `snapshot_state`: what is stored of a particle's state `u`: `snapshot_state(u)` (default `identity`, the state; e.g.
  `u -> moments(eq, u)` stores the macroscopic moments of a kinetic state). The blow-up check runs on the stored values.
- `params`: the parameters of the run, the dataset's `params` (they name its cache file in PDEStudioCore's pipeline).
- `xmins`, `xmaxs`: the extent of the dataset (default: the box of the grid's geometry).

# Returns
- The run as a PDEStudioCore `LSimData`: `sim.x[k]`, `sim.u[k]`, `sim.t[k]` of the stored snapshots (the ones reached: a
  run that stopped early has fewer than `snapshots + 1`), the metadata of the run (`run_metadata`: versions, backend,
  steps, the time reached, `:stopped_early`, `:finite`, `:restarts`) and its statistics, each registered with the
  dimensions it keeps: `:Runtime` (wall time of the loop in seconds) and `:k_step` (steps), both over all attempts of a
  resumed run, `:final_particle_count` (numbers), `:particle_count` (the particles stored per snapshot) and every name
  of `snapshot_fields` (one value per particle and snapshot, aligned with `sim.x[k]`).

The loop stops early with a warning that gives the time reached if the time step collapses (below `1e-12 tmax`)
or is not finite, or if a stored snapshot is not finite (a blown-up state; this also covers fixed time steps
and equations with constant wave speeds, whose step never collapses). Every stored snapshot is checked,
the final one included. The last stored state is the one at that time, and the snapshots that were not
reached are not part of the dataset. Each blow-up is reported once (the final slot of a stopped run is not checked again).
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
    adaptivity::Union{Nothing, ParticleAdaptivity} = nothing,
    checkpoint::Union{Nothing, Checkpointing} = nothing,
    snapshot_fields = DEFAULT_SNAPSHOT_FIELDS,
    snapshot_state::F = identity,
    params::AbstractDict = ParamDict(),
    xmins = nothing,
    xmaxs = nothing
) where {D, M, T, F}
    snapshots >= 1 || throw(ArgumentError("snapshots must be ≥ 1 (the number of output intervals), got $snapshots"))
    # the stored value of a state (the state, or `snapshot_state` of it: e.g. the moments of a kinetic state)
    R  = snapshot_state === identity ? State{M, T} : Base.promote_op(snapshot_state, State{M, T})
    isconcretetype(R) || (R = typeof(snapshot_state(zero(State{M, T}))))
    xs = Vector{Vector{Space{D, T}}}(undef, snapshots + 1)
    us = Vector{Vector{R}}(undef, snapshots + 1)
    ts = Vector{T}(undef, snapshots + 1)
    fields = _field_storage(snapshot_fields, pg, snapshots + 1)
    div_interp = hasproperty(timestepper, :divergence_interpolator) ? timestepper.divergence_interpolator : NoFallbackGrad()

    # (the content form of the Lagrangian movers keeps the scheme's volumes between the steps: the management must not
    # replace them all with the estimator's)
    own_volumes = adaptivity !== nothing && keeps_own_volumes(timestepper, pg)
    step!(t, d) = (timestepper(eq, pg, t, d);
                   adaptivity === nothing || adapt_particles!(pg, adaptivity, _extract_order(div_interp); own_volumes = own_volumes); nothing)
    save!(k, t) = (saveData!(xs, us, ts, k, pg, t, remove_ghosts; fields, snapshot_state); us[k])
    resume = _resume!(ck -> restore_checkpoint!(pg, ck), checkpoint, xs, us, ts, fields)
    k_step, elapsed_time = _time_loop(step!, () -> get_time_step(pg, eq, div_interp), save!,
                                      () -> isfinite(_max_abs(host_view(pg.rhos, pg.meta.N))),
                                      T(tmax), T(dt), is_cfl, Int(snapshots), show_progress, progress_interval;
                                      _checkpoint_kw(checkpoint, () -> checkpoint_state(pg), xs, us, ts, fields, resume)...)
    return _run_result(xs, us, ts, fields, k_step, elapsed_time, pg, tmax, params, checkpoint, xmins, xmaxs)
end

# The checkpoint keywords of `_time_loop`: the resumed loop state, the writer of a checkpoint (the solver state from
# `state()`) and its interval
_checkpoint_kw(::Nothing, state, xs, us, ts, fields, resume) = (; resume)
_checkpoint_kw(ck::Checkpointing, state, xs, us, ts, fields, resume) =
    (; resume, checkpoint! = loop -> _write_checkpoint(ck, state(), loop, xs, us, ts, fields), every = ck.every)

"""
    solve_equation(timestepper::BlockTimeStepper, eq, pg, bs::BlockState, tmax, dt; reduce, kwargs...)

The time loop of [`solve_equation`](@ref) for the block pipeline: the block state `bs` is advanced, and the
stored snapshots hold `reduce(Y, N, nb, i)` per particle (the macroscopic moments, e.g.
`(Y, N, nb, i) -> block_moments(bk, Y, N, nb, i)`; a kernel function, run on the storage of the state and copied to the
host), never the full distribution; `adaptivity` is applied between steps on every block. The other keywords and the
returned `LSimData` as for the standard pipeline (a `checkpoint` saves and restores the block state with the
grid).
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
    adaptivity::Union{Nothing, ParticleAdaptivity} = nothing,
    checkpoint::Union{Nothing, Checkpointing} = nothing,
    snapshot_fields = DEFAULT_SNAPSHOT_FIELDS,
    params::AbstractDict = ParamDict(),
    xmins = nothing,
    xmaxs = nothing
) where {D, W, T}
    snapshots >= 1 || throw(ArgumentError("snapshots must be ≥ 1 (the number of output intervals), got $snapshots"))
    # the type of the reduction from particle 1's blocks (a host copy of nb states, not of the state)
    R  = typeof(reduce(Array(copy(view(parent(bs.data), 1:bs.N[]:(bs.nb * bs.N[])))), 1, bs.nb, 1))
    xs = Vector{Vector{Space{D, T}}}(undef, snapshots + 1)
    us = Vector{Vector{R}}(undef, snapshots + 1)
    ts = Vector{T}(undef, snapshots + 1)
    fields = _field_storage(snapshot_fields, pg, snapshots + 1)
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
    save!(k, t) = (_save_block_data!(xs, us, ts, fields, k, pg, t, remove_ghosts, reduce_all!()); us[k])
    finite()    = all(m -> all(isfinite, m), reduce_all!())
    resume = _resume!(ck -> restore_checkpoint!(pg, bs, ck), checkpoint, xs, us, ts, fields)
    k_step, elapsed_time = _time_loop(step!, () -> get_time_step(pg, eq, div_interp, bs), save!, finite,
                                      T(tmax), T(dt), is_cfl, Int(snapshots), show_progress, progress_interval;
                                      _checkpoint_kw(checkpoint, () -> checkpoint_state(pg, bs), xs, us, ts, fields, resume)...)
    return _run_result(xs, us, ts, fields, k_step, elapsed_time, pg, tmax, params, checkpoint, xmins, xmaxs)
end

"""
    solve_equation(devs::BlockDevices, eq, tmax, dt; reduce_partial, partial_type, kwargs...)

The time loop of a block state split over devices: `devs(t, dt)` advances, and the stored snapshots hold the sum
over the parts of `reduce_partial(Y, N, nb, blocks, i)` (a kernel function of a part's blocks, the moments of a block
range: `(Y, N, nb, blocks, i) -> block_moments(bk, Y, N, nb, i, blocks)`), of type `partial_type`, computed on every
device and summed on the host (`host_reduction`). No adaptivity. The other keywords and the returned
`LSimData` as for the block state (a `checkpoint` holds the blocks of all parts: it resumes on any split, or on
a single block state; the snapshot fields are those of the first part's grid, the same cloud on every part).
"""
function solve_equation(devs::BlockDevices{D, W, T}, eq, tmax::Real, dt::Real; reduce_partial, partial_type::Type{P}, is_cfl::Bool = false,
                        snapshots::Integer = 10, remove_ghosts::Bool = false, show_progress::Bool = true, progress_interval::Real = 1.0,
                        checkpoint::Union{Nothing, Checkpointing} = nothing, snapshot_fields = DEFAULT_SNAPSHOT_FIELDS,
                        params::AbstractDict = ParamDict(), xmins = nothing, xmaxs = nothing) where {D, W, T, P}
    snapshots >= 1 || throw(ArgumentError("snapshots must be ≥ 1 (the number of output intervals), got $snapshots"))
    pg = first(devs.parts).pg                         # (every part holds the same cloud: the fields are read from the first)
    xs = Vector{Vector{Space{D, T}}}(undef, snapshots + 1)
    us = Vector{Vector{P}}(undef, snapshots + 1)
    ts = Vector{T}(undef, snapshots + 1)
    fields = _field_storage(snapshot_fields, pg, snapshots + 1)
    moments()   = host_reduction(reduce_partial, devs, P)
    step!(t, d) = (devs(t, d); nothing)
    save!(k, t) = (_save_block_data!(xs, us, ts, fields, k, pg, t, remove_ghosts, moments()); us[k])
    finite()    = all(m -> all(isfinite, m), moments())
    resume = _resume!(ck -> restore_checkpoint!(devs, ck), checkpoint, xs, us, ts, fields)
    k_step, elapsed_time = _time_loop(step!, () -> get_time_step(devs), save!, finite,
                                      T(tmax), T(dt), is_cfl, Int(snapshots), show_progress, progress_interval;
                                      _checkpoint_kw(checkpoint, () -> checkpoint_state(devs), xs, us, ts, fields, resume)...)
    return _run_result(xs, us, ts, fields, k_step, elapsed_time, pg, tmax, params, checkpoint, xmins, xmaxs)
end

# Kernel body (see HyperCloudCore's Backend.jl): the reduction of particle `i` over its blocks
@inline function _block_reduce_all!(i, out, Y, N, nb, reduce)
    @inbounds out[i] = reduce(Y, N, nb, i)
    return nothing
end
# The stored data of a block state: positions, the per-particle reductions (`moments`, on the host) and the snapshot fields
function _save_block_data!(xs, us, ts, fields, k::Int, pg::ParticleGrid{D, W, T}, t, remove_ghosts::Bool, moments) where {D, W, T}
    N  = pg.meta.N
    ts[k] = T(t)
    pos = host_view(get_positions(pg), N)
    idx = remove_ghosts ? findall(!, host_view(pg.core.is_boundary, N)) : collect(1:N)
    xs[k] = [pos[i] for i in idx]
    us[k] = [moments[i] for i in idx]
    _save_fields!(fields, k, pg, idx)
    return nothing
end
# The block-major state on the host (a view of a host array, a copy of a device array)
_host_blocks(bs::BlockState) = host_view(parent(bs.data), bs.nb * bs.N[])

# A test hook: the time loop throws an `InterruptException` right after the step with this number (counted over all
# attempts of a resumed run), before that step's snapshot and checkpoint, as a run that is killed there (0: never)
const _INTERRUPT_AT_STEP = Ref(0)

# The time loop itself: `step!(t, dt)` advances, `cfl_step()` is the CFL step, `save!(k, t)` stores snapshot `k` and
# returns the stored states, `state_finite()` says whether the current state is finite. Returns `(k_step, elapsed)`.
# Checkpoints: `resume` is the loop state of a checkpoint (`_resume!`; the solver state and the snapshots are restored
# already), `checkpoint!(loop)` writes one with the loop state `loop`, every `every` seconds of wall time at the end of a
# step (not after a stop). `elapsed` includes the wall time of the attempts before a resume.
function _time_loop(step!, cfl_step, save!, state_finite, tmax_val::T, dt_inp::T, is_cfl::Bool, snapshots::Int,
                    show_progress::Bool, progress_interval::Real; resume = nothing, checkpoint! = nothing,
                    every::Real = Inf) where {T}
    t_snap = range(zero(T), tmax_val, length=snapshots+1)
    if resume === nothing
        snap_counter = 1
        t = zero(T)
        k_step = 0

        u0_scale = _max_abs(save!(snap_counter, t))
        snap_counter += 1
        growth_warned = false
        elapsed_before, restarts = 0.0, 0
    else
        t, k_step, snap_counter = T(resume.t), Int(resume.k_step), Int(resume.snap_counter)
        u0_scale, growth_warned = resume.u0_scale, Bool(resume.growth_warned)
        elapsed_before, restarts = Float64(resume.elapsed), Int(resume.restarts) + 1
    end
    t_start = time()
    last_checkpoint = t_start

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
        # (relative: no fixed time scale; and not below the resolution of the time itself, 4 eps(T) tmax: a Float32 run
        # whose CFL step fell into (1e-12, 4 eps] tmax used to leave through the end-of-run test below, without a message)
        elseif !(isfinite(dt_step) && dt_step > max(T(1e-12), 4 * eps(T)) * tmax_val)
            @warn "solve_equation stopped at t = $t before tmax = $tmax_val after $k_step steps: the time step " *
                  "is $(isfinite(dt_step) ? "collapsed" : "not finite") (dt = $dt_step" *
                  (isfinite(dt_inp) && dt_inp > zero(T) ? "); the state has probably blown up." :
                   ", from $(is_cfl ? "the CFL number" : "the fixed step") $dt_inp, which must be positive and finite).") *
                  " The data end at t = $t."
            stopped = true
            break
        end
        if tmax_val - t <= max(T(1e-12), 4 * eps(T)) * tmax_val; break; end   # tmax reached up to round-off (relative)
        dt_val = min(dt_step, tmax_val - t)

        step!(t, dt_val)

        t += dt_val
        k_step += 1
        k_step == _INTERRUPT_AT_STEP[] && throw(InterruptException())      # (tests: killed before the snapshot and checkpoint)

        blown_up = false
        while snap_counter <= snapshots && t >= t_snap[snap_counter]
            state = save!(snap_counter, t)
            blown_up, growth_warned = _check_snapshot(state, t, tmax_val, k_step, u0_scale, growth_warned)
            snap_counter += 1
            blown_up && break
        end
        blown_up && (stopped = true; break)

        if checkpoint! !== nothing && time() - last_checkpoint >= every
            checkpoint!((; t, k_step, snap_counter, u0_scale, growth_warned, elapsed = elapsed_before + (time() - t_start), restarts))
            last_checkpoint = time()
        end

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

    return k_step, elapsed_before + elapsed_time
end
