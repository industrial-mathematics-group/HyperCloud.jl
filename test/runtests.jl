using HyperCloud
using StaticArrays
using Test
using LinearAlgebra
using Dates

const HC = HyperCloud

# GLMakie needs a display. On headless CI run with HYPERCLOUD_GUI_TESTS=false
# (or wrap the job in xvfb-run). `using` must stay at top level.
const RUN_GUI_TESTS = get(ENV, "HYPERCLOUD_GUI_TESTS", "true") == "true"
if RUN_GUI_TESTS
    using PDEStudio
    using GLMakie
end

# Execution backend of the HyperCloudCore per-particle loops: polyester (default), threads or ka_cpu
# (KernelAbstractions kernels on the host).
const TEST_BACKEND = get(ENV, "HYPERCLOUD_BACKEND", "polyester")
if TEST_BACKEND == "polyester"
    set_backend!(PolyesterBackend())
elseif TEST_BACKEND == "threads"
    set_backend!(ThreadsBackend())
elseif TEST_BACKEND == "ka_cpu"
    set_backend!(HyperCloudCore.KA.CPU())
else
    error("HYPERCLOUD_BACKEND=$(TEST_BACKEND) is not supported (polyester, threads, ka_cpu)")
end
@info "Testing with $(get_backend()) on $(Threads.nthreads()) threads"

# =========================================================================
# 0. MOCK MODULES (must be defined at top-level scope)
# =========================================================================

module MockPDEStudio
    using HyperCloud
    using Base

    launch_plotter() = "mock_figure"
    set_sim_config!(conf) = nothing
    force_simulation() = nothing
    dummy_config() = "mock_config"
end

module MockGLMakie
    activate!() = nothing
end

module MockWGLMakie
    activate!() = nothing
end

module MockBonito
    configure_server!(; listen_url, listen_port, proxy_url) = nothing
end

# =========================================================================
# 1. HELPERS
# =========================================================================

"Dispatch to the pipeline named in `params[:sim_func_name]`."
run_sim(params) = getfield(HyperCloud, params[:sim_func_name])(params)
@eval HyperCloud dummy_func(params) = $(NoSimData)()
"Spacetime vector `(x..., t)` as expected by the analytic closures."
spacetime(x, t) = SVector(Tuple(x)..., Float64(t))

"""
    l1_error(params; component = 1) -> (err, sim_data)

Runs the simulation described by `params` and returns the mean absolute error of
the final snapshot against `analytical_solution(params)`. Returns `(NaN, nothing)`
if the run failed. The runners catch every exception and return `nothing`, so
callers must always check `sim !== nothing`.
"""
function l1_error(params; component::Int = 1)
    exact = analytical_solution(params)
    sim = run_sim(params)
    sim === nothing && return (NaN, nothing)
    xs, us, t = sim.x[end], sim.u[end], sim.t[end]
    err = sum(i -> abs(exact(spacetime(xs[i], t))[component] - us[i][component]), eachindex(xs))
    return (err / length(xs), sim)
end

"Midpoint-rule quadrature (robust to the discontinuities in exact solutions)."
function integrate(f, a, b; n::Int = 20_000)
    h = (b - a) / n
    return h * sum(i -> f(a + (i - 0.5) * h), 1:n)
end

function test_domain(mins, maxs, periodic)
    conf = Dict{Symbol, Any}(:domain => :rectangular, :mins => mins, :maxs => maxs, :periodic => periodic)
    return build_domain(conf, Dict{Symbol, Any}(:T => Float64, :D => length(mins)))
end

function test_grid(mins, maxs, Ns; M::Int = 1, randomness::Float64 = 0.0, seed::Int = 42)
    D = length(mins)
    ctx = Dict{Symbol, Any}(:T => Float64, :D => D, :M => M)
    ctx[:Domain] = test_domain(mins, maxs, true)
    ctx[:WeightConf] = Dict{Symbol, Any}(:name => :exponential, :alpha => 1.0, :range => 2.5)
    grid_conf = Dict{Symbol, Any}(:Ns => Ns, :randomness_factor => ntuple(_ -> randomness, D), :seed => seed)
    return build_particle_grid(grid_conf, ctx), ctx
end

# --- Reusable configurations ----------------------------------------------

const UPWIND1 = Dict{Symbol, Any}(
    :Time_stepper => :RK2,
    :Scheme_name => :Upwind, :Scheme_order => 1, :Scheme_MLS_order => 1,
    :Scheme_upwind_alg_nd => :Classic, :Flux_name => :Upwind,
    :MOOD_criterion => :none,
)

"1D periodic linear advection of a Gaussian; the workhorse for accuracy tests."
linear_1d_gauss(N; tmax = 0.1, width = 0.1) = Dict{Symbol, Any}(
    :PDE_name => :linear, :PDE_velocities => ((1.0,),),
    :sim_func_name => :run_direct_simulation,
    :tmax => tmax, :snapshots => 2,
    :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.5,), :IC_width => width,
    :Time_CFL => 0.2,
)

"1D periodic Burgers with a Gaussian pulse (breaking time ≈ 0.23, so keep tmax below that)."
burgers_1d_gauss(N; tmax = 0.15) = Dict{Symbol, Any}(
    :PDE_name => :burgers, :PDE_D => 1,
    :sim_func_name => :run_direct_simulation,
    :tmax => tmax, :snapshots => 2,
    :Grid_domain => :rectangular, :Grid_mins => (-1.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.0,), :IC_width => 0.2,
    :Time_CFL => 0.2,
)

"2D linear advection along the diagonal on the unit square."
function linear_2d_base(periodic::Bool)
    d = Dict{Symbol, Any}(
        :PDE_name => :linear, :PDE_D => 2, :PDE_velocities => ((1.0,), (1.0,)),
        :sim_func_name => :run_direct_simulation,
        :tmax => 0.1, :snapshots => 2,
        :Grid_domain => :rectangular, :Grid_mins => (0.0, 0.0), :Grid_maxs => (1.0, 1.0),
        :Grid_Ns => (20, 20), :Grid_periodic => periodic,
        :Grid_randomness_factor => (0.0, 0.0), :Grid_seed => 42, :Grid_mover => :none,
        :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
        :Time_CFL => 0.2,
    )
    if !periodic
        # NOTE: boundary conditions are read from `:Grid_bc` (`:Grid_bc_map` is an alias).
        d[:Grid_bc] = Dict{Int, AbstractBoundaryCondition}(i => OutflowBC() for i in 1:4)
    end
    return d
end

"1D BGK model (γ = 3: one distribution, `M = Nv`) with a smooth periodic density/momentum/energy wave."
bgk_wave_params(N; Nv = 16, tau = Inf, stepper = :RK3, order = 3, T = Float64) = Dict{Symbol, Any}(
    :PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 3.0, :PDE_Nv => Nv, :PDE_vmax => 6.0, :PDE_tau => tau,
    :sim_func_name => :run_bgk_simulation, :T => T,
    :tmax => 0.2, :snapshots => 2,
    :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :sine, :IC_a => (0.2, 0.1, 0.1), :IC_period => (1.0,), :IC_c_offset => (1.0, 0.0, 0.5),
    :Time_CFL => 0.2, :Time_stepper => stepper,
    :Scheme_name => :MUSCL, :Scheme_order => order, :Flux_name => :Rusanov, :Limiter_name => :none,
    :MOOD_criterion => :none,
)

"Sod shock tube for the 1D BGK model with γ = 1.4 (Chu-reduced pair f, h: `M = 2 Nv`)."
bgk_sod_params(N; Nv = 16, tau = 1e-5) = Dict{Symbol, Any}(
    :PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 1.4, :PDE_Nv => Nv, :PDE_vmax => 6.0, :PDE_tau => tau,
    :sim_func_name => :run_kinetic_simulation,                      # forwarded to run_bgk_simulation
    :tmax => 0.2, :snapshots => 1,
    :Grid_domain => :rectangular, :Grid_mins => (-0.5,), :Grid_maxs => (0.5,), :Grid_Ns => (N,),
    :Grid_periodic => false, :Grid_bc => Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC()),
    :Grid_randomness_factor => (0.0,), :Grid_seed => 10, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :riemann, :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (0.125, 0.0, 0.25), :IC_p0 => (0.0,), :IC_n => (1.0,),
    :Time_CFL => 0.2, :Time_stepper => :ARS222,
    :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard,
    :MOOD_criterion => :none,
)

"Mean density error of a BGK run against the exact Euler solution with the same γ (`(err, sim)`)."
function euler_l1(params)
    exact = euler_limit_solution(params)
    sim = run_sim(params)
    sim === nothing && return (NaN, nothing)
    xs, us, t = sim.x[end], sim.u[end], sim.t[end]
    return sum(i -> abs(exact(spacetime(xs[i], t))[1] - us[i][1]), eachindex(xs)) / length(xs), sim
end

"Totals Σ_i U_i of the stored macroscopic snapshots (equal volumes on a regular lattice)."
totals(sim, snap) = sum(sim.u[snap])

const IMEX_STEPPERS = (:ARS222, :ARS233, :IMEXEuler, :PRSSP3, :SSP332)

# Master-dict entries that currently fail the sweep. They are reported as "Broken"
# instead of failing. Delete an entry once it is fixed (Julia then reports "Unexpected Pass").
const KNOWN_BROKEN_METHODS = Dict{String, String}()

# =========================================================================
# MAIN TEST SUITE: test groups
# =========================================================================
# The suite is dominated by compilation (every method is a new combination of scheme, MOOD and tableau types,
# compiled anew; compilation is single threaded). The groups are independent and run in separate processes in
# parallel:
#   HYPERCLOUD_TEST_JOBS=n     at most n processes (default: the CPU threads, at most the number of groups and
#                              one per 2.5 GiB of free memory; a group peaks at 1.4-2 GiB); 1 runs every group in
#                              this process, one after the other.
#   HYPERCLOUD_TEST_GROUP=name runs one group in this process (what the parallel runner starts).
const TEST_GROUPS = [
    "units"    => ["units.jl"],
    "sims-bgk" => ["sims_bgk.jl"],
    "sweep-a"  => ["sims_sweep.jl"],
    "sweep-b"  => ["sims_sweep.jl"],
    "sims"     => ["sims.jl"],
    "examples" => ["examples.jl"],
]
const TEST_GROUP = get(ENV, "HYPERCLOUD_TEST_GROUP", "")
TEST_GROUP == "" || TEST_GROUP in first.(TEST_GROUPS) ||
    error("HYPERCLOUD_TEST_GROUP=$(TEST_GROUP) is not a test group ($(join(first.(TEST_GROUPS), ", ")))")

# The master-dictionary sweep is split over the two `sweep` groups (all methods when run in one process).
function _sweep_cases(cases)
    h = cld(length(cases), 2)
    return TEST_GROUP == "sweep-a" ? cases[1:h] : TEST_GROUP == "sweep-b" ? cases[h+1:end] : cases
end

function run_group(name; label = name)
    @testset "$label" begin
        for file in Dict(TEST_GROUPS)[name]
            include(file)
        end
    end
end

# Runs the groups in child processes (same julia flags, e.g. --check-bounds and coverage, the same project and
# environment, so HYPERCLOUD_BACKEND and HYPERCLOUD_GUI_TESTS carry over) and prints each group's output when it
# finishes: in full if it failed, else its test summary.
function run_groups_in_parallel(jobs)
    names = first.(TEST_GROUPS)
    threads = max(1, Threads.nthreads() ÷ jobs)
    cmd(name) = addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=$threads $(@__FILE__)`,
                       "HYPERCLOUD_TEST_GROUP" => name)
    logs = Dict(name => tempname() for name in names)
    queue, running, failed = copy(names), Dict{String, Tuple{Base.Process, IOStream, Float64}}(), String[]
    println("Running $(length(names)) test groups in up to $jobs processes ($threads thread(s) each)")
    while !isempty(queue) || !isempty(running)
        while length(running) < jobs && !isempty(queue)
            name = popfirst!(queue)
            io = open(logs[name], "w")
            running[name] = (run(pipeline(cmd(name); stdout = io, stderr = io); wait = false), io, time())
        end
        sleep(0.5)
        for (name, (proc, io, t0)) in collect(running)
            process_exited(proc) || continue
            close(io)
            delete!(running, name)
            out = read(logs[name], String)
            ok = success(proc)
            ok || push!(failed, name)
            println("\n", "="^30, " test group `$name`: ", ok ? "passed" : "FAILED", " in $(round(time() - t0; digits = 1)) s")
            i = findlast("Test Summary:", out)
            print(ok && i !== nothing ? out[first(i):end] : out)
        end
    end
    @testset "HyperCloud Tests" begin
        @testset "$name" for name in names
            @test !(name in failed)
        end
    end
end

const TEST_JOBS = parse(Int, get(ENV, "HYPERCLOUD_TEST_JOBS",
    string(clamp(min(Sys.CPU_THREADS, Int(Sys.free_memory() ÷ (5 * 2^29))), 1, length(TEST_GROUPS)))))

if TEST_GROUP != ""
    run_group(TEST_GROUP)
    println("test group `$(TEST_GROUP)`: peak memory $(Sys.maxrss() ÷ 2^20) MiB")
elseif TEST_JOBS <= 1
    @testset "HyperCloud Tests" begin
        for name in first.(TEST_GROUPS)
            name == "sweep-b" && continue            # `sweep-a` runs every method in one process
            run_group(name; label = name == "sweep-a" ? "sweep" : name)
        end
    end
else
    run_groups_in_parallel(TEST_JOBS)
end
