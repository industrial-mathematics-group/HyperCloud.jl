# Round-2 probe: time-loop early-stop detection and BGK tau = 0 runs.
# Run from the repo root: julia -t 2 --project=test <this file>
using HyperCloud, StaticArrays, Test, LinearAlgebra
const HC = HyperCloud
enable_progress_bar!(false)

const UPWIND1 = Dict{Symbol, Any}(
    :Time_stepper => :RK2, :Scheme_name => :Upwind, :Scheme_order => 1, :Scheme_MLS_order => 1,
    :Scheme_upwind_alg_nd => :Classic, :Flux_name => :Upwind, :MOOD_criterion => :none)

lin1d(N; tmax = 0.1, vel = 1.0) = Dict{Symbol, Any}(
    :PDE_name => :linear, :PDE_velocities => ((vel,),), :sim_func_name => :run_direct_simulation,
    :tmax => tmax, :snapshots => 2,
    :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.5,), :IC_width => 0.1, :Time_CFL => 0.2)

warned(logs) = any(l -> l.level == Base.CoreLogging.Warn && occursin("solve_equation stopped", string(l.message)), logs)

bgk_sod(N; Nv = 16, tau = 0.0, stepper = :ARS222) = Dict{Symbol, Any}(
    :PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 1.4, :PDE_Nv => Nv, :PDE_vmax => 6.0, :PDE_tau => tau,
    :sim_func_name => :run_bgk_simulation, :tmax => 0.2, :snapshots => 1,
    :Grid_domain => :rectangular, :Grid_mins => (-0.5,), :Grid_maxs => (0.5,), :Grid_Ns => (N,),
    :Grid_periodic => false, :Grid_bc => Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC()),
    :Grid_randomness_factor => (0.0,), :Grid_seed => 10, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :riemann, :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (0.125, 0.0, 0.25), :IC_p0 => (0.0,), :IC_n => (1.0,),
    :Time_CFL => 0.2, :Time_stepper => stepper,
    :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard,
    :MOOD_criterion => :none)

@testset "probe_runs" begin
@testset "A blown-up run with constant wave speeds is reported" begin
    # Linear advection: max_eigenvalue is a constant, so the CFL step never becomes non-finite.
    for (label, extra) in (("CFL 5", Dict{Symbol, Any}(:Time_CFL => 5.0)),
                           ("fixed dt", Dict{Symbol, Any}(:Time_dt => 0.1)))
        p = merge(lin1d(50; tmax = 20.0), UPWIND1, extra)
        label == "fixed dt" && delete!(p, :Time_CFL)
        logs, sim = Test.collect_test_logs(() -> run_direct_simulation(p))
        @test sim !== nothing
        sim === nothing && continue
        blown = !all(u -> isfinite(u[1]) && abs(u[1]) < 10, sim.u[end])
        println("$label: t[end] = $(sim.t[end]), final state blown up = $blown, ",
                "max |u| = $(maximum(u -> abs(u[1]), sim.u[end])), early-stop warning = $(warned(logs))")
        @test blown          # the run did blow up
        @test warned(logs)   # ... and the user should be told (expected to FAIL at d4fe5e4)
    end
end

@testset "A run with zero wave speeds reaches tmax" begin
    # Stationary data: zero advection speed (CFL step Inf). Before the fix the loop took one step of
    # tmax - t; now an infinite step is treated as a blow-up.
    p = merge(lin1d(50; tmax = 0.1, vel = 0.0), UPWIND1)
    logs, sim = Test.collect_test_logs(() -> run_direct_simulation(p))
    @test sim !== nothing
    if sim !== nothing
        println("zero speed: t = $(sim.t), warning = $(warned(logs))")
        @test sim.t[end] ≈ 0.1          # expected to FAIL at d4fe5e4
        @test !warned(logs)
    end
    # Burgers with u ≡ 0 (Gauss of amplitude 0)
    pb = merge(lin1d(50; tmax = 0.1), UPWIND1, Dict{Symbol, Any}(:PDE_name => :burgers, :PDE_D => 1, :IC_a => (0.0,)))
    logs, simb = Test.collect_test_logs(() -> run_direct_simulation(pb))
    @test simb !== nothing
    simb === nothing || (println("Burgers u=0: t = $(simb.t), warning = $(warned(logs))"); @test simb.t[end] ≈ 0.1)
end

@testset "BGK τ = 0 end to end with every IMEX tableau ($st)" for st in (:ARS222, :ARS233, :IMEXEuler, :SSP332, :PRSSP3)
    p = bgk_sod(200; stepper = st)
    exact = euler_limit_solution(p)
    sim = run_bgk_simulation(p)
    @test sim !== nothing
    if sim !== nothing
        us, xs, t = sim.u[end], sim.x[end], sim.t[end]
        @test t ≈ 0.2
        @test all(u -> all(isfinite, u), us)
        err = sum(i -> abs(exact(SVector(xs[i][1], t))[1] - us[i][1]), eachindex(xs)) / length(xs)
        println("BGK τ = 0, $st: L1(ρ) vs Euler = $err")
        @test err < 0.03
    end
end

@testset "BGK zero viscosity (τ ≡ 0) with ARS222" begin
    p = bgk_sod(200)
    delete!(p, :PDE_tau); p[:PDE_viscosity] = 0.0
    sim = run_bgk_simulation(p)
    @test sim !== nothing && all(u -> all(isfinite, u), sim.u[end])
end

end
