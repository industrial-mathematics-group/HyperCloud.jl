# Probe for d3a0415 (final-snapshot check, infinite CFL step with host state check, fixed-dt handling).
# Run from the repo root: HYPERCLOUD_GUI_TESTS=false julia -t 2 --project=test <this> [old|new]
#   old = re-include f982101's src/time_integration.jl into HyperCloud
using HyperCloud, Test, Logging
const S = @__DIR__
which = get(ARGS, 1, "new")
which == "old" && Base.include(HyperCloud, joinpath(S, "old_time_integration.jl"))
set_backend!(PolyesterBackend())

const UPWIND1 = Dict{Symbol, Any}(:Time_stepper => :RK2, :Scheme_name => :Upwind, :Scheme_order => 1,
    :Scheme_MLS_order => 1, :Scheme_upwind_alg_nd => :Classic, :Flux_name => :Upwind, :MOOD_criterion => :none)
linear_1d_gauss(N; tmax = 0.1, width = 0.1) = Dict{Symbol, Any}(
    :PDE_name => :linear, :PDE_velocities => ((1.0,),), :sim_func_name => :run_direct_simulation,
    :tmax => tmax, :snapshots => 2, :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,),
    :Grid_Ns => (N,), :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.5,), :IC_width => width, :Time_CFL => 0.2)
burgers_1d_gauss(N; tmax = 0.15) = Dict{Symbol, Any}(
    :PDE_name => :burgers, :PDE_D => 1, :sim_func_name => :run_direct_simulation,
    :tmax => tmax, :snapshots => 2, :Grid_domain => :rectangular, :Grid_mins => (-1.0,), :Grid_maxs => (1.0,),
    :Grid_Ns => (N,), :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.0,), :IC_width => 0.2, :Time_CFL => 0.2)

maxabs(u) = maximum(s -> maximum(abs, s), u)
function report(label, p)
    logs, sim = Test.collect_test_logs(() -> run_direct_simulation(p))
    warns = [first(split(string(l.message), ';')) for l in logs if l.level >= Logging.Warn]
    if sim === nothing
        println(rpad(label, 40), " sim = nothing; warns = ", warns)
        return sim, warns
    end
    println(rpad(label, 40), " t = ", round.(sim.t; sigdigits = 4), " | final max|u| = ", maxabs(sim.u[end]),
            " | #warn = ", length(warns), " ", warns)
    return sim, warns
end
println("=== code: $which ===")

# The three new tests of test/sims.jl:79-92, as @testsets (they must fail on `old`)
pb = merge(linear_1d_gauss(50; tmax = 100.0), UPWIND1, Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => 10))
@testset "new sims.jl tests ($which)" begin
    pb1 = merge(pb, Dict{Symbol, Any}(:snapshots => 1))
    @test_logs (:warn, r"is not finite \(blown up\)") match_mode = :any run_direct_simulation(pb1)
    pnan = merge(burgers_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:IC_a => (NaN,), :snapshots => 1))
    simnan = @test_logs (:warn, r"is not finite \(blown up\)") match_mode = :any run_direct_simulation(pnan)
    @test simnan !== nothing && simnan.t[end] == 0.0
    pinf = merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:Time_dt => Inf)); delete!(pinf, :Time_CFL)
    siminf = @test_logs (:warn, r"time step is not finite") match_mode = :any run_direct_simulation(pinf)
    @test siminf !== nothing && siminf.t == [0.0, 0.0]
end

# Extra scenarios
for snaps in (10, 2, 1)
    report("lin CFL5 snaps=$snaps", merge(pb, Dict{Symbol, Any}(:snapshots => snaps)))
end
report("lin CFL5 snaps=1 Float32", merge(pb, Dict{Symbol, Any}(:snapshots => 1, :T => Float32)))
report("lin CFL5 snaps=1 remove_ghosts=false", merge(pb, Dict{Symbol, Any}(:snapshots => 1, :remove_ghosts => false)))
report("Burgers NaN snaps=4", merge(burgers_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:IC_a => (NaN,), :snapshots => 4)))
report("lin vel 0 CFL (finite, Inf step)", merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:PDE_velocities => ((0.0,),))))
report("lin CFL = Inf", merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:Time_CFL => Inf)))
report("lin CFL = -0.2", merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:Time_CFL => -0.2)))
report("lin vel 0, CFL = -0.2 (-Inf step)", merge(linear_1d_gauss(50), UPWIND1,
       Dict{Symbol, Any}(:PDE_velocities => ((0.0,),), :Time_CFL => -0.2)))
for dtv in (Inf, -Inf, NaN, -0.01, 0.0)
    p = merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:Time_dt => dtv)); delete!(p, :Time_CFL)
    report("lin fixed dt = $dtv", p)
end
sod = euler_1d_sod()
for snaps in (1, 4)
    p = merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_dt => 0.05,
              :Limiter_name => :none, :tmax => 0.2, :snapshots => snaps)); delete!(p, :Time_CFL)
    report("Sod fixed dt 0.05 nolim snaps=$snaps", p)
end
# Legitimate runs: no warnings expected
report("Sod default (RK2MUSCL2_VK) snaps=1", merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :snapshots => 1)))
report("Burgers CFL 0.2 tmax 2 snaps=1", merge(burgers_1d_gauss(50; tmax = 2.0), UPWIND1, Dict{Symbol, Any}(:snapshots => 1)))
