# Probe: all-NaN state -> get_time_step returns Inf (NaN wave speeds are skipped) -> new code jumps to tmax
# Run: HYPERCLOUD_GUI_TESTS=false julia -t 2 --project=test probe_nan.jl [old|new]
using HyperCloud, Test, Logging
const S = @__DIR__
which = get(ARGS, 1, "new")
which == "old" && Base.include(HyperCloud, joinpath(S, "old_time_integration.jl"))
set_backend!(PolyesterBackend())
const UPWIND1 = Dict{Symbol, Any}(:Time_stepper => :RK2, :Scheme_name => :Upwind, :Scheme_order => 1,
    :Scheme_MLS_order => 1, :Scheme_upwind_alg_nd => :Classic, :Flux_name => :Upwind, :MOOD_criterion => :none)
burgers(; tmax = 0.15, a = 1.0, cfl = 0.2, snaps = 2) = merge(UPWIND1, Dict{Symbol, Any}(
    :PDE_name => :burgers, :PDE_D => 1, :sim_func_name => :run_direct_simulation, :tmax => tmax, :snapshots => snaps,
    :Grid_domain => :rectangular, :Grid_mins => (-1.0,), :Grid_maxs => (1.0,), :Grid_Ns => (50,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (a,), :IC_b => (0.0,), :IC_width => 0.2, :Time_CFL => cfl))
maxabs(u) = maximum(s -> maximum(abs, s), u)
function report(label, p)
    logs, sim = Test.collect_test_logs(() -> run_direct_simulation(p))
    warns = [first(split(string(l.message), ';')) for l in logs if l.level >= Logging.Warn]
    sim === nothing && return println(rpad(label, 40), " nothing ", warns)
    println(rpad(label, 40), " t = ", round.(sim.t; sigdigits = 4), " k_step = ", sim.stats[:k_step], " | final max|u| = ",
            maxabs(sim.u[end]), " | warns = ", warns)
end
println("=== code: $which ===")
for snaps in (1, 4)
    report("Burgers IC_a = NaN snaps=$snaps", burgers(; a = NaN, snaps))
end
for snaps in (1, 4), cfl in (3.0, 8.0)
    report("Burgers CFL $cfl tmax 2 snaps=$snaps", burgers(; tmax = 2.0, cfl, snaps))
end
