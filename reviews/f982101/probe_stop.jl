# Probe: early-stop / blow-up detection of solve_equation (f982101 vs d4fe5e4)
# Run: HYPERCLOUD_GUI_TESTS=false julia -t 2 --project=test probe_stop.jl [old|new]
using HyperCloud, Test, Logging
const HC = HyperCloud
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

maxabs(u) = maximum(s -> maximum(abs, s), u)
function report(label, p)
    logs, sim = Test.collect_test_logs(() -> run_direct_simulation(p))
    warns = [first(split(string(l.message), ';')) for l in logs if l.level >= Logging.Warn]
    if sim === nothing
        println(rpad(label, 46), " sim = nothing; warns = ", warns)
        return (nothing, warns)
    end
    println(rpad(label, 46), " t = ", round.(sim.t; sigdigits = 4), " | final max|u| = ", maxabs(sim.u[end]),
            " | finite = ", all(s -> all(isfinite, s), sim.u[end]), " | warns = ", warns)
    return (sim, warns)
end
println("=== code: $which ===")
for (snaps, tmax) in ((10, 100.0), (2, 100.0), (1, 100.0), (1, 20.0))
    report("lin CFL5 snaps=$snaps tmax=$tmax", merge(linear_1d_gauss(50; tmax), UPWIND1,
           Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => snaps)))
end
report("lin vel 0 CFL", merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:PDE_velocities => ((0.0,),))))
for dtv in (Inf, -0.01, 0.0)
    p = merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:Time_dt => dtv)); delete!(p, :Time_CFL)
    report("lin fixed dt = $dtv", p)
end
sod = euler_1d_sod()
for snaps in (1, 4, 51)
    p = merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_CFL => 4.0,
              :Limiter_name => :none, :tmax => 0.2, :snapshots => snaps))
    report("Sod CFL4 nolim snaps=$snaps", p)
end
for snaps in (1, 4)
    p = merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_dt => 0.05,
              :Limiter_name => :none, :tmax => 0.2, :snapshots => snaps)); delete!(p, :Time_CFL)
    report("Sod fixed dt 0.05 nolim snaps=$snaps", p)
end
