using HyperCloud, StaticArrays, Test
enable_progress_bar!(false)
include(joinpath(@__DIR__, "rt_helpers.jl"))
function main()
    pb1 = merge(linear_1d_gauss(50; tmax = 100.0), UPWIND1, Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => 1))
    sim = run_direct_simulation(pb1)
    println("pb1 (blows up, snapshots=1): t_end=", sim.t[end], " finite=", all(u -> all(isfinite, u), sim.u[end]), " md=", sim.metadata)
    sod1 = euler_1d_sod()
    pf = merge(sod1.shared, sod1.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_dt => 0.05,
               :Limiter_name => :none, :tmax => 0.2, :snapshots => 1))
    delete!(pf, :Time_CFL)
    sim = run_direct_simulation(pf)
    println("pf (fixed dt blow-up): t_end=", sim.t[end], " finite=", all(u -> all(isfinite, u), sim.u[end]), " stopped_early=", sim.metadata[:stopped_early])
    # Float32 normal run
    p32 = merge(linear_1d_gauss(50; tmax = 0.3), UPWIND1, Dict{Symbol, Any}(:T => Float32))
    sim = run_direct_simulation(p32)
    println("Float32 normal: t_end=", sim.t[end], " md=", sim.metadata)
    p32 = merge(linear_1d_gauss(50; tmax = 1e-13), UPWIND1)
    sim = run_direct_simulation(p32)
    println("tmax=1e-13: t_end=", sim.t[end], " stopped_early=", sim.metadata[:stopped_early])
end
main()
# @sim edge cases
@eval Main my_cfg() = HyperCloud.SimulationConfig(merge(linear_1d_gauss(30), UPWIND1), Dict{Symbol,Dict{Symbol,Any}}(:a => Dict{Symbol,Any}()), [:a])
for ex in (:(HyperCloud.@sim my_cfg dry_run = true dry_run = false), :(HyperCloud.@sim my_cfg calculate_stats = false dry_run = true),
           :(HyperCloud.@sim my_cfg bogus_option = 1), :(HyperCloud.@sim my_cfg (dry_run = true)), :(HyperCloud.@sim my_cfg dry_run => true))
    try
        r = Core.eval(Main, ex); println(ex, " -> ", typeof(r), " ", r isa RunSummary ? r.dry_run : "")
    catch e
        println(ex, " -> ERROR ", typeof(e), ": ", first(sprint(showerror, e), 250))
    end
end
