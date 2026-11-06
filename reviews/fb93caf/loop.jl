using HyperCloud, StaticArrays, Test, Logging
const HC = HyperCloud
include(joinpath(@__DIR__, "helpers.jl"))
function show_run(name, p)
    logs, sim = Test.collect_test_logs(() -> run_sim(p); min_level = Logging.Warn)
    ws = [string(l.level, ": ", first(l.message, 160)) for l in logs]
    tt = sim === nothing ? "nothing" : string(sim.t)
    tt = length(tt) > 120 ? tt[1:60] * " … " * tt[end-50:end] : tt
    fin = sim === nothing ? "-" : all(u -> all(isfinite, u), sim.u[end]) ? "finite" : "NONFINITE"
    println("## $name: #warn = $(length(ws)), t = $tt, final $fin, k_step = $(sim === nothing ? "-" : sim.stats[:k_step])")
    foreach(w -> println("     ", w), ws)
end
run_sim(p) = getfield(HyperCloud, p[:sim_func_name])(p)
fixed(p, dt) = (q = merge(p, Dict{Symbol, Any}(:Time_dt => dt)); delete!(q, :Time_CFL); q)
lin = merge(linear_1d_gauss(50), UPWIND1)
show_run("lin CFL5 tmax100 snaps10", merge(linear_1d_gauss(50; tmax = 100.0), UPWIND1, Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => 10)))
show_run("lin CFL5 tmax100 snaps1", merge(linear_1d_gauss(50; tmax = 100.0), UPWIND1, Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => 1)))
show_run("lin CFL5 tmax100 snaps2", merge(linear_1d_gauss(50; tmax = 100.0), UPWIND1, Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => 2)))
show_run("lin CFL5 tmax100 snaps10 Float32", merge(linear_1d_gauss(50; tmax = 100.0), UPWIND1, Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => 10, :T => Float32)))
show_run("lin CFL5 tmax100 snaps10 ghosts kept", merge(linear_1d_gauss(50; tmax = 100.0), UPWIND1, Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => 10, :remove_ghosts => false)))
for s in (1, 4)
    show_run("burgers NaN snaps$s", merge(burgers_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:IC_a => (NaN,), :snapshots => s)))
end
for c in (Inf, -Inf, NaN, -0.2, 0.0, 1e-14)
    show_run("lin CFL=$c", merge(lin, Dict{Symbol, Any}(:Time_CFL => c)))
end
zero_v = merge(lin, Dict{Symbol, Any}(:PDE_velocities => ((0.0,),)))
for c in (0.2, Inf, -0.2, 0.0)
    show_run("zero velocity CFL=$c", merge(zero_v, Dict{Symbol, Any}(:Time_CFL => c)))
end
for dt in (Inf, -Inf, NaN, -0.01, 0.0, 1e-13, 0.1, 0.03)
    show_run("lin fixed dt=$dt", fixed(lin, dt))
end
show_run("lin fixed dt=0.1 tmax=0.3 snaps3 (round-off end)", fixed(merge(lin, Dict{Symbol, Any}(:tmax => 0.3, :snapshots => 3)), 0.1))
show_run("lin fixed dt=0.1 tmax=0.3 snaps3 Float32", fixed(merge(lin, Dict{Symbol, Any}(:tmax => 0.3, :snapshots => 3, :T => Float32)), 0.1))
sod = euler_1d_sod()
for s in (1, 4)
    show_run("Sod no limiter fixed dt=0.05 snaps$s", fixed(merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Limiter_name => :none, :tmax => 0.2, :snapshots => s)), 0.05))
end
show_run("Sod no limiter CFL4 snaps4", merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_CFL => 4.0, :Limiter_name => :none, :tmax => 0.2, :snapshots => 4)))
show_run("Sod no limiter CFL4 snaps1", merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_CFL => 4.0, :Limiter_name => :none, :tmax => 0.2, :snapshots => 1)))
show_run("Sod VK snaps1 (legit)", merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :snapshots => 1)))
show_run("Sod VK snaps51 (legit)", merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,))))
show_run("BGK wave CFL5 (blow-up, constant speeds)", merge(bgk_wave_params(32), Dict{Symbol, Any}(:Time_CFL => 5.0, :tmax => 20.0, :snapshots => 4)))
show_run("BGK wave CFL=Inf", merge(bgk_wave_params(32), Dict{Symbol, Any}(:Time_CFL => Inf)))
show_run("Sod lagrangian + adaptivity (legit)", merge(sod.shared, sod.methods[:RK2MUSCL2_U1_EPD1], Dict{Symbol, Any}(:Grid_mover => :lagrangian, :Adapt_enabled => true, :Grid_Ns => (100,), :snapshots => 4)))
show_run("Sod lagrangian + adaptivity, no limiter CFL4", merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_mover => :lagrangian, :Adapt_enabled => true, :Grid_Ns => (100,), :snapshots => 4, :Time_CFL => 4.0, :Limiter_name => :none)))
