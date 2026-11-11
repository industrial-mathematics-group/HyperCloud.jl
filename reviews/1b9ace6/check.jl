using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
const BCD = Dict{Int, AbstractBoundaryCondition}
const HC = HyperCloud
showres(lbl, f) = try; r = f(); println(lbl, " -> ", r === nothing ? "nothing" : typeof(r)); catch e; println(lbl, " -> ", typeof(e), ": ", first(sprint(showerror, e), 220)); end
box(bcs...) = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = BCD(1 => bcs[1], 2 => bcs[end]))
# user operators
myop(eq, U, wall) = U
struct MyOp end
(::MyOp)(eq::EulerEquation, U, wall) = U
showres("user function op on Euler", () -> HC._check_wall_ops(EulerEquation(Val(1), Float64), box(MirrorBC(myop))))
showres("user struct op on Euler", () -> HC._check_wall_ops(EulerEquation(Val(1), Float64), box(MirrorBC(MyOp()))))
showres("user struct op on Burgers", () -> HC._check_wall_ops(HC.BurgersEquation{1, Float64}(), box(MirrorBC(MyOp()))))
showres("MirrorBC() (no op)", () -> HC._check_wall_ops(HC.BurgersEquation{1, Float64}(), box(MirrorBC())))
showres("SlipWall Euler Float32 domain", () -> HC._check_wall_ops(EulerEquation(Val(1), Float32), box(slip_wall())))
nodes, weights = velocity_grid(:uniform, 8, 6.0)
bgk = BGKEquation(Val(1), nodes, weights, 1.4, ConstantRelaxationTime(1e-2))
for θw in (1e-4, 1e-2, 50.0, Inf)
    showres("DiffuseWall($θw) Nv=8", () -> HC._check_wall_ops(bgk, box(diffuse_wall(θw))))
    θw == Inf && continue
    Uw = SVector(1.0, 0.0, 0.5 * 5 * θw)
    Mw, ok = equilibrium(bgk, Uw)
    @printf("    equilibrium ok=%s, moments rel err %.2e (θ requested %g, got %.4g)\n", ok, maximum(abs.(moments(bgk, Mw) - Uw) ./ abs.(Uw .+ 1e-30)), θw, 2 * moments(bgk, Mw)[3] / (5 * moments(bgk, Mw)[1]))
end
showres("DiffuseWall velocity (5.9,) Nv=8 vmax 6", () -> HC._check_wall_ops(bgk, box(diffuse_wall(1.0; velocity = (5.9,)))))
# kinetic system run (Burgers kinetic) with slip walls instead of periodic
c = burgers_kinetic()
t = first(generate_tasks(c))
p = merge(t, Dict{Symbol, Any}(:Grid_periodic => false, :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()), :Grid_Ns => (50,), :tmax => 0.05, :snapshots => 1))
logs, sim = Test.collect_test_logs(() -> HC.run_kinetic_simulation(p))
println("kinetic Burgers with slip_wall: sim === nothing: ", sim === nothing)
for l in logs; l.level >= Base.CoreLogging.Warn || continue; println("  [", l.level, "] ", first(string(l.message), 200)); for (k, v) in l.kwargs; println("    ", k, " => ", first(v isa Tuple && v[1] isa Exception ? sprint(showerror, v[1]) : string(v), 300)); end; end
# @sim duplicate option message
try; macroexpand(Main, :(HyperCloud.@sim euler_1d_sod dry_run = true dry_run = false calculate_stats = true calculate_stats = false)); catch e; println("@sim dup: ", sprint(showerror, e)); end
