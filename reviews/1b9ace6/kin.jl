using HyperCloud, StaticArrays, Printf, Test; enable_progress_bar!(false); const BCD = Dict{Int, AbstractBoundaryCondition}; const HC = HyperCloud
c = burgers_kinetic()
ts = [t for t in generate_tasks(c) if any(k -> startswith(string(k), "Kinetic_"), keys(t))]
println(length(ts), " kinetic tasks; ", ts[1][:sim_func_name])
p = merge(ts[1], Dict{Symbol, Any}(:Grid_periodic => false, :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()), :Grid_Ns => (50,), :tmax => 0.05, :snapshots => 1))
logs, sim = Test.collect_test_logs(() -> HC.run_kinetic_simulation(p))
println("kinetic Burgers with slip_wall: sim === nothing: ", sim === nothing)
for l in logs; l.level >= Base.CoreLogging.Warn || continue; println("  [", l.level, "] ", first(string(l.message), 200)); for (k, v) in l.kwargs; println("    ", k, " => ", first(v isa Tuple && v[1] isa Exception ? sprint(showerror, v[1]) : string(v), 300)); end; end
nodes, weights = velocity_grid(:uniform, 8, 6.0)
bgk = BGKEquation(Val(1), nodes, weights, 1.4, ConstantRelaxationTime(1e-2))
feq, _ = equilibrium(bgk, SVector(1.0, 0.0, 2.5))
for θw in (1e-4, 1e-2)
    g = DiffuseWall(θw)(bgk, feq, WallPoint(SVector(1.0), SVector(1.0), 0.05, SVector(0.0)))
    println("DiffuseWall($θw) ghost finite: ", all(isfinite, g), "  ghost θ of the emitted half: ", (m = moments(bgk, State{16,Float64}(ntuple(k -> velocity_node(bgk, (k-1)%8+1)[1][1] < 0 ? g[k] : 0.0, 16))); 2 * (m[3] - 0.5m[2]^2/m[1]) / (5m[1]) ))
end
