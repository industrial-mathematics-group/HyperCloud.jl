using HyperCloud, StaticArrays, Printf, Test; enable_progress_bar!(false); const BCD = Dict{Int, AbstractBoundaryCondition}; const HC = HyperCloud
# kinetic system run (Burgers kinetic) with slip walls instead of periodic
c = burgers_kinetic()
t = first(generate_tasks(c))
p = merge(t, Dict{Symbol, Any}(:Grid_periodic => false, :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()), :Grid_Ns => (50,), :tmax => 0.05, :snapshots => 1))
logs, sim = Test.collect_test_logs(() -> HC.run_kinetic_simulation(p))
println("kinetic Burgers with slip_wall: sim === nothing: ", sim === nothing)
for l in logs; l.level >= Base.CoreLogging.Warn || continue; println("  [", l.level, "] ", first(string(l.message), 200)); for (k, v) in l.kwargs; println("    ", k, " => ", first(v isa Tuple && v[1] isa Exception ? sprint(showerror, v[1]) : string(v), 300)); end; end
for l in logs; l.level >= Base.CoreLogging.Warn || continue; println("  [", l.level, "] ", first(string(l.message), 200)); for (k, v) in l.kwargs; println("    ", k, " => ", first(v isa Tuple && v[1] isa Exception ? sprint(showerror, v[1]) : string(v), 300)); end; end
