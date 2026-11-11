using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
const BCD = Dict{Int, AbstractBoundaryCondition}
include(joinpath("/tmp/brev3/r3/probes", "helpers.jl"))
h(p) = HyperCloud.PDEStudioCore._param_hash(p)
task(c) = first(generate_tasks(c))
c05, c04 = euler_1d_piston(), euler_1d_piston(speed = 0.4)
println("hash speed 0.5 vs 0.4 differ: ", h(task(c05)) != h(task(c04)))
println("hash speed 0.5 vs 1//2 equal: ", h(task(c05)) == h(task(euler_1d_piston(speed = 1//2))), "  shape: ", euler_1d_piston(speed = 1//2).shared[:Grid_shape])
println("hash default vs speed=0.5f0 equal: ", h(task(c05)) == h(task(euler_1d_piston(speed = 0.5f0))))
println("shape string: ", c05.shared[:Grid_shape])
for (s, f) in ((-0.5, 1.0), (0.0, 1.0), (0.5, 0.1), (0.5, 1.49), (0.5, 1.5))
    try
        c = euler_1d_piston(speed = s, face = f)
        p = merge(c.shared, c.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:snapshots => 1, :Grid_Ns => (150,)))
        logs, sim = Test.collect_test_logs(() -> run_direct_simulation(p))
        ex = exact_piston(p)
        errs = [l for l in logs if l.level >= Base.CoreLogging.Warn]
        if sim === nothing
            println("speed=$s face=$f: run failed: ", isempty(errs) ? "" : first(string(errs[1].message, " ", get(errs[1].kwargs, :exception, "")), 300))
        else
            xs = [x[1] for x in sim.x[end]]; t = sim.t[end]
            L1 = sum(abs(ex(SVector(xs[i], t))[1] - sim.u[end][i][1]) for i in eachindex(xs)) / length(xs)
            @printf("speed=%g face=%g: ran to t=%.3f, L1ρ vs exact_piston %.3e, ρ range [%.3f, %.3f], max x %.4f, piston_state %s, warnings %d\n", s, f, t, L1,
                    minimum(u[1] for u in sim.u[end]), maximum(u[1] for u in sim.u[end]), maximum(xs), string(round.(HyperCloud.piston_state(1.0, 1.0, s, 1.4), digits = 4)), length(errs))
        end
    catch e
        println("speed=$s face=$f: ", typeof(e), ": ", first(sprint(showerror, e), 200))
    end
end
# speed 0.8 face 1.2, N = 300 / 600 (example convergence to its own reference)
for N in (300, 600)
    c = euler_1d_piston(speed = 0.8, face = 1.2)
    p = merge(c.shared, c.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:snapshots => 1, :Grid_Ns => (N,)))
    sim = run_direct_simulation(p); ex = exact_piston(p)
    xs = [x[1] for x in sim.x[end]]; t = sim.t[end]
    L1 = sum(abs(ex(SVector(xs[i], t))[1] - sim.u[end][i][1]) for i in eachindex(xs)) / length(xs)
    ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, 0.8, 1.4)
    @printf("speed 0.8 face 1.2 N=%d: L1ρ %.3e, plateau %s vs (%.4f, -0.8, %.4f), face %.4f max x %.4f\n", N, L1,
            string(round.(plateau(sim, 1.2 - S * t + 0.05, 1.2 - 0.8t - 0.03), digits = 4)), ρ₂, p₂, 1.2 - 0.8t, maximum(xs))
end
# Float32 example run
c = euler_1d_piston()
p32 = merge(c.shared, c.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:T => Float32, :snapshots => 1))
sim = run_direct_simulation(p32)
@printf("Float32 example: eltype %s, plateau %s, metadata %s\n", eltype(sim.u[end][1]), string(round.(plateau(sim, 0.6, 0.8), digits = 4)), string(sim.metadata))
p64 = merge(c.shared, c.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:snapshots => 1))
s64 = run_direct_simulation(p64)
@printf("Float64 example plateau %s\n", string(round.(plateau(s64, 0.6, 0.8), digits = 4)))
println("piston_state bracket: ", [round.(HyperCloud.piston_state(1.0, 1.0, w, 1.4), sigdigits = 6) for w in (1e-6, 1.0, 50.0, 1e3)])
# piston_state vs an independent formula: p2/p1 from the shock Mach number
for w in (0.5, 0.8, 5.0)
    ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, w, 1.4)
    c1 = sqrt(1.4); Ms = S / c1
    @printf("w=%g: p2 %.6f vs 1+2γ/(γ+1)(M²-1) = %.6f; ρ2 %.6f vs (γ+1)M²/((γ-1)M²+2) = %.6f; mass ρ2 (S - w) = %.6f vs S = %.6f\n", w, p₂,
            1 + 2 * 1.4 / 2.4 * (Ms^2 - 1), ρ₂, 2.4Ms^2 / (0.4Ms^2 + 2), ρ₂ * (S - w), S)
end
