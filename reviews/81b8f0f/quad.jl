using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
const BCD = Dict{Int, AbstractBoundaryCondition}
include(joinpath(@__DIR__, "helpers.jl"))
function main()
    for (lbl, bcs, uR) in (("hot diffuse 1/2, uniform IC", (diffuse_wall(1.0), diffuse_wall(2.0)), (1.0, 0.0, 2.5)), ("slip/slip, jump IC", (slip_wall(), slip_wall()), (0.5, 0.0, 1.0)))
        for N in (50, 100, 200, 400)
            p = merge(bgk_wall_base(; Ns = N, tmax = 0.5), Dict{Symbol, Any}(:Grid_bc => BCD(1 => bcs[1], 2 => bcs[2]), :IC_uR => uR))
            sim = run_bgk_simulation(p)
            xs = [x[1] for x in sim.x[end]]; o = sortperm(xs); xs = xs[o]
            trap(us) = (u = [us[i][1] for i in o]; sum((u[i] + u[i+1]) / 2 * (xs[i+1] - xs[i]) for i in 1:length(xs)-1))
            mean(us) = sum(u[1] for u in us) / length(us)
            @printf("%s N=%d (n=%d, dx=%.4g, ends %.3g %.3g): Δmean %.3e  Δtrapezoid %.3e\n", lbl, N, length(xs), xs[2]-xs[1], xs[1], xs[end],
                    mean(sim.u[end]) - mean(sim.u[1]), trap(sim.u[end]) - trap(sim.u[1]))
        end
    end
end
main()
