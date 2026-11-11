using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
const BCD = Dict{Int, AbstractBoundaryCondition}
include(joinpath(@__DIR__, "helpers.jl"))
function main()
    for bc in (MirrorBC(), ExtrapolationBC(1), OutflowBC())
        sim = run_bgk_simulation(merge(bgk_wall_base(), Dict{Symbol, Any}(:Grid_bc => BCD(1 => bc, 2 => bc))))
        @printf("equilibrium test with %s: %.3e\n", nameof(typeof(bc)), maximum(maximum(abs.(u - SVector(1.0, 0.0, 2.5))) for u in sim.u[end]))
    end
    # hot-wall test with a wrong operator: accommodation 0 (pure specular) at the hot wall -> would the test catch it?
    θ(u) = 2 * (u[3] - 0.5 * u[2]^2 / u[1]) / (5 * u[1])
    sim = run_bgk_simulation(merge(bgk_wall_base(; Ns = 50, tmax = 0.5), Dict{Symbol, Any}(:Grid_bc => BCD(1 => diffuse_wall(1.0), 2 => diffuse_wall(2.0; accommodation = 0.0)))))
    @printf("hot wall with α = 0: θ near hot wall %.4f\n", θ(sim.u[end][end]))
end
main()
