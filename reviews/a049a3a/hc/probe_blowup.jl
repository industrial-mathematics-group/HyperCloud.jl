# Probe: a run that blows up (unstable CFL) - does the runner report it?
using HyperCloud, Test, Logging
enable_progress_bar!(false)
sod = euler_1d_sod()
p = merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_CFL => 4.0,
          :Limiter_name => :none, :tmax => 0.2, :snapshots => 4))
sim = run_direct_simulation(p)
@testset "blow-up is reported" begin
    @test sim === nothing || (all(isfinite, sim.t) && sim.t[end] ≈ 0.2 && all(u -> all(isfinite, u), sim.u[end]))
    if sim !== nothing
        @info "blow-up run" repr(sim.t) length(sim.t) sim.stats[:k_step][1] any(u -> !all(isfinite, u), sim.u[end])
    end
end
