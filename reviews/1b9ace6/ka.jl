using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
set_backend!(HyperCloudCore.KA.CPU())
println("backend: ", get_backend())
const BCD = Dict{Int, AbstractBoundaryCondition}
include(joinpath("/tmp/brev3/r3/probes", "helpers.jl"))
w = 0.5
ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, w, 1.4)
tube = "intersect(Tagged(Cuboid((T(0),), (T(1.5),)), 1), Moving(Tagged(HalfSpace((T(1),), (T(1),)), 2), translating((-T(0.5),))))"
p = merge(bgk_wall_base(; Ns = 300, tau = 1e-4, tmax = 0.3), Dict{Symbol, Any}(:Grid_domain => :shape, :Grid_shape => tube,
          :Grid_mins => (0.0,), :Grid_maxs => (1.5,), :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()),
          :Limiter_name => :VK, :Limiter_mode => :hard))
sim = run_bgk_simulation(p)
t = sim.t[end]
ρ, u, pr = plateau(sim, 1 - S * t + 0.06, 1 - w * t - 0.08)
@printf("KA BGK piston slip: ρ=%.4f (%.2f%%) u=%.4f p=%.4f (%.2f%%) metadata backend=%s finite=%s\n", ρ, 100(ρ/ρ₂-1), u, pr, 100(pr/p₂-1), sim.metadata[:backend], sim.metadata[:finite])
p = Dict{Symbol, Any}(
    :PDE_name => :bgk, :PDE_D => 2, :PDE_gamma => 1.4, :PDE_Nv => 8, :PDE_vmax => 6.0, :PDE_tau => 1e-2,
    :sim_func_name => :run_bgk_simulation, :tmax => 0.1, :snapshots => 1, :T => Float32,
    :Grid_domain => :spherical, :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (24, 24),
    :Grid_periodic => false, :Grid_randomness_factor => (0.0, 0.0), :Grid_seed => 1, :Grid_mover => :none,
    :Grid_bc => BCD(1 => slip_wall()), :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
    :IC_name => :riemann, :IC_uL => (1.0, 0.0, 0.0, 2.5), :IC_uR => (1.0, 0.0, 0.0, 2.5), :IC_p0 => (0.0, 0.0), :IC_n => (1.0, 0.0),
    :Time_CFL => 0.2, :Time_stepper => :ARS222,
    :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :none, :MOOD_criterion => :none)
sim = run_bgk_simulation(p)
@printf("KA Float32 disk at rest slip: dev %.3e, eltype %s\n", maximum(maximum(abs.(u - SVector{4,Float32}(1, 0, 0, 2.5))) for u in sim.u[end]), eltype(sim.u[end][1]))
