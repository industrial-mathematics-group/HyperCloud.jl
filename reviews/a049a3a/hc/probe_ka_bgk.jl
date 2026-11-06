# Probe: BGK runs with many equations on the KernelAbstractions CPU backend (PoCL)
using HyperCloud, Test, Logging
enable_progress_bar!(false)
set_backend!(HyperCloudCore.KA.CPU())
@info "backend" get_backend()
p2 = Dict{Symbol, Any}(
    :PDE_name => :bgk, :PDE_D => 2, :PDE_gamma => 2.0, :PDE_Nv => 6, :PDE_vmax => 4.0, :PDE_tau => Inf,
    :sim_func_name => :run_bgk_simulation, :tmax => 0.02, :snapshots => 1,
    :Grid_domain => :rectangular, :Grid_mins => (0.0, 0.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (16, 16),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0, 0.0), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
    :IC_name => :sine, :IC_a => (0.2, 0.05, 0.05, 0.1), :IC_period => (1.0, 1.0), :IC_c_offset => (1.0, 0.0, 0.0, 0.5),
    :Time_CFL => 0.2, :Time_stepper => :RK2,
    :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :none, :MOOD_criterion => :none)
sod = bgk_1d_sod()
p1 = merge(sod.shared, sod.methods[:ARS222MUSCL2_VK], Dict{Symbol, Any}(:tmax => 0.02, :Grid_Ns => (100,)))   # Nv = 24, gamma 1.4: 48 equations
@testset "BGK on KA CPU" begin
    for (name, p) in (("2D, 36 equations", p2), ("1D bgk_1d_sod, 48 equations", p1))
        sim = with_logger(ConsoleLogger(stderr, Logging.Error)) do; run_bgk_simulation(p); end
        @info name ok = sim !== nothing
        @test sim !== nothing
    end
end
