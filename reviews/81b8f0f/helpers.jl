
euler_wall_base(; Ns = 200, tmax = 0.3) = Dict{Symbol, Any}(
    :PDE_name => :euler, :PDE_D => 1, :PDE_gamma => 1.4,
    :sim_func_name => :run_direct_simulation, :tmax => tmax, :snapshots => 1,
    :Grid_Ns => (Ns,), :Grid_periodic => false, :Grid_randomness_factor => (0.0,), :Grid_seed => 1, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
    :Time_CFL => 0.2, :Time_stepper => :RK3,
    :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard,
    :MOOD_criterion => :none)

bgk_wall_base(; Ns = 50, tau = 1e-2, tmax = 0.2) = Dict{Symbol, Any}(
    :PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 1.4, :PDE_Nv => 16, :PDE_vmax => 6.0, :PDE_tau => tau,
    :sim_func_name => :run_bgk_simulation, :tmax => tmax, :snapshots => 1,
    :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (Ns,),
    :Grid_periodic => false, :Grid_randomness_factor => (0.0,), :Grid_seed => 1, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
    :IC_name => :riemann, :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (1.0, 0.0, 2.5), :IC_p0 => (0.5,), :IC_n => (1.0,),
    :Time_CFL => 0.2, :Time_stepper => :ARS222,
    :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :none, :MOOD_criterion => :none)

# Mean of (ρ, u, p) over the particles with x in (lo, hi)
function plateau(sim, lo, hi)
    xs, us = [x[1] for x in sim.x[end]], sim.u[end]
    idx = [i for i in eachindex(xs) if lo < xs[i] < hi]
    @test length(idx) >= 10
    ρ = sum(us[i][1] for i in idx) / length(idx)
    u = sum(us[i][2] / us[i][1] for i in idx) / length(idx)
    p = sum(0.4 * (us[i][3] - 0.5 * us[i][2]^2 / us[i][1]) for i in idx) / length(idx)
    return ρ, u, p
end

