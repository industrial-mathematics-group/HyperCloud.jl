const UPWIND1 = Dict{Symbol, Any}(
    :Time_stepper => :RK2,
    :Scheme_name => :Upwind, :Scheme_order => 1, :Scheme_MLS_order => 1,
    :Scheme_upwind_alg_nd => :Classic, :Flux_name => :Upwind,
    :MOOD_criterion => :none,
)

"1D periodic linear advection of a Gaussian; the workhorse for accuracy tests."
linear_1d_gauss(N; tmax = 0.1, width = 0.1) = Dict{Symbol, Any}(
    :PDE_name => :linear, :PDE_velocities => ((1.0,),),
    :sim_func_name => :run_direct_simulation,
    :tmax => tmax, :snapshots => 2,
    :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.5,), :IC_width => width,
    :Time_CFL => 0.2,
)

"1D periodic Burgers with a Gaussian pulse (breaking time ≈ 0.23, so keep tmax below that)."
burgers_1d_gauss(N; tmax = 0.15) = Dict{Symbol, Any}(
    :PDE_name => :burgers, :PDE_D => 1,
    :sim_func_name => :run_direct_simulation,
    :tmax => tmax, :snapshots => 2,
    :Grid_domain => :rectangular, :Grid_mins => (-1.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.0,), :IC_width => 0.2,
    :Time_CFL => 0.2,
)

"2D linear advection along the diagonal on the unit square."
function linear_2d_base(periodic::Bool)
    d = Dict{Symbol, Any}(
        :PDE_name => :linear, :PDE_D => 2, :PDE_velocities => ((1.0,), (1.0,)),
        :sim_func_name => :run_direct_simulation,
        :tmax => 0.1, :snapshots => 2,
        :Grid_domain => :rectangular, :Grid_mins => (0.0, 0.0), :Grid_maxs => (1.0, 1.0),
        :Grid_Ns => (20, 20), :Grid_periodic => periodic,
        :Grid_randomness_factor => (0.0, 0.0), :Grid_seed => 42, :Grid_mover => :none,
        :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
        :Time_CFL => 0.2,
    )
    if !periodic
        # NOTE: boundary conditions are read from `:Grid_bc` (`:Grid_bc_map` is an alias).
        d[:Grid_bc] = Dict{Int, AbstractBoundaryCondition}(i => OutflowBC() for i in 1:4)
    end
    return d
end

"1D BGK model (γ = 3: one distribution, `M = Nv`) with a smooth periodic density/momentum/energy wave."
bgk_wave_params(N; Nv = 16, tau = Inf, stepper = :RK3, order = 3, T = Float64) = Dict{Symbol, Any}(
    :PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 3.0, :PDE_Nv => Nv, :PDE_vmax => 6.0, :PDE_tau => tau,
    :sim_func_name => :run_bgk_simulation, :T => T,
    :tmax => 0.2, :snapshots => 2,
    :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :sine, :IC_a => (0.2, 0.1, 0.1), :IC_period => (1.0,), :IC_c_offset => (1.0, 0.0, 0.5),
    :Time_CFL => 0.2, :Time_stepper => stepper,
    :Scheme_name => :MUSCL, :Scheme_order => order, :Flux_name => :Rusanov, :Limiter_name => :none,
    :MOOD_criterion => :none,
)

"Sod shock tube for the 1D BGK model with γ = 1.4 (Chu-reduced pair f, h: `M = 2 Nv`)."
bgk_sod_params(N; Nv = 16, tau = 1e-5) = Dict{Symbol, Any}(
    :PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 1.4, :PDE_Nv => Nv, :PDE_vmax => 6.0, :PDE_tau => tau,
    :sim_func_name => :run_kinetic_simulation,                      # forwarded to run_bgk_simulation
    :tmax => 0.2, :snapshots => 1,
    :Grid_domain => :rectangular, :Grid_mins => (-0.5,), :Grid_maxs => (0.5,), :Grid_Ns => (N,),
    :Grid_periodic => false, :Grid_bc => Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC()),
    :Grid_randomness_factor => (0.0,), :Grid_seed => 10, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :riemann, :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (0.125, 0.0, 0.25), :IC_p0 => (0.0,), :IC_n => (1.0,),
    :Time_CFL => 0.2, :Time_stepper => :ARS222,
    :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :VK, :Limiter_mode => :hard,
    :MOOD_criterion => :none,
)

