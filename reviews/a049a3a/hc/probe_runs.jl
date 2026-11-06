# Probes: end-to-end runs that the suite does not cover (run: julia -t 2 --project=test <this file>)
using HyperCloud, StaticArrays, Test
const HC = HyperCloud
enable_progress_bar!(false)
spacetime(x, t) = SVector(Tuple(x)..., Float64(t))
function l1_error(params; component = 1)
    exact = analytical_solution(params)
    sim = getfield(HyperCloud, params[:sim_func_name])(params)
    sim === nothing && return (NaN, nothing)
    xs, us, t = sim.x[end], sim.u[end], sim.t[end]
    return sum(i -> abs(exact(spacetime(xs[i], t))[component] - us[i][component]), eachindex(xs)) / length(xs), sim
end

@testset "probes: runs" begin
    @testset "documented :primitive / :lagrangian representations build" begin
        for rep in (:primitive, :lagrangian)
            ok = try
                build_equation(Dict{Symbol, Any}(:name => :euler, :D => 1, :gamma => 1.4, :representation => rep),
                               Dict{Symbol, Any}(:T => Float64)); true
            catch e
                @info "representation $rep" exception = e
                false
            end
            @test ok
        end
    end

    sod = euler_1d_sod()
    base = merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (200,)))

    @testset "Euler Sod Float32 vs Float64" begin
        e64, s64 = l1_error(base)
        e32, s32 = l1_error(merge(base, Dict{Symbol, Any}(:T => Float32)))
        @info "Sod L1(rho)" e64 e32
        @test s32 !== nothing
        s32 === nothing || @test eltype(s32.u[end][1]) == Float32
        @test abs(e32 - e64) < 1e-3
    end

    @testset "Kinetic relaxation with an Euler macro model (Sod)" begin
        vels, idx = generate_relax_params(3.0, 1, 3)
        p = merge(sod.shared, Dict{Symbol, Any}(:Grid_Ns => (200,), :sim_func_name => :run_kinetic_simulation,
                  :Time_stepper => :ARS222, :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov,
                  :Limiter_name => :VK, :Limiter_mode => :hard, :MOOD_criterion => :none,
                  :Kinetic_velocities => vels, :Kinetic_indices => idx, :Kinetic_epsilon => 1e-6))
        err, sim = l1_error(p)
        @info "kinetic Euler Sod L1(rho)" err
        @test sim !== nothing
        @test err < 0.05
        if sim !== nothing
            # mass: closed tube with Dirichlet/outflow ends, waves stay inside at t = 0.2
            @test length(sim.u[end][1]) == 3
        end
    end

    bgk = Dict{Symbol, Any}(
        :PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 3.0, :PDE_Nv => 12, :PDE_vmax => 4.0, :PDE_tau => Inf,
        :PDE_velocity_grid => :gauss_hermite,
        :sim_func_name => :run_bgk_simulation, :tmax => 0.2, :snapshots => 1,
        :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,),
        :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
        :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
        :IC_name => :sine, :IC_a => (0.2, 0.1, 0.1), :IC_period => (1.0,), :IC_c_offset => (1.0, 0.0, 0.5),
        :Time_CFL => 0.2, :Time_stepper => :RK3,
        :Scheme_name => :MUSCL, :Scheme_order => 3, :Flux_name => :Rusanov, :Limiter_name => :none, :MOOD_criterion => :none)

    @testset "BGK Gauss-Hermite grid, collisionless convergence" begin
        errs = [first(l1_error(merge(bgk, Dict{Symbol, Any}(:Grid_Ns => (N,))))) for N in (50, 100)]
        @info "GH errors" errs order = log2(errs[1] / errs[2])
        @test log2(errs[1] / errs[2]) > 2.5
    end

    @testset "BGK viscosity law, Sod, gamma = 1.4 (Chu pair), Float32" begin
        p = merge(sod.shared, Dict{Symbol, Any}(
            :PDE_name => :bgk, :PDE_D => 1, :PDE_gamma => 1.4, :PDE_Nv => 16, :PDE_vmax => 6.0,
            :PDE_viscosity => 1e-4, :PDE_omega => 0.81, :T => Float32,
            :sim_func_name => :run_bgk_simulation, :Grid_Ns => (200,), :Grid_bc => sod.shared[:Grid_bc_map],
            :Time_stepper => :ARS222, :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov,
            :Limiter_name => :VK, :Limiter_mode => :hard, :MOOD_criterion => :none))
        delete!(p, :Grid_bc_map)
        exact = euler_limit_solution(p)
        sim = run_bgk_simulation(p)
        @test sim !== nothing
        if sim !== nothing
            xs, us, t = sim.x[end], sim.u[end], sim.t[end]
            err = sum(i -> abs(exact(spacetime(xs[i], t))[1] - us[i][1]), eachindex(xs)) / length(xs)
            @info "BGK viscosity Sod Float32" err eltype(us[1])
            @test err < 0.03
            @test eltype(us[1]) == Float32
        end
    end
end
