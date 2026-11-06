# Group `sims`: initial conditions, snapshots, convergence orders, Burgers, the Sod sweep, Float32, movers,
# backend helpers, the memory limit, spherical WENO, the run_all_simulations driver.

@testset "Simulations" begin
    @testset "Initial conditions (2D linear advection)" begin
        cases = [
            ("Box", true, Dict{Symbol, Any}(:IC_name => :box, :IC_u_bg => (0.1,), :IC_u_box => (1.0,),
                                            :IC_mins => (0.25, 0.25), :IC_maxs => (0.75, 0.75))),
            ("Gauss", true, Dict{Symbol, Any}(:IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.5, 0.5), :IC_width => 0.1)),
            ("Sine", true, Dict{Symbol, Any}(:IC_name => :sine, :IC_a => (1.0,), :IC_period => (1.0, 1.0), :IC_c_offset => (0.5,))),
            ("Riemann", false, Dict{Symbol, Any}(:IC_name => :riemann, :IC_uL => (1.0,), :IC_uR => (0.0,),
                                                 :IC_p0 => (0.5, 0.5), :IC_n => (1.0, 1.0))),
            ("Smoothed Riemann", false, Dict{Symbol, Any}(:IC_name => :s_riemann, :IC_uL => (1.0,), :IC_uR => (0.0,),
                                                          :IC_p0 => (0.5, 0.5), :IC_n => (1.0, 0.0), :IC_width => 0.05)),
            ("Quadrant Riemann", false, Dict{Symbol, Any}(:IC_name => :q_riemann, :IC_p0 => (0.5, 0.5),
                                                          :IC_states => ((1.0,), (2.0,), (3.0,), (4.0,)))),
        ]

        @testset "IC: $name" for (name, periodic, ic) in cases
            params = merge(linear_2d_base(periodic), ic, UPWIND1)
            exact = analytical_solution(params)
            sim = run_direct_simulation(params)
            @test sim !== nothing
            if sim !== nothing
                x0, u0 = sim.x[1], sim.u[1]
                # The IC was actually applied to every particle
                @test all(i -> isapprox(u0[i][1], exact(spacetime(x0[i], 0.0))[1]; atol = 1e-12), eachindex(x0))

                # The run reaches tmax, stays finite and (nearly) inside the initial bounds
                @test sim.t[end] ≈ params[:tmax]
                lo, hi = extrema(u[1] for u in u0)
                slack = 0.05 * (hi - lo) + 1e-12
                uend = [u[1] for u in sim.u[end]]
                @test all(isfinite, uend)
                @test lo - slack <= minimum(uend)
                @test maximum(uend) <= hi + slack
            end
        end
    end

    @testset "Snapshot bookkeeping" begin
        params = merge(linear_1d_gauss(50; tmax = 0.1), UPWIND1)
        params[:snapshots] = 4
        sim = run_direct_simulation(params)
        @test sim !== nothing
        if sim !== nothing
            @test length(sim.t) == 5
            @test sim.t[1] == 0
            @test sim.t[end] ≈ 0.1
            @test issorted(sim.t)
            @test all(k -> sim.t[k] >= (k - 1) * 0.1 / 4 - 1e-12, eachindex(sim.t))
            @test all(x -> length(x) == length(sim.x[1]), sim.x)
            @test sim.x[1] == sim.x[end]   # Grid_mover = :none keeps particles fixed
        end
    end

    @testset "Early stop is reported, snapshots = 0 is rejected" begin
        # Sod with MUSCL2 and no limiter at CFL 4 blows up in the first step: the loop stops with a warning
        # (instead of silently), records no NaN time and the runner still returns the data up to that point.
        sod = euler_1d_sod()
        p = merge(sod.shared, sod.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_CFL => 4.0,
                  :Limiter_name => :none, :tmax => 0.2, :snapshots => 4))
        sim = @test_logs (:warn, r"solve_equation stopped at t = .* before tmax") match_mode = :any run_direct_simulation(p)
        @test sim !== nothing
        if sim !== nothing
            @test all(isfinite, sim.t)
            @test sim.t[end] < 0.2
        end

        # A non-finite step (here a NaN fixed dt) stops at once without recording a NaN time
        pn = merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:Time_dt => NaN))
        delete!(pn, :Time_CFL)
        simn = @test_logs (:warn, r"time step is not finite") match_mode = :any run_direct_simulation(pn)
        @test simn !== nothing && simn.t == [0.0, 0.0]

        # Blow-ups the step doesn't see (constant wave speeds): unstable linear advection at CFL 5 first grows
        # (warned once), then overflows; the run stops at the first non-finite snapshot
        pb = merge(linear_1d_gauss(50; tmax = 100.0), UPWIND1, Dict{Symbol, Any}(:Time_CFL => 5.0, :snapshots => 10))
        logsb, simb = Test.collect_test_logs(() -> run_direct_simulation(pb))
        warnsb = [l.message for l in logsb if l.level == Base.CoreLogging.Warn]
        @test simb !== nothing && all(isfinite, simb.t) && simb.t[end] < 100.0
        # each blow-up is reported once: the growth warning and one stop (the final slot of a stopped run is not
        # checked again; it used to repeat "not finite" without "stopped before tmax")
        @test length(warnsb) == 2 && occursin("times larger than at t = 0", warnsb[1]) && occursin("stopped before tmax", warnsb[2])
        # ... also when the final slot is the only one after t = 0
        pb1 = merge(pb, Dict{Symbol, Any}(:snapshots => 1))
        @test_logs (:warn, r"is not finite \(blown up\)") match_mode = :any run_direct_simulation(pb1)
        # A NaN state has an infinite CFL step (NaN wave speeds are skipped): it stops, it doesn't jump to tmax
        pnan = merge(burgers_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:IC_a => (NaN,), :snapshots => 1))
        logsnan, simnan = Test.collect_test_logs(() -> run_direct_simulation(pnan))
        @test count(l -> l.level == Base.CoreLogging.Warn, logsnan) == 1
        @test any(l -> occursin("is not finite (blown up)", l.message), logsnan)
        @test simnan !== nothing && simnan.t[end] == 0.0
        # An infinite CFL number is a configuration error: it stops at once instead of one silent step to tmax
        pcfl = merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:Time_CFL => Inf))
        simcfl = @test_logs (:warn, r"from the CFL number Inf, which must be positive and finite") match_mode = :any run_direct_simulation(pcfl)
        @test simcfl !== nothing && simcfl.t == [0.0, 0.0]
        # A fixed-step blow-up with snapshots = 1 (the final slot is the only check): Sod without limiter at dt = 0.05
        sod1 = euler_1d_sod()
        pf = merge(sod1.shared, sod1.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:Grid_Ns => (100,), :Time_dt => 0.05,
                   :Limiter_name => :none, :tmax => 0.2, :snapshots => 1))
        delete!(pf, :Time_CFL)
        @test_logs (:warn, r"not finite") match_mode = :any run_direct_simulation(pf)
        # An infinite fixed step is not a step
        pinf = merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:Time_dt => Inf))
        delete!(pinf, :Time_CFL)
        siminf = @test_logs (:warn, r"time step is not finite") match_mode = :any run_direct_simulation(pinf)
        @test siminf !== nothing && siminf.t == [0.0, 0.0]

        # No fixed time scale: tmax = 1e-13 is integrated (the absolute round-off end 1e-12 returned the IC at t = 0)
        ptiny = merge(linear_1d_gauss(50; tmax = 1e-13), UPWIND1)
        simtiny = run_direct_simulation(ptiny)
        @test simtiny !== nothing && simtiny.t[end] ≈ 1e-13

        # No wave moves (zero velocity: infinite CFL step): the run reaches tmax without a warning
        pz = merge(linear_1d_gauss(50), UPWIND1, Dict{Symbol, Any}(:PDE_velocities => ((0.0,),)))
        logs, simz = Test.collect_test_logs(() -> run_direct_simulation(pz))
        @test simz !== nothing && simz.t[end] ≈ 0.1
        @test !any(l -> l.level >= Base.CoreLogging.Warn, logs)

        p0 = merge(linear_1d_gauss(50), UPWIND1)
        p0[:snapshots] = 0
        logs, sim0 = Test.collect_test_logs(() -> run_direct_simulation(p0))
        @test sim0 === nothing
        errs = [first(l.kwargs[:exception]) for l in logs if haskey(l.kwargs, :exception)]
        @test length(errs) == 1 && errs[1] isa ArgumentError && occursin("snapshots must be ≥ 1", errs[1].msg)
    end

    @testset "Convergence order: $label" for (label, method, min_order) in [
        ("RK2 Upwind1", last(create_method(:RK2, :Upwind, 1; Scheme_upwind_alg_nd = :Classic)), 0.7),
        ("RK2 MUSCL2",  last(create_method(:RK2, :MUSCL, 2)), 1.8),
        ("RK4 MUSCL4",  last(create_method(:RK4, :MUSCL, 4)), 3.5),
    ]
        # Measured on N = 100 → 200: 0.80, 2.06, 4.36
        errs = [first(l1_error(merge(linear_1d_gauss(N), method))) for N in (100, 200)]
        @test all(isfinite, errs)
        observed = log2(errs[1] / errs[2])
        @info "Observed convergence order" label errs observed
        @test observed > min_order
    end

    @testset "Burgers (smooth, pre-shock) vs exact: $label" for (label, method) in [
        ("direct RK2 MUSCL2",     last(create_method(:RK2, :MUSCL, 2))),
        ("kinetic ARS222 MUSCL2", last(create_method(:ARS222, :MUSCL, 2))),
    ]
        vels, idx = generate_relax_params(2.0, 1, 1)
        params = merge(burgers_1d_gauss(200), method, Dict{Symbol, Any}(
            :Kinetic_velocities => vels, :Kinetic_indices => idx, :Kinetic_epsilon => 1e-6))
        err, sim = l1_error(params)
        @test sim !== nothing
        @test err < 0.03
    end

    @testset "Sod shock tube: every configured method vs exact" begin
        config = euler_1d_sod()
        γ = config.shared[:PDE_gamma]
        cases = sort!([(k, v) for (k, v) in config.methods if haskey(v, :Time_stepper)]; by = first)

        @testset "$name" for (name, m) in cases
            err, sim = l1_error(merge(config.shared, m))
            @test sim !== nothing
            @test err < 0.1
            if sim !== nothing
                us = sim.u[end]
                @test minimum(u -> u[1], us) > 0                                          # density
                @test minimum(u -> (γ - 1) * (u[3] - 0.5 * u[2]^2 / u[1]), us) > 0        # pressure
            end
        end
    end

    @testset "Float32 end-to-end" begin
        p64 = merge(linear_1d_gauss(100), UPWIND1)
        p32 = merge(p64, Dict{Symbol, Any}(:T => Float32))
        err64, _   = l1_error(p64)
        err32, s32 = l1_error(p32)
        @test s32 !== nothing
        @test isfinite(err32)
        @test abs(err32 - err64) < 1e-3
    end

    @testset "Lagrangian mover uses the PDE's fluid velocity" begin
        config = euler_1d_sod()
        params = merge(config.shared, config.methods[:RK2MUSCL2_U1_EPD1], Dict{Symbol, Any}(:Grid_mover => :lagrangian))
        err, sim = l1_error(params)
        @test sim !== nothing
        @test err < 0.1
        sim === nothing || @test sim.x[end] != sim.x[1]           # the particles moved with the flow
    end

    @testset "Execution backend helpers" begin
        @test storage_for(PolyesterBackend()) === Array
        @test storage_for(ThreadsBackend()) === Array
        @test storage_for(KABackend(HyperCloudCore.KA.CPU())) === Array
        previous = get_backend()
        @test with_backend(get_backend, ThreadsBackend()) isa ThreadsBackend
        @test_throws ErrorException with_backend(() -> error("boom"), ThreadsBackend())
        @test get_backend() === previous                            # restored, also after an error
    end

    @testset "Memory limit" begin
        params = merge(linear_1d_gauss(100), UPWIND1)
        try
            set_memory_limit!(64 * 2^20; tolerance = 0.1)
            @test get_memory_limit() == (; limit = 64 * 2^20, tolerance = 0.1, slots_per_particle = nothing)
            sim = run_sim(params)
            @test sim !== nothing
            if sim !== nothing
                @test sim.stats[:reserved_particles][1] >= length(sim.x[end])
                @test sim.stats[:reserved_bytes][1] <= 0.9 * 64 * 2^20
            end
            set_memory_limit!(10_000)                               # cannot hold 100 particles
            @test (@test_logs (:error, r"Error during Direct Simulation") match_mode = :any run_sim(params)) === nothing
        finally
            set_memory_limit!(nothing)
        end
        @test get_memory_limit() === nothing
        @test_throws ArgumentError set_memory_limit!(0)
    end

    @testset "Spherical WENO & progress logging" begin
        # Disable the progress bar and force frequent @info logs
        enable_progress_bar!(false)
        set_progress_interval!(0.001)
        try
            params = Dict{Symbol, Any}(
                :PDE_name => :linear, :PDE_velocities => ((1.0,),),
                :sim_func_name => :run_direct_simulation,
                :tmax => 0.05, :snapshots => 2,
                :Grid_domain => :spherical, :Grid_mins => (0.1,), :Grid_maxs => (1.0,),
                :Grid_Ns => (250,),   # large enough that the ETA log fires
                :Grid_periodic => false, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
                :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
                :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.5,), :IC_width => 0.1,
                :Time_CFL => 0.1, :Time_stepper => :RK2,
                :Scheme_name => :WENO, :Scheme_order => 2, :Scheme_MLS_order => 0,
                :Flux_name => :Upwind, :Limiter_name => :none,
                :MOOD_criterion => :none,
            )

            @test_logs (:info, r"Using .* for parallel runs!") (:info, r"Simulation Progress: .*") match_mode = :any begin
                sim = run_direct_simulation(params)
                @test sim !== nothing
                @test sim.t[end] ≈ 0.05
                @test length(sim.x[end]) == 251
            end
        finally
            enable_progress_bar!(true)
            set_progress_interval!(1.0)
        end
    end

    @testset "run_all_simulations driver (with stats)" begin
        methods = Dict{Symbol, Dict{Symbol, Any}}(
            :analytical_solution => Dict{Symbol, Any}(),
            :Upwind => copy(UPWIND1),
        )
        conf = SimulationConfig(linear_1d_gauss(50), methods, [:analytical_solution, :Upwind];
                                reference = :analytical_solution)
        old_path = get_save_path()
        set_save_path!(mktempdir())          # never reuse or write results in the working directory
        try
            run_all_simulations(conf; force_overwrite = true, calculate_stats = true)
            tasks = generate_tasks(conf)
            @test length(tasks) == 1                                   # the reference method is not run
            p = only(tasks)
            @test does_sim_data_exist(p)
            if does_sim_data_exist(p)
                sim = load_sim_data(p, Val(:lagrangian))
                @test sim.t[end] ≈ 0.1
                exact = analytical_solution(p)
                xs, us = sim.x[end], sim.u[end]
                err = sum(i -> abs(exact(spacetime(xs[i], sim.t[end]))[1] - us[i][1]), eachindex(xs)) / length(xs)
                @test err < 0.15                                      # Upwind1 at N = 50: 0.079
            end
        finally
            set_save_path!(old_path)
        end
    end
end
