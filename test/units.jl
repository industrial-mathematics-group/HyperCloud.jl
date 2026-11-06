# Group `units`: macros and launchers, configuration builders, the method factory, equation kernels, kinetic
# relaxation, the BGK equation and the analytical closures.

# -------------------------------------------------------------------------
# 2. MACROS & LAUNCHERS
# -------------------------------------------------------------------------
@testset "Macro utilities & simulation launchers" begin
    # Save global state so later testsets (e.g. Examples) are not affected
    old_root   = HyperCloud.CONFIG_ROOT_OVERWRITE[]
    old_paths  = HyperCloud.CONFIG_SEARCH_PATHS[]
    old_bonito = HyperCloud.BONITO_IS_CONFIGURED[]

    test_dir = mktempdir()
    mock_configs_dir = joinpath(test_dir, "mock_configs")
    mkdir(mock_configs_dir)

    try
        HyperCloud.CONFIG_ROOT_OVERWRITE[] = test_dir
        HyperCloud.CONFIG_SEARCH_PATHS[] = ["mock_configs"]
        
        @eval Main begin
            function my_local_sim()
                shared = Dict{Symbol, Any}(:Time_tmax => 0.0, :Time_snapshots => 0, :sim_func_name => :dummy_func)
                return SimulationConfig(shared, Dict(), Symbol[])
            end
        end

        plugin_name = "test_plugin"
        open(joinpath(mock_configs_dir, "$(plugin_name).jl"), "w") do f
            write(f, """
            function test_plugin()
                shared = Dict{Symbol, Any}(:Time_tmax => 0.0, :Time_snapshots => 0, :sim_func_name => :dummy_func)
                return SimulationConfig(shared, Dict(), Symbol[])
            end
            """)
        end

        @testset "@sim resolution (local function, string, symbol, plugin file)" begin
            # Any exception here is recorded as a test error, so no `@test true` is needed
            @eval Main @sim my_local_sim
            @eval Main @sim "my_local_sim"
            @eval Main @sim :my_local_sim
            @eval Main @sim test_plugin
            @test_throws ErrorException @eval Main @sim missing_sim_config
        end

        @testset "@backup" begin
            @eval Main @backup test_plugin
            backup_dir = joinpath(mock_configs_dir, "backups")
            @test isdir(backup_dir)
            files = readdir(backup_dir)
            @test length(files) == 1
            @test startswith(files[1], "test_plugin_")
            @test endswith(files[1], ".jl")
        end

        @testset "@plot backend routing" begin
            HyperCloud.BONITO_IS_CONFIGURED[] = false

            # No backend: warns but still returns the figure
            res = @test_logs (:warn, r"No Makie backend found!") HyperCloud._launch_and_display_plotter(MockPDEStudio)
            @test res == "mock_figure"

            # GLMakie
            @eval MockPDEStudio GLMakie = $MockGLMakie
            @test HyperCloud._launch_and_display_plotter(MockPDEStudio) == "mock_figure"

            # WGLMakie without Bonito
            @eval MockPDEStudio WGLMakie = $MockWGLMakie
            res = @test_logs (:warn, r"WGLMakie is loaded, but Bonito is not!") (:info, r"Bonito server listens on") HyperCloud._launch_and_display_plotter(MockPDEStudio)
            @test res == "mock_figure"

            # WGLMakie + Bonito configures the server exactly once
            @eval MockPDEStudio Bonito = $MockBonito
            @test HyperCloud.BONITO_IS_CONFIGURED[] == false
            res = @test_logs (:info, r"Bonito server listens on") HyperCloud._launch_and_display_plotter(MockPDEStudio)
            @test res == "mock_figure"
            @test HyperCloud.BONITO_IS_CONFIGURED[] == true

            @test (@eval MockPDEStudio @plot dummy_config) == "mock_figure"
        end
    finally
        HyperCloud.CONFIG_ROOT_OVERWRITE[] = old_root
        HyperCloud.CONFIG_SEARCH_PATHS[]  = old_paths
        HyperCloud.BONITO_IS_CONFIGURED[] = old_bonito
    end
end

# -------------------------------------------------------------------------
# 3. CONFIGURATION BUILDERS (point tests + error paths)
# -------------------------------------------------------------------------
@testset "Configuration builders" begin
    ctx1 = Dict{Symbol, Any}(:T => Float64, :D => 1, :M => 1)
    ctxT = Dict{Symbol, Any}(:T => Float64)

    @testset "build_equation" begin
        eu = build_equation(Dict{Symbol, Any}(:name => :euler, :D => 2, :gamma => 1.4), ctxT)
        @test eu isa EulerEquation
        @test HC.get_D(eu) == 2
        @test HC.get_M(eu) == 4
        @test eu.gamma ≈ 1.4

        eu32 = build_equation(Dict{Symbol, Any}(:name => :euler, :D => 3, :gamma => 1.4), Dict{Symbol, Any}(:T => Float32))
        @test HC.get_M(eu32) == 5
        @test eu32.gamma isa Float32

        @test build_equation(Dict{Symbol, Any}(:name => :burgers, :D => 3), ctxT) isa BurgersEquation{3, Float64}

        lin = build_equation(Dict{Symbol, Any}(:name => :linear, :velocities => ((1.0,), (-2.0,))), ctxT)
        @test lin isa LinearAdvection
        @test HC.get_D(lin) == 2
        @test HC.get_M(lin) == 1

        @test_throws ErrorException build_equation(Dict{Symbol, Any}(:D => 1), ctxT)
        @test_throws ErrorException build_equation(Dict{Symbol, Any}(:name => :navier_stokes), ctxT)

        # Only the conservative representation exists; the others raise a clear error (no UndefVarError)
        eu_conf = Dict{Symbol, Any}(:name => :euler, :D => 1, :gamma => 1.4)
        @test build_equation(merge(eu_conf, Dict(:representation => :conservative)), ctxT).rep isa Conservative
        for rep in (:primitive, :lagrangian, :bogus)
            @test_throws r"not supported; only :conservative" build_equation(merge(eu_conf, Dict(:representation => rep)), ctxT)
            @test_throws ArgumentError build_equation(merge(eu_conf, Dict(:representation => rep)), ctxT)
        end
    end

    @testset "build_ic" begin
        ic = build_ic(Dict{Symbol, Any}(:name => :gauss, :a => (2.0,), :b => (0.5,), :width => 0.1), ctx1)
        @test ic isa Gauss
        @test ic(Space{1, Float64}(0.5))[1] ≈ 2.0
        @test ic(Space{1, Float64}(0.6))[1] ≈ 2.0 * exp(-1.0)

        @test_throws ErrorException build_ic(Dict{Symbol, Any}(:a => (1.0,)), ctx1)
        # Requires the fixed fallback signature in src/ICs/_main.jl
        @test_throws ErrorException build_ic(Dict{Symbol, Any}(:name => :does_not_exist), ctx1)
        # q_riemann needs one state per quadrant (a short tuple gave a BoundsError only when evaluated)
        ctx2 = Dict{Symbol, Any}(:T => Float64, :D => 2, :M => 1)
        @test_throws ArgumentError build_ic(Dict{Symbol, Any}(:name => :q_riemann, :states => ((1.0,), (2.0,)), :p0 => (0.0, 0.0)), ctx2)
        @test build_ic(Dict{Symbol, Any}(:name => :q_riemann, :states => ((1.0,), (2.0,), (3.0,), (4.0,)), :p0 => (0.0, 0.0)), ctx2) isa HC.QRiemann
    end

    @testset "ICs keep the requested precision ($T)" for T in (Float32, Float64)
        ctx = Dict{Symbol, Any}(:T => T, :D => 2, :M => 1)
        confs = [
            Dict{Symbol, Any}(:name => :gauss, :a => (1.0,), :b => (0.5, 0.5), :width => 0.1),
            Dict{Symbol, Any}(:name => :box, :u_bg => (0.1,), :u_box => (1.0,), :mins => (0.25, 0.25), :maxs => (0.75, 0.75)),
            Dict{Symbol, Any}(:name => :sine, :a => (1.0,), :period => (1.0, 1.0), :c_offset => (0.5,)),
            Dict{Symbol, Any}(:name => :riemann, :uL => (1.0,), :uR => (0.0,), :p0 => (0.5, 0.5), :n => (1.0, 1.0)),
            Dict{Symbol, Any}(:name => :s_riemann, :uL => (1.0,), :uR => (0.0,), :p0 => (0.5, 0.5), :n => (1.0, 0.0), :width => 0.05),
            Dict{Symbol, Any}(:name => :q_riemann, :p0 => (0.5, 0.5), :states => ((1.0,), (2.0,), (3.0,), (4.0,))),
        ]
        pos = Space{2, T}(0.3, 0.6)
        for conf in confs
            ic = build_ic(conf, ctx)
            @test eltype(ic(pos)) == T
        end
    end

    @testset "build_domain" begin
        geom = test_domain((0.0, -1.0), (2.0, 1.0), true)
        @test Tuple(geom.mins) == (0.0, -1.0)
        @test Tuple(geom.maxs) == (2.0, 1.0)
        # Pre-built domains are passed through untouched
        @test build_domain(Dict{Symbol, Any}(:domain => geom), ctx1) === geom
        @test_throws ErrorException build_domain(Dict{Symbol, Any}(:domain => :hexagon, :mins => (0.0,), :maxs => (1.0,)), ctx1)
        @test_throws ErrorException build_domain(Dict{Symbol, Any}(:mins => (0.0,)), ctx1)   # missing :domain
    end

    @testset "parse_bc" begin
        bcs = HC.parse_bc(Dict{Symbol, Any}(:bc => Dict(1 => :outflow, 2 => "fixed_dirichlet", 3 => OutflowBC())))
        @test bcs[1] isa OutflowBC
        @test bcs[2] isa FixedDirichlet
        @test bcs[3] isa OutflowBC
        @test isempty(HC.parse_bc(Dict{Symbol, Any}()))
        @test HC.parse_bc(Dict{Symbol, Any}(:bc => "Dict(1 => :outflow)"))[1] isa OutflowBC
        @test_throws ErrorException HC.parse_bc(Dict{Symbol, Any}(:bc => Dict(1 => :no_such_bc)))
    end

    @testset "build_weights" begin
        ctx = Dict{Symbol, Any}(:T => Float64, :max_dx => 0.1)
        @test build_weights(Dict{Symbol, Any}(:name => :exponential, :range => 2.5, :alpha => 1.0), ctx) isa HC.ExponentialWeightFunction
        @test ctx[:interp_range] ≈ 0.25
        @test build_weights(Dict{Symbol, Any}(:name => :inverse, :range => 3.0, :alpha => 2.0), ctx) isa HC.InverseWeightFunction
        @test ctx[:interp_range] ≈ 0.3

        @test_throws ErrorException build_weights(Dict{Symbol, Any}(:range => 2.5, :alpha => 1.0), ctx)
        @test_throws ErrorException build_weights(Dict{Symbol, Any}(:name => :exponential, :alpha => 1.0), ctx)
        @test_throws ErrorException build_weights(Dict{Symbol, Any}(:name => :gaussian, :range => 2.5, :alpha => 1.0), ctx)
    end

    @testset "build_tableau / build_scheme" begin
        for s in (:Euler, :RK2, :RK3, :RK4)
            @test build_tableau(Dict(:stepper => s), ctxT) isa HC.IMEXButcherTableau
        end
        for s in IMEX_STEPPERS
            @test build_tableau(Dict(:stepper => s), ctxT) isa HC.IMEXButcherTableau
        end
        @test_throws ErrorException build_tableau(Dict(:stepper => :RK42), ctxT)
        @test_throws ErrorException build_scheme(Dict{Symbol, Any}(:name => :Spectral), ctx1)
    end

    @testset "build_flux / build_limiter / build_mood" begin
        @test build_flux(Dict{Symbol, Any}(:name => :Rusanov), ctxT) isa HC.RusanovFlux
        @test build_flux(Dict{Symbol, Any}(:name => :Upwind), ctxT) isa HC.UpwindFlux
        @test_throws ErrorException build_flux(Dict{Symbol, Any}(:name => :Roe), ctxT)

        @test build_limiter(Dict{Symbol, Any}(), ctxT) isa HC.NoLimiter   # missing name ⇒ :none
        for (name, L) in ((:minmod, HC.MinmodLimiter), (:superbee, HC.SuperbeeLimiter),
                          (:VK, HC.VenkatakrishnanLimiter), (:BJ, HC.BarthJespersenLimiter))
            @test build_limiter(Dict{Symbol, Any}(:name => name, :mode => :hard), ctxT) isa L
            @test_throws KeyError build_limiter(Dict{Symbol, Any}(:name => name), ctxT)   # :mode is mandatory
        end
        @test_throws ErrorException build_limiter(Dict{Symbol, Any}(:name => :vanLeer, :mode => :hard), ctxT)

        pg, _ = test_grid((0.0,), (1.0,), (20,))
        mctx = Dict{Symbol, Any}(:T => Float64, :Grid => pg)
        @test build_mood(Dict{Symbol, Any}(), mctx) isa HC.MOOD
        for crit in (:U1, :U2), strat in (:EPD0, :SEPD0, :EPD1, :EPD2)
            @test build_mood(Dict{Symbol, Any}(:criterion => crit, :strategy => strat, :delta_relax => 1e-4), mctx) isa HC.MOOD
        end
        @test build_mood(Dict{Symbol, Any}(:criterion => :only, :strategy => :EPD1), mctx) isa HC.MOOD
        @test_throws KeyError build_mood(Dict{Symbol, Any}(:criterion => :U2, :strategy => :EPD1), mctx)   # :delta_relax is mandatory
        @test_throws ErrorException build_mood(Dict{Symbol, Any}(:criterion => :U3, :strategy => :EPD1), mctx)
        @test_throws ErrorException build_mood(Dict{Symbol, Any}(:criterion => :U2, :strategy => :EPD9, :delta_relax => 1e-4), mctx)
    end

    @testset "Upwind scheme is strictly first order" begin
        sctx = Dict{Symbol, Any}(:T => Float64, :D => 1, :M => 1,
                                 :Flux => build_flux(Dict{Symbol, Any}(:name => :Rusanov), ctxT))
        @test build_scheme(Dict{Symbol, Any}(:name => :Upwind, :order => 1, :upwind_alg_nd => :Classic), sctx) isa HC.UpwindDivergence
        @test_throws AssertionError build_scheme(Dict{Symbol, Any}(:name => :Upwind, :order => 2, :upwind_alg_nd => :Classic), sctx)
    end

    @testset "build_particle_grid" begin
        pg, ctx = test_grid((0.0,), (1.0,), (20,))
        @test ctx[:max_dx] ≈ 0.05
        @test ctx[:interp_range] ≈ 2.5 * 0.05

        # Jittered grids are reproducible for a fixed seed
        pa, _ = test_grid((0.0, 0.0), (1.0, 1.0), (8, 8); randomness = 0.2, seed = 10)
        pb, _ = test_grid((0.0, 0.0), (1.0, 1.0), (8, 8); randomness = 0.2, seed = 10)
        pc, _ = test_grid((0.0, 0.0), (1.0, 1.0), (8, 8); randomness = 0.2, seed = 11)
        @test pa.core.positions[1:pa.meta.N] == pb.core.positions[1:pb.meta.N]
        @test pa.core.positions[1:pa.meta.N] != pc.core.positions[1:pc.meta.N]
    end

    @testset "set_initial_conditions! (direct)" begin
        pg, _ = test_grid((0.0,), (1.0,), (16,))
        ic = Gauss(State{1, Float64}(1.0), Space{1, Float64}(0.5), 0.2)
        set_initial_conditions!(pg, BurgersEquation{1, Float64}(), ic)
        @test all(i -> pg.rhos[i] ≈ ic(pg.core.positions[i]), 1:pg.meta.N)
    end
end

# -------------------------------------------------------------------------
# 4. METHOD FACTORY
# -------------------------------------------------------------------------
@testset "Method factory" begin
    @testset "create_method labels" begin
        @test first(create_method(:RK2, :MUSCL, 2)) == "RK2MUSCL2"
        @test first(create_method(:RK2, :MUSCL, 2; limiter = :VK)) == "RK2MUSCL2(VK)"
        @test first(create_method(:RK2, :MUSCL, 2; mood_criterion = :U2, mood_strategy = :EPD1)) == "RK2MUSCL2(EPD1)"
        @test first(create_method(:RK2, :MUSCL, 2; mood_criterion = :U1, mood_strategy = :EPD2)) == "RK2MUSCL2(EPD2,U1)"
        @test first(create_method(:RK4, :MUSCL, 3; main_flux = :Upwind, limiter = :minmod)) == "RK4MUSCL3(Upwind,minmod)"
        @test first(create_method(:RK2, :MUSCL, 2; custom_label = "mine")) == "mine"
        # non-default limiter modes and MOOD relaxations are part of the label (they used to collide); the defaults
        # (`:hard`, 0) leave the existing labels as they are
        @test first(create_method(:RK2, :MUSCL, 2; limiter = :VK, limiter_mode = :soft)) == "RK2MUSCL2(VK soft)"
        @test first(create_method(:RK2, :MUSCL, 2; limiter = :VK, limiter_mode = :hard)) == "RK2MUSCL2(VK)"
        @test first(create_method(:RK2, :MUSCL, 2; mood_criterion = :U2, MOOD_delta_relax = 1e-4)) == "RK2MUSCL2(EPD1,δ=0.0001)"
        @test first(create_method(:RK2, :MUSCL, 2; mood_criterion = :U2, MOOD_delta_relax = 0.0)) == "RK2MUSCL2(EPD1)"
    end

    @testset "create_method pipeline inference and kwargs" begin
        for ts in IMEX_STEPPERS
            @test last(create_method(ts, :MUSCL, 2))[:sim_func_name] === :run_kinetic_simulation
        end
        for ts in (:Euler, :RK2, :RK3, :RK4)
            @test last(create_method(ts, :MUSCL, 2))[:sim_func_name] === :run_direct_simulation
        end
        @test last(create_method(:RK2, :MUSCL, 2; sim_func_name = :custom))[:sim_func_name] === :custom

        d = last(create_method(:RK2, :Upwind, 1; Scheme_upwind_alg_nd = "Classic", MOOD_delta_relax = 1e-4))
        @test d[:Scheme_upwind_alg_nd] === :Classic   # strings are converted to Symbols
        @test d[:MOOD_delta_relax] == 1e-4
        @test d[:Time_stepper] === :RK2
        @test d[:Scheme_order] == 1
    end

    @testset "get_master_method_dict consistency" begin
        md = get_master_method_dict()
        @test haskey(md, "Analytical Solution")
        @test md["ARS222MUSCL2"][:Time_stepper] === :ARS222        # the kinetic method of `burgers_kinetic`
        for (name, d) in md
            haskey(d, :Time_stepper) || continue
            expected = d[:Time_stepper] in IMEX_STEPPERS ? :run_kinetic_simulation : :run_direct_simulation
            @test d[:sim_func_name] === expected
            @test startswith(name, string(d[:Time_stepper], d[:Scheme_name], d[:Scheme_order]))
        end
    end

    @testset "generate_relax_params" begin
        vels, idx = generate_relax_params(2.0, 2, 2)
        @test idx == [1, 5, 9]
        @test vels[1] == (2.0, -2.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0)
        @test vels[2] == (0.0, 0.0, 2.0, -2.0, 0.0, 0.0, 2.0, -2.0)

        v1, i1 = generate_relax_params(1.5, 1, 3)
        @test i1 == [1, 3, 5, 7]
        @test v1 == ((1.5, -1.5, 1.5, -1.5, 1.5, -1.5),)
        @test generate_relax_params(2, 1, 1) == generate_relax_params(2.0, 1, 1)   # an integer speed (was a MethodError)
    end
end

# -------------------------------------------------------------------------
# 5. EQUATION KERNELS
# -------------------------------------------------------------------------
@testset "Equation kernels" begin
    @testset "Linear advection" begin
        eq = LinearAdvection(((2.0,), (-3.0,)), Float64)
        U = State{1, Float64}(1.5)
        @test eq.max_eigs ≈ SVector(2.0, 3.0)
        F = @inferred HC.flux(eq, U)
        @test F[1][1] ≈ 3.0
        @test F[2][1] ≈ -4.5
        @test (@inferred HC.max_eigenvalue(eq, U, 2)) ≈ 3.0
        @test HC.kinetic_wave_speed(eq, 2, 1) ≈ -3.0
        @test HC.velocity(eq, U, 1) ≈ SMatrix{1, 1}(2.0)
        @test HC.prim2cons(eq, U) == U
        @test HC.cons2prim(eq, U) == U
    end

    @testset "Burgers" begin
        eq = BurgersEquation{2, Float64}()
        u = State{1, Float64}(-3.0)
        F = @inferred HC.flux(eq, u)
        @test F[1][1] ≈ 4.5
        @test F[2][1] ≈ 4.5
        @test (@inferred HC.max_eigenvalue(eq, u, 1)) ≈ 3.0
        @test HC.velocity(eq, u, 2) == SMatrix{1, 1}(-3.0)
        @test HC.get_D(eq) == 2
        @test HC.get_M(eq) == 1
    end

    @testset "Euler (2D, conservative)" begin
        γ = 1.4
        eq = EulerEquation(Val(2), Float64, γ, Conservative())
        ρ, u, v, p = 1.2, 0.3, -0.4, 0.9
        V = State{4, Float64}(ρ, u, v, p)
        E = p / (γ - 1) + 0.5 * ρ * (u^2 + v^2)

        U = @inferred HC.prim2cons(eq, V)
        @test U ≈ [ρ, ρ * u, ρ * v, E]
        @test (@inferred HC.cons2prim(eq, U)) ≈ V

        F = @inferred HC.flux(eq, U)
        @test F[1] ≈ [ρ * u, ρ * u^2 + p, ρ * u * v, u * (E + p)]
        @test F[2] ≈ [ρ * v, ρ * u * v, ρ * v^2 + p, v * (E + p)]

        c = sqrt(γ * p / ρ)
        @test (@inferred HC.max_eigenvalue(eq, U, 1)) ≈ abs(u) + c
        @test HC.max_eigenvalue(eq, U, 2) ≈ abs(v) + c

        # Positivity floors in cons2prim
        Vbad = HC.cons2prim(eq, State{4, Float64}(1.0, 0.0, 0.0, -1.0))
        @test Vbad[4] ≈ HC._euler_floor(Float64)       # the same floor as the wave speeds (was 1e-7 vs 1e-9)

        # Admissibility (MOOD PAD): positive density and pressure, finite
        @test HC.is_admissible(eq, U)
        @test !HC.is_admissible(eq, State{4, Float64}(1.0, 0.0, 0.0, -1.0))            # p < 0
        @test !HC.is_admissible(eq, State{4, Float64}(1.0, 2.0, 0.0, 1.0))             # kinetic energy > E
        @test !HC.is_admissible(eq, State{4, Float64}(-1.0, 0.0, 0.0, 1.0))            # ρ < 0
        @test !HC.is_admissible(eq, State{4, Float64}(NaN, 0.0, 0.0, 1.0))
    end
end

# -------------------------------------------------------------------------
# 6. KINETIC RELAXATION
# -------------------------------------------------------------------------
@testset "Kinetic relaxation" begin
    @testset "Kin2Macro" begin
        km = HC.Kin2Macro([1, 3, 6])
        @test km.ranges == (1:2, 3:5)
        @test km.k_to_m == (1, 1, 2, 2, 2)
        @test km(4) == 2
        @test km(SVector(1.0, 2.0, 3.0, 4.0, 5.0)) ≈ [3.0, 12.0]
    end

    @testset "Relaxation source term (D = $D)" for D in (1, 2)
        eq_macro = BurgersEquation{D, Float64}()
        vels, idx = generate_relax_params(2.0, D, 1)
        NK = 2D
        ε = 1e-2
        ctx = Dict{Symbol, Any}(:T => Float64, :D => D, :M => 1, :Equation => eq_macro)
        eq_kin, st = build_kinetic_system(Dict{Symbol, Any}(:velocities => vels, :epsilon => ε, :indices => idx), ctx)

        @test HC.get_M(eq_kin) == NK
        @test eq_kin.max_eigs ≈ fill(2.0, D)
        @test st.inv_epsilon ≈ 1 / ε
        # `:indices` as a tuple, as every other key (was a `Vector{Int}` type assertion)
        _, st_t = build_kinetic_system(Dict{Symbol, Any}(:velocities => vels, :epsilon => ε, :indices => Tuple(idx)), ctx)
        @test st_t.km == st.km

        pg, _ = test_grid(ntuple(_ -> 0.0, D), ntuple(_ -> 1.0, D), ntuple(_ -> 8, D); M = NK)
        ic = Gauss(State{1, Float64}(0.8), Space{D, Float64}(ntuple(_ -> 0.5, D)), 0.3)
        set_initial_conditions!(pg, st, ic, eq_macro)

        idxs = 1:pg.meta.N
        feq  = [pg.rhos[i] for i in idxs]
        umac = [ic(pg.core.positions[i]) for i in idxs]

        # Zeroth moment: the Maxwellian carries exactly the macroscopic state
        @test all(i -> st.km(feq[i])[1] ≈ umac[i][1], idxs)

        # First moment: Σ_k λ_k^d f_k reproduces the physical flux in every direction
        @test all(Iterators.product(idxs, 1:D)) do (i, d)
            Fd = HC.flux(eq_macro, umac[i])[d][1]
            λf = sum(HC.kinetic_wave_speed(eq_kin, d, k) * feq[i][k] for k in 1:NK)
            isapprox(λf, Fd; atol = 1e-12)
        end

        # Equilibrium is a fixed point of both the source and the implicit solve
        @test all(i -> norm(HC.evaluate_source(st, feq[i], i, pg, 0.0)) < 1e-10, idxs)
        @test all(i -> HC.implicit_solve(st, feq[i], 0.1, i, pg, 0.0) ≈ feq[i], idxs)

        # A moment-free perturbation decays exactly as δ / (1 + Δt/ε) and conserves the macro state
        δ  = State{NK, Float64}(ntuple(k -> isodd(k) ? 0.1 : -0.1, NK))
        Δt = 0.05
        p  = first(idxs)
        Y  = feq[p] + δ
        Yn = HC.implicit_solve(st, Y, Δt, p, pg, 0.0)
        @test Yn ≈ feq[p] + δ / (1 + Δt / ε)
        @test st.km(Yn) ≈ st.km(Y)
        @test HC.evaluate_source(st, Y, p, pg, 0.0) ≈ -δ / ε
    end
end

# -------------------------------------------------------------------------
# 6b. BGK EQUATION
# -------------------------------------------------------------------------
@testset "BGK equation" begin
    "Macroscopic state (ρ, ρu..., E) from density, velocity and temperature."
    macro_state(eq, ρ, u, θ) = SVector(ρ, (ρ .* u)..., 0.5 * ρ * sum(abs2, u) + 0.5 * (HC.get_D(eq) + eq.K) * ρ * θ)

    @testset "Velocity grids" begin
        v, w = velocity_grid(:uniform, 8, 4.0)
        @test v ≈ -v[end:-1:1]
        @test all(≈(1.0), w) && sum(w) ≈ 8.0
        @test v[1] ≈ -3.5 && v[end] ≈ 3.5
        # Gauss–Hermite: exact for polynomials times the Gaussian of the grid's scale
        Nv = 10
        x, W = velocity_grid(:gauss_hermite, Nv, 1.0)
        @test maximum(abs, x) ≈ 1.0
        σ = 1 / maximum(eigvals(SymTridiagonal(zeros(Nv), [sqrt(j / 2) for j in 1:(Nv - 1)])))   # node scale
        for n in 0:(Nv - 1)
            exact = sqrt(π) * σ^(2n + 1) * prod(1.0:2:(2n - 1); init = 1.0) / 2.0^n
            @test sum(W .* exp.(-(x ./ σ) .^ 2) .* x .^ (2n)) ≈ exact rtol = 1e-12
        end
        @test eltype(velocity_grid(:gauss_hermite, 6, 3.0, Float32)[1]) == Float32
        @test_throws ArgumentError velocity_grid(:uniform, 1, 1.0)
        @test_throws ArgumentError velocity_grid(:uniform, 8, -1.0)
        @test_throws ArgumentError velocity_grid(:chebyshev, 8, 1.0)
    end

    @testset "Construction: γ fixes the internal degrees of freedom" begin
        nodes, w = velocity_grid(:uniform, 6, 5.0)
        for (D, γ, K, M) in ((1, 3.0, 0, 6), (1, 1.4, 4, 12), (1, 5 / 3, 2, 12), (2, 2.0, 0, 36), (2, 1.4, 3, 72), (3, 5 / 3, 0, 216))
            eq = BGKEquation(Val(D), nodes, w, γ, ConstantRelaxationTime(1.0))
            @test eq.K ≈ K
            @test HC.get_M(eq) == M
            @test num_velocities(eq) == 6^D
            @test eq.gamma ≈ γ
            @test euler_limit(eq).gamma ≈ γ
            @test HC.max_eigenvalue(eq, zero(State{M, Float64}), 1) ≈ 5 * 5 / 6   # outermost midpoint node
        end
        @test_throws ArgumentError BGKEquation(Val(1), nodes, w, 4.0, ConstantRelaxationTime(1.0))
        eqc = BGKEquation(Val(2), nodes, w, 2.0, ConstantRelaxationTime(1.0); center = (1.0, 0.0))
        @test eqc.max_speed ≈ [1 + 5 * 5 / 6, 5 * 5 / 6]
        v, wk = velocity_node(eqc, 8)                                 # node (2, 2) of the 6×6 grid
        @test v ≈ [1.0 + nodes[2], nodes[2]] && wk ≈ w[2]^2
    end

    @testset "build_equation(:bgk) ($T)" for T in (Float32, Float64)
        ctx = Dict{Symbol, Any}(:T => T)
        conf = Dict{Symbol, Any}(:name => :bgk, :D => 1, :Nv => 16, :vmax => 6.0, :gamma => 1.4, :tau => 1e-3)
        eq = build_equation(conf, ctx)
        @test eq isa BGKEquation{1, 32, T}
        @test eq.relaxation == ConstantRelaxationTime{T}(T(1e-3))
        eqv = build_equation(merge(delete!(copy(conf), :tau), Dict(:viscosity => 1e-2, :omega => 0.81)), ctx)
        @test eqv.relaxation isa ViscosityRelaxationTime{T}
        @test relaxation_time(eqv.relaxation, T(2), T(1.5)) ≈ T(1e-2 * 1.5^(0.81 - 1) / 2)
        eqg = build_equation(merge(conf, Dict(:velocity_grid => :gauss_hermite, :gamma => 3.0)), ctx)
        @test HC.get_M(eqg) == 16 && maximum(abs, eqg.nodes) ≈ 6
        @test HC.get_M(build_equation(delete!(copy(conf), :gamma), ctx)) == 16   # default γ = (D + 2) / D
        @test_throws ErrorException build_equation(delete!(copy(conf), :tau), ctx)
        @test_throws ErrorException build_equation(merge(conf, Dict(:viscosity => 1.0)), ctx)
    end

    @testset "Initial state the velocity grid cannot represent: warned once" begin
        T = Float64
        # γ = 3, 16 nodes up to vmax = 6: a state at rest is represented to round-off (quiet); u = 5.5 puts the
        # Maxwellian against the edge of the grid (moment error O(1), the run would start elsewhere: one warning)
        eq = build_equation(Dict{Symbol, Any}(:name => :bgk, :D => 1, :Nv => 16, :vmax => 6.0, :gamma => 3.0, :tau => 1.0),
                            Dict{Symbol, Any}(:T => T))
        pg, _ = test_grid((0.0,), (1.0,), (8,); M = HC.get_M(eq))
        rest = Riemann(State{3, T}(1, 0, 0.5), State{3, T}(1, 0, 0.5), Space{1, T}(0.5), Space{1, T}(1))
        fast = Riemann(State{3, T}(1, 0, 0.5), State{3, T}(1, 5.5, 5.5^2 / 2 + 0.5), Space{1, T}(0.5), Space{1, T}(1))
        @test_logs set_initial_conditions!(pg, eq, rest)
        @test_logs (:warn, r"cannot represent the macroscopic state of \d+ particle") set_initial_conditions!(pg, eq, fast)
    end

    @testset "Conservative discrete Maxwellian (D = $D, γ = $γ, $kind, $T)" for (D, γ, Nv, kind) in
            ((1, 3.0, 24, :uniform), (1, 1.4, 16, :uniform), (1, 5 / 3, 12, :gauss_hermite), (2, 2.0, 10, :uniform), (2, 1.4, 8, :gauss_hermite)),
            T in (Float32, Float64)
        nodes, w = velocity_grid(kind, Nv, 6.0, T)
        eq = BGKEquation(Val(D), nodes, w, γ, ConstantRelaxationTime(T(1e-2)))
        tol = T == Float64 ? 1e-11 : 1e-4
        for (ρ, u, θ) in ((1.0, ntuple(_ -> 0.0, D), 1.0), (0.125, ntuple(d -> 0.3 * d, D), 0.8), (2.0, ntuple(d -> -0.5, D), 0.4))
            U = SVector{D + 2, T}(macro_state(eq, ρ, u, θ))
            f, ok = equilibrium(eq, U)
            @test ok
            @test f isa State{HC.get_M(eq), T}
            @test isapprox(moments(eq, f), U; rtol = tol)
            @test all(>(0), f)
            @test is_admissible(eq, f)
            @test HC.fluid_velocity(eq, f) ≈ SVector{D, T}(u) atol = 10tol
            @test relaxation_time(eq, U) == T(1e-2)
        end
        # On a grid that resolves it, the discrete equilibrium is the continuous Maxwellian at the nodes
        if kind === :uniform && D == 1
            U = SVector{3, T}(macro_state(eq, 1.0, (0.2,), 1.0))
            f, _ = equilibrium(eq, U)
            vs = [velocity_node(eq, k)[1][1] for k in 1:Nv]
            G = @. exp(-(vs - 0.2)^2 / 2) / sqrt(2π)
            @test isapprox(f[1:Nv], G; atol = T == Float64 ? 1e-7 : 1e-5)
            eq.K > 0 && @test isapprox(f[(Nv + 1):(2Nv)], 0.5 * eq.K .* f[1:Nv]; rtol = 1e-5)
        end
        # Non-physical states are reported, not solved
        @test !equilibrium(eq, SVector{D + 2, T}(macro_state(eq, -1.0, ntuple(_ -> 0.0, D), 1.0)))[2]
        @test !equilibrium(eq, SVector{D + 2, T}(macro_state(eq, 1.0, ntuple(_ -> 0.0, D), -1.0)))[2]
        @test !equilibrium(eq, SVector{D + 2, T}(ntuple(_ -> T(NaN), D + 2)))[2]
    end

    @testset "Collision term" begin
        nodes, w = velocity_grid(:uniform, 16, 6.0)
        eq = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(1e-2))
        M  = HC.get_M(eq)
        imp, expl = BGKRelaxation(eq), ExplicitBGKRelaxation(eq)
        pg, _ = test_grid((0.0,), (1.0,), (4,); M = M)
        feq, _ = equilibrium(eq, SVector(1.0, 0.3, 2.0))
        δ = State{M, Float64}(ntuple(k -> 0.01 * sin(k), M))
        f = feq + δ
        Uf = moments(eq, f)
        Mf, _ = equilibrium(eq, Uf)

        @test norm(HC.evaluate_source(imp, feq, 1, pg, 0.0)) < 1e-9
        @test HC.implicit_solve(imp, feq, 0.1, 1, pg, 0.0) ≈ feq
        S = HC.evaluate_source(imp, f, 1, pg, 0.0)
        @test S ≈ (Mf - f) / 1e-2
        @test HC.evaluate_source(expl, f, 1, pg, 0.0) == S
        @test norm(moments(eq, S)) < 1e-9 * norm(Uf) / 1e-2              # mass, momentum and energy conserved
        # The implicit solve is the exact solution of f_new = f + Δt S(f_new)
        Δt = 0.05
        fn = HC.implicit_solve(imp, f, Δt, 1, pg, 0.0)
        @test fn ≈ f + Δt * HC.evaluate_source(imp, fn, 1, pg, 0.0)
        @test moments(eq, fn) ≈ Uf
        @test HC.implicit_solve(imp, f, 1e8, 1, pg, 0.0) ≈ Mf                # stiff limit: projection onto equilibrium
        # A non-physical state is left alone
        bad = -feq
        @test HC.implicit_solve(imp, bad, Δt, 1, pg, 0.0) == bad
        @test iszero(HC.evaluate_source(imp, bad, 1, pg, 0.0))
        @test !is_admissible(eq, bad)
        # Viscosity law: τ = μ_ref θ^(ω-1) / ρ
        eqv = BGKEquation(Val(1), nodes, w, 1.4, ViscosityRelaxationTime(1e-3, 0.5))
        ρ, θ = Uf[1], 2 * (Uf[3] - 0.5 * Uf[2]^2 / Uf[1]) / (5 * Uf[1])
        @test relaxation_time(eqv, Uf) ≈ 1e-3 * θ^(-0.5) / ρ

        # Source selection follows the tableau
        @test HC.bgk_sources(eq, IMEX_ARS222_Tableau(Float64)) == ((), (BGKRelaxation(eq),))
        @test HC.bgk_sources(eq, RK2_SSP_Tableau(Float64)) == ((ExplicitBGKRelaxation(eq),), ())
        eq_free = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(Inf))
        @test HC.bgk_sources(eq_free, IMEX_ARS222_Tableau(Float64)) == ((), ())

        # τ = 0: the implicit collision is the projection onto the equilibrium (it was Inf / Inf = NaN);
        # the explicit form cannot divide by τ = 0, and τ < 0 is rejected at construction.
        for T in (Float64, Float32)
            nodesT, wT = velocity_grid(:uniform, 16, 6.0, T)
            eq0 = BGKEquation(Val(1), nodesT, wT, 1.4, ConstantRelaxationTime(0.0))
            pgT, _ = test_grid((0.0,), (1.0,), (4,); M = M)
            fT = State{M, T}(f)
            MfT, _ = equilibrium(eq0, moments(eq0, fT))
            f0 = HC.implicit_solve(BGKRelaxation(eq0), fT, T(0.05), 1, pgT, zero(T))
            @test all(isfinite, f0) && f0 ≈ MfT
            @test HC.bgk_sources(eq0, IMEX_ARS222_Tableau(T)) == ((), (BGKRelaxation(eq0),))
            @test_throws ArgumentError HC.bgk_sources(eq0, RK2_SSP_Tableau(T))
            eqμ0 = BGKEquation(Val(1), nodesT, wT, 1.4, ViscosityRelaxationTime(0.0, 0.5))
            @test_throws ArgumentError HC.bgk_sources(eqμ0, RK2_SSP_Tableau(T))
        end
        @test_throws ArgumentError BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(-1e-3))
        @test_throws ArgumentError BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(NaN))
        @test_throws ArgumentError BGKEquation(Val(1), nodes, w, 1.4, ViscosityRelaxationTime(-1e-3, 0.5))
    end

    @testset "Numerical flux is the exact upwind flux per velocity" begin
        nodes, w = velocity_grid(:uniform, 6, 3.0)
        for D in (1, 2)
            eq = BGKEquation(Val(D), nodes, w, (D + 2) / D, ConstantRelaxationTime(1.0))
            M = HC.get_M(eq)
            fL = Flux{D, M, Float64}(ntuple(d -> State{M, Float64}(ntuple(k -> 1 + 0.1k + d, M)), D))
            fR = Flux{D, M, Float64}(ntuple(d -> State{M, Float64}(ntuple(k -> 2 - 0.05k * d, M)), D))
            FL = Flux{D, M, Float64}(ntuple(d -> HC.flux(eq, fL[d])[d], D))
            FR = Flux{D, M, Float64}(ntuple(d -> HC.flux(eq, fR[d])[d], D))
            exact = Flux{D, M, Float64}(ntuple(D) do d
                State{M, Float64}(ntuple(M) do k
                    v = velocity_node(eq, k)[1][d]
                    max(v, 0) * fL[d][k] + min(v, 0) * fR[d][k]
                end)
            end)
            for F in (RusanovFlux(), UpwindFlux(), AlignedRusanovFlux())
                @test F(fL, fR, FL, FR, eq) ≈ exact
            end
            @test HC.flux(eq, fL[1])[D] ≈ [velocity_node(eq, k)[1][D] * fL[1][k] for k in 1:M]
        end
    end

    @testset "Free transport closure" begin
        nodes, w = velocity_grid(:uniform, 16, 6.0)
        geom = test_domain((0.0,), (1.0,), true)
        ic = Sine(State{3, Float64}(0.2, 0.1, 0.1), Space{1, Float64}(1.0), State{3, Float64}(1.0, 0.0, 0.5))
        eq = BGKEquation(Val(1), nodes, w, 3.0, ConstantRelaxationTime(Inf))
        exact = analytic_closure(eq, ic, geom)
        @test all(x -> exact(SVector(x, 0.0)) ≈ ic(SVector(x)), (0.1, 0.37, 0.8))
        @test exact(SVector(0.3, 0.25)) ≈ exact(SVector(1.3, 0.25))                     # periodic wrap
        @test integrate(x -> exact(SVector(x, 0.25))[1], 0.0, 1.0; n = 2000) ≈ 1.0 rtol = 1e-10   # mass is transported
        eq_coll = BGKEquation(Val(1), nodes, w, 3.0, ConstantRelaxationTime(1e-3))
        @test_logs (:warn, r"No Analytical Solution") analytic_closure(eq_coll, ic, geom)
    end
end

# -------------------------------------------------------------------------
# 7. ANALYTICAL CLOSURES
# -------------------------------------------------------------------------
@testset "Analytical closures" begin

    @testset "Linear advection" begin
        @testset "1D scalar with periodic wrap" begin
            eq = LinearAdvection(((2.0,),), Float64)
            ic = Gauss(State{1, Float64}(1.0), Space{1, Float64}(0.0), 1.0)
            exact = analytic_closure(eq, ic, test_domain((-5.0,), (5.0,), true))
            @test exact(SVector(0.0, 0.0))[1] ≈ 1.0   # t = 0 matches the IC
            @test exact(SVector(2.0, 1.0))[1] ≈ 1.0   # peak at x = 2
            @test exact(SVector(0.0, 5.0))[1] ≈ 1.0   # moved 10 = one period
        end

        @testset "2D scalar with wrap in both directions" begin
            eq = LinearAdvection(((1.0,), (0.5,)), Float64)
            ic = Gauss(State{1, Float64}(1.0), Space{2, Float64}(1.8, 0.0), 0.5)
            exact = analytic_closure(eq, ic, test_domain((-2.0, -2.0), (2.0, 2.0), true))
            @test exact(SVector(-1.8, 0.2, 0.4))[1] ≈ 1.0   # peak crossed x = 2 and re-entered at -2
            @test exact(SVector(1.8, 0.0, 8.0))[1] ≈ 1.0    # shifted by (8, 4) = (2, 1) periods
        end

        @testset "1D two-component diagonal system" begin
            eq = LinearAdvection(((1.0, -1.0),), Float64)
            @test HC.get_M(eq) == 2
            ic = Gauss(State{2, Float64}(1.0, 2.0), Space{1, Float64}(0.0), 0.5)
            exact = analytic_closure(eq, ic, test_domain((-5.0,), (5.0,), true))
            @test exact(SVector(1.0, 1.0))[1] ≈ 1.0                          # component 1 moves right
            @test exact(SVector(-1.0, 1.0))[2] ≈ 2.0                         # component 2 moves left
            @test exact(SVector(1.0, 1.0))[2] ≈ 2.0 * exp(-16.0) atol = 1e-12
        end

        @testset "2D kinetic velocity set (repeated eigenvalues of Σ A_d)" begin
            vels, _ = generate_relax_params(1.0, 2, 1)
            eq = LinearAdvection(vels, Float64)
            ic = Gauss(State{4, Float64}(1.0, 2.0, 3.0, 4.0), Space{2, Float64}(0.0, 0.0), 0.3)
            exact = analytic_closure(eq, ic, test_domain((-3.0, -3.0), (3.0, 3.0), true))
            peaks = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
            for k in 1:4
                @test exact(SVector(peaks[k]..., 1.0))[k] ≈ k
            end
        end

        @testset "Commutation check is relative ($T, entries O($s))" for (T, s) in ((Float64, 100.0), (Float32, 1.0))
            # A₂ = p(A₁) commutes with A₁, but the round-off of the products (2e-11 resp. 7e-7) exceeded the old
            # absolute tolerance 1e-12
            A1 = SMatrix{3, 3, T}(1.1, 0.7, 0.3, 0.7, 2.2, 0.5, 0.3, 0.5, 0.9) .* T(s)
            A2 = A1 * A1 ./ T(37) .+ T(0.3) .* A1
            eq = LinearAdvection(SVector{2, SMatrix{3, 3, T, 9}}((A1, A2)), T)
            geom = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-3.0, -3.0), :maxs => (3.0, 3.0),
                                                  :periodic => true), Dict{Symbol, Any}(:T => T, :D => 2))
            ic = Gauss(State{3, T}(1, 0, 0), Space{2, T}(0, 0), T(0.3))
            exact = analytic_closure(eq, ic, geom)
            @test exact(SVector{3, T}(0.1, -0.2, 0)) ≈ ic(Space{2, T}(0.1, -0.2))
            @test eltype(exact(SVector{3, T}(0.1, -0.2, 0.01))) == T
            # Non-commuting matrices are still rejected
            B = SMatrix{3, 3, T}(0, 1, 0, 1, 0, 0, 0, 0, 1) .* T(s)
            eq_nc = LinearAdvection(SVector{2, SMatrix{3, 3, T, 9}}((A1, B)), T)
            @test_throws r"do not commute" analytic_closure(eq_nc, ic, geom)
        end

        @testset "Precision is preserved" begin
            eq = LinearAdvection(((1.0,),), Float32)
            ic = Gauss(State{1, Float32}(1.0f0), Space{1, Float32}(0.0f0), 0.5)
            exact = analytic_closure(eq, ic, test_domain((-1.0,), (1.0,), true))
            @test eltype(exact(SVector(0.5f0, 0.1f0))) == Float32
        end
    end

    @testset "Burgers" begin
        eq    = BurgersEquation{1, Float64}()
        geom  = test_domain((-5.0,), (5.0,), false)
        eq2   = BurgersEquation{2, Float64}()
        geom2 = test_domain((-5.0, -5.0), (5.0, 5.0), (false, false))

        @testset "1D Riemann" begin
            # Shock: uL = 2, uR = 0, s = 1
            ex = analytic_closure(eq, Riemann(State{1}(2.0), State{1}(0.0), Space{1}(0.0), Space{1}(1.0)), geom)
            @test ex(SVector(0.5, 1.0))[1] ≈ 2.0
            @test ex(SVector(1.5, 1.0))[1] ≈ 0.0

            # Shifted interface, plus t = 0 returns the IC
            ex = analytic_closure(eq, Riemann(State{1}(2.0), State{1}(0.0), Space{1}(1.0), Space{1}(1.0)), geom)
            @test ex(SVector(1.9, 1.0))[1] ≈ 2.0
            @test ex(SVector(2.1, 1.0))[1] ≈ 0.0
            @test ex(SVector(0.5, 0.0))[1] ≈ 2.0
            @test ex(SVector(1.5, 0.0))[1] ≈ 0.0

            # Rarefaction: fan spans [0, 2] at t = 1
            ex = analytic_closure(eq, Riemann(State{1}(0.0), State{1}(2.0), Space{1}(0.0), Space{1}(1.0)), geom)
            @test ex(SVector(-1.0, 1.0))[1] ≈ 0.0
            @test ex(SVector(1.0, 1.0))[1] ≈ 1.0
            @test ex(SVector(3.0, 1.0))[1] ≈ 2.0
        end

        @testset "1D Box" begin
            # Top-hat: background 0, box 2 on [0, 2] → t_int = 2
            tophat = analytic_closure(eq, HC.Box(State{1}(0.0), State{1}(2.0), Space{1}(0.0), Space{1}(2.0)), geom)
            @test tophat(SVector(-1.0, 1.0))[1] ≈ 0.0   # background
            @test tophat(SVector(1.0, 1.0))[1] ≈ 1.0    # rarefaction fan (x - xs)/t
            @test tophat(SVector(2.5, 1.0))[1] ≈ 2.0    # plateau
            @test tophat(SVector(3.5, 1.0))[1] ≈ 0.0    # ahead of the shock at x = 3
            @test tophat(SVector(2.0, 4.0))[1] ≈ 0.5    # after interaction: decaying fan
            @test tophat(SVector(6.0, 4.0))[1] ≈ 0.0    # ahead of the shock at x ≈ 5.657

            # Well: background 2, box 0 on [0, 2] → t_int = 2
            well = analytic_closure(eq, HC.Box(State{1}(2.0), State{1}(0.0), Space{1}(0.0), Space{1}(2.0)), geom)
            @test well(SVector(0.5, 1.0))[1] ≈ 2.0
            @test well(SVector(1.5, 1.0))[1] ≈ 0.0
            @test well(SVector(3.0, 1.0))[1] ≈ 1.0
            @test well(SVector(5.0, 1.0))[1] ≈ 2.0

            # Mass relative to the background is conserved, before and after t_int.
            # This pins down the shock positions without hand-computing them.
            for (ex, ug, mass0) in ((tophat, 0.0, 4.0), (well, 2.0, -4.0)), t in (0.5, 1.0, 2.0, 4.0)
                @test integrate(x -> ex(SVector(x, t))[1] - ug, -10.0, 15.0) ≈ mass0 atol = 1e-3
            end
        end

        @testset "Gauss: characteristic relation u = u₀(x − u t) holds" begin
            t = 0.3   # below the breaking times (≈ 0.58 in 1D, ≈ 0.41 in 2D)

            ic1 = Gauss(State{1, Float64}(1.0), Space{1, Float64}(0.0), 0.5)
            ex1 = analytic_closure(eq, ic1, geom)
            @test all(range(-2.0, 2.0; length = 81)) do x
                u = ex1(SVector(x, t))[1]
                isapprox(u, ic1(Space{1, Float64}(x - u * t))[1]; atol = 1e-8)
            end

            ic2 = Gauss(State{1, Float64}(1.0), Space{2, Float64}(0.0, 0.0), 0.5)
            ex2 = analytic_closure(eq2, ic2, geom2)
            grid = range(-1.5, 1.5; length = 21)
            @test all(Iterators.product(grid, grid)) do (x, y)
                u = ex2(SVector(x, y, t))[1]
                isapprox(u, ic2(Space{2, Float64}(x - u * t, y - u * t))[1]; atol = 1e-8)
            end

            # Point checks on a wider pulse (width 1 ⇒ breaking time ≈ 0.82, so t = 0.5 is still smooth)
            ic3 = Gauss(State{1, Float64}(1.0), Space{2, Float64}(0.0, 0.0), 1.0)
            ex3 = analytic_closure(eq2, ic3, geom2)
            @test ex3(SVector(0.0, 0.0, 0.0))[1] ≈ 1.0
            @test ex3(SVector(1.0, 0.0, 0.0))[1] ≈ exp(-1.0)
            @test ex3(SVector(0.5, 0.5, 0.5))[1] ≈ 1.0   # peak travels along (1, 1)
        end

        @testset "Gauss: breaking time, warn once, Float32 tolerance" begin
            # u₀ = exp(-x²) breaks at t_b = e^{1/2} / √2 ≈ 1.166 (the pulse of `burgers_kinetic`)
            ic = Gauss(State{1, Float64}(1.0), Space{1, Float64}(0.0), 1.0)
            ex = analytic_closure(eq, ic, geom)
            xs = range(-3.0, 4.0; length = 141)
            @test_logs min_level = Base.CoreLogging.Warn foreach(x -> ex(SVector(x, 1.1)), xs)
            logs, _ = Test.collect_test_logs() do
                foreach(t -> foreach(x -> ex(SVector(x, t)), xs), (1.5, 2.0))
            end
            @test count(l -> occursin("breaking time", string(l.message)), logs) == 1
            @test count(l -> occursin("did not converge", string(l.message)), logs) <= 1
            @test all(l -> l.level == Base.CoreLogging.Warn, logs) && length(logs) <= 2

            # The reference of `burgers_kinetic` stops before breaking and conserves mass there
            p = copy(burgers_kinetic().shared)
            @test p[:tmax] < exp(0.5) / sqrt(2.0)
            exk = analytical_solution(p)
            m0 = integrate(x -> exk(SVector(x, 0.0))[1], -5.0, 5.0; n = 4000)
            @test integrate(x -> exk(SVector(x, p[:tmax]))[1], -5.0, 5.0; n = 4000) ≈ m0 rtol = 1e-6

            # Float32: the Newton tolerance is relative, so it converges without warnings before breaking
            ic32 = Gauss(State{1, Float32}(1.0f0), Space{1, Float32}(0.0f0), 1.0f0)
            ex32 = analytic_closure(BurgersEquation{1, Float32}(), ic32, geom)
            xs32 = range(-2.0f0, 2.0f0; length = 41)
            @test_logs min_level = Base.CoreLogging.Warn foreach(x -> ex32(SVector(x, 0.5f0)), xs32)
            @test all(x -> eltype(ex32(SVector(x, 0.5f0))) == Float32, xs32)
            @test maximum(x -> abs(ex32(SVector(x, 0.5f0))[1] - ex(SVector(Float64(x), 0.5))[1]), xs32) < 1e-6
        end

        @testset "2D Riemann" begin
            n_diag = normalize(Space{2, Float64}(1.0, 1.0))
            shock = analytic_closure(eq2, Riemann(State{1}(2.0), State{1}(0.0), Space{2}(0.0, 0.0), n_diag), geom2)
            # planar shock speed sqrt(2) along the normal
            @test shock(SVector(0.5, 0.5, 1.0))[1] ≈ 2.0
            @test shock(SVector(2.0, 2.0, 1.0))[1] ≈ 0.0

            rare = analytic_closure(eq2, Riemann(State{1}(0.0), State{1}(2.0), Space{2}(0.0, 0.0), Space{2}(1.0, 0.0)), geom2)
            @test rare(SVector(-1.0, 5.0, 1.0))[1] ≈ 0.0
            @test rare(SVector(1.0, -3.0, 1.0))[1] ≈ 1.0
            @test rare(SVector(3.0, 0.0, 1.0))[1] ≈ 2.0
        end

        @testset "Riemann closures are invariant under (uL, uR, n) → (uR, uL, −n)" begin
            # Both parametrisations describe the same initial data …
            a1 = Riemann(State{1}(2.0), State{1}(0.0), Space{1}(0.0), Space{1}(1.0))
            b1 = Riemann(State{1}(0.0), State{1}(2.0), Space{1}(0.0), Space{1}(-1.0))
            @test all(x -> a1(Space{1}(x)) == b1(Space{1}(x)), (-1.0, 0.5, 1.0))

            nd = normalize(Space{2, Float64}(1.0, 1.0))
            a2 = Riemann(State{1}(2.0), State{1}(0.0), Space{2}(0.0, 0.0), nd)
            b2 = Riemann(State{1}(0.0), State{1}(2.0), Space{2}(0.0, 0.0), -nd)
            pts = (Space{2}(-1.0, -0.5), Space{2}(0.5, 0.5), Space{2}(1.0, 0.5), Space{2}(2.0, 2.0))
            @test all(p -> a2(p) == b2(p), pts)

            # … so the exact solutions must agree as well (shock for a, mirrored shock for b)
            ea1, eb1 = analytic_closure(eq, a1, geom), analytic_closure(eq, b1, geom)
            @test all(x -> ea1(SVector(x, 1.0)) ≈ eb1(SVector(x, 1.0)), (-1.0, 0.5, 1.5, 3.0))
            ea2, eb2 = analytic_closure(eq2, a2, geom2), analytic_closure(eq2, b2, geom2)
            @test all(p -> ea2(SVector(p..., 1.0)) ≈ eb2(SVector(p..., 1.0)), pts)

            # Same for a rarefaction, which exercises the fan branch
            r1 = analytic_closure(eq, Riemann(State{1}(0.0), State{1}(2.0), Space{1}(0.0), Space{1}(1.0)), geom)
            r1m = analytic_closure(eq, Riemann(State{1}(2.0), State{1}(0.0), Space{1}(0.0), Space{1}(-1.0)), geom)
            @test all(x -> r1(SVector(x, 1.0)) ≈ r1m(SVector(x, 1.0)), (-1.0, 0.5, 1.0, 1.5, 3.0))
        end

        @testset "Riemann closure: unnormalized normals and D = 3" begin
            nd = normalize(Space{2, Float64}(1.0, 1.0))
            e_unit = analytic_closure(eq2, Riemann(State{1}(2.0), State{1}(0.0), Space{2}(0.0, 0.0), nd), geom2)
            e_long = analytic_closure(eq2, Riemann(State{1}(2.0), State{1}(0.0), Space{2}(0.0, 0.0), Space{2}(3.0, 3.0)), geom2)
            pts = (Space{2}(0.5, 0.5), Space{2}(0.7, 0.8), Space{2}(2.0, 2.0))
            @test all(p -> e_unit(SVector(p..., 1.0)) ≈ e_long(SVector(p..., 1.0)), pts)

            eq3 = BurgersEquation{3, Float64}()
            geom3 = test_domain((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), false)
            ex3 = analytic_closure(eq3, Riemann(State{1}(2.0), State{1}(0.0), Space{3}(0.0, 0.0, 0.0), Space{3}(1.0, 0.0, 0.0)), geom3)
            @test ex3(SVector(0.5, 2.0, -3.0, 1.0))[1] ≈ 2.0   # behind the shock at x = 1
            @test ex3(SVector(1.5, -1.0, 4.0, 1.0))[1] ≈ 0.0   # ahead of it
        end
    end

    @testset "Euler: Sod shock tube" begin
        eq = EulerEquation(Val(1), Float64, 1.4, Conservative())
        SOD_LEFT  = State{3, Float64}(1.0, 0.0, 2.5)
        SOD_RIGHT = State{3, Float64}(0.125, 0.0, 0.25)
        exact_sod = analytic_closure(eq, Riemann(SOD_LEFT, SOD_RIGHT, Space{1}(0.0), Space{1}(1.0)),
                                     test_domain((-5.0,), (5.0,), false))
        prim(x, t = 0.2) = HC.cons2prim(eq, exact_sod(SVector(x, t)))

        @testset "Initial states" begin
            @test exact_sod(SVector(-1.0, 0.0)) ≈ SOD_LEFT
            @test exact_sod(SVector(1.0, 0.0)) ≈ SOD_RIGHT
        end

        @testset "Star-region values (Toro, Test 1)" begin
            # At t = 0.2: rarefaction [-0.237, -0.014], contact 0.185, shock 0.350
            VL = prim(0.1)
            @test VL[1] ≈ 0.42632 rtol = 1e-3
            @test VL[2] ≈ 0.92745 rtol = 1e-3
            @test VL[3] ≈ 0.30313 rtol = 1e-3
            VR = prim(0.25)
            @test VR[1] ≈ 0.26557 rtol = 1e-3
            @test VR[2] ≈ 0.92745 rtol = 1e-3
            @test VR[3] ≈ 0.30313 rtol = 1e-3
            @test prim(-0.3) ≈ [1.0, 0.0, 1.0]
            @test prim(0.4)  ≈ [0.125, 0.0, 0.1]

            # Density decreases monotonically through the rarefaction fan
            ρ_fan = [prim(x)[1] for x in range(-0.23, -0.02; length = 20)]
            @test issorted(ρ_fan; rev = true)
            @test all(ρ -> 0.42632 < ρ < 1.0, ρ_fan)
        end

        @testset "Mirrored setup (states swapped, n = -1) gives the same solution" begin
            # Moving left state (Toro's modified Sod: u_L = 0.75) so the momentum sign matters
            UL = State{3, Float64}(1.0, 0.75, 2.5 + 0.5 * 0.75^2)
            geom = test_domain((-5.0,), (5.0,), false)
            ea = analytic_closure(eq, Riemann(UL, SOD_RIGHT, Space{1}(0.0), Space{1}(1.0)), geom)
            eb = analytic_closure(eq, Riemann(SOD_RIGHT, UL, Space{1}(0.0), Space{1}(-1.0)), geom)
            xs = range(-0.45, 0.45; length = 37)
            @test all(x -> ea(SVector(x, 0.0)) ≈ eb(SVector(x, 0.0)), (-0.3, 0.3))   # same data
            @test all(x -> isapprox(ea(SVector(x, 0.2)), eb(SVector(x, 0.2)); rtol = 1e-8), xs)
        end

        @testset "Conservation over [-1, 1] at t = 0.2" begin
            U(x) = exact_sod(SVector(x, 0.2))
            @test integrate(x -> U(x)[1], -1.0, 1.0) ≈ 1.125 atol = 1e-3   # mass
            @test integrate(x -> U(x)[2], -1.0, 1.0) ≈ 0.18  atol = 1e-3   # momentum gains (p_L − p_R)·t
            @test integrate(x -> U(x)[3], -1.0, 1.0) ≈ 2.75  atol = 1e-3   # energy
        end
    end

    @testset "Euler: vacuum generation (Toro, Sec. 4.6)" begin
        eq = EulerEquation(Val(1), Float64, 1.4, Conservative())
        geom = test_domain((-1.0,), (1.0,), false)
        cons(ρ, u, p) = State{3, Float64}(ρ, ρ * u, p / 0.4 + 0.5 * ρ * u^2)
        t = 0.1   # the fan heads (-0.72/0.72 and -0.62/0.91) are still inside [-1, 1]

        # Symmetric: u = ∓6 with c = √1.4, 2 (c_L + c_R) / (γ - 1) = 11.8 < 12, so a vacuum opens at x = 0
        sym = analytic_closure(eq, Riemann(cons(1.0, -6.0, 1.0), cons(1.0, 6.0, 1.0), Space{1}(0.0), Space{1}(1.0)), geom)
        U(x) = sym(SVector(x, t))
        @test U(0.0) == zeros(3)
        @test all(x -> U(x) ≈ U(-x) .* (1, -1, 1), (0.1, 0.3, 0.5, 0.7, 0.9))
        @test all(x -> U(x)[1] > 0, (-0.5, 0.5))                                  # inside the fans
        @test U(0.9) ≈ cons(1.0, 6.0, 1.0)
        # Conservation on [-1, 1]: the boundary fluxes are ρu = ±6, ρu² + p = 37, u (E + p) = ±129
        @test integrate(x -> U(x)[1], -1.0, 1.0) ≈ 2.0 - 12t atol = 1e-4
        @test integrate(x -> U(x)[2], -1.0, 1.0) ≈ 0.0 atol = 1e-10
        @test integrate(x -> U(x)[3], -1.0, 1.0) ≈ 41.0 - 2 * 6 * 21.5 * t atol = 1e-3

        # Asymmetric data: mass leaves through both ends at 5 and 4, i.e. M(t) = 1.5 - 9 t; mirroring agrees
        UL, UR = cons(1.0, -5.0, 1.0), cons(0.5, 8.0, 0.4)
        asym = analytic_closure(eq, Riemann(UL, UR, Space{1}(0.0), Space{1}(1.0)), geom)
        @test integrate(x -> asym(SVector(x, t))[1], -1.0, 1.0) ≈ 1.5 - 9t atol = 1e-4
        mirr = analytic_closure(eq, Riemann(UR, UL, Space{1}(0.0), Space{1}(-1.0)), geom)
        @test all(x -> isapprox(asym(SVector(x, t)), mirr(SVector(x, t)); atol = 1e-12), range(-0.95, 0.95; length = 39))
    end

    @testset "Euler: exact Riemann solver vs Toro's Tests 1-5 ($T), also at small pressure scales" for T in (Float64, Float32)
        # Toro, Riemann Solvers, Table 4.2 (γ = 1.4): (ρL, uL, pL, ρR, uR, pR) -> (p*, u*). Scaled copies (p × 1e-8,
        # u × 1e-4: the same solution in other units) test the relative tolerances; the absolute tolerance and floor
        # 1e-9 gave p* = 1e-9 instead of 1.9e-11 for Test 2 there
        eq = EulerEquation(Val(1), T, T(1.4), Conservative())
        geom = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-1.0,), :maxs => (1.0,), :periodic => false),
                            Dict{Symbol, Any}(:T => T, :D => 1))
        cons(ρ, u, p) = State{3, T}(ρ, ρ * u, p / 0.4 + 0.5 * ρ * u^2)
        pres(U) = T(0.4) * (U[3] - U[2]^2 / (2U[1]))
        toro = [((1.0, 0.0, 1.0, 0.125, 0.0, 0.1), (0.30313, 0.92745)), ((1.0, -2.0, 0.4, 1.0, 2.0, 0.4), (0.00189, 0.0)),
                ((1.0, 0.0, 1000.0, 1.0, 0.0, 0.01), (460.894, 19.5975)), ((1.0, 0.0, 0.01, 1.0, 0.0, 100.0), (46.0950, -6.19633)),
                ((5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950), (1691.64, 8.68975))]
        for (ps_scale, u_scale) in ((1.0, 1.0), (1e-8, 1e-4)), ((ρL, uL, pL, ρR, uR, pR), (ps, us)) in toro
            ex = analytic_closure(eq, Riemann(cons(ρL, u_scale * uL, ps_scale * pL), cons(ρR, u_scale * uR, ps_scale * pR),
                                              Space{1, T}(0), Space{1, T}(1)), geom)
            t = T(0.01)
            for δ in (-1e-4, 1e-4)        # just left and right of the contact x/t = u*: pressure p*, velocity u*
                U = ex(SVector{2, T}(T(u_scale * (us + δ) * t), t))
                @test isapprox(pres(U), ps_scale * ps; rtol = ps < 0.01 ? 3e-3 : 2e-4)   # Toro gives 3 digits for Test 2
                @test isapprox(U[2] / U[1], u_scale * us; atol = 2e-4 * u_scale * max(1, abs(us)))
            end
        end
    end

    @testset "Missing closure warns" begin
        eq2 = EulerEquation(Val(2), Float64, 1.4, Conservative())
        ic  = Gauss(State{4, Float64}(1.0, 0.0, 0.0, 2.5), Space{2, Float64}(0.0, 0.0), 1.0)
        @test_logs (:warn, r"No Analytical Solution") analytic_closure(eq2, ic, test_domain((-1.0, -1.0), (1.0, 1.0), true))
    end
end

# -------------------------------------------------------------------------
# 8. FULL SIMULATIONS
# -------------------------------------------------------------------------
