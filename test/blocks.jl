# Group `sims-bgk`: the BGK model on HyperCloudCore's block pipeline (BGKBlocks + BGKBlockRelaxation) against the
# SVector pipeline (BGKEquation + BGKRelaxation) with the same scheme and tableau, through the Core API.

"Periodic 1-D grid on [0, 1) with `M` components per particle."
function bgk_block_grid(N, M; range = 2.5, periodic = true)
    dx   = 1.0 / N
    geom = periodic ? get_rectangular_domain(Float64, (0.0,), (1.0,); is_periodic = true) :
                      get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC()))
    return ParticleGrid(geom, (dx,), ExponentialWeightFunction(1.0, range * dx), M)
end

"Macroscopic state (ρ, ρu, E) of the smooth wave / the Sod data at `x`."
wave_state(x) = SVector(1.0 + 0.2 * sinpi(2x), 0.1 + 0.1 * sinpi(2x), 1.2 + 0.1 * sinpi(2x))
sod_state(x)  = x < 0.5 ? SVector(1.0, 0.0, 2.5) : SVector(0.125, 0.0, 0.25)

"Both pipelines from the same initial data to `tend`; returns the largest difference of the moments and both grids."
function bgk_block_runs(eq, W, state, tableau; N = 60, tend = 0.1, cfl = 0.3, periodic = true, limiter = MinmodLimiter(:hard),
                        explicit = false)
    M  = HC.get_M(eq)
    bk = BGKBlocks(eq, Val(W))
    # the SVector pipeline
    pg_s = bgk_block_grid(N, M; periodic = periodic)
    for i in 1:pg_s.meta.N
        f, ok = equilibrium(eq, state(pg_s.core.positions[i][1]))
        ok || error("initial state not admissible")
        pg_s.rhos[i] = f
    end
    sources_s = HC._is_collisionless(eq.relaxation) ? () : (explicit ? (ExplicitBGKRelaxation(eq),) : (BGKRelaxation(eq),))
    ts_s = GeneralRKTimeStepper(eq, MUSCL(Float64, 1, M, 2, limiter, RusanovFlux()), MOOD(NoMOOD()), sources_s, tableau)
    # the block pipeline
    pg_b = bgk_block_grid(N, W; periodic = periodic)
    bs   = BlockState(bk, pg_b)
    set_block_states!((x, b) -> equilibrium_block(bk, state(x[1]), b), bs, pg_b)
    sources_b = HC._is_collisionless(eq.relaxation) ? () :
                (explicit ? (BlockSourceTerm(ExplicitBGKBlockRelaxation(bk)),) : (BlockSourceTerm(BGKBlockRelaxation(bk)),))
    ts_b = BlockTimeStepper(bk, MUSCL(Float64, 1, W, 2, limiter, RusanovFlux()), sources_b, tableau)
    # the same initial data
    dev0 = maximum(maximum(abs.(block_view(bs, b)[i] - pg_s.rhos[i][((b - 1) * W + 1):(b * W)])) for b in 1:bs.nb, i in 1:pg_s.meta.N)
    dt = cfl * (1 / N) / eq.max_speed[1]
    t  = 0.0
    while t < tend - 1e-12
        d = min(dt, tend - t)
        ts_s(eq, pg_s, t, d)
        ts_b(bk, pg_b, bs, t, d)
        t += d
    end
    dev = maximum(maximum(abs.(block_moments(bk, bs, i) - moments(eq, pg_s.rhos[i]))) for i in 1:pg_s.meta.N)
    return dev0, dev, pg_s, pg_b, bs
end

bk_of(bs, eq) = BGKBlocks(eq, Val(length(block_view(bs, 1)[1])))

@testset "BGK on the block pipeline" begin
    nodes, w = velocity_grid(:uniform, 16, 6.0)

    @testset "Construction and the block API" begin
        eq = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(1e-2))       # M = 32 (f and h)
        bk = BGKBlocks(eq, Val(8))
        @test bk isa HyperbolicPDE{1, 8, Float64} && HyperCloudCore.num_blocks(bk) == 4 && HC.num_velocities(bk) == 16
        @test_throws ArgumentError BGKBlocks(eq, Val(5))
        U = SVector(1.0, 0.3, 2.0)
        f, _ = equilibrium(eq, U)
        for b in 1:4
            fb = State{8, Float64}(f[(8b - 7):(8b)])
            @test equilibrium_block(bk, U, b) == fb                                     # the same components
            @test HyperCloudCore.block_flux(bk, fb, b)[1] == State{8, Float64}(HC.flux(eq, f)[1][(8b - 7):(8b)])
        end
        @test HyperCloudCore.block_max_eigenvalue(bk, State{8, Float64}(f[1:8]), 1, 2) == eq.max_speed[1]
        α, ok = equilibrium_parameters(eq, U)
        @test ok && all(k -> equilibrium_component(eq, α, k) == f[k], 1:32)
        @test !equilibrium_parameters(eq, SVector(-1.0, 0.0, 1.0))[2]
        @test_throws ErrorException equilibrium_block(bk, SVector(-1.0, 0.0, 1.0), 1)
        # moments from a block state
        pg = bgk_block_grid(4, 8)
        bs = BlockState(bk, pg)
        set_block_states!((x, b) -> equilibrium_block(bk, U, b), bs, pg)
        @test block_moments(bk, bs, 2) ≈ U atol = 1e-13
        @test block_moments(bk, bs, 2) == moments(eq, f)                              # the same summation order
        # the per-velocity upwind flux of a block, also through UpwindFlux and AlignedRusanovFlux
        fL = Flux{1, 8, Float64}((State{8, Float64}(ntuple(j -> 1 + 0.1j, 8)),))
        fR = Flux{1, 8, Float64}((State{8, Float64}(ntuple(j -> 2 - 0.05j, 8)),))
        for b in (1, 3)
            w_b = BlockOf(bk, b)
            FL = Flux{1, 8, Float64}((HC.flux(w_b, fL[1])[1],)); FR = Flux{1, 8, Float64}((HC.flux(w_b, fR[1])[1],))
            exact = State{8, Float64}(ntuple(8) do j
                v = velocity_node(eq, (b - 1) * 8 + j)[1][1]
                max(v, 0) * fL[1][j] + min(v, 0) * fR[1][j]
            end)
            for F in (RusanovFlux(), UpwindFlux(), AlignedRusanovFlux())
                @test F(fL, fR, FL, FR, w_b)[1] ≈ exact
            end
        end
        # the reduction and the solve are kernel functions: no allocation, inferred
        st = BGKBlockRelaxation(bk)
        Y  = parent(bs.data)
        prm = HyperCloudCore.block_reduce(st, 2, Y, 4, 4, pg, 0.0, 0.01)
        @test prm.ok && prm.alpha ≈ α && prm.lambda ≈ 0.01 / 1e-2
        @test (@inferred HyperCloudCore.block_reduce(st, 2, Y, 4, 4, pg, 0.0, 0.01)) isa HyperCloudCore.reduction_type(st)
        alloc_reduce(st, Y, pg) = (HyperCloudCore.block_reduce(st, 2, Y, 4, 4, pg, 0.0, 0.01); @allocated HyperCloudCore.block_reduce(st, 2, Y, 4, 4, pg, 0.0, 0.01))
        alloc_solve(st, f, prm, pg) = (HyperCloudCore.block_implicit_solve(st, f, 2, prm, 0.01, 2, pg, 0.0); @allocated HyperCloudCore.block_implicit_solve(st, f, 2, prm, 0.01, 2, pg, 0.0))
        @test alloc_reduce(st, Y, pg) == 0
        @test alloc_solve(st, block_view(bs, 2)[2], prm, pg) == 0
        @test (@inferred HyperCloudCore.block_implicit_solve(st, block_view(bs, 2)[2], 2, prm, 0.01, 2, pg, 0.0)) isa State{8, Float64}
        # an equilibrium block is a fixed point of the solve and a zero of the source
        g = HyperCloudCore.block_implicit_solve(st, block_view(bs, 3)[2], 3, prm, 0.01, 2, pg, 0.0)
        @test g ≈ block_view(bs, 3)[2] atol = 1e-14
        @test HyperCloudCore.block_source(ExplicitBGKBlockRelaxation(bk), block_view(bs, 3)[2], 3, prm, 2, pg, 0.0) ≈ zero(State{8, Float64}) atol = 1e-12
    end

    @testset "Collisionless transport: blocks reproduce the SVector run (γ = 3, M = Nv = 16, W = 4)" begin
        eq = BGKEquation(Val(1), nodes, w, 3.0, ConstantRelaxationTime(Inf))
        dev0, dev, pg_s, _, bs = bgk_block_runs(eq, 4, wave_state, RK3_SSP_Tableau(Float64))
        @test dev0 == 0.0
        @test dev < 1e-13
    end

    @testset "Collisions (IMEX ARS222): the smooth wave and the Sod tube, W = 8 of M = 32, τ = 1e-2 and 1e-5" begin
        for (τ, state, periodic, tend) in ((1e-2, wave_state, true, 0.1), (1e-5, sod_state, false, 0.05))
            eq = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(τ))
            dev0, dev, pg_s, pg_b, bs = bgk_block_runs(eq, 8, state, IMEX_ARS222_Tableau(Float64); periodic = periodic, tend = tend,
                                                       limiter = VenkatakrishnanLimiter(:hard))
            @test dev0 == 0.0
            @test dev < 1e-11
            # physical and conserved (periodic): the moments stay positive, mass momentum energy of the lattice sum kept
            ρθ(U) = (U[1], 2 * (U[3] - 0.5 * U[2]^2 / U[1]) / (5 * U[1]))
            @test all(i -> all(>(0), ρθ(block_moments(bk_of(bs, eq), bs, i))), 1:pg_b.meta.N)
        end
    end

    @testset "Explicit collision (RK2, τ = 0.05) and τ = 0 (projection) agree with the SVector run" begin
        eq = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(0.05))
        dev0, dev, _, _, _ = bgk_block_runs(eq, 8, wave_state, RK2_SSP_Tableau(Float64); explicit = true)
        @test dev0 == 0.0 && dev < 1e-13
        eq0 = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(0.0))
        dev0, dev, _, _, _ = bgk_block_runs(eq0, 8, wave_state, IMEX_ARS222_Tableau(Float64); tend = 0.05)
        @test dev0 == 0.0 && dev < 1e-12
        # conservation of the periodic lattice sums with collisions (W = 8, τ = 1e-2)
        eqc = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(1e-2))
        _, _, _, pg_b, bs = bgk_block_runs(eqc, 8, wave_state, IMEX_ARS222_Tableau(Float64); tend = 0.1)
        bk = BGKBlocks(eqc, Val(8))
        tot = sum(block_moments(bk, bs, i) for i in 1:pg_b.meta.N)
        tot0 = sum(wave_state(pg_b.core.positions[i][1]) for i in 1:pg_b.meta.N)
        @test tot ≈ tot0 rtol = 1e-12
    end

    @testset "Configuration pipeline: :PDE_block_width runs the block form, the moments agree with the standard run" begin
        p = bgk_sod_params(120)                                            # ARS222, VK, τ = 1e-5, Dirichlet / outflow
        sim_s = run_bgk_simulation(p)
        sim_b = run_bgk_simulation(merge(p, Dict{Symbol, Any}(:PDE_block_width => 8)))
        @test sim_s !== nothing && sim_b !== nothing
        if sim_s !== nothing && sim_b !== nothing
            @test sim_b.x[end] == sim_s.x[end] && sim_b.t[end] == sim_s.t[end]
            @test length(sim_b.u[end][1]) == 3                               # the moments
            @test maximum(maximum(abs.(ub - us)) for (ub, us) in zip(sim_b.u[end], sim_s.u[end])) < 1e-9
            @test sim_b.metadata[:finite] && !sim_b.metadata[:stopped_early]
        end
        # collisionless wave, RK3, forwarded by the kinetic runner
        pw = merge(bgk_wave_params(50), Dict{Symbol, Any}(:sim_func_name => :run_kinetic_simulation, :PDE_block_width => 4))
        sim_w = run_sim(pw)
        sim_w0 = run_sim(bgk_wave_params(50))
        @test sim_w !== nothing && sim_w0 !== nothing
        sim_w !== nothing && sim_w0 !== nothing &&
            @test maximum(maximum(abs.(ub - us)) for (ub, us) in zip(sim_w.u[end], sim_w0.u[end])) < 1e-12
        # refusals (the runner logs the error and returns nothing)
        for bad in (Dict{Symbol, Any}(:PDE_block_width => 5),                         # 5 does not divide 32
                    Dict{Symbol, Any}(:PDE_block_width => 8, :MOOD_criterion => :u2, :MOOD_strategy => :EPD1),
                    Dict{Symbol, Any}(:PDE_block_width => 8, :save_relax => true),
                    Dict{Symbol, Any}(:PDE_block_width => 8, :Grid_mover => :lagrangian))
            @test (@test_logs (:info, r"") (:error, r"Error during BGK") match_mode = :any run_bgk_simulation(merge(p, bad))) === nothing
        end
    end

    @testset "Mixture on blocks: the block API against the single state, the configuration pipeline" begin
        lam = [19.96 1.153; 1.153 17.52]
        mix = BGKMixture(Val(1), (1.0, 10.0), lam, (velocity_grid(:uniform, 16, 8.0), velocity_grid(:uniform, 12, 4.0)); epsilon = 1e-2, kappa = 2e-2)
        @test_throws ArgumentError MixtureBlocks(mix, Val(16))                                  # 16 does not divide 24
        mb = MixtureBlocks(mix, Val(8))                                                           # 4 + 3 blocks
        @test mb isa HyperbolicPDE{1, 8, Float64} && HyperCloudCore.num_blocks(mb) == 7 && HC.num_species(mb) == 2
        @test [species_of_block(mb, b) for b in 1:7] == [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3)]
        U0 = mixture_state(mix, (1.0, 0.3), (0.5, -0.2), (1.0, 2.0))
        f  = HC._cat(ntuple(s -> first(equilibrium(mix.species[s], SVector{3}(U0[(3s - 2):(3s)]))), 2))
        f  = f .* (1 .+ 0.05 .* sin.(1:56))                                                        # out of equilibrium
        f  = State{56, Float64}(f)
        slice(b) = State{8, Float64}(f[(8b - 7):(8b)])
        for b in 1:7
            @test HyperCloudCore.block_flux(mb, slice(b), b)[1] == State{8, Float64}(HC.flux(mix, f)[1][(8b - 7):(8b)])
        end
        # the block state of f and its moments
        pg = bgk_block_grid(4, 8)
        bs = BlockState(mb, pg)
        set_block_states!((x, b) -> slice(b), bs, pg)
        @test block_moments(mb, bs, 3) == moments(mix, f)
        # the collision: the block solve of every block equals the slice of the single-state solve (same Newton, same order)
        st  = MixtureBlockRelaxation(mb)
        Y   = parent(bs.data)
        Δ   = 0.01
        prm = HyperCloudCore.block_reduce(st, 3, Y, 4, 7, pg, 0.0, Δ)
        @test prm.ok
        g_full = HC.implicit_solve(MixtureRelaxation(mix), f, Δ, 1, pg, 0.0)
        for b in 1:7
            g_b = HyperCloudCore.block_implicit_solve(st, slice(b), b, prm, Δ, 3, pg, 0.0)
            @test maximum(abs.(g_b - State{8, Float64}(g_full[(8b - 7):(8b)]))) < 1e-14
        end
        prm0 = HyperCloudCore.block_reduce(ExplicitMixtureBlockRelaxation(mb), 3, Y, 4, 7, pg, 0.0, 0.0)
        S_full = HC.evaluate_source(ExplicitMixtureRelaxation(mix), f, 1, pg, 0.0)
        for b in 1:7
            S_b = HyperCloudCore.block_source(ExplicitMixtureBlockRelaxation(mb), slice(b), b, prm0, 3, pg, 0.0)
            @test maximum(abs.(S_b - State{8, Float64}(S_full[(8b - 7):(8b)]))) < 1e-10 * maximum(abs.(S_full))
        end
        # kernel rules
        alloc_r(st, Y, pg) = (HyperCloudCore.block_reduce(st, 3, Y, 4, 7, pg, 0.0, 0.01); @allocated HyperCloudCore.block_reduce(st, 3, Y, 4, 7, pg, 0.0, 0.01))
        alloc_s(st, f, prm, pg) = (HyperCloudCore.block_implicit_solve(st, f, 5, prm, 0.01, 3, pg, 0.0); @allocated HyperCloudCore.block_implicit_solve(st, f, 5, prm, 0.01, 3, pg, 0.0))
        @test alloc_r(st, Y, pg) == 0
        @test alloc_s(st, slice(5), prm, pg) == 0
        @test (@inferred HyperCloudCore.block_reduce(st, 3, Y, 4, 7, pg, 0.0, 0.01)) isa HyperCloudCore.reduction_type(st)
        @test (@inferred HyperCloudCore.block_implicit_solve(st, slice(5), 5, prm, 0.01, 3, pg, 0.0)) isa State{8, Float64}
        # the configuration pipeline: the two-species wave, block form against the single state
        pm = mixture_wave_params(50; tmax = 0.1)
        sim_s = run_bgk_simulation(pm)
        sim_b = run_bgk_simulation(merge(pm, Dict{Symbol, Any}(:PDE_block_width => 8)))
        @test sim_s !== nothing && sim_b !== nothing
        if sim_s !== nothing && sim_b !== nothing
            @test length(sim_b.u[end][1]) == 6
            @test maximum(maximum(abs.(ub - us)) for (ub, us) in zip(sim_b.u[end], sim_s.u[end])) < 1e-10
            t0, t1 = sum(sim_b.u[1]), sum(sim_b.u[end])
            @test t1[1] ≈ t0[1] rtol = 1e-12
            @test t1[4] ≈ t0[4] rtol = 1e-12
            @test t1[3] + 10 * t1[6] ≈ t0[3] + 10 * t0[6] rtol = 1e-12
        end
        @test run_bgk_simulation(merge(pm, Dict{Symbol, Any}(:PDE_block_width => 16))) === nothing          # 16 ∤ 24
    end
end
