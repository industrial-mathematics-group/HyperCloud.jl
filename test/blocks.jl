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

# Calls left in the optimised code of `f(types...)`: the kernel functions of the blocks must inline their closures (one kept
# as a call captured the equation, and on a GPU the closure was built in local memory at every call)
noninlined_calls(f, types) = count(s -> s isa Expr && s.head === :invoke, first(Base.code_typed(f, types))[1].code)
function closure_calls(w::BlockOf{D, W, T}) where {D, W, T}
    U = State{W, T}(ntuple(i -> T(i), W)); F = HC.flux(w, U)
    a = HyperCloudCore.ALEFrame(w, zero(SVector{D, T}))
    return (noninlined_calls(HC.flux, (typeof(w), typeof(U))),
            noninlined_calls((x...) -> RusanovFlux()(x...), (typeof(F), typeof(F), typeof(F), typeof(F), typeof(w))),
            noninlined_calls((x...) -> RusanovFlux()(x...), (typeof(F), typeof(F), typeof(F), typeof(F), typeof(a))))
end

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
        @test closure_calls(BlockOf(bk, 2)) == (0, 0, 0)
        # one component per block: UpwindFlux is the per-velocity flux too (it took the scalar Murman–Roe path, which
        # needs the velocity, and failed), and the transport matrix is the node speed
        bk1 = BGKBlocks(eq, Val(1))
        w1 = BlockOf(bk1, 5)
        g = Flux{1, 1, Float64}((State{1, Float64}(1.0),))
        G = Flux{1, 1, Float64}((HC.flux(w1, g[1])[1],))
        @test UpwindFlux()(g, g, G, G, w1) == RusanovFlux()(g, g, G, G, w1)
        @test HyperCloudCore.velocity(w1, g[1], 1)[1, 1] == velocity_node(eq, 5)[1][1]
        # the reduction and the solve are kernel functions: no allocation, inferred
        st = BGKBlockRelaxation(bk)
        Y  = parent(bs.data)
        prm = HyperCloudCore.block_reduce(st, 2, Y, 4, 4, pg, 0.0, 0.01)
        @test prm.ok && prm.alpha ≈ α && prm.lambda ≈ 0.01 / 1e-2
        @test (@inferred HyperCloudCore.block_reduce(st, 2, Y, 4, 4, pg, 0.0, 0.01)) isa HyperCloudCore.reduction_type(st)
        @test HyperCloudCore.block_partial(st, 2, Y, 4, 4, 1:2, pg, 0.0) + HyperCloudCore.block_partial(st, 2, Y, 4, 4, 3:4, pg, 0.0) ≈ U atol = 1e-13
        # a partial reads its own blocks only (the f values in blocks 1:2, the h values in 3:4: a device owning some
        # blocks has only those)
        p34 = HyperCloudCore.block_partial(st, 2, Y, 4, 4, 3:4, pg, 0.0)
        @test p34[1] == 0 && p34[2] == 0 && p34[3] > 0
        Y2 = copy(Y)
        Y2[2 * 4 + 2] = 2 * Y2[2 * 4 + 2]                                              # block 3 of particle 2
        @test HyperCloudCore.block_partial(st, 2, Y2, 4, 4, 1:2, pg, 0.0) == HyperCloudCore.block_partial(st, 2, Y, 4, 4, 1:2, pg, 0.0)
        @test HyperCloudCore.block_partial(st, 2, Y2, 4, 4, 3:4, pg, 0.0)[3] > p34[3]
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
                    Dict{Symbol, Any}(:PDE_block_width => -8), Dict{Symbol, Any}(:PDE_block_width => 8.5))
            @test (@test_logs (:error, r"Error during BGK") match_mode = :any run_bgk_simulation(merge(p, bad))) === nothing
        end
        @test HC._block_width(Dict{Symbol, Any}()) == 0 && HC._block_width(Dict{Symbol, Any}(:block_width => 0)) == 0
        # the blocks per launch of the passes with scratch: any value gives the same run (bit for bit: the same arithmetic
        # per particle and block), a negative one is refused
        sim_1 = run_bgk_simulation(merge(p, Dict{Symbol, Any}(:PDE_block_width => 8, :PDE_blocks_per_launch => 1)))
        sim_0 = run_bgk_simulation(merge(p, Dict{Symbol, Any}(:PDE_block_width => 8, :PDE_blocks_per_launch => 0)))
        @test sim_1 !== nothing && sim_0 !== nothing && sim_b !== nothing
        if sim_1 !== nothing && sim_0 !== nothing && sim_b !== nothing
            @test sim_1.u[end] == sim_b.u[end] && sim_0.u[end] == sim_b.u[end]
        end
        @test (@test_logs (:error, r"Error during BGK") match_mode = :any run_bgk_simulation(merge(p, Dict{Symbol, Any}(:PDE_block_width => 8, :PDE_blocks_per_launch => -1)))) === nothing
        @test HC._blocks_per_launch(Dict{Symbol, Any}()) === :auto && HC._blocks_per_launch(Dict{Symbol, Any}(:blocks_per_launch => 0)) == 0
        # the blocks split over host parts (:PDE_devices on the Array storage): the single block run to round-off (the
        # collision's moments are summed in a different order), the same cloud
        for nd in (2, 3)
            sim_d = run_bgk_simulation(merge(p, Dict{Symbol, Any}(:PDE_block_width => 8, :PDE_devices => nd)))
            @test sim_d !== nothing && sim_b !== nothing
            if sim_d !== nothing && sim_b !== nothing
                @test maximum(maximum(abs.(ud - ub)) for (ud, ub) in zip(sim_d.u[end], sim_b.u[end])) < 1e-10
                @test sim_d.x[end] == sim_b.x[end] && sim_d.t[end] == sim_b.t[end]
            end
        end
        @test HC._block_devices(Dict{Symbol, Any}(:devices => 3)) == [0, 1, 2] && HC._block_devices(Dict{Symbol, Any}(:devices => [1, 3])) == [1, 3]
        @test HC._block_devices(Dict{Symbol, Any}()) === nothing && HC._device_ranges(7, 3) == [1:3, 4:5, 6:7]
        for bad in (true, [0, 0], [-1], Int[])
            @test_throws ArgumentError HC._block_devices(Dict{Symbol, Any}(:devices => bad))
        end
        # the keys of the block form without a block width are refused (they were ignored: one device, the standard form)
        for kv in (:PDE_devices => 2, :PDE_blocks_per_launch => 4)
            @test (@test_logs (:error, r"Error during BGK") match_mode = :any run_bgk_simulation(merge(p, Dict{Symbol, Any}(kv)))) === nothing
        end
        @test (@test_logs (:error, r"Error during BGK") match_mode = :any run_bgk_simulation(merge(p, Dict{Symbol, Any}(:PDE_block_width => 8, :PDE_devices => 0)))) === nothing
        # the additive forms of the single gas: the moments of two block ranges add up, the velocity and the admissibility
        # follow from the sum
        bkd = BGKBlocks(BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(1e-2)), Val(8))
        nbd = HyperCloudCore.num_blocks(bkd)
        Yd  = [equilibrium_block(bkd, SVector(1.2, 0.3, 2.0), b) for b in 1:nbd]            # one particle, N = 1
        pv  = HyperCloudCore.block_fluid_velocity_partial(bkd, Yd, 1, nbd, 1:2, 1) + HyperCloudCore.block_fluid_velocity_partial(bkd, Yd, 1, nbd, 3:nbd, 1)
        @test pv ≈ block_moments(bkd, Yd, 1, nbd, 1) atol = 1e-14
        @test HyperCloudCore.finalize_fluid_velocity(bkd, pv) ≈ HyperCloudCore.block_fluid_velocity(bkd, Yd, 1, nbd, 1)
        @test HyperCloudCore.finalize_admissible(bkd, pv) == HyperCloudCore.block_admissible(bkd, Yd, 1, nbd, 1) == true
        @test !HyperCloudCore.finalize_admissible(bkd, -pv)
        # a width on an equation without a block form is refused, not ignored
        p_lin = merge(linear_1d_gauss(40), UPWIND1, Dict{Symbol, Any}(:PDE_block_width => 4))
        @test (@test_logs (:error, r"Error during Direct") match_mode = :any run_sim(p_lin)) === nothing
        # the collision's parameter arrays follow the storage; a memory limit reserves the block stepper and the state
        bk8 = BGKBlocks(BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(1e-2)), Val(8))
        @test parent(HC._block_sources(bk8, IMEX_ARS222_Tableau(Float64), Array)[1].params) isa Vector
        try
            set_memory_limit!(64 * 2^20; tolerance = 0.1)
            sim_m = run_bgk_simulation(merge(p, Dict{Symbol, Any}(:PDE_block_width => 8)))
            @test sim_m !== nothing
            if sim_m !== nothing
                @test sim_m.stats[:reserved_particles][1] >= length(sim_m.x[end])
                @test sim_m.stats[:reserved_bytes][1] <= 0.9 * 64 * 2^20
            end
        finally
            set_memory_limit!(nothing)
        end
    end

    @testset "Mixture on blocks: the block API against the single state, the configuration pipeline" begin
        lam = [19.96 1.153; 1.153 17.52]
        mix = BGKMixture(Val(1), (1.0, 10.0), lam, (velocity_grid(:uniform, 16, 8.0), velocity_grid(:uniform, 12, 4.0)); epsilon = 1e-2, kappa = 2e-2)
        @test_throws ArgumentError MixtureBlocks(mix, Val(16))                                  # 16 does not divide 24
        mb = MixtureBlocks(mix, Val(8))                                                           # 4 + 3 blocks
        @test mb isa HyperbolicPDE{1, 8, Float64} && HyperCloudCore.num_blocks(mb) == 7 && HC.num_species(mb) == 2
        @test closure_calls(BlockOf(mb, 6)) == (0, 0, 0)
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
        # the additive form of the reduction: partials over block ranges add up to the species moments
        pa = HyperCloudCore.block_partial(st, 3, Y, 4, 7, 1:3, pg, 0.0)
        pb = HyperCloudCore.block_partial(st, 3, Y, 4, 7, 4:7, pg, 0.0)
        @test pa + pb ≈ moments(mix, f) atol = 1e-13
        # the additive forms of the velocity and the admissibility (a split state): the same sums, the same results
        pv = HyperCloudCore.block_fluid_velocity_partial(mb, Y, 4, 7, 1:3, 3) + HyperCloudCore.block_fluid_velocity_partial(mb, Y, 4, 7, 4:7, 3)
        @test pv ≈ pa + pb atol = 1e-13
        @test HyperCloudCore.finalize_fluid_velocity(mb, pv) ≈ HyperCloudCore.block_fluid_velocity(mb, Y, 4, 7, 3)
        @test HyperCloudCore.finalize_admissible(mb, pv) == HyperCloudCore.block_admissible(mb, Y, 4, 7, 3) == true
        @test HyperCloudCore.finalize_reduction(st, 3, pa + pb, pg, 0.0, Δ).ok
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
        # the mixture split over three host parts (ranges 1:3, 4:5, 6:7: the first species' blocks on two parts, the
        # second's too): the single block run to round-off
        sim_d = run_bgk_simulation(merge(pm, Dict{Symbol, Any}(:PDE_block_width => 8, :PDE_devices => 3)))
        @test sim_d !== nothing && sim_b !== nothing
        if sim_d !== nothing && sim_b !== nothing
            @test maximum(maximum(abs.(ud - ub)) for (ud, ub) in zip(sim_d.u[end], sim_b.u[end])) < 1e-10
        end
    end

    @testset "MOOD on the block form: U1 and PAD through the configuration pipeline, the cross-block admissibility" begin
        nodes, w = velocity_grid(:uniform, 16, 6.0)
        eq = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(1e-5))
        bk = BGKBlocks(eq, Val(8))
        pg = bgk_block_grid(4, 8)
        bs = BlockState(bk, pg)
        set_block_states!((x, b) -> equilibrium_block(bk, SVector(1.0, 0.3, 2.0), b), bs, pg)
        Y = parent(bs.data)
        @test HyperCloudCore.block_admissible(bk, Y, 4, 4, 2)
        Y[2] = -3 * Y[2]                                                              # block 1 of particle 2: negative mass
        @test !HyperCloudCore.block_admissible(bk, Y, 4, 4, 2) && HyperCloudCore.block_admissible(bk, Y, 4, 4, 3)
        lam = [19.96 1.153; 1.153 17.52]
        mix = BGKMixture(Val(1), (1.0, 10.0), lam, (velocity_grid(:uniform, 16, 8.0), velocity_grid(:uniform, 12, 4.0)); epsilon = 1e-2)
        mb  = MixtureBlocks(mix, Val(8))
        bsm = BlockState(mb, pg)
        mix_block(b) = begin
            s, bl = species_of_block(mb, b)
            HC._species_equilibrium_block(mb, first(equilibrium_parameters(mix.species[s], SVector(1.0, 0.0, 1.5))), s, bl)
        end
        set_block_states!((x, b) -> mix_block(b), bsm, pg)
        Ym = parent(bsm.data)
        @test HyperCloudCore.block_admissible(mb, Ym, 4, 7, 2)
        Ym[4 * 4 + 2] = -3 * Ym[4 * 4 + 2]                                            # block 5 (species 2, its first f block: the bulk of the mass) of particle 2
        @test !HyperCloudCore.block_admissible(mb, Ym, 4, 7, 2) && HyperCloudCore.block_admissible(mb, Ym, 4, 7, 1)
        # the Sod tube without a limiter needs MOOD: U1 and PAD, block form against the standard run
        p = merge(bgk_sod_params(120), Dict{Symbol, Any}(:Limiter_name => :none, :Scheme_order => 3))
        for crit in ((:MOOD_criterion => :U1, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-3), (:MOOD_criterion => :PAD, :MOOD_strategy => :EPD1))
            pm = merge(p, Dict{Symbol, Any}(crit...))
            sim_s = run_bgk_simulation(pm)
            sim_b = run_bgk_simulation(merge(pm, Dict{Symbol, Any}(:PDE_block_width => 8)))
            @test sim_s !== nothing && sim_b !== nothing
            if sim_s !== nothing && sim_b !== nothing
                @test maximum(maximum(abs.(ub - us)) for (ub, us) in zip(sim_b.u[end], sim_s.u[end])) < 1e-8
                @test all(u -> u[1] > 0 && 0.4 * (u[3] - 0.5 * u[2]^2 / u[1]) > 0, sim_b.u[end])
            end
        end
        # U2 on blocks: the block's gradients are regathered for its curvature test; the standard run to round-off
        pu = merge(p, Dict{Symbol, Any}(:MOOD_criterion => :U2, :MOOD_strategy => :EPD1, :MOOD_delta_relax => 1e-3))
        sim_s = run_bgk_simulation(pu)
        sim_b = run_bgk_simulation(merge(pu, Dict{Symbol, Any}(:PDE_block_width => 8)))
        @test sim_s !== nothing && sim_b !== nothing
        if sim_s !== nothing && sim_b !== nothing
            @test maximum(maximum(abs.(ub - us)) for (ub, us) in zip(sim_b.u[end], sim_s.u[end])) < 1e-8
        end
    end

    @testset "Moving grids on the block form: a prescribed mover as the standard run, the Lagrangian Sod in the ALE frame" begin
        p = merge(bgk_wave_params(50; tau = 1e-2, stepper = :ARS222, order = 2),
                  Dict{Symbol, Any}(:Grid_mover => :sinusoidal, :Grid_mover_amplitude => 0.2, :tmax => 0.1))
        sim_s = run_sim(p)
        sim_b = run_sim(merge(p, Dict{Symbol, Any}(:PDE_block_width => 4)))
        @test sim_s !== nothing && sim_b !== nothing
        if sim_s !== nothing && sim_b !== nothing
            @test sim_b.x[end] == sim_s.x[end]                                     # the particles moved the same way
            @test any(x -> x != sim_s.x[1][1], sim_s.x[end][1:1]) || sim_s.x[end] != sim_s.x[1]
            @test maximum(maximum(abs.(ub - us)) for (ub, us) in zip(sim_b.u[end], sim_s.u[end])) < 1e-10
        end
        # the Lagrangian Sod: the block form runs in the ALE frame with the mean velocity (the frozen frame), positive and
        # close to the standard run with the conservative Lagrangian frame
        pl = merge(bgk_sod_params(200), Dict{Symbol, Any}(:Grid_mover => :lagrangian, :Grid_volumes => :mls, :Grid_volumes_degree => 2,
                                                         :Grid_randomness_factor => (0.2,), :snapshots => 2))
        sim_s = run_bgk_simulation(pl)
        sim_b = run_bgk_simulation(merge(pl, Dict{Symbol, Any}(:PDE_block_width => 8)))
        @test sim_s !== nothing && sim_b !== nothing
        if sim_s !== nothing && sim_b !== nothing
            @test sim_b.metadata[:finite] && !sim_b.metadata[:stopped_early]
            @test all(u -> u[1] > 0 && 0.4 * (u[3] - 0.5 * u[2]^2 / u[1]) > 0, sim_b.u[end])
            @test sim_b.x[end] != sim_b.x[1]                                        # the cloud moved
            xs = [x[1] for x in sim_b.x[end]]; pm = sortperm(xs); xs = xs[pm]; ρ = [u[1] for u in sim_b.u[end]][pm]
            hs = [i == 1 ? xs[2] - xs[1] : i == length(xs) ? xs[end] - xs[end - 1] : (xs[i + 1] - xs[i - 1]) / 2 for i in eachindex(xs)]
            x0 = [x[1] for x in sim_b.x[1]]; pm0 = sortperm(x0); x0 = x0[pm0]; ρ0 = [u[1] for u in sim_b.u[1]][pm0]
            h0 = [i == 1 ? x0[2] - x0[1] : i == length(x0) ? x0[end] - x0[end - 1] : (x0[i + 1] - x0[i - 1]) / 2 for i in eachindex(x0)]
            @test abs(sum(hs .* ρ) - sum(h0 .* ρ0)) < 1e-2                         # the frozen frame's gain (standard: 1.3e-4 with the conservative frame)
            # against the standard Lagrangian run, read at the block run's positions (both clouds move with the fluid)
            xs_s = [x[1] for x in sim_s.x[end]]; ps = sortperm(xs_s); xs_s = xs_s[ps]; ρ_s = [u[1] for u in sim_s.u[end]][ps]
            interp(x) = (k = searchsortedfirst(xs_s, x); k <= 1 ? ρ_s[1] : k > length(xs_s) ? ρ_s[end] : ρ_s[k - 1] + (ρ_s[k] - ρ_s[k - 1]) * (x - xs_s[k - 1]) / (xs_s[k] - xs_s[k - 1]))
            @test sum(abs(ρ[i] - interp(xs[i])) for i in eachindex(xs)) / length(xs) < 0.02
        end
    end

    @testset "Adaptivity on the block form: the Lagrangian Sod with particle management, every block managed alike" begin
        pl = merge(bgk_sod_params(200), Dict{Symbol, Any}(:Grid_mover => :lagrangian, :Grid_volumes => :mls, :Grid_volumes_degree => 2,
                                                         :Grid_randomness_factor => (0.2,), :snapshots => 2, :Adapt_enabled => true,
                                                         :PDE_block_width => 8))
        sim = run_bgk_simulation(pl)
        @test sim !== nothing
        if sim !== nothing
            @test sim.metadata[:finite] && !sim.metadata[:stopped_early]
            @test all(u -> u[1] > 0 && 0.4 * (u[3] - 0.5 * u[2]^2 / u[1]) > 0, sim.u[end])
            xs = sort([x[1] for x in sim.x[end]])
            hs = diff(xs)
            @test 0.3 * (1 / 200) < minimum(hs) && maximum(hs) < 2.2 * (1 / 200)                     # the spacing rules held
        end
        # without the adaptivity the Lagrangian cloud violates the spacing rules (compressed at the shock, stretched in the fan)
        sim0 = run_bgk_simulation(merge(pl, Dict{Symbol, Any}(:Adapt_enabled => false)))
        @test sim0 !== nothing
        if sim0 !== nothing
            hs0 = diff(sort([x[1] for x in sim0.x[end]]))
            @test minimum(hs0) < 0.5 * (1 / 200) || maximum(hs0) > 1.6 * (1 / 200)
            @test length(sim0.x[end]) == length(sim0.x[1])
        end
    end

    @testset "Specular reflection on blocks off the node grid: oblique and moving walls as the single state, also for the mixture" begin
        nodes, w = velocity_grid(:uniform, 12, 6.0)
        eq = BGKEquation(Val(2), nodes, w, 2.0, ConstantRelaxationTime(1e-2))            # 2-D, M = 144 (no h)
        bk = BGKBlocks(eq, Val(16))                                                      # 9 blocks
        f, _ = equilibrium(eq, SVector(1.2, 0.3, -0.2, 2.4))
        f = State{144, Float64}(f .* (1 .+ 0.05 .* sin.(1:144)))
        Y = [State{16, Float64}(f[(16b - 15):(16b)]) for b in 1:9]
        for (n, wv) in ((SVector(1.0, 0.0), SVector(0.0, 0.0)), (SVector(1.0, 0.0), SVector(-0.5, 0.0)),
                        (SVector(0.6, 0.8), SVector(0.0, 0.0)), (SVector(0.6, 0.8), SVector(0.1, -0.2)))
            wall = WallPoint(SVector(0.0, 0.0), n, 0.02, wv)
            g_std = SlipWall()(eq, f, wall)
            prm = HyperCloudCore.block_wall_reduce(SlipWall(), bk, Y, 1, 9, 1, wall)
            g_blk = vcat((HyperCloudCore.block_wall_apply(SlipWall(), bk, Y[b], b, prm, wall, Y, 1, 9, 1) for b in 1:9)...)
            @test maximum(abs.(g_std - g_blk)) < 1e-13
            @test prm.flip == (n == SVector(1.0, 0.0) && wv[1] == 0.0)
            d_std = DiffuseWall(1.0; accommodation = 0.5)(eq, f, wall)
            prm_d = HyperCloudCore.block_wall_reduce(DiffuseWall(1.0; accommodation = 0.5), bk, Y, 1, 9, 1, wall)
            d_blk = vcat((HyperCloudCore.block_wall_apply(DiffuseWall(1.0; accommodation = 0.5), bk, Y[b], b, prm_d, wall, Y, 1, 9, 1) for b in 1:9)...)
            @test maximum(abs.(d_std - d_blk)) < 1e-13
        end
        # the mixture at an oblique moving wall
        lam = [19.96 1.153; 1.153 17.52]
        mix = BGKMixture(Val(1), (1.0, 10.0), lam, (velocity_grid(:uniform, 16, 8.0), velocity_grid(:uniform, 12, 4.0)); epsilon = 1e-2)
        mb  = MixtureBlocks(mix, Val(8))
        U0  = mixture_state(mix, (1.0, 0.3), (0.5, -0.2), (1.0, 2.0))
        fm  = HC._cat(ntuple(s -> first(equilibrium(mix.species[s], SVector{3}(U0[(3s - 2):(3s)]))), 2))
        fm  = State{56, Float64}(fm .* (1 .+ 0.05 .* sin.(1:56)))
        Ym  = [State{8, Float64}(fm[(8b - 7):(8b)]) for b in 1:7]
        wallm = WallPoint(SVector(0.0), SVector(1.0), 0.02, SVector(-0.3))
        g_std = SlipWall()(mix, fm, wallm)
        prm = HyperCloudCore.block_wall_reduce(SlipWall(), mb, Ym, 1, 7, 1, wallm)
        g_blk = vcat((HyperCloudCore.block_wall_apply(SlipWall(), mb, Ym[b], b, prm, wallm, Ym, 1, 7, 1) for b in 1:7)...)
        @test maximum(abs.(g_std - g_blk)) < 1e-13
    end
end
