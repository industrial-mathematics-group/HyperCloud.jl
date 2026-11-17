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
end

