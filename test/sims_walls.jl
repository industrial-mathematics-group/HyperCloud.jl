# Group `walls`: the wall operators for MirrorBC (SlipWall for Euler and BGK, DiffuseWall for BGK) on static and
# moving boundaries, against exact Rankine–Hugoniot states and the kinetic flux balance.

const BCD = Dict{Int, AbstractBoundaryCondition}

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

@testset "Walls" begin
    @testset "Operators" begin
        eq = EulerEquation(Val(2), Float64)
        wall = WallPoint(SVector(0.0, 0.0), SVector(0.0, 1.0), 0.1, SVector(0.0, 0.0))
        U = prim2cons(eq, State{4, Float64}(2.0, 0.3, -0.4, 1.5))
        V = cons2prim(eq, SlipWall()(eq, U, wall))
        @test V ≈ State{4, Float64}(2.0, 0.3, 0.4, 1.5)                        # the normal velocity flips, the rest stays
        moving = WallPoint(SVector(0.0, 0.0), SVector(0.0, 1.0), 0.1, SVector(0.0, 0.2))
        V = cons2prim(eq, SlipWall()(eq, U, moving))
        @test V ≈ State{4, Float64}(2.0, 0.3, 0.8, 1.5)                        # relative to the wall: -0.4 - 0.2 -> +0.6, plus 0.2
        @test_throws ArgumentError DiffuseWall(0.0)
        @test_throws ArgumentError DiffuseWall(1.0; accommodation = 2)
        @test slip_wall() isa MirrorBC{1, SlipWall, true} && diffuse_wall(1.5; degree = 2) isa MirrorBC{2, <:DiffuseWall, true}
        # BGK: specular reflection flips the node index along the wall axis; the diffuse wall returns an
        # equilibrium at its own temperature into itself (exact discrete flux balance)
        nodes, weights = velocity_grid(:uniform, 8, 6.0)
        bgk = BGKEquation(Val(1), nodes, weights, 1.4, ConstantRelaxationTime(1e-2))
        f, _ = equilibrium(bgk, SVector(1.0, 0.3, 2.5 + 0.5 * 0.09))
        w1 = WallPoint(SVector(1.0), SVector(1.0), 0.05, SVector(0.0))
        g = SlipWall()(bgk, f, w1)
        @test all(k -> g[k] == f[9 - k] && g[8 + k] == f[17 - k], 1:8)
        feq, _ = equilibrium(bgk, SVector(1.0, 0.0, 2.5))
        @test DiffuseWall(1.0)(bgk, feq, w1) ≈ feq atol = 1e-13
        @test DiffuseWall(1.0; accommodation = 0.3)(bgk, feq, w1) ≈ feq atol = 1e-13
        # a hotter wall emits a hotter half: the outgoing (v < 0, into the fluid at x < 1) components change
        hot = DiffuseWall(2.0)(bgk, feq, w1)
        @test all(k -> (velocity_node(bgk, k)[1][1] > 0) == (hot[k] == feq[k]), 1:8)
        @test moments(bgk, hot)[1] > 0
    end

    @testset "Euler: a gas at u = 1 reflected by a slip wall reaches the Rankine–Hugoniot state" begin
        ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, 1.0, 1.4)
        E₁ = 2.5 + 0.5
        p = merge(euler_wall_base(), Dict{Symbol, Any}(:Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,),
                  :Grid_bc => BCD(1 => FixedDirichlet(), 2 => slip_wall()),
                  :IC_name => :riemann, :IC_uL => (1.0, 1.0, E₁), :IC_uR => (1.0, 1.0, E₁), :IC_p0 => (0.5,), :IC_n => (1.0,)))
        sim = run_direct_simulation(p)
        @test sim !== nothing
        if sim !== nothing
            t = sim.t[end]
            ρ, u, pr = plateau(sim, 1 - (S - 1) * t + 0.08, 0.98)
            @test abs(ρ - ρ₂) < 0.01 && abs(pr - p₂) < 0.01 && abs(u) < 0.01   # measured 2.0774 / 2.9267 / 4e-4 (exact 2.0792 / 2.9266 / 0)
        end
    end

    @testset "Euler: a piston driven into a gas at rest (moving slip wall) matches Rankine–Hugoniot" begin
        w = 0.5
        ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, w, 1.4)
        piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-w,)))
        tube   = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), piston)
        p = merge(euler_wall_base(; Ns = 300), Dict{Symbol, Any}(:Grid_domain => :shape, :Grid_shape => tube, :Grid_mins => (0.0,), :Grid_maxs => (1.5,),
                  :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()),
                  :IC_name => :riemann, :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (1.0, 0.0, 2.5), :IC_p0 => (0.5,), :IC_n => (1.0,)))
        sim = run_direct_simulation(p)
        @test sim !== nothing
        if sim !== nothing
            t = sim.t[end]
            face = 1 - w * t
            xs = [x[1] for x in sim.x[end]]
            @test face - 0.01 < maximum(xs) < face                                # the cloud follows the piston
            ρ, u, pr = plateau(sim, 1 - S * t + 0.06, face - 0.03)
            @test abs(ρ - ρ₂) < 0.01 && abs(pr - p₂) < 0.01 && abs(u + w) < 0.005   # measured 1.4895 / 1.7602 / -0.5000
            @test all(u -> u[1] > 0.99, sim.u[end])
        end
        # the example wraps this set-up; its reference is the exact plateau solution
        conf = euler_1d_piston()
        @test conf.reference == :exact_piston
        ex = exact_piston(conf.shared)
        @test ex(SVector(0.2, 0.3))[1] == 1.0 && ex(SVector(0.8, 0.3))[1] ≈ ρ₂
        # a piston face on the end of the canvas has no ghosts of its own: refused
        bad = intersect(Tagged(Cuboid((0.0,), (1.0,)), 1), piston)
        @test_throws ArgumentError ParticleGrid(GeometricDomain(bad; mins = (0.0,), maxs = (1.0,)), (0.01,), ExponentialWeightFunction(1.0, 0.025), 3)
    end

    @testset "BGK: an equilibrium gas between walls at its temperature stays in equilibrium" begin
        for bc in (diffuse_wall(1.0), slip_wall(), diffuse_wall(1.0; accommodation = 0.5))
            sim = run_bgk_simulation(merge(bgk_wall_base(), Dict{Symbol, Any}(:Grid_bc => BCD(1 => bc, 2 => bc))))
            @test sim !== nothing
            sim !== nothing && @test maximum(maximum(abs.(u - SVector(1.0, 0.0, 2.5))) for u in sim.u[end]) < 1e-12
        end
    end

    @testset "BGK: a hot diffuse wall heats the gas, the mass error shrinks with the resolution" begin
        θ(u) = 2 * (u[3] - 0.5 * u[2]^2 / u[1]) / (5 * u[1])                  # D + K = 5 for γ = 1.4 in 1-D
        drift = Float64[]
        for N in (50, 100)
            sim = run_bgk_simulation(merge(bgk_wall_base(; Ns = N, tmax = 0.5), Dict{Symbol, Any}(:Grid_bc => BCD(1 => diffuse_wall(1.0), 2 => diffuse_wall(2.0)))))
            @test sim !== nothing
            sim === nothing && continue
            us = sim.u[end]
            @test 1.7 < θ(us[end]) < 2.0 && abs(θ(us[1]) - 1) < 0.01            # measured 1.83 / 1.86 next to the hot wall
            @test all(x -> 0.99 < x < 2.0, θ.(us)) && θ(us[end - 5]) > θ(us[5])    # (the profile is not strictly monotone: 1e-3 ripples at the cold wall)
            push!(drift, abs(sum(u[1] for u in us) / length(us) - 1))
        end
        length(drift) == 2 && @test drift[2] < 0.8 * drift[1] && drift[2] < 0.01   # measured 7.4e-3 / 5.0e-3 / 3.2e-3 (N = 50 / 100 / 200)
    end

    @testset "BGK: a piston (τ = 1e-4, isothermal diffuse wall) reproduces the Euler plateau within 1 %" begin
        w = 0.5
        ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, w, 1.4)
        piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-w,)))
        tube   = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), piston)
        p = merge(bgk_wall_base(; Ns = 300, tau = 1e-4, tmax = 0.3), Dict{Symbol, Any}(:Grid_domain => :shape, :Grid_shape => tube,
                  :Grid_mins => (0.0,), :Grid_maxs => (1.5,), :Grid_bc => BCD(1 => slip_wall(), 2 => diffuse_wall(1.0)),
                  :Limiter_name => :VK, :Limiter_mode => :hard))
        sim = run_bgk_simulation(p)
        @test sim !== nothing
        if sim !== nothing
            t = sim.t[end]
            ρ, u, pr = plateau(sim, 1 - S * t + 0.06, 1 - w * t - 0.08)
            @test abs(ρ - ρ₂) / ρ₂ < 0.012 && abs(pr - p₂) / p₂ < 0.012 && abs(u + w) < 0.012   # measured 1.482 / 1.747 / -0.493
        end
    end
end
