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

# Net mass flux through a wall (normal n into the wall, velocity w) of a BGK ghost state g for the mirrored interior
# state f: the incident half (c > 0) from f, the emitted half from g. Zero for a wall that holds the gas.
function wall_mass_flux(eq, f, g, n, w)
    Φ = 0.0
    for k in 1:HyperCloud.num_velocities(eq)
        v, wk = velocity_node(eq, k)
        c = sum((v - w) .* n)
        Φ += wk * c * (c > 0 ? f[k] : g[k])
    end
    return Φ
end

# Mass of the final state minus the initial mass (trapezoid rule over the sorted 1-D cloud: the mean of the point
# values is not the mass, the wall particles at both ends count half)
function mass_change(sim)
    xs = [x[1] for x in sim.x[end]]
    o = sortperm(xs)
    trap(us) = sum((us[o[i]][1] + us[o[i + 1]][1]) / 2 * (xs[o[i + 1]] - xs[o[i]]) for i in 1:length(o) - 1)
    return trap(sim.u[end]) - trap(sim.u[1])
end

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
        # the runners check the operators before the first step (inside the kernel a MethodError, or no wall at all)
        box1 = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = BCD(1 => slip_wall(), 2 => slip_wall()))
        lin  = LinearAdvection(((1.0,),), Float64)
        @test_throws ArgumentError HyperCloud._check_wall_ops(lin, box1)
        @test HyperCloud._check_wall_ops(EulerEquation(Val(1), Float64), box1) === nothing
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
        boxd(op) = get_rectangular_domain(Float64, (0.0,), (1.0,); bc_map = BCD(1 => MirrorBC(op), 2 => MirrorBC(op)))
        @test HyperCloud._check_wall_ops(bgk, boxd(DiffuseWall(2.0))) === nothing
        @test_throws ArgumentError HyperCloud._check_wall_ops(bgk, boxd(DiffuseWall(1.0; velocity = (0.1, 0.2))))
        @test_throws ArgumentError HyperCloud._check_wall_ops(bgk, boxd(DiffuseWall(1.0; velocity = (NaN,))))   # no Maxwellian
        @test_throws ArgumentError HyperCloud._check_wall_ops(bgk, boxd(DiffuseWall(1e-4)))    # too cold for 8 nodes: emits nothing
        @test_throws ArgumentError HyperCloud._check_wall_ops(EulerEquation(Val(1), Float64), boxd(DiffuseWall(1.0)))
        hot = DiffuseWall(2.0)(bgk, feq, w1)
        @test all(k -> (velocity_node(bgk, k)[1][1] > 0) == (hot[k] == feq[k]), 1:8)
        @test moments(bgk, hot)[1] > 0
        # Off the node grid (a shifted grid, an oblique normal, a moving wall) the specular reflection keeps the mass
        # flux through the wall at zero and reflects an equilibrium into the equilibrium of the reflected state.
        # A drifting gas, so that the identity or an even reflection would fail.
        reflected(eq, f, n, w) = (U = moments(eq, f); u = SVector(U[2:end-1]...) / U[1];
                                  ur = u - 2 * sum((u - w) .* n) * n; (U[1], ur, U[end] - 0.5 * U[1] * sum(abs2, u)))
        function check_reflection(eq, f, n, w; tol = 1e-3)
            g = SlipWall()(eq, f, WallPoint(zero(n), n, 0.05, w))
            @test abs(wall_mass_flux(eq, f, g, n, w)) < 1e-13
            ρ, ur, ei = reflected(eq, f, n, w)
            G = moments(eq, g)
            @test abs(G[1] - ρ) < tol && maximum(abs.(SVector(G[2:end-1]...) / G[1] - ur)) < tol &&
                  abs(G[end] - 0.5 * G[1] * sum(abs2, SVector(G[2:end-1]...) / G[1]) - ei) < tol
        end
        nodes16, weights16 = velocity_grid(:uniform, 16, 6.0)
        shifted = BGKEquation(Val(1), nodes16, weights16, 1.4, ConstantRelaxationTime(1e-2); center = (0.5,))
        fs, _ = equilibrium(shifted, SVector(1.0, 0.2, 2.5 + 0.5 * 0.04))
        # (16 nodes: the scaled emitted half is off by 2e-3 in 1-D, 2e-5 with 64 nodes)
        check_reflection(shifted, fs, SVector(1.0), SVector(0.0); tol = 5e-3)   # used to reflect about the grid centre 0.5
        bgk16 = BGKEquation(Val(1), nodes16, weights16, 1.4, ConstantRelaxationTime(1e-2))
        f16, _ = equilibrium(bgk16, SVector(1.0, 0.3, 2.5 + 0.5 * 0.09))
        check_reflection(bgk16, f16, SVector(1.0), SVector(0.5); tol = 5e-3)    # moving wall: u' = 2w - u = 0.7
        nodes12, weights12 = velocity_grid(:uniform, 12, 6.0)
        bgk2 = BGKEquation(Val(2), nodes12, weights12, 1.4, ConstantRelaxationTime(1e-2))
        f2, _ = equilibrium(bgk2, SVector(1.0, 0.2, -0.1, 2.5 + 0.5 * 0.05))
        for θ in (0.3, π / 4)                                                # oblique; the diagonal is a node permutation
            check_reflection(bgk2, f2, SVector(cos(θ), sin(θ)), SVector(0.0, 0.0))
        end
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
        # its shape is a string, so the piston speed is in the parameter hash (a Moving instance's motion is a
        # closure, which the hash does not see), and it builds in the run's float type
        @test euler_1d_piston(speed = 0.4).shared[:Grid_shape] != conf.shared[:Grid_shape]
        @test euler_1d_piston(speed = 0.4).shared[:piston_speed] == 0.4
        @test_throws ArgumentError euler_1d_piston(speed = -0.5)
        @test_throws ArgumentError euler_1d_piston(face = 0.3)
        p32 = merge(conf.shared, conf.methods[:RK2MUSCL2_VK], Dict{Symbol, Any}(:T => Float32, :Grid_Ns => (60,), :tmax => 0.05, :snapshots => 1))
        sim32 = run_direct_simulation(p32)
        @test sim32 !== nothing && eltype(sim32.u[end][1]) == Float32
        @test HyperCloud.piston_state(1.0, 1.0, 50.0, 1.4)[2] > 1000      # a fast piston is bracketed (capped at 1000 p₁)
        # a piston face on the end of the canvas has no ghosts of its own: refused
        bad = intersect(Tagged(Cuboid((0.0,), (1.0,)), 1), piston)
        @test_throws ArgumentError ParticleGrid(GeometricDomain(bad; mins = (0.0,), maxs = (1.0,)), (0.01,), ExponentialWeightFunction(1.0, 0.025), 3)
    end

    # (a smoke test of the runs: an equilibrium at rest is even in v, so any even reflection keeps it; the operator
    # checks above discriminate)
    @testset "BGK: an equilibrium gas between walls at its temperature stays in equilibrium" begin
        for bc in (diffuse_wall(1.0), slip_wall(), diffuse_wall(1.0; accommodation = 0.5))
            sim = run_bgk_simulation(merge(bgk_wall_base(), Dict{Symbol, Any}(:Grid_bc => BCD(1 => bc, 2 => bc))))
            @test sim !== nothing
            sim !== nothing && @test maximum(maximum(abs.(u - SVector(1.0, 0.0, 2.5))) for u in sim.u[end]) < 1e-12
        end
    end

    @testset "BGK: a hot diffuse wall heats the gas, the mass error shrinks with the resolution" begin
        # control: specular walls with a pressure jump in the gas conserve the mass to second order (measured
        # -3.9e-5 at N = 100), so the drift below is the diffuse wall's
        simc = run_bgk_simulation(merge(bgk_wall_base(; Ns = 100, tmax = 0.5), Dict{Symbol, Any}(:Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()),
                                  :IC_uR => (0.5, 0.0, 1.0))))
        @test simc !== nothing
        simc !== nothing && @test abs(mass_change(simc)) < 2e-4
        θ(u) = 2 * (u[3] - 0.5 * u[2]^2 / u[1]) / (5 * u[1])                  # D + K = 5 for γ = 1.4 in 1-D
        drift = Float64[]
        for N in (50, 100)
            sim = run_bgk_simulation(merge(bgk_wall_base(; Ns = N, tmax = 0.5), Dict{Symbol, Any}(:Grid_bc => BCD(1 => diffuse_wall(1.0), 2 => diffuse_wall(2.0)))))
            @test sim !== nothing
            sim === nothing && continue
            us = sim.u[end]
            @test 1.7 < θ(us[end]) < 2.0 && abs(θ(us[1]) - 1) < 0.01            # measured 1.83 / 1.86 next to the hot wall
            @test all(x -> 0.99 < x < 2.0, θ.(us)) && θ(us[end - 5]) > θ(us[5])    # (the profile is not strictly monotone: 1e-3 ripples at the cold wall)
            push!(drift, abs(mass_change(sim)))
        end
        length(drift) == 2 && @test drift[2] < 0.8 * drift[1] && drift[2] < 0.01   # measured 1.15e-2 / 7.1e-3 / 4.2e-3 (N = 50 / 100 / 200)
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
        # the same with a specular face: a moving wall takes the reflection off the node grid inside the kernel (the
        # nearest-node reflection was 9 % low in ρ and 13 % in p)
        ps = merge(p, Dict{Symbol, Any}(:Grid_Ns => (200,), :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall())))
        sims = run_bgk_simulation(ps)
        @test sims !== nothing
        if sims !== nothing
            t = sims.t[end]
            ρ, u, pr = plateau(sims, 1 - S * t + 0.06, 1 - w * t - 0.08)
            @test abs(ρ - ρ₂) / ρ₂ < 0.02 && abs(pr - p₂) / p₂ < 0.02 && abs(u + w) < 0.02
        end
    end
end
