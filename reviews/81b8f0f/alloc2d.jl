using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
const BCD = Dict{Int, AbstractBoundaryCondition}
function allocs()
    for T in (Float64, Float32), D in (1, 2)
        nodes, weights = velocity_grid(:uniform, 8, 6.0, T)
        eq = BGKEquation(Val(D), nodes, weights, 1.4, ConstantRelaxationTime(1e-2))
        U = SVector{D+2,T}(1, ntuple(_->T(0.1), D)..., T(2.6))
        f, _ = equilibrium(eq, U)
        n = D == 1 ? SVector{1,T}(1) : SVector{2,T}(0.6, 0.8)
        wp = WallPoint(zero(SVector{D,T}), n, T(0.05), zero(SVector{D,T}))
        wpm = WallPoint(zero(SVector{D,T}), n, T(0.05), SVector{D,T}(ntuple(_->T(0.2), D)))
        s, d, dm = SlipWall(), DiffuseWall(1.0; accommodation = 0.5), DiffuseWall(1.0; velocity = ntuple(_->0.1, D))
        s(eq, f, wp); d(eq, f, wp); dm(eq, f, wpm); s(eq, f, wpm)
        a = (@allocated s(eq, f, wp)), (@allocated s(eq, f, wpm)), (@allocated d(eq, f, wp)), (@allocated dm(eq, f, wpm))
        t = @elapsed for _ in 1:100; s(eq, f, wp); end
        println("D=$D $T allocs slip/slip-moving/diffuse α=.5/diffuse moving = $a; slip oblique $(round(t/100*1e6, digits=1)) µs per ghost (M=$(length(f)))")
    end
end
allocs()
function disk(bc; Ns = 24, Nv = 8, tmax = 0.1)
    p = Dict{Symbol, Any}(
        :PDE_name => :bgk, :PDE_D => 2, :PDE_gamma => 1.4, :PDE_Nv => Nv, :PDE_vmax => 6.0, :PDE_tau => 1e-2,
        :sim_func_name => :run_bgk_simulation, :tmax => tmax, :snapshots => 1,
        :Grid_domain => :spherical, :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (Ns, Ns),
        :Grid_periodic => false, :Grid_randomness_factor => (0.0, 0.0), :Grid_seed => 1, :Grid_mover => :none,
        :Grid_bc => bc,
        :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
        :IC_name => :riemann, :IC_uL => (1.0, 0.0, 0.0, 2.5), :IC_uR => (1.0, 0.0, 0.0, 2.5), :IC_p0 => (0.0, 0.0), :IC_n => (1.0, 0.0),
        :Time_CFL => 0.2, :Time_stepper => :ARS222,
        :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :none, :MOOD_criterion => :none)
    t = @elapsed sim = run_bgk_simulation(p)
    sim === nothing && return println("failed")
    us = sim.u[end]
    dev = maximum(maximum(abs.(u - SVector(1.0, 0.0, 0.0, 2.5))) for u in us)
    @printf("  dev from rest equilibrium %.3e, mean ρ-1 %.3e, wall %.1f s (incl. compile), steps %d\n", dev, sum(u[1] for u in us)/length(us) - 1, t, sim.metadata[:steps])
end
for (lbl, bc) in (("diffuse", BCD(1 => diffuse_wall(1.0))), ("slip", BCD(1 => slip_wall())), ("diffuse α=0.5", BCD(1 => diffuse_wall(1.0; accommodation=0.5))))
    println("2-D disk, gas at rest, ", lbl); disk(bc); disk(bc)
end
