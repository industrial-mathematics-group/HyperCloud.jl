using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
const BCD = Dict{Int, AbstractBoundaryCondition}
include(joinpath("/tmp/brev3/r3/probes", "helpers.jl"))
which = ARGS[1]

function disk(bc; Ns = 24, Nv = 8, tmax = 0.1, u0 = (0.0, 0.0), tau = 1e-2, kind = :rest)
    E0 = 2.5 + 0.5 * sum(abs2, u0)
    uL = kind == :rest ? (1.0, u0..., E0) : (1.0, 0.0, 0.0, 2.5)
    uR = kind == :rest ? (1.0, u0..., E0) : (0.25, 0.0, 0.0, 0.25)
    p = Dict{Symbol, Any}(
        :PDE_name => :bgk, :PDE_D => 2, :PDE_gamma => 1.4, :PDE_Nv => Nv, :PDE_vmax => 6.0, :PDE_tau => tau,
        :sim_func_name => :run_bgk_simulation, :tmax => tmax, :snapshots => 1,
        :Grid_domain => :spherical, :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (Ns, Ns),
        :Grid_periodic => false, :Grid_randomness_factor => (0.0, 0.0), :Grid_seed => 1, :Grid_mover => :none,
        :Grid_bc => bc,
        :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
        :IC_name => :riemann, :IC_uL => uL, :IC_uR => uR, :IC_p0 => (0.0, 0.0), :IC_n => (1.0, 0.0),
        :Time_CFL => 0.2, :Time_stepper => :ARS222,
        :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :none, :MOOD_criterion => :none)
    t = @elapsed sim = run_bgk_simulation(p)
    sim === nothing && return println("  failed")
    us0, us = sim.u[1], sim.u[end]
    mean(f, us) = sum(f, us) / length(us)
    dev = kind == :rest ? maximum(maximum(abs.(u[1:4] - SVector(1.0, u0..., E0))) for u in us) : NaN
    @printf("  dev %.3e, Δmean ρ %.3e, Δmean E %.3e, mean ρu %.3e %.3e (t0 %.3e), %.1f s, steps %d, M=%d\n", dev,
            mean(u -> u[1], us) - mean(u -> u[1], us0), mean(u -> u[4], us) - mean(u -> u[4], us0),
            mean(u -> u[2], us), mean(u -> u[3], us), mean(u -> u[2], us0), t, sim.metadata[:steps], length(us[1]))
end

if which == "disk"
    for (lbl, bc) in (("diffuse", BCD(1 => diffuse_wall(1.0))), ("slip", BCD(1 => slip_wall())), ("diffuse α=0.5", BCD(1 => diffuse_wall(1.0; accommodation = 0.5))))
        println("2-D disk, gas at rest, ", lbl); disk(bc)
    end
    for (lbl, bc) in (("slip", BCD(1 => slip_wall())), ("diffuse α=0.5", BCD(1 => diffuse_wall(1.0; accommodation = 0.5))))
        println("2-D disk, gas drifting u=(0.3,0.1), τ=1e-2, t=0.5, ", lbl); disk(bc; u0 = (0.3, 0.1), tmax = 0.5)
        println("2-D disk, Riemann (ρ,p)=(1,1)|(0.25,0.1) rarefied τ=1, t=0.5, ", lbl); disk(bc; kind = :riemann, tau = 1.0, tmax = 0.5)
        println("2-D disk, Riemann τ=1e-3, t=0.5, ", lbl); disk(bc; kind = :riemann, tau = 1e-3, tmax = 0.5)
    end
elseif which == "disk_res"
    for (Ns, Nv) in ((24, 8), (48, 8), (24, 16))
        println("2-D disk drifting, slip, Ns=$Ns Nv=$Nv"); disk(BCD(1 => slip_wall()); Ns = Ns, Nv = Nv, u0 = (0.3, 0.1), tmax = 0.5)
    end
elseif which == "bgkpiston"
    w = 0.5
    ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, w, 1.4)
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-w,)))
    tube   = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), piston)
    for (lbl, bc2, vc) in (("diffuse", diffuse_wall(1.0), 0.0), ("slip", slip_wall(), 0.0), ("slip, vcenter=-0.5 (flip)", slip_wall(), -0.5))
        p = merge(bgk_wall_base(; Ns = 300, tau = 1e-4, tmax = 0.3), Dict{Symbol, Any}(:Grid_domain => :shape, :Grid_shape => tube,
                  :Grid_mins => (0.0,), :Grid_maxs => (1.5,), :Grid_bc => BCD(1 => slip_wall(), 2 => bc2), :PDE_vcenter => (vc,),
                  :Limiter_name => :VK, :Limiter_mode => :hard))
        sim = run_bgk_simulation(p)
        sim === nothing && (println("bgk piston $lbl failed"); continue)
        t = sim.t[end]
        ρ, u, pr = plateau(sim, 1 - S * t + 0.06, 1 - w * t - 0.08)
        @printf("BGK piston %s: ρ=%.4f (%.2f%%) u=%.4f p=%.4f (%.2f%%)\n", lbl, ρ, 100(ρ/ρ₂-1), u, pr, 100(pr/p₂-1))
    end
elseif which == "vcenter"
    for (vc, u0) in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.3), (0.3, 0.3))
        E = 2.5 + 0.5u0^2
        sim = run_bgk_simulation(merge(bgk_wall_base(; tmax = 0.5), Dict{Symbol, Any}(:PDE_vcenter => (vc,), :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()),
                                 :IC_uL => (1.0, u0, E), :IC_uR => (1.0, u0, E))))
        us = sim.u[end]
        @printf("vcenter=%.1f u0=%.1f slip walls: max dev from rest eq %.3e, mean ρ-1 %.3e, mean ρu %.3e, mean E-E0 %.3e\n", vc, u0, maximum(maximum(abs.(u - SVector(1.0, 0.0, 2.5))) for u in us),
                sum(u[1] for u in us)/length(us)-1, sum(u[2] for u in us)/length(us), sum(u[3] for u in us)/length(us) - E)
    end
elseif which == "hot"
    for (lbl, bcs, uR) in (("hot diffuse 1/2", (diffuse_wall(1.0), diffuse_wall(2.0)), (1.0, 0.0, 2.5)), ("slip/slip, jump IC", (slip_wall(), slip_wall()), (0.5, 0.0, 1.0)))
        for N in (50, 100, 200)
            p = merge(bgk_wall_base(; Ns = N, tmax = 0.5), Dict{Symbol, Any}(:Grid_bc => BCD(1 => bcs[1], 2 => bcs[2]), :IC_uR => uR))
            sim = run_bgk_simulation(p)
            xs = [x[1] for x in sim.x[end]]; o = sortperm(xs); xs = xs[o]
            trap(us) = (u = [us[i][1] for i in o]; sum((u[i] + u[i+1]) / 2 * (xs[i+1] - xs[i]) for i in 1:length(xs)-1))
            θ(u) = 2 * (u[3] - 0.5 * u[2]^2 / u[1]) / (5 * u[1])
            @printf("%s N=%d: Δtrapezoid %.3e  θ at x=1 %.4f\n", lbl, N, trap(sim.u[end]) - trap(sim.u[1]), θ(sim.u[end][o[end]]))
        end
    end
end
if which == "pos"
    for (lbl, bc) in (("slip", BCD(1 => slip_wall())), ("diffuse", BCD(1 => diffuse_wall(1.0))))
        for (tau, tm) in ((1.0, 0.5), (1e-2, 0.5))
            p = Dict{Symbol, Any}(
                :PDE_name => :bgk, :PDE_D => 2, :PDE_gamma => 1.4, :PDE_Nv => 8, :PDE_vmax => 6.0, :PDE_tau => tau,
                :sim_func_name => :run_bgk_simulation, :tmax => tm, :snapshots => 5, :save_relax => true,
                :Grid_domain => :spherical, :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (24, 24),
                :Grid_periodic => false, :Grid_randomness_factor => (0.0, 0.0), :Grid_seed => 1, :Grid_mover => :none,
                :Grid_bc => bc, :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
                :IC_name => :riemann, :IC_uL => (1.0, 0.0, 0.0, 2.5), :IC_uR => (0.25, 0.0, 0.0, 0.25), :IC_p0 => (0.0, 0.0), :IC_n => (1.0, 0.0),
                :Time_CFL => 0.2, :Time_stepper => :ARS222,
                :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :none, :MOOD_criterion => :none)
            sim = run_bgk_simulation(p)
            sim === nothing && (println("failed"); continue)
            r = [minimum(minimum(u) for u in us) / maximum(maximum(u) for u in us) for us in sim.u]
            @printf("%s τ=%g: min f / max f over the particles per snapshot: %s\n", lbl, tau, join((@sprintf("%.2e", x) for x in r), " "))
        end
    end
end
