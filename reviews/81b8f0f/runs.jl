using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
const BCD = Dict{Int, AbstractBoundaryCondition}
include(joinpath(@__DIR__, "helpers.jl"))
which = ARGS[1]
if which == "euler"
    ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, 1.0, 1.4)
    @printf("RH reflect: ρ2=%.6f p2=%.6f S=%.6f\n", ρ₂, p₂, S)
    # independent check of RH reflection: wall frame, gas at u=1 -> rest
    E₁ = 3.0
    p = merge(euler_wall_base(), Dict{Symbol, Any}(:Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,),
              :Grid_bc => BCD(1 => FixedDirichlet(), 2 => slip_wall()),
              :IC_name => :riemann, :IC_uL => (1.0, 1.0, E₁), :IC_uR => (1.0, 1.0, E₁), :IC_p0 => (0.5,), :IC_n => (1.0,)))
    sim = run_direct_simulation(p); t = sim.t[end]
    @printf("reflect plateau: %s (t=%.4f)\n", string(round.(plateau(sim, 1 - (S - 1) * t + 0.08, 0.98), digits=5)), t)
    w = 0.5
    ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, w, 1.4)
    @printf("RH piston: ρ2=%.6f p2=%.6f S=%.6f\n", ρ₂, p₂, S)
    for T in (Float64, Float32), N in (150, 300, 600)
        T == Float32 && N != 300 && continue
        piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-w,)))
        tube   = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), piston)
        p = merge(euler_wall_base(; Ns = N), Dict{Symbol, Any}(:T => T, :Grid_domain => :shape, :Grid_shape => tube, :Grid_mins => (0.0,), :Grid_maxs => (1.5,),
                  :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()),
                  :IC_name => :riemann, :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (1.0, 0.0, 2.5), :IC_p0 => (0.5,), :IC_n => (1.0,)))
        sim = run_direct_simulation(p)
        sim === nothing && (println("piston $T $N failed"); continue)
        t = sim.t[end]; face = 1 - w*t
        ex = exact_piston(p)
        xs = [x[1] for x in sim.x[end]]
        L1 = sum(abs(ex(SVector(xs[i], t))[1] - sim.u[end][i][1]) for i in eachindex(xs)) / length(xs)
        mass = sum(u[1] for u in sim.u[end]) * 1.5 / N    # nominal volume
        @printf("piston %s N=%d: plateau %s, max x - face = %.4e (dx=%.4e), L1ρ=%.4e, mass≈%.5f (exact %.5f) md=%s\n", T, N,
            string(round.(plateau(sim, 1 - S * t + 0.06, face - 0.03), digits=5)), maximum(xs) - face, 1.5/N, L1, mass, 1.0*1.0, string(sim.metadata[:stopped_early]))
    end
elseif which == "example"
    conf = euler_1d_piston()
    old = get_save_path(); set_save_path!(mktempdir())
    for N in (150, 300, 600)
        conf2 = SimulationConfig(merge(conf.shared, Dict{Symbol,Any}(:Grid_Ns => (N,))), conf.methods, conf.active_methods; reference = conf.reference)
        s = run_all_simulations(conf2; force_overwrite = true, calculate_stats = false)
        for p in generate_tasks(conf2)
            sim = load_sim_data(p)
            t = sim.t[end]; ex = exact_piston(p)
            xs = [x[1] for x in sim.x[end]]
            L1 = sum(abs(ex(SVector(xs[i], t))[1] - sim.u[end][i][1]) for i in eachindex(xs)) / length(xs)
            @printf("example N=%d %s: L1ρ=%.4e t=%.3f\n", N, p[:method_name], L1, t)
        end
    end
    set_save_path!(old)
elseif which == "bgk"
    for bc in (diffuse_wall(1.0), slip_wall(), diffuse_wall(1.0; accommodation = 0.5))
        sim = run_bgk_simulation(merge(bgk_wall_base(), Dict{Symbol, Any}(:Grid_bc => BCD(1 => bc, 2 => bc))))
        @printf("equilibrium %s: %.3e\n", typeof(bc.op), maximum(maximum(abs.(u - SVector(1.0, 0.0, 2.5))) for u in sim.u[end]))
    end
    θ(u) = 2 * (u[3] - 0.5 * u[2]^2 / u[1]) / (5 * u[1])
    for N in (50, 100, 200, 400)
        sim = run_bgk_simulation(merge(bgk_wall_base(; Ns = N, tmax = 0.5), Dict{Symbol, Any}(:Grid_bc => BCD(1 => diffuse_wall(1.0), 2 => diffuse_wall(2.0)))))
        us = sim.u[end]
        mean_ρ = sum(u[1] for u in us) / length(us)
        xs = [x[1] for x in sim.x[end]]
        @printf("hot wall N=%d: θ near hot %.4f, cold %.4f, mass drift %.3e  x range [%.4f, %.4f]  mom sum %.3e\n", N, θ(us[end]), θ(us[1]), mean_ρ - 1, minimum(xs), maximum(xs), sum(u[2] for u in us)/length(us))
    end
    # same but walls both at gas temperature yet gas moving? mass drift control: hot wall with slip walls (no flux) -> none
    w = 0.5
    ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, w, 1.4)
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-w,)))
    tube   = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), piston)
    for (lbl, bc2) in (("diffuse", diffuse_wall(1.0)), ("slip(nearest)", slip_wall()))
        p = merge(bgk_wall_base(; Ns = 300, tau = 1e-4, tmax = 0.3), Dict{Symbol, Any}(:Grid_domain => :shape, :Grid_shape => tube,
                  :Grid_mins => (0.0,), :Grid_maxs => (1.5,), :Grid_bc => BCD(1 => slip_wall(), 2 => bc2),
                  :Limiter_name => :VK, :Limiter_mode => :hard))
        sim = run_bgk_simulation(p)
        sim === nothing && (println("bgk piston $lbl failed"); continue)
        t = sim.t[end]
        ρ, u, pr = plateau(sim, 1 - S * t + 0.06, 1 - w * t - 0.08)
        @printf("BGK piston %s: ρ=%.4f (%.2f%%) u=%.4f p=%.4f (%.2f%%)\n", lbl, ρ, 100(ρ/ρ₂-1), u, pr, 100(pr/p₂-1))
    end
elseif which == "vcenter"
    for vc in (0.0, 0.5)
        sim = run_bgk_simulation(merge(bgk_wall_base(; tmax=0.5), Dict{Symbol, Any}(:PDE_vcenter => (vc,), :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()))))
        us = sim.u[end]
        @printf("vcenter=%.1f slip walls equilibrium at rest: max dev %.3e, mean ρ-1 %.3e, mean ρu %.3e\n", vc, maximum(maximum(abs.(u - SVector(1.0, 0.0, 2.5))) for u in us), sum(u[1] for u in us)/length(us)-1, sum(u[2] for u in us)/length(us))
    end
elseif which == "other"
    for (name, uL) in ((:linear, (1.0,)), (:burgers, (1.0,)))
        p = Dict{Symbol,Any}(:PDE_name => name, :PDE_D => 1, :PDE_velocities => (1.0,), :Limiter_mode => :hard, :sim_func_name => :run_direct_simulation, :tmax => 0.1, :snapshots => 1,
            :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (50,), :Grid_periodic => false,
            :Grid_randomness_factor => (0.0,), :Grid_seed => 1, :Grid_mover => :none, :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()),
            :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
            :IC_name => :riemann, :IC_uL => uL, :IC_uR => uL, :IC_p0 => (0.5,), :IC_n => (1.0,),
            :Time_CFL => 0.2, :Time_stepper => :RK2, :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :VK, :MOOD_criterion => :none)
        logs, sim = Test.collect_test_logs(() -> run_direct_simulation(p))
        println(name, ": sim === nothing: ", sim === nothing)
        for l in logs; println("  [", l.level, "] ", first(string(l.message), 300)); for (k,v) in l.kwargs; println("    ", k, " => ", first(v isa Exception ? sprint(showerror, v) : (v isa Tuple && v[1] isa Exception ? sprint(showerror, v[1]) : string(v)), 400)); end; end
    end
end
