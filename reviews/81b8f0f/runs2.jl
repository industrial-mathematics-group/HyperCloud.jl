using HyperCloud, StaticArrays, Printf, Test
enable_progress_bar!(false)
const BCD = Dict{Int, AbstractBoundaryCondition}
include(joinpath(@__DIR__, "helpers.jl"))
function example()
    old = get_save_path(); set_save_path!(mktempdir())
    for N in (150, 300, 600)
        conf = euler_1d_piston()
        conf.shared[:Grid_Ns] = (N,)
        run_all_simulations(conf; force_overwrite = true, calculate_stats = false)
        for p in generate_tasks(conf)
            sim = load_sim_data(p)
            t = sim.t[end]; ex = exact_piston(p)
            xs = [x[1] for x in sim.x[end]]
            L1 = sum(abs(ex(SVector(xs[i], t))[1] - sim.u[end][i][1]) for i in eachindex(xs)) / length(xs)
            @printf("example N=%d %s: L1ρ=%.4e t=%.3f\n", N, string(get(p, :method_name, get(p, :Method_name, "?"))), L1, t)
        end
    end
    set_save_path!(old)
end
function f32()
    w = 0.5f0
    piston = Moving(Tagged(HalfSpace((1f0,), (1f0,)), 2), translating((-w,)))
    tube   = intersect(Tagged(Cuboid((0f0,), (1.5f0,)), 1), piston)
    p = merge(euler_wall_base(; Ns = 300), Dict{Symbol, Any}(:T => Float32, :Grid_domain => :shape, :Grid_shape => tube, :Grid_mins => (0.0,), :Grid_maxs => (1.5,),
              :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()),
              :IC_name => :riemann, :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (1.0, 0.0, 2.5), :IC_p0 => (0.5,), :IC_n => (1.0,)))
    sim = run_direct_simulation(p)
    sim === nothing && return println("F32 piston failed")
    ρ₂, p₂, S = HyperCloud.piston_state(1.0, 1.0, 0.5, 1.4); t = sim.t[end]
    @printf("F32 piston plateau %s eltype %s\n", string(round.(plateau(sim, 1 - S * t + 0.06, 1 - 0.5t - 0.03), digits=5)), eltype(eltype(sim.u[end])))
    # example with T = Float32
    conf = euler_1d_piston(); conf.shared[:T] = Float32
    p = merge(conf.shared, conf.methods[:RK2MUSCL2_VK])
    println("example with :T => Float32: ", run_direct_simulation(p) === nothing ? "fails" : "runs")
end
# control: is the mass drift wall specific? same non-equilibrium gas, slip (specular) walls vs diffuse walls at the gas temperature
function control()
    for (lbl, bcs) in (("slip/slip", (slip_wall(), slip_wall())), ("diffuse1/diffuse1", (diffuse_wall(1.0), diffuse_wall(1.0))), ("diffuse1/diffuse2", (diffuse_wall(1.0), diffuse_wall(2.0))))
        for N in (50, 100, 200)
            # a pressure jump in the middle (Sod-like, gentle) so that the gas moves against the walls
            p = merge(bgk_wall_base(; Ns = N, tmax = 0.5), Dict{Symbol, Any}(:Grid_bc => BCD(1 => bcs[1], 2 => bcs[2]),
                      :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (0.5, 0.0, 1.0)))
            sim = run_bgk_simulation(p)
            us0, us = sim.u[1], sim.u[end]
            @printf("BGK %s N=%d: mean ρ change %.3e\n", lbl, N, (sum(u[1] for u in us) - sum(u[1] for u in us0)) / length(us))
        end
    end
    for N in (50, 100, 200)
        p = merge(euler_wall_base(; Ns = N, tmax = 0.5), Dict{Symbol, Any}(:Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,),
                  :Grid_bc => BCD(1 => slip_wall(), 2 => slip_wall()),
                  :IC_name => :riemann, :IC_uL => (1.0, 0.0, 2.5), :IC_uR => (0.5, 0.0, 1.0), :IC_p0 => (0.5,), :IC_n => (1.0,)))
        sim = run_direct_simulation(p)
        @printf("Euler slip/slip N=%d: mean ρ change %.3e\n", N, (sum(u[1] for u in sim.u[end]) - sum(u[1] for u in sim.u[1])) / length(sim.u[end]))
    end
end
getfield(Main, Symbol(ARGS[1]))()
