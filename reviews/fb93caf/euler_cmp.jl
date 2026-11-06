# Euler runs whose results may change with the floor 1e-7 -> 1e-9; saves final states for comparison
using HyperCloud, StaticArrays, Serialization, Logging
tag = ARGS[1]; backend = length(ARGS) > 1 ? ARGS[2] : "polyester"
backend == "ka_cpu" && set_backend!(HyperCloudCore.KA.CPU())
res = Dict{String, Any}()
function rec(name, p)
    sim = nothing; nw = Ref(0)
    lg = Base.CoreLogging.SimpleLogger(devnull, Logging.Warn)
    t = @elapsed sim = getfield(HyperCloud, p[:sim_func_name])(p)
    res[name] = sim === nothing ? nothing : (t = sim.t, u = [Vector{Float64}.(collect.(s)) for s in (sim.u[end],)][1], x = sim.x[end])
    println(name, ": ", sim === nothing ? "FAILED" : "t_end=$(sim.t[end]) min rho=$(minimum(u[1] for u in sim.u[end])) ($(round(t, digits=1)) s)")
end
sod = euler_1d_sod()
for T in (Float64, Float32), m in (:RK2Upwind1, :RK2MUSCL2_VK, :RK2MUSCL2_U2_EPD1, :RK2MUSCL2_U2_EPD2, :RK2MUSCL2_U1_EPD1, :RK4MUSCL4_U2_EPD1)
    rec("sod $m $T", merge(sod.shared, sod.methods[m], Dict{Symbol, Any}(:snapshots => 1, :T => T)))
end
for k in 1:2
    rec("sod lagr+adapt run$k", merge(sod.shared, sod.methods[:RK2MUSCL2_U1_EPD1], Dict{Symbol, Any}(:Grid_mover => :lagrangian, :Adapt_enabled => true, :Grid_Ns => (100,), :snapshots => 1)))
end
cons(ρ, u, p) = (ρ, ρ*u, p/0.4 + 0.5*ρ*u^2)
for (nm, uL, uR) in (("toro2", cons(1.0, -2.0, 0.4), cons(1.0, 2.0, 0.4)), ("vacuum6", cons(1.0, -6.0, 1.0), cons(1.0, 6.0, 1.0)), ("vacuum20", cons(1.0, -20.0, 1.0), cons(1.0, 20.0, 1.0)), ("lowp1e-8", cons(1.0, 0.0, 1e-6), cons(0.125, 0.0, 1e-8))),
    T in (Float64, Float32), m in (:RK2MUSCL2_U2_EPD1, :RK2Upwind1)
    rec("$nm $m $T", merge(sod.shared, sod.methods[m], Dict{Symbol, Any}(:snapshots => 1, :T => T, :tmax => 0.1,
        :IC_uL => uL, :IC_uR => uR, :Grid_bc_map => Dict{Int, AbstractBoundaryCondition}(1 => OutflowBC(), 2 => OutflowBC()))))
end
# 2-D: quadrant Riemann problem (Lax-Liu configuration 3) with MOOD U2 EPD1 (PAD) and VK
c2(ρ, u, v, p) = (ρ, ρ*u, ρ*v, p/0.4 + 0.5*ρ*(u^2 + v^2))
# binary order: index 1 + (x>x0) + 2(y>y0)
q3 = (c2(0.138, 1.206, 1.206, 0.029), c2(0.5323, 0.0, 1.206, 0.3), c2(0.5323, 1.206, 0.0, 0.3), c2(1.5, 0.0, 0.0, 1.5))
base2 = Dict{Symbol, Any}(:PDE_name => :euler, :PDE_D => 2, :PDE_gamma => 1.4, :sim_func_name => :run_direct_simulation,
    :tmax => 0.1, :snapshots => 1, :Grid_domain => :rectangular, :Grid_mins => (0.0, 0.0), :Grid_maxs => (1.0, 1.0),
    :Grid_Ns => (50, 50), :Grid_periodic => false, :Grid_bc => Dict{Int, AbstractBoundaryCondition}(i => OutflowBC() for i in 1:4),
    :Grid_randomness_factor => (0.2, 0.2), :Grid_seed => 10, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
    :IC_name => :q_riemann, :IC_p0 => (0.5, 0.5), :IC_states => q3, :Time_CFL => 0.2)
for T in (Float64, Float32), m in (:RK2MUSCL2_U2_EPD1, :RK2MUSCL2_VK)
    rec("2d q3 $m $T", merge(base2, sod.methods[m], Dict{Symbol, Any}(:T => T)))
end
# 2-D near vacuum: two quadrant pairs moving apart (strong 2-D rarefaction)
qv = (c2(1.0, -3.0, -3.0, 0.4), c2(1.0, 3.0, -3.0, 0.4), c2(1.0, -3.0, 3.0, 0.4), c2(1.0, 3.0, 3.0, 0.4))
for T in (Float64, Float32)
    rec("2d vac $T", merge(base2, sod.methods[:RK2MUSCL2_U2_EPD1], Dict{Symbol, Any}(:T => T, :IC_states => qv, :tmax => 0.05)))
end
serialize("/tmp/review_pr5/probes/euler_$(tag)_$(backend).jls", res)
