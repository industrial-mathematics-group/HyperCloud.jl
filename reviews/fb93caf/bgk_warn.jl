using HyperCloud, StaticArrays, Test, Logging
const HC = HyperCloud
include(joinpath(@__DIR__, "helpers.jl"))
struct ConstIC{U} <: HC.InitialCondition; U::U; end
(c::ConstIC)(x) = c.U
function grid(D, N, M, T)
    ctx = Dict{Symbol, Any}(:T => T, :D => D, :M => M)
    ctx[:Domain] = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => ntuple(_ -> 0.0, D), :maxs => ntuple(_ -> 1.0, D), :periodic => true), Dict{Symbol, Any}(:T => T, :D => D))
    ctx[:WeightConf] = Dict{Symbol, Any}(:name => :exponential, :alpha => 1.0, :range => 2.5)
    build_particle_grid(Dict{Symbol, Any}(:Ns => ntuple(_ -> N, D), :randomness_factor => ntuple(_ -> 0.0, D), :seed => 1), ctx)
end
function case(name, D, T, gamma, Nv, vmax, ρ, u, θ; N = 8)
    eq = build_equation(Dict{Symbol, Any}(:name => :bgk, :D => D, :gamma => gamma, :Nv => Nv, :vmax => vmax, :tau => 1.0), Dict{Symbol, Any}(:T => T))
    # E = ρ|u|²/2 + ρθ/(γ-1)
    U = SVector{D + 2, T}(T(ρ), (T(ρ) .* T.(u))..., T(ρ * sum(abs2, u) / 2 + ρ * θ / (gamma - 1)))
    pg = grid(D, N, HC.get_M(eq), T)
    logs, _ = Test.collect_test_logs(() -> set_initial_conditions!(pg, eq, ConstIC(U)); min_level = Logging.Warn)
    f, _ = equilibrium(eq, U)
    r = maximum(abs.(moments(eq, f) .- U)) / maximum(abs, U)
    println(rpad(name, 48), " r = ", rpad(string(Float64(r)), 24), " warns: ", length(logs), length(logs) > 0 ? "  [" * first(logs[1].message, 90) * "]" : "")
end
for T in (Float64, Float32)
    println("== $T")
    case("1D γ=3 Nv16 vmax6 rest", 1, T, 3.0, 16, 6.0, 1.0, (0.0,), 1.0)
    case("1D γ=3 Nv16 vmax6 wave max (ρ1.2,u.08,θ…)", 1, T, 3.0, 16, 6.0, 1.2, (0.1/1.2,), 0.8)
    case("1D γ=3 Nv16 vmax6 u=2", 1, T, 3.0, 16, 6.0, 1.0, (2.0,), 1.0)
    case("1D γ=3 Nv16 vmax6 u=3", 1, T, 3.0, 16, 6.0, 1.0, (3.0,), 1.0)
    case("1D γ=3 Nv16 vmax6 u=4", 1, T, 3.0, 16, 6.0, 1.0, (4.0,), 1.0)
    case("1D γ=3 Nv16 vmax6 u=5.5", 1, T, 3.0, 16, 6.0, 1.0, (5.5,), 1.0)
    case("1D γ=3 Nv16 vmax6 θ=0.01", 1, T, 3.0, 16, 6.0, 1.0, (0.0,), 0.01)
    case("1D γ=3 Nv16 vmax6 θ=0.01 u=0.37", 1, T, 3.0, 16, 6.0, 1.0, (0.37,), 0.01)
    case("1D γ=3 Nv16 vmax6 θ=10", 1, T, 3.0, 16, 6.0, 1.0, (0.0,), 10.0)
    case("1D γ=1.4 Nv16 vmax6 Sod L", 1, T, 1.4, 16, 6.0, 1.0, (0.0,), 1.0)
    case("1D γ=1.4 Nv16 vmax6 Sod R", 1, T, 1.4, 16, 6.0, 0.125, (0.0,), 0.8)
    case("1D γ=1.4 Nv32 vmax8 ρ=1e-6", 1, T, 1.4, 32, 8.0, 1e-6, (0.0,), 1.0)
    case("1D γ=1.4 Nv32 vmax8 ρ=1e6", 1, T, 1.4, 32, 8.0, 1e6, (0.5,), 1.0)
    case("1D γ=1.4 Nv32 vmax8 u=1e-6 (tiny momentum)", 1, T, 1.4, 32, 8.0, 1.0, (1e-6,), 1.0)
    case("2D γ=2 Nv12 vmax6 rest", 2, T, 2.0, 12, 6.0, 1.0, (0.0, 0.0), 1.0)
    case("2D γ=2 Nv12 vmax6 u=(2,1)", 2, T, 2.0, 12, 6.0, 1.0, (2.0, 1.0), 1.0)
    case("2D γ=2 Nv12 vmax6 u=(4.5,0)", 2, T, 2.0, 12, 6.0, 1.0, (4.5, 0.0), 1.0)
end
# cost: 2-D, 100x100 particles, Nv = 16 per axis
eq = build_equation(Dict{Symbol, Any}(:name => :bgk, :D => 2, :gamma => 2.0, :Nv => 16, :vmax => 6.0, :tau => 1.0), Dict{Symbol, Any}(:T => Float64))
pg = grid(2, 100, HC.get_M(eq), Float64)
ic = ConstIC(SVector(1.0, 0.3, 0.1, 0.5 * 0.1 + 1.0))
set_initial_conditions!(pg, eq, ic)
t1 = @elapsed set_initial_conditions!(pg, eq, ic)
U = ic(nothing); 
t2 = @elapsed for _ in 1:pg.meta.N; equilibrium(eq, U); end
t3 = @elapsed for _ in 1:pg.meta.N; f, _ = equilibrium(eq, U); moments(eq, f); end
println("cost 2-D 100x100, M = $(HC.get_M(eq)): set_initial_conditions! $(round(t1, digits=3)) s; equilibrium loop $(round(t2, digits=3)) s; +moments $(round(t3, digits=3)) s")
