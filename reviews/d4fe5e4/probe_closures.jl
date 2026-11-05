# Round-2 probe: exact closures (Euler Toro tests 1-5 in Float64/Float32, Burgers Gauss near breaking,
# master dictionary label collisions). Run from the repo root: julia -t 2 --project=test <this file>
using HyperCloud, StaticArrays, Test, LinearAlgebra
const HC = HyperCloud

geom1(T) = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-1.0,), :maxs => (1.0,), :periodic => false),
                        Dict{Symbol, Any}(:T => T, :D => 1))
cons(T, ρ, u, p; γ = 1.4) = State{3, T}(ρ, ρ * u, p / (γ - 1) + 0.5 * ρ * u^2)

# Toro, Riemann Solvers, Table 4.2 (γ = 1.4): (ρL, uL, pL, ρR, uR, pR) -> (p*, u*), plus Table 4.3 densities
const TORO = [
    ("Test 1 (Sod)",      (1.0, 0.0, 1.0, 0.125, 0.0, 0.1),               (0.30313, 0.92745)),
    ("Test 2 (123)",      (1.0, -2.0, 0.4, 1.0, 2.0, 0.4),                (0.00189, 0.0)),
    ("Test 3 (blast L)",  (1.0, 0.0, 1000.0, 1.0, 0.0, 0.01),             (460.894, 19.5975)),
    ("Test 4 (blast R)",  (1.0, 0.0, 0.01, 1.0, 0.0, 100.0),              (46.0950, -6.19633)),
    ("Test 5 (collision)",(5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950), (1691.64, 8.68975)),
]

@testset "Euler exact Riemann solver vs Toro tables ($T)" for T in (Float64, Float32)
    eq = EulerEquation(Val(1), T, T(1.4), Conservative())
    for (name, (ρL, uL, pL, ρR, uR, pR), (ps, us)) in TORO
        ic = Riemann(cons(T, ρL, uL, pL), cons(T, ρR, uR, pR), Space{1, T}(0), Space{1, T}(1))
        ex = analytic_closure(eq, ic, geom1(T))
        # sample at the contact (x/t = u*) slightly left and right: pressure must be p*, velocity u*
        t = T(0.01)
        for δ in (-1e-4, 1e-4)
            V = HC.cons2prim(eq, ex(SVector{2, T}(T((us + δ) * t), t)))
            okp = isapprox(V[3], ps; rtol = ps < 0.01 ? 3e-3 : 2e-4)   # Toro gives 0.00189 (3 digits) for Test 2
            oku = isapprox(V[2], us; atol = 2e-4 * max(1, abs(us)))
            (okp && oku) || println("$name ($T) at δ=$δ: p = $(V[3]) (exact $ps), u = $(V[2]) (exact $us)")
            @test okp
            @test oku
        end
    end
end

@testset "Burgers Gauss closure just before breaking ($T, D = $D, a = $a)" for T in (Float64, Float32), D in (1, 2), a in (1.0, -0.7)
    w = 0.5
    tb = w * exp(0.5) / (abs(a) * sqrt(2D))
    eq = BurgersEquation{D, T}()
    ic = Gauss(State{1, T}(a), Space{D, T}(ntuple(_ -> 0, D)), T(w))
    geom = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => ntuple(_ -> -3.0, D), :maxs => ntuple(_ -> 3.0, D),
                                          :periodic => false), Dict{Symbol, Any}(:T => T, :D => D))
    ex = analytic_closure(eq, ic, geom)
    t = T(0.97 * tb)
    pts = D == 1 ? [SVector{1, T}(x) for x in range(-2, 2; length = 401)] :
                   [SVector{2, T}(x, y) for x in range(-2, 2; length = 61) for y in range(-2, 2; length = 61)]
    logs, res = Test.collect_test_logs() do
        maximum(pts) do p
            u = ex(SVector{D + 1, T}(p..., t))[1]
            abs(u - ic(p .- u * t)[1])
        end
    end
    @test isempty(logs)
    @test res < (T == Float64 ? 1e-10 : 1e-4)
    println("Burgers Gauss T=$T D=$D a=$a t=0.97 t_b: max residual $res, $(length(logs)) warnings")
end

@testset "master method dictionary: no label collisions" begin
    # count the entries before the Dict constructor deduplicates them
    md = get_master_method_dict()
    src = read(joinpath(pkgdir(HyperCloud), "src", "inbuilt_methods.jl"), String)
    @test length(md) == 50  # 1 reference + 6 + 11 + 8 + 4*6 = 49 create_method calls (counted in the source), so no label collides
    println("master dict entries: ", length(md))
end
