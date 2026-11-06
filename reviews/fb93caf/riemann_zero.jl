using HyperCloud, StaticArrays
cons(ρ, u, p) = State{3, Float64}(ρ, ρ*u, p/0.4 + 0.5*ρ*u^2)
geom = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-1.0,), :maxs => (1.0,), :periodic => false), Dict{Symbol, Any}(:T => Float64, :D => 1))
eq = EulerEquation(Val(1), Float64, 1.4, Conservative())
for (nm, L, R) in (("pL=0 pR=1", (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)), ("pL=1 pR=0", (1.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
                   ("Noh p=0", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0)), ("Noh p=1e-6", (1.0, 1.0, 1e-6), (1.0, -1.0, 1e-6)),
                   ("pL=0 colliding", (1.0, 1.0, 0.0), (1.0, -1.0, 1.0)), ("pL=1e-300", (1.0, 0.0, 1e-300), (1.0, 0.0, 1.0)))
    r = try
        ex = analytic_closure(eq, Riemann(cons(L...), cons(R...), Space{1}(0.0), Space{1}(1.0)), geom)
        vals = [ex(SVector(x, 0.1)) for x in (-0.05, -0.01, 0.0, 0.01, 0.05)]
        string(all(v -> all(isfinite, v), vals) ? "finite" : "NON-FINITE", "  U(0, 0.1) = ", ex(SVector(0.0, 0.1)))
    catch e
        "throws " * sprint(showerror, e)[1:min(end, 80)]
    end
    println(rpad(nm, 18), r)
end
