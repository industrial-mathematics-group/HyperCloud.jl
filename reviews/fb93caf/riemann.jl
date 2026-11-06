# Exact Riemann solver probe: closure (whatever version is loaded) + instrumented replicas of the old and new Newton loops
using HyperCloud, StaticArrays, Printf
const HC = HyperCloud
function newton(T, ρL, uL, pL, ρR, uR, pR; new::Bool, γ = T(1.4))
    cL = sqrt(γ*pL/ρL); cR = sqrt(γ*pR/ρR)
    pf(p) = begin
        fL = p > pL ? (p - pL) * sqrt((T(2) / ((γ + 1) * ρL)) / (p + pL * (γ - 1) / (γ + 1))) : (T(2) * cL / (γ - 1)) * ((p / pL)^((γ - 1) / (T(2) * γ)) - 1)
        fR = p > pR ? (p - pR) * sqrt((T(2) / ((γ + 1) * ρR)) / (p + pR * (γ - 1) / (γ + 1))) : (T(2) * cR / (γ - 1)) * ((p / pR)^((γ - 1) / (T(2) * γ)) - 1)
        fL + fR + (uR - uL)
    end
    vac = T(2)*(cL+cR)/(γ-1) <= uR - uL
    us = abs(uL)+abs(uR)+cL+cR; pfl = eps(T)*min(pL,pR)
    p = T(0.5)*(pL+pR); it = 0; why = :maxit
    for _ in 1:(vac ? 0 : 100)
        f = pf(p)
        if new
            abs(f) <= 16*eps(T)*us && (why = :tol; break)
        else
            abs(f) < T(1e-9) && (why = :tol; break)
        end
        d = (pf(p*T(1.001)) - f)/(p*T(0.001))
        if new
            d > 0 || (why = :slope; break)
            p = max(pfl, p - f/d)
        else
            p = max(T(1e-9), p - f/(d + T(1e-9)))
        end
        it += 1
    end
    return p, it, why, abs(pf(p))/us
end
cons(T, ρ, u, p) = State{3, T}(ρ, ρ*u, p/T(0.4) + T(0.5)*ρ*u^2)
pres(U) = 0.4*(U[3] - U[2]^2/(2U[1]))
geomf(T) = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-1.0,), :maxs => (1.0,), :periodic => false), Dict{Symbol, Any}(:T => T, :D => 1))
function closure_pstar(T, s, us_ref)
    eq = EulerEquation(Val(1), T, T(1.4), Conservative())
    ex = analytic_closure(eq, Riemann(cons(T, s[1:3]...), cons(T, s[4:6]...), Space{1,T}(0), Space{1,T}(1)), geomf(T))
    t = T(1e-2); U = ex(SVector{2,T}(T(us_ref*t), t))   # at the contact
    return pres(U), U, ex
end
toro = [("T1", (1.0, 0.0, 1.0, 0.125, 0.0, 0.1), 0.30313, 0.92745),
        ("T2", (1.0, -2.0, 0.4, 1.0, 2.0, 0.4), 0.00189, 0.0),
        ("T3", (1.0, 0.0, 1000.0, 1.0, 0.0, 0.01), 460.894, 19.5975),
        ("T4", (1.0, 0.0, 0.01, 1.0, 0.0, 100.0), 46.0950, -6.19633),
        ("T5", (5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950), 1691.64, 8.68975)]
extra = [("nearvac", (1.0, -3.7416, 1.0, 1.0, 3.7416, 1.0), NaN, 0.0),      # 2c/(γ-1) = 3.7417: barely no vacuum
         ("nearvac2", (1.0, -3.74165, 1.0, 1.0, 3.74165, 1.0), NaN, 0.0),
         ("strong1e10", (1.0, 0.0, 1e10, 1.0, 0.0, 1.0), NaN, NaN),
         ("strong1e-10", (1.0, 0.0, 1e-10, 1.0, 0.0, 1.0), NaN, NaN),
         ("collide", (1.0, 100.0, 1.0, 1.0, -100.0, 1.0), NaN, 0.0),
         ("dens1e6", (1e6, 0.0, 1.0, 1.0, 0.0, 1.0), NaN, NaN),
         ("equal", (1.0, 0.3, 1.0, 1.0, 0.3, 1.0), 1.0, 0.3)]
for T in (Float64, Float32)
    println("==== $T")
    for (ps, vs) in ((1.0, 1.0), (1e-8, 1e-4), (1e8, 1e4), (1e-20, 1e-10))
        for (name, s, pref, uref) in vcat(toro, extra)
            sc = T.((s[1], vs*s[2], ps*s[3], s[4], vs*s[5], ps*s[6]))
            all(isfinite, sc) && all(x -> x == 0 || abs(x) > floatmin(T), sc) || continue
            pn, itn, whyn, rn = newton(T, sc...; new = true)
            po, ito, whyo, ro = newton(T, sc...; new = false)
            ustar = isnan(uref) ? 0.0 : vs*uref
            pc = try closure_pstar(T, sc, ustar)[1] catch e; NaN end
            relerr(p) = isnan(pref) ? NaN : abs(p - ps*pref)/(ps*pref)
            @printf("%-12s scale p%.0e: new p*=%.6e it=%3d %-5s res=%.1e err=%.1e | old p*=%.6e it=%3d %-5s err=%.1e | closure p(contact)=%.6e\n",
                    name, ps, pn, itn, whyn, rn, relerr(pn), po, ito, whyo, relerr(po), pc)
        end
    end
end
# t tiny but positive (removed 1e-9 cut-off)
for T in (Float64, Float32)
    eq = EulerEquation(Val(1), T, T(1.4), Conservative())
    ex = analytic_closure(eq, Riemann(cons(T, 1.0, 0.0, 1.0), cons(T, 0.125, 0.0, 0.1), Space{1,T}(0), Space{1,T}(1)), geomf(T))
    for t in (0.0, 1e-300, 1e-40, 1e-12, 1e-9), x in (-0.1, 0.0, 1e-12, 0.1)
        U = ex(SVector{2,T}(T(x), T(t)))
        println("$T t=$t x=$x -> ", U, all(isfinite, U) ? "" : "  NON-FINITE")
    end
end
