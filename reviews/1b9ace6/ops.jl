using HyperCloud, StaticArrays, Printf, LinearAlgebra
const HC = HyperCloud

function bgk(D, Nv; vmax = 6.0, center = ntuple(_ -> 0.0, D), kind = :uniform, T = Float64, γ = 1.4)
    nodes, weights = velocity_grid(kind, Nv, vmax, T)
    BGKEquation(Val(D), nodes, weights, γ, ConstantRelaxationTime(1e-2); center = center)
end

# Half-range wall flux (incident from f, emitted from g) of mass, momentum, energy; and the "central" flux of the
# full ghost: Σ w c (f + g)/2
function wallflux(eq, f, g, n, w; full = false)
    D = length(n); T = eltype(n); NV = HC.num_velocities(eq)
    F = zero(SVector{D + 2, T})
    for k in 1:NV
        v, wk = velocity_node(eq, k)
        c = sum((v - w) .* n)
        val = full ? (f[k] + g[k]) / 2 : (c > 0 ? f[k] : g[k])
        hv  = HC._has_h(eq) ? (full ? (f[k + NV] + g[k + NV]) / 2 : (c > 0 ? f[k + NV] : g[k + NV])) : zero(T)
        e = T(0.5) * sum(abs2, v) * val + hv
        F += wk * c * SVector{D + 2, T}(val, (val * v)..., e)
    end
    F
end
# moments of the exact specular reflection of f's macroscopic state
function exact_reflect(eq, f, n, w)
    U = moments(eq, f); D = length(n)
    u = SVector{D}(U[2:D+1]) / U[1]; ur = u - 2 * sum((u - w) .* n) * n
    SVector(U[1], (U[1] * ur)..., U[end] + 0.5U[1] * (sum(abs2, ur) - sum(abs2, u)))
end
tang(F, n) = (D = length(n); m = SVector{D}(F[2:D+1]); norm(m - sum(m .* n) * n))

function report(lbl, eq, f, n, w; op = SlipWall())
    T = eltype(n); D = length(n)
    wp = WallPoint(zero(SVector{D, T}), n, T(0.05), w)
    g = op(eq, f, wp)
    F = wallflux(eq, f, g, n, w); Fc = wallflux(eq, f, g, n, w; full = true)
    Ue = exact_reflect(eq, f, n, w); G = moments(eq, g)
    D2 = length(n); ρg, ug = G[1], SVector{D2}(G[2:D2+1]) / G[1]; eg = G[end] / G[1] - 0.5sum(abs2, ug)
    ρe, ue = Ue[1], SVector{D2}(Ue[2:D2+1]) / Ue[1]; ee = Ue[end] / Ue[1] - 0.5sum(abs2, ue)
    @printf("%-46s mass flux %9.2e | central %9.2e | tang mom flux %9.2e | energy flux %9.2e | ghost vs exact refl.: dρ %8.1e du %8.1e de_int/e %8.1e | min g/max f %9.2e\n",
            lbl, F[1], Fc[1], tang(F, n), F[end], ρg - ρe, norm(ug - ue), eg / ee - 1, minimum(g) / maximum(f))
    return g
end

println("== Equilibrium states (drifting)")
eq1 = bgk(1, 16); f1, _ = equilibrium(eq1, SVector(1.0, 0.3, 2.5 + 0.5 * 0.09))
report("1D static on-axis (flip)", eq1, f1, SVector(1.0), SVector(0.0))
for wv in (0.5, 0.2, 0.375, 1e-12)
    report("1D moving w=$wv", eq1, f1, SVector(1.0), SVector(wv))
end
report("1D moving w=-0.5, n=-1", eq1, f1, SVector(-1.0), SVector(-0.5))
for c in (0.5, 0.3)
    eqc = bgk(1, 16; center = (c,)); fc, _ = equilibrium(eqc, SVector(1.0, 0.0, 2.5))
    report("1D vcenter=$c gas at rest static", eqc, fc, SVector(1.0), SVector(0.0))
    report("1D vcenter=$c wall moving at $c (flip)", eqc, fc, SVector(1.0), SVector(c))
end
eqc = bgk(1, 64; center = (0.5,)); fc, _ = equilibrium(eqc, SVector(1.0, 0.2, 2.5 + 0.02))
report("1D vcenter=0.5, Nv=64, drift 0.2", eqc, fc, SVector(1.0), SVector(0.0))
eqg = bgk(1, 16; kind = :gauss_hermite); fg, _ = equilibrium(eqg, SVector(1.0, 0.3, 2.5 + 0.045))
report("1D Gauss-Hermite 16 moving w=0.5", eqg, fg, SVector(1.0), SVector(0.5))
eq2 = bgk(2, 12); f2, _ = equilibrium(eq2, SVector(1.0, 0.2, -0.1, 2.5 + 0.5 * 0.05))
for θ in (0.0, π / 8, π / 6, 0.3, π / 4, 1.0)
    report(@sprintf("2D Nv=12 oblique θ=%.3f", θ), eq2, f2, SVector(cos(θ), sin(θ)), SVector(0.0, 0.0))
end
report("2D Nv=12 θ=0.3 moving w=(0.2,0.1)", eq2, f2, SVector(cos(0.3), sin(0.3)), SVector(0.2, 0.1))
eq2b = bgk(2, 24); f2b, _ = equilibrium(eq2b, SVector(1.0, 0.2, -0.1, 2.5 + 0.5 * 0.05))
report("2D Nv=24 oblique θ=0.3", eq2b, f2b, SVector(cos(0.3), sin(0.3)), SVector(0.0, 0.0))
eq2g = bgk(2, 12; kind = :gauss_hermite); f2g, _ = equilibrium(eq2g, SVector(1.0, 0.2, -0.1, 2.5 + 0.5 * 0.05))
report("2D GH Nv=12 oblique θ=0.3", eq2g, f2g, SVector(cos(0.3), sin(0.3)), SVector(0.0, 0.0))
eq3 = bgk(3, 8; γ = 5 / 3); f3, ok3 = equilibrium(eq3, SVector(1.0, 0.2, -0.1, 0.05, 1.5 + 0.5 * 0.0525))
n3 = normalize(SVector(1.0, 0.5, 0.3))
report("3D Nv=8 γ=5/3 oblique", eq3, f3, n3, SVector(0.0, 0.0, 0.0))
eqf = bgk(2, 12; T = Float32); ff, _ = equilibrium(eqf, SVector{4, Float32}(1, 0.2, -0.1, 2.5 + 0.025))
report("2D Float32 oblique θ=0.3", eqf, ff, SVector{2, Float32}(cos(0.3), sin(0.3)), SVector{2, Float32}(0, 0))
# fast drift near vmax: equilibrium not representable
fq, _ = equilibrium(eq2, SVector(1.0, 3.5, 0.0, 2.5 + 0.5 * 3.5^2))
report("2D Nv=12 fast drift u=3.5 θ=0.3", eq2, fq, SVector(cos(0.3), sin(0.3)), SVector(0.0, 0.0))

println("\n== Non-equilibrium states")
# two beams in 1D (narrow Maxwellians at ±1.5) on a shifted grid / moving wall
beam(eq, u0, θ0) = equilibrium(eq, SVector(0.5, 0.5u0, 0.5 * (0.5u0^2 + 0.5 * 5 * θ0)))[1]
fb = beam(eq1, 1.5, 0.1) + beam(eq1, -1.5, 0.1)
report("1D two-beam static flip", eq1, fb, SVector(1.0), SVector(0.0))
report("1D two-beam moving w=0.2", eq1, fb, SVector(1.0), SVector(0.2))
eqc = bgk(1, 16; center = (0.5,))
fbc = beam(eqc, 1.5, 0.1) + beam(eqc, -1.5, 0.1)
report("1D two-beam vcenter=0.5 static", eqc, fbc, SVector(1.0), SVector(0.0))
# hot/cold half Maxwellians in 1D: θ=2 for v>0 (towards the wall), θ=0.5 for v<0
fh = equilibrium(eq1, SVector(1.0, 0.0, 0.5 * 5 * 2.0))[1]; fcld = equilibrium(eq1, SVector(1.0, 0.0, 0.5 * 5 * 0.5))[1]
fhc = State{length(fh), Float64}(ntuple(k -> (kk = k > 16 ? k - 16 : k; velocity_node(eq1, kk)[1][1] > 0 ? fh[k] : fcld[k]), length(fh)))
report("1D hot/cold half-Maxwellian moving w=0.2", eq1, fhc, SVector(1.0), SVector(0.2))
report("1D hot/cold half-Maxwellian moving w=0.5", eq1, fhc, SVector(1.0), SVector(0.5))
# 2D two beams, oblique
fb2 = equilibrium(eq2, SVector(0.5, 0.75, 0.0, 0.5 * (0.5 * 2.25 + 0.5 * 5 * 0.15)))[1] +
      equilibrium(eq2, SVector(0.5, 0.0, 0.75, 0.5 * (0.5 * 2.25 + 0.5 * 5 * 0.15)))[1]
for θ in (0.3, π / 6)
    report(@sprintf("2D two-beam oblique θ=%.3f", θ), eq2, fb2, SVector(cos(θ), sin(θ)), SVector(0.0, 0.0))
end
# 2D hot/cold half in the wall normal direction
fh2 = equilibrium(eq2, SVector(1.0, 0.0, 0.0, 0.5 * 5 * 2.0))[1]; fc2 = equilibrium(eq2, SVector(1.0, 0.0, 0.0, 0.5 * 5 * 0.5))[1]
n = SVector(cos(0.3), sin(0.3))
fhc2 = State{length(fh2), Float64}(ntuple(k -> (kk = k > 144 ? k - 144 : k; sum(velocity_node(eq2, kk)[1] .* n) > 0 ? fh2[k] : fc2[k]), length(fh2)))
report("2D hot/cold half oblique θ=0.3", eq2, fhc2, n, SVector(0.0, 0.0))
# a sparse distribution: one node with all the mass, others 1e-8 (strongly non-equilibrium)
fs = State{288, Float64}(ntuple(k -> k == 7 + 12 * 5 ? 10.0 : (k == 7 + 12 * 5 + 144 ? 10.0 : 1e-8), 288))
report("2D single-node f oblique θ=0.3", eq2, fs, n, SVector(0.0, 0.0))
# negative ρ (equilibrium fails): fallback
fneg = -f2
report("2D f = -feq (equilibrium fails: fallback)", eq2, fneg, n, SVector(0.0, 0.0))

println("\n== DiffuseWall α < 1 off-axis (specular part)")
for α in (0.0, 0.5, 1.0)
    report("2D diffuse α=$α θ=0.3 drifting eq", eq2, f2, n, SVector(0.0, 0.0); op = DiffuseWall(1.0; accommodation = α))
    report("1D diffuse α=$α moving w=0.5 vcenter 0", eq1, f1, SVector(1.0), SVector(0.5); op = DiffuseWall(1.0; accommodation = α))
end

println("\n== Continuity of the switch (w → grid centre)")
for wv in (0.0, 1e-14, 1e-8, 1e-4)
    g = SlipWall()(eq1, fb, WallPoint(SVector(0.0), SVector(1.0), 0.05, SVector(wv)))
    g0 = SlipWall()(eq1, fb, WallPoint(SVector(0.0), SVector(1.0), 0.05, SVector(0.0)))
    @printf("w=%g: max |g - g(w=0)| = %.3e\n", wv, maximum(abs.(g - g0)))
end

println("\n== Allocation and cost per ghost")
function cost(lbl, eq, f, n, w)
    T = eltype(n); D = length(n)
    wp = WallPoint(zero(SVector{D, T}), n, T(0.05), w)
    s = SlipWall(); s(eq, f, wp)
    a = @allocated s(eq, f, wp)
    N = 200
    t = @elapsed for _ in 1:N; s(eq, f, wp); end
    @printf("%-40s M=%4d alloc %d B, %.1f µs per ghost\n", lbl, length(f), a, t / N * 1e6)
end
cost("1D Nv=16 flip", eq1, f1, SVector(1.0), SVector(0.0))
cost("1D Nv=16 moving", eq1, f1, SVector(1.0), SVector(0.5))
cost("2D Nv=12 oblique", eq2, f2, n, SVector(0.0, 0.0))
cost("2D Nv=24 oblique", eq2b, f2b, SVector(cos(0.3), sin(0.3)), SVector(0.0, 0.0))
cost("3D Nv=8 oblique", eq3, f3, n3, SVector(0.0, 0.0, 0.0))
cost("2D Float32 oblique", eqf, ff, SVector{2, Float32}(cos(0.3), sin(0.3)), SVector{2, Float32}(0, 0))
wp = WallPoint(SVector(0.0, 0.0), n, 0.05, SVector(0.0, 0.0))
t = @elapsed for _ in 1:200; equilibrium(eq2b, moments(eq2b, f2b)); end
@printf("one equilibrium() 2D Nv=24: %.1f µs\n", t / 200 * 1e6)
