using HyperCloud, StaticArrays, Printf
const HC = HyperCloud
println("== Euler SlipWall")
for (D,T) in ((1,Float64),(2,Float32),(3,Float32),(3,Float64))
    eq = EulerEquation(Val(D), T)
    n = SVector{D,T}(ntuple(d->T(1/sqrt(D)), D))
    w = SVector{D,T}(ntuple(d->T(0.1d), D))
    V = State{D+2,T}(T(1.3), ntuple(d->T(0.2d-0.3), D)..., T(0.9))
    U = prim2cons(eq, V)
    wp = WallPoint(zero(SVector{D,T}), n, T(0.1), w)
    G = SlipWall()(eq, U, wp)
    Vg = cons2prim(eq, G)
    u = SVector{D,T}(V[2:D+1]...); ug = SVector{D,T}(Vg[2:D+1]...)
    @printf("D=%d %s type=%s  (u+ug)/2·n=%.3e w·n=%.3e  tang diff=%.2e rho,p kept: %s alloc=%d\n", D, T, eltype(G),
        sum((u+ug)/2 .* n), sum(w .* n), maximum(abs.((u - sum(u.*n)*n) - (ug - sum(ug.*n)*n))), (Vg[1]≈V[1] && Vg[end]≈V[end]),
        @allocated SlipWall()(eq, U, wp))
end
println("== Linear advection / Burgers")
for mk in (() -> HC.build_equation(Dict{Symbol,Any}(:name=>:linear_advection, :D=>1), Dict{Symbol,Any}(:T=>Float64)),)
    try; eq = mk(); println(typeof(eq)); SlipWall()(eq, State{1,Float64}(1.0), WallPoint(SVector(0.0),SVector(1.0),0.1,SVector(0.0))); catch e; println("err: ", sprint(showerror, e)[1:min(200,end)]); end
end

println("== BGK SlipWall / DiffuseWall")
function bgk(D, Nv; vmax=6.0, center=ntuple(_->0.0,D), kind=:uniform, T=Float64, γ=1.4)
    nodes, weights = velocity_grid(kind, Nv, vmax, T)
    BGKEquation(Val(D), nodes, weights, γ, ConstantRelaxationTime(1e-2); center=center)
end
# flux of moments through the wall (relative to w): sum_k w_k ((v-w)·n) ψ_k g_k for the ghost combined with the mirrored? Use the half-range construction:
# net flux at the wall = sum over c>0 of the mirrored f + sum over c<0 of the ghost
function wallflux(eq, f, g, n, w)
    D = length(n); T = eltype(n)
    NV = HC.num_velocities(eq)
    F = zero(SVector{D+2,T})
    for k in 1:NV
        v, wk = velocity_node(eq, k)
        c = sum((v - w) .* n)
        val = c > 0 ? f[k] : g[k]
        e = T(0.5)*sum(abs2, v)*val + (HC._has_h(eq) ? (c > 0 ? f[k+NV] : g[k+NV]) : zero(T))
        F += wk * c * SVector{D+2,T}(val, (val*v)..., e)
    end
    F
end
eq = bgk(1, 16)
f, _ = equilibrium(eq, SVector(1.0, 0.3, 2.5+0.5*0.09))
for (lbl, wv) in (("static", 0.0), ("w=0.5", 0.5), ("w=0.2", 0.2))
    w = SVector(wv); n = SVector(1.0)
    g = SlipWall()(eq, f, WallPoint(SVector(1.0), n, 0.05, w))
    Fl = wallflux(eq, f, g, n, w)
    @printf("1D slip %s: mass flux rel wall %.3e (should be 0), ghost moments %s\n", lbl, Fl[1] - 0, string(round.(moments(eq,g), digits=4)))
    # effective reflected normal velocity: mean velocity of the mirrored+ghost combination
end
# vcenter shifted grid
for c in (0.5, 0.3)
    eqc = bgk(1, 16; center=(c,))
    f, ok = equilibrium(eqc, SVector(1.0, 0.0, 2.5))
    n = SVector(1.0); w = SVector(0.0)
    g = SlipWall()(eqc, f, WallPoint(SVector(1.0), n, 0.05, w))
    gd = DiffuseWall(1.0)(eqc, f, WallPoint(SVector(1.0), n, 0.05, w))
    @printf("vcenter=%.2f: slip mass flux %.3e mom of ghost %.4f (mirrored %.4f); diffuse |g-f|=%.3e mass flux %.3e\n", c, wallflux(eqc,f,g,n,w)[1], moments(eqc,g)[2], moments(eqc,f)[2], maximum(abs.(gd-f)), wallflux(eqc,f,gd,n,w)[1])
end
# 2-D oblique normals
eq2 = bgk(2, 12)
f2, ok = equilibrium(eq2, SVector(1.0, 0.2, -0.1, 2.5+0.5*0.05))
@show ok
for θ in (0.0, π/8, π/6, π/4, 0.3)
    n = SVector(cos(θ), sin(θ)); w = SVector(0.0, 0.0)
    g = SlipWall()(eq2, f2, WallPoint(SVector(0.0,0.0), n, 0.05, w))
    F = wallflux(eq2, f2, g, n, w)
    mg, mf = moments(eq2, g), moments(eq2, f2)
    # exact specular: rho, E kept; momentum reflected
    uf = SVector(mf[2], mf[3]); ue = uf - 2*sum(uf .* n)*n
    @printf("2D slip θ=%.3f: mass flux %.3e, ρ ghost/mirr %.5f, E ratio %.5f, mom err vs exact reflect %.3e, distinct targets %d of %d\n", θ, F[1], mg[1]/mf[1], mg[4]/mf[4], maximum(abs.(SVector(mg[2],mg[3]) - ue)),
        length(unique([HC._nearest_node(eq2, (v = velocity_node(eq2,k)[1]; v - 2*sum(v.*n)*n)) for k in 1:HC.num_velocities(eq2)])), HC.num_velocities(eq2))
    gd = DiffuseWall(1.0)(eq2, f2, WallPoint(SVector(0.0,0.0), n, 0.05, w))
    gd5 = DiffuseWall(1.0; accommodation=0.5)(eq2, f2, WallPoint(SVector(0.0,0.0), n, 0.05, w))
    @printf("    diffuse mass flux %.3e, α=0.5 mass flux %.3e\n", wallflux(eq2,f2,gd,n,w)[1], wallflux(eq2,f2,gd5,n,w)[1])
end
# Diffuse moving wall, 1D
f, _ = equilibrium(eq, SVector(1.0, 0.0, 2.5))
for wv in (0.0, 0.5, -0.5), α in (1.0, 0.5, 0.0)
    w = SVector(wv); n = SVector(1.0)
    g = DiffuseWall(1.0; accommodation=α)(eq, f, WallPoint(SVector(1.0), n, 0.05, w))
    F = wallflux(eq, f, g, n, w)
    @printf("1D diffuse w=%.1f α=%.1f: mass flux rel wall %.3e\n", wv, α, F[1])
end
# gauss-hermite symmetry
eqg = bgk(1, 16; kind=:gauss_hermite)
f, _ = equilibrium(eqg, SVector(1.0, 0.0, 2.5))
g = DiffuseWall(1.0)(eqg, f, WallPoint(SVector(1.0), SVector(1.0), 0.05, SVector(0.0)))
@printf("GH diffuse equilibrium |g-f| = %.3e; nodes asym %.3e\n", maximum(abs.(g-f)), maximum(abs.(eqg.nodes .+ reverse(eqg.nodes))))
# Float32
eqf = bgk(1, 16; T=Float32)
f, _ = equilibrium(eqf, SVector{3,Float32}(1, 0, 2.5))
wp = WallPoint(SVector(1f0), SVector(1f0), 0.05f0, SVector(0f0))
g = DiffuseWall(1.0)(eqf, f, wp); gs = SlipWall()(eqf, f, wp)
println("F32 diffuse eltype ", eltype(g), " |g-f| ", maximum(abs.(g-f)), " slip ", eltype(gs), " alloc ", (@allocated DiffuseWall(1.0)(eqf, f, wp)), " ", (@allocated SlipWall()(eqf, f, wp)))
# velocity length mismatch
try; DiffuseWall(1.0; velocity=(0.1,0.2))(eq, f, WallPoint(SVector(1.0), SVector(1.0), 0.05, SVector(0.0))) ; println("no error for length-2 velocity in 1-D"); catch e; println("velocity length mismatch: ", typeof(e)); end
println("isbits DiffuseWall: ", isbits(DiffuseWall(1.0; velocity=(0.1,))), " ", isbits(DiffuseWall(1.0)))
