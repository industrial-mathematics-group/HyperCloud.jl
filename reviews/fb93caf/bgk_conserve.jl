using HyperCloud, StaticArrays
for (u, θ) in ((5.5, 1.0), (0.0, 0.01), (0.37, 0.01))
    eq = build_equation(Dict{Symbol, Any}(:name => :bgk, :D => 1, :gamma => 3.0, :Nv => 16, :vmax => 6.0, :tau => 1.0), Dict{Symbol, Any}(:T => Float64))
    U = SVector(1.0, u, u^2/2 + θ/2)
    f, _ = equilibrium(eq, U); U1 = moments(eq, f)
    f2, _ = equilibrium(eq, U1); U2 = moments(eq, f2)
    println("u=$u θ=$θ: |moments(f) - U| = ", maximum(abs.(U1 .- U)), ";  collision at f: |moments(M[f]) - moments(f)| = ", maximum(abs.(U2 .- U1)), ", |M[f] - f| = ", maximum(abs.(f2 .- f)))
end
