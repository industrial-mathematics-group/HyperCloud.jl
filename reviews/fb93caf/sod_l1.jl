using Serialization, StaticArrays, HyperCloud
sod = euler_1d_sod()
p = merge(sod.shared, sod.methods[:RK2Upwind1])
ex = analytical_solution(p)
for f in ARGS
    r = deserialize(f)
    println("== ", f)
    for k in sort(filter(k -> startswith(k, "sod RK"), collect(keys(r))))
        d = r[k]; t = Float64(d.t[end])
        l1 = sum(i -> abs(ex(SVector(Float64(d.x[i][1]), t))[1] - d.u[i][1]), eachindex(d.x)) / length(d.x)
        println(rpad(k, 32), " L1(rho) = ", round(l1, sigdigits = 6))
    end
end
