using Serialization, StaticArrays, HyperCloud
a = deserialize(ARGS[1]); b = deserialize(ARGS[2])
for k in sort(collect(keys(a)))
    ra, rb = a[k], get(b, k, nothing)
    if ra === nothing || rb === nothing; println(rpad(k, 40), " one failed: ", ra === nothing, " ", rb === nothing); continue; end
    same_t = ra.t == rb.t
    if length(ra.u) != length(rb.u); println(rpad(k, 40), " N differs ", length(ra.u), " vs ", length(rb.u)); continue; end
    d = maximum(i -> maximum(abs.(ra.u[i] .- rb.u[i])), eachindex(ra.u))
    sc = maximum(i -> maximum(abs.(ra.u[i])), eachindex(ra.u))
    println(rpad(k, 40), " t equal: ", same_t, "  max|Δu| = ", d, "  (rel ", d / sc, ")", d == 0 ? "  IDENTICAL" : "")
end
