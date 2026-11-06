using HyperCloud
d = get_master_method_dict()
println(length(d))
for k in sort(collect(keys(d))); println(k, "  ", hash(sort(collect(d[k]), by = x -> string(first(x))))); end
