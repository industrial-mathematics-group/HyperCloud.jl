using HyperCloud
conf = euler_1d_piston()
for p in generate_tasks(conf)
    println(HyperCloud.PDEStudioCore._param_hash(p))
end
println(typeof(conf.shared[:Grid_shape]))
