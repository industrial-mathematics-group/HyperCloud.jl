using HyperCloud
conf = euler_1d_piston()
p = first(generate_tasks(conf))
h(p) = HyperCloud.PDEStudioCore._param_hash(p)
h0 = h(p)
for w in (0.5, 0.4)
    piston = Moving(Tagged(HalfSpace((1.0,), (1.0,)), 2), translating((-w,)))
    q = copy(p); q[:Grid_shape] = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), piston)
    println("w=$w same hash as example: ", h(q) == h0)
end
for xf in (1.0, 0.9)
    piston = Moving(Tagged(HalfSpace((xf,), (1.0,)), 2), translating((-0.5,)))
    q = copy(p); q[:Grid_shape] = intersect(Tagged(Cuboid((0.0,), (1.5,)), 1), piston)
    println("face=$xf same hash: ", h(q) == h0)
end
println(repr(p[:Grid_shape])[1:min(400,end)])
s = repr(p[:Grid_shape]); println(s[400:end])
