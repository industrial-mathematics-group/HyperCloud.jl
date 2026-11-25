# What identifies HyperCloud's values in a run's parameters (`src/param_identity.jl`). Needs PDEStudioCore with
# `param_identity` (0.9); with an earlier version nothing is declared and the testset is skipped.
@testset "Parameter identity of configuration values" begin
    if !isdefined(PDEStudioCore, :param_identity)
        @test_skip "PDEStudioCore $(pkgversion(PDEStudioCore)) has no param_identity"
    else
        HCC = HyperCloudCore
        text(v) = val2str(get_clean_params(Dict(:v => v))[:v])
        hash_of(v) = calculate_hash(Dict(:v => v))

        # Boundary conditions with their defaults keep the stored form of earlier versions (and so the names of the
        # stored runs): these texts are the ones in existing data folders
        bcs = Dict{Int, AbstractBoundaryCondition}(1 => FixedDirichlet(), 2 => OutflowBC())
        @test text(bcs) == "Dict{Int64, AbstractBoundaryCondition}(1 => FixedDirichlet, 2 => OutflowBC)"
        @test text(slip_wall()) == "MirrorBC(op=SlipWall)"
        @test text(diffuse_wall(300.0)) == "MirrorBC(op=DiffuseWall(temperature=300.0, velocity=Nothing, accommodation=1.0))"
        # ... and what isn't a default tells runs apart: the degree and boundedness of a mirror, an extrapolation's degree
        @test text(MirrorBC(SlipWall(); degree = 2)) == "MirrorBC(op=SlipWall, degree=2, bounded=true)"
        @test hash_of(MirrorBC(SlipWall(); bounded = false)) != hash_of(slip_wall())
        @test text(ExtrapolationBC(1)) == "ExtrapolationBC{1}" && hash_of(ExtrapolationBC(1)) != hash_of(ExtrapolationBC(2))

        # Shapes are data; a domain is identified by its inputs, not by its functions or derived fields
        shape = HCC.Cuboid((0.0, 0.0), (1.0, 1.0))
        @test text(shape) == "Cuboid(mins=Float64[0.0, 0.0], maxs=Float64[1.0, 1.0], open=Bool[false, false])"
        walls = Dict{Int, AbstractBoundaryCondition}(1 => slip_wall())
        domain(s) = HCC.GeometricDomain(s; bc_map = walls)
        @test hash_of(domain(shape)) == hash_of(domain(HCC.Cuboid((0.0, 0.0), (1.0, 1.0))))   # (built again)
        @test hash_of(domain(shape)) != hash_of(domain(HCC.Cuboid((0.0, 0.0), (1.0, 2.0))))
        @test !occursin("volume", text(domain(shape))) && !occursin("is_interior", text(domain(shape)))
        # unset fields keep their stored form (`Nothing`), also a domain without a shape
        box = HCC.GeometricDomain(:box, HCC.SVector(0.0), HCC.SVector(1.0), HCC.SVector(false), x -> true, x -> 1, walls)
        @test occursin("shape=Nothing", text(box)) && calculate_hash(Dict(:Grid_domain => box)) isa String
        @test text(diffuse_wall(300.0; velocity = (0.1,))) == "MirrorBC(op=DiffuseWall(temperature=300.0, velocity=Float64[0.1], accommodation=1.0))"

        # A moving piece whose motion is a function can't be compared: refused; the string form is data
        moving = HCC.Moving(HCC.Tagged(HCC.HalfSpace((1.0,), (1.0,)), 2), HCC.translating((-0.5,)))
        @test_throws r":Grid_shape.motion is a function" calculate_hash(Dict(:Grid_shape => moving))
        @test calculate_hash(Dict(:Grid_shape => "Moving(Tagged(HalfSpace((T(1),), (T(1),)), 2), translating((-T(0.5),)))")) isa String

        # Every run of the shipped examples is made of data or declared values
        for config in (advection_2d(), burgers_kinetic(), euler_1d_sod(), bgk_1d_sod(), euler_1d_piston())
            @test all(t -> calculate_hash(t) isa String, generate_tasks(config))
        end
    end
end
