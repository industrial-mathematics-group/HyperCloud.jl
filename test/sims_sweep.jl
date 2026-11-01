# Groups `sweep-a`, `sweep-b`: every method of the master dictionary on a 1-D advection, split over the two
# groups by `_sweep_cases` (each method is a new type combination, i.e. a compilation of its stepper).

@testset "Simulations" begin
    @testset "Master method dictionary sweep (1D linear advection)" begin
        # Replaces the copy-pasted per-scheme runs: every registered method must run and stay accurate.
        vels, idx = generate_relax_params(2.0, 1, 1)
        base = merge(linear_1d_gauss(200), Dict{Symbol, Any}(
            :Kinetic_velocities => vels, :Kinetic_indices => idx, :Kinetic_epsilon => 1e-6))
        cases = sort!([(k, v) for (k, v) in get_master_method_dict() if haskey(v, :Time_stepper)]; by = first)

        @testset "$name" for (name, m) in _sweep_cases(cases)
            err, sim = l1_error(merge(base, m))
            if haskey(KNOWN_BROKEN_METHODS, name)
                @test_broken sim !== nothing && err < 0.05
            else
                @test sim !== nothing
                @test err < 0.05
            end
        end
    end
end
