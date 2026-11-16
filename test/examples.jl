# Group `examples`: the exported examples, and the plotter when the GUI tests are on.
#
# Every example runs through the pipeline (`run_all_simulations`, as `@sim` does) with a short `tmax`, in a
# temporary save directory: results cached in the working directory would otherwise be reused, so the examples
# would never run again. Every task must store data that reaches `tmax` and stays close to the reference.

# (example, tmax, overrides of the shared parameters, bound on the mean error of component 1 vs the reference).
# Measured: advection 0.0175, Burgers 4.3e-4 (direct) / 9.1e-4 (kinetic), Sod 0.006-0.012, BGK 0.011-0.015.
const EXAMPLE_CHECKS = [
    (advection_2d,    0.5,  Dict{Symbol, Any}(:Grid_Ns => (30, 30)), 0.035),
    (burgers_kinetic, 0.5,  Dict{Symbol, Any}(),                     0.003),
    (euler_1d_sod,    0.05, Dict{Symbol, Any}(),                     0.025),
    (bgk_1d_sod,      0.05, Dict{Symbol, Any}(:Grid_Ns => (165,)),   0.03),
    (euler_1d_piston, 0.1,  Dict{Symbol, Any}(:Grid_Ns => (150,)),   0.03),   # the shock is smeared over a few particles
]

"Mean absolute error of component 1 of the last stored snapshot against `exact`."
function final_error(exact, sim)
    xs, us, t = sim.x[end], sim.u[end], sim.t[end]
    return sum(i -> abs(exact(spacetime(xs[i], t))[1] - us[i][1]), eachindex(xs)) / length(xs)
end

set_save_path!(mktempdir())    # not restored: the plotter below may still be running from it
enable_progress_bar!(false)

@testset "Examples" begin
    @testset "$(nameof(example))" for (example, tmax, overrides, bound) in EXAMPLE_CHECKS
        conf = example()
        # Every active method is defined (or is a reference method, which the pipeline skips)
        @test all(m -> haskey(conf.methods, m) || is_reference_method(m), conf.defaults)
        merge!(conf.shared, Dict{Symbol, Any}(:tmax => tmax), overrides)
        tasks = generate_tasks(conf)
        @test !isempty(tasks)
        @test all(p -> haskey(p, :sim_func_name) && haskey(p, :Time_stepper), tasks)

        summary = run_all_simulations(conf; force_overwrite = true, calculate_stats = true)
        @test isempty(summary.failed) && isempty(summary.empty) && isempty(summary.stats_failed)   # nothing silently lost

        reference = conf.reference === nothing ? nothing : getfield(HyperCloud, conf.reference)
        for p in tasks
            @test does_sim_data_exist(p)
            does_sim_data_exist(p) || continue
            sim = load_sim_data(p, Val(:lagrangian))
            @test sim.t[end] ≈ tmax
            if reference !== nothing
                err = final_error(reference(p), sim)
                @info "$(nameof(example)): error vs $(conf.reference)" p[:Time_stepper] p[:Scheme_order] err
                @test err < bound
            end
        end
    end

    if RUN_GUI_TESTS
        @plot euler_1d_sod
        reset_plotter!()
    end
end
