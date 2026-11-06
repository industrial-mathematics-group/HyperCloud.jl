# Probes for the exported examples and the time loop (run: julia -t 2 --project=test <this file>)
using HyperCloud, StaticArrays, Test
const PSC = HyperCloud.PDEStudioCore
set_save_path!(mktempdir())
enable_progress_bar!(false)

@testset "probes: examples and solve_equation" begin
    @testset "burgers_kinetic: every default method exists and runs" begin
        conf = burgers_kinetic()
        md = conf.methods
        for m in conf.defaults
            @info "method defined?" m haskey(md, m)
            @test haskey(md, m)
        end
        tasks = generate_tasks(conf)
        bad = filter(p -> !haskey(p, :Time_stepper), tasks)
        @info "tasks without a time stepper" length(bad) length(tasks)
        @test isempty(bad)
        if !isempty(bad)
            p = merge(first(bad), Dict{Symbol, Any}(:tmax => 0.01))
            st = Base.invokelatest(PSC._run_task, p, true, Dict{String, Any}())
            @info "status of the ARS222MUSCL2 task" st
            @test st === :ran
        end
    end

    @testset "snapshots = 0 keeps the final state" begin
        base = Dict{Symbol, Any}(
            :PDE_name => :linear, :PDE_velocities => ((1.0,),),
            :sim_func_name => :run_direct_simulation, :tmax => 0.1, :snapshots => 0,
            :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (50,),
            :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
            :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
            :IC_name => :gauss, :IC_a => (1.0,), :IC_b => (0.5,), :IC_width => 0.1,
            :Time_CFL => 0.2, :Time_stepper => :RK2, :Scheme_name => :Upwind, :Scheme_order => 1,
            :Scheme_MLS_order => 1, :Scheme_upwind_alg_nd => :Classic, :Flux_name => :Upwind, :MOOD_criterion => :none)
        sim = run_direct_simulation(base)
        @test sim !== nothing
        if sim !== nothing
            @info "snapshots = 0" sim.t length(sim.u)
            @test sim.t[end] ≈ 0.1
        end
    end
end
