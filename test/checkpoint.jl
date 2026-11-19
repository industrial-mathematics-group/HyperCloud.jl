# Checkpoints: a run killed in the middle (the test hook `_INTERRUPT_AT_STEP`, as a preempted cluster job) and run again
# resumes from its checkpoint file and ends with the result of the run that went through, bit for bit.

# The run of `params` straight, then killed after step `kills...` (one attempt per kill) and finished by one more attempt,
# with checkpoints every step in `dir`; returns both results and what the attempts returned
function killed_and_resumed(params, dir; kills = (7,))
    set_checkpointing!(nothing)
    straight = run_sim(params)
    set_checkpointing!(dir; every = 1e-9)                         # (every step)
    attempts = Any[]
    try
        for k in kills
            HC._INTERRUPT_AT_STEP[] = k
            push!(attempts, run_sim(params))                       # the runner catches the interrupt: `nothing`
        end
        HC._INTERRUPT_AT_STEP[] = 0
        resumed = run_sim(params)
        return straight, resumed, attempts
    finally
        HC._INTERRUPT_AT_STEP[] = 0
        set_checkpointing!(nothing)
    end
end
same_result(a, b) = a !== nothing && b !== nothing && isequal(a.x, b.x) && isequal(a.u, b.u) && isequal(a.t, b.t)
stat(sim, name) = sim.stats[name][1]

@testset "Checkpoints" begin
    @testset "Settings and the Checkpointing handle" begin
        @test get_checkpointing() === nothing
        set_checkpointing!("ckdir"; every = 60)
        @test get_checkpointing() == (; dir = "ckdir", every = 60.0)
        ck = HC.run_checkpoint(linear_1d_gauss(20))
        @test ck.path == joinpath("ckdir", calculate_hash(linear_1d_gauss(20)) * ".jld2") && ck.every == 60.0 && HC.restarts(ck) == 0
        set_checkpointing!(nothing)
        @test get_checkpointing() === nothing && HC.run_checkpoint(linear_1d_gauss(20)) === nothing
        @test_throws ArgumentError set_checkpointing!("ckdir"; every = 0)
        @test_throws ArgumentError Checkpointing("x.jld2"; every = -1)
        @test_throws ArgumentError Checkpointing("")
        @test remove_checkpoint(nothing) === nothing
    end

    mktempdir() do dir
        @testset "Direct runner: killed twice, resumed, the same result (static cloud, Float32 too)" begin
            p = merge(linear_1d_gauss(40; tmax = 0.2), UPWIND1, Dict{Symbol, Any}(:snapshots => 4))
            straight, resumed, attempts = killed_and_resumed(p, dir; kills = (5, 11))
            @test stat(straight, :k_step) > 11                                             # (both kills happen before the end)
            @test all(isnothing, attempts)
            @test same_result(straight, resumed)
            @test resumed.metadata[:restarts] == 2 && !haskey(straight.metadata, :restarts)
            @test stat(resumed, :k_step) == stat(straight, :k_step)
            @test isempty(readdir(dir))                                                     # the finished run removed its file
            p32 = merge(p, Dict{Symbol, Any}(:T => Float32))
            straight, resumed, _ = killed_and_resumed(p32, dir; kills = (12,))
            @test same_result(straight, resumed) && eltype(resumed.t) == Float32
        end

        @testset "A moving cloud with adaptivity: the Lagrangian Sod, the particle count changes" begin
            config = euler_1d_sod()
            p = merge(config.shared, config.methods[:RK2MUSCL2_U1_EPD1],
                      Dict{Symbol, Any}(:Grid_mover => :lagrangian, :Adapt_enabled => true, :snapshots => 3))
            straight, resumed, _ = killed_and_resumed(p, dir; kills = (40,))
            @test same_result(straight, resumed)
            @test stat(resumed, :final_particle_count) == stat(straight, :final_particle_count)
            @test straight.x[end] != straight.x[1]
        end

        @testset "Block form: one state and a split over host parts" begin
            p = merge(bgk_sod_params(80), Dict{Symbol, Any}(:PDE_block_width => 8, :snapshots => 3))
            straight, resumed, _ = killed_and_resumed(p, dir; kills = (15,))
            @test same_result(straight, resumed) && resumed.metadata[:restarts] == 1
            pd = merge(p, Dict{Symbol, Any}(:PDE_devices => 2))
            straight, resumed, _ = killed_and_resumed(pd, dir; kills = (15,))
            @test same_result(straight, resumed)
        end

        @testset "The file: complete, of this run, set aside when unreadable" begin
            p = merge(linear_1d_gauss(40; tmax = 0.2), UPWIND1, Dict{Symbol, Any}(:snapshots => 4))
            set_checkpointing!(dir; every = 1e-9)
            HC._INTERRUPT_AT_STEP[] = 10
            try
                @test run_sim(p) === nothing
            finally
                HC._INTERRUPT_AT_STEP[] = 0
                set_checkpointing!(nothing)
            end
            path = joinpath(dir, calculate_hash(p) * ".jld2")
            @test isfile(path) && !isfile(path * ".tmp")
            data = HC.JLD2.load(path)
            @test data["key"] == calculate_hash(p) && data["loop"].k_step == 9 && data["state"].N == 40
            slots = data["snapshot_slots"]                                                 # t = 0 and those reached by step 9
            @test length(slots) >= 2 && slots == collect(1:length(slots)) && data["loop"].snap_counter == length(slots) + 1
            @test issorted(data["ts"]) && data["ts"][end] <= data["loop"].t && length(data["xs"][1]) == 40
            @test data["hypercloud"] == string(pkgversion(HyperCloud))
            # another run's key: refused
            @test_throws ArgumentError HC._read_checkpoint(Checkpointing(path; key = "another run"))
            # unreadable: moved aside, the run starts from t = 0
            write(path, "not a JLD2 file")
            @test (@test_logs (:warn, r"not resumed") match_mode = :any HC._read_checkpoint(Checkpointing(path; key = calculate_hash(p)))) === nothing
            @test !isfile(path) && isfile(path * ".unreadable")
            rm(path * ".unreadable")
        end
    end
end
