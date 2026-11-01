# Group `sims-bgk`: BGK simulations (collisionless convergence, Euler limit, conservation, explicit vs IMEX).

@testset "Simulations" begin
    @testset "BGK" begin
        @testset "Collisionless transport converges at the scheme's order" begin
            errs = [first(l1_error(bgk_wave_params(N))) for N in (50, 100)]
            @test all(isfinite, errs)
            @test errs[2] < 5e-5
            @test log2(errs[1] / errs[2]) > 2.7                     # MUSCL3 / RK3 against the exact DVM solution
        end

        @testset "Euler limit: Sod at τ = 1e-5 (IMEX, forwarded by run_kinetic_simulation)" begin
            err, sim = euler_l1(bgk_sod_params(200))
            @test sim !== nothing
            if sim !== nothing
                us = sim.u[end]
                @test length(us[1]) == 3                              # macroscopic output
                @test err < 0.025                                     # the Euler scheme itself: about 0.016
                @test minimum(u -> u[1], us) > 0
                @test minimum(u -> 0.4 * (u[3] - 0.5 * u[2]^2 / u[1]), us) > 0
            end
            # A larger τ is further from the Euler solution (viscous, heat-conducting profiles)
            err_far, _ = euler_l1(bgk_sod_params(200; tau = 1e-2))
            @test err_far > 0.025
        end

        @testset "Mass, momentum and energy are conserved (periodic lattice, IMEX)" begin
            sim = run_sim(bgk_wave_params(64; tau = 1e-2, stepper = :ARS222, order = 2))
            @test sim !== nothing
            sim !== nothing && @test isapprox(totals(sim, length(sim.u)), totals(sim, 1); rtol = 1e-12, atol = 1e-12)
        end

        @testset "Explicit collision (RK2) agrees with IMEX for a non-stiff τ" begin
            pe = merge(bgk_wave_params(64; tau = 0.05, stepper = :RK2, order = 2), Dict{Symbol, Any}(:sim_func_name => :run_direct_simulation))
            pi_ = bgk_wave_params(64; tau = 0.05, stepper = :ARS222, order = 2)
            se, si = run_sim(pe), run_sim(pi_)
            @test se !== nothing && si !== nothing
            if se !== nothing && si !== nothing
                @test maximum(i -> abs(se.u[end][i][1] - si.u[end][i][1]), eachindex(se.u[end])) < 1e-3
                @test isapprox(totals(se, length(se.u)), totals(se, 1); rtol = 1e-12, atol = 1e-12)
                # Relaxation damps the free-transport solution
                ex = analytical_solution(merge(pe, Dict{Symbol, Any}(:PDE_tau => Inf)))
                @test maximum(i -> abs(ex(spacetime(se.x[end][i], se.t[end]))[1] - se.u[end][i][1]), eachindex(se.x[end])) > 1e-3
            end
        end

        @testset ":save_relax keeps the distributions" begin
            p = bgk_wave_params(32; tau = 1e-2, stepper = :ARS222, order = 2)
            s_mac, s_kin = run_sim(p), run_sim(merge(p, Dict{Symbol, Any}(:save_relax => true)))
            @test s_mac !== nothing && s_kin !== nothing
            if s_mac !== nothing && s_kin !== nothing
                eq = build_equation(HC.extract_namespace(p, :PDE), Dict{Symbol, Any}(:T => Float64))
                @test length(s_kin.u[end][1]) == HC.get_M(eq) == 16
                @test all(i -> moments(eq, s_kin.u[end][i]) ≈ s_mac.u[end][i], eachindex(s_mac.u[end]))
            end
        end

        @testset "Float32 end-to-end" begin
            e64, _ = l1_error(bgk_wave_params(50))
            e32, s32 = l1_error(bgk_wave_params(50; T = Float32))
            @test s32 !== nothing
            s32 !== nothing && @test eltype(s32.u[end][1]) == Float32
            @test abs(e32 - e64) < 1e-4
        end

        @testset "2D: collisionless convergence and conservation" begin
            p2 = Dict{Symbol, Any}(
                :PDE_name => :bgk, :PDE_D => 2, :PDE_gamma => 2.0, :PDE_Nv => 6, :PDE_vmax => 4.0, :PDE_tau => Inf,
                :sim_func_name => :run_bgk_simulation, :tmax => 0.1, :snapshots => 1,
                :Grid_domain => :rectangular, :Grid_mins => (0.0, 0.0), :Grid_maxs => (1.0, 1.0),
                :Grid_periodic => true, :Grid_randomness_factor => (0.0, 0.0), :Grid_seed => 42, :Grid_mover => :none,
                :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 2.5,
                :IC_name => :sine, :IC_a => (0.2, 0.05, 0.05, 0.1), :IC_period => (1.0, 1.0), :IC_c_offset => (1.0, 0.0, 0.0, 0.5),
                :Time_CFL => 0.2, :Time_stepper => :RK2,
                :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :none, :MOOD_criterion => :none)
            errs = [first(l1_error(merge(p2, Dict{Symbol, Any}(:Grid_Ns => (N, N))))) for N in (16, 32)]
            @test all(isfinite, errs)
            @test log2(errs[1] / errs[2]) > 1.5
            sim = run_sim(merge(p2, Dict{Symbol, Any}(:Grid_Ns => (16, 16), :PDE_tau => 1e-2, :Time_stepper => :ARS222)))
            @test sim !== nothing
            sim !== nothing && @test isapprox(totals(sim, length(sim.u)), totals(sim, 1); rtol = 1e-12, atol = 1e-12)
        end
    end
end
