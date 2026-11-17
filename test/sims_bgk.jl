# Group `sims-bgk`: BGK simulations (collisionless convergence, Euler limit, conservation, explicit vs IMEX).

const MIXTURE_LAMBDA = [19.96 1.153; 1.153 17.52]      # He–Ar collision constants of the paper (masses 1 and 10 here)

"1-D two-species mixture (masses 1 and 10) with a smooth periodic wave of both species' number densities and energies."
mixture_wave_params(N; epsilon = 1e-2, kappa = epsilon, masses = (1.0, 10.0), lambda = MIXTURE_LAMBDA, Nv = (16, 12), vmax = (8.0, 4.0),
                    gamma = 5 / 3, stepper = :ARS222, tmax = 0.1,
                    a = (0.1, 0.0, 0.152, 0.03, 0.0, 0.0069), c = (1.0, 0.2, 1.52, 0.3, -0.03, 0.069)) = Dict{Symbol, Any}(
    :PDE_name => :bgk_mixture, :PDE_D => 1, :PDE_masses => masses, :PDE_lambda => lambda, :PDE_epsilon => epsilon, :PDE_kappa => kappa,
    :PDE_Nv => Nv, :PDE_vmax => vmax, :PDE_gamma => gamma,
    :sim_func_name => :run_bgk_simulation, :tmax => tmax, :snapshots => 2,
    :Grid_domain => :rectangular, :Grid_mins => (0.0,), :Grid_maxs => (1.0,), :Grid_Ns => (N,),
    :Grid_periodic => true, :Grid_randomness_factor => (0.0,), :Grid_seed => 42, :Grid_mover => :none,
    :Weight_name => :exponential, :Weight_alpha => 1.0, :Weight_range => 5.5,
    :IC_name => :sine, :IC_a => a, :IC_period => (1.0,), :IC_c_offset => c,
    :Time_CFL => 0.2, :Time_stepper => stepper,
    :Scheme_name => :MUSCL, :Scheme_order => 2, :Flux_name => :Rusanov, :Limiter_name => :none,
    :MOOD_criterion => :none,
)

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

        @testset "Sod on Lagrangian particles: sharp contact, mass kept (per-velocity frame flux, LagrangianFrame)" begin
            # On Lagrangian particles the generic Rusanov of the moving frame smeared the contact (width 0.135 against
            # 0.036 static at N = 800 with EPD2) and the cloud gained mass at the shock at a rate independent of N
            # (+0.1 to 0.5 % by t = 0.2; study/bgk-lagrangian-contact of the simulations). With the per-velocity
            # dissipation of the frame and HyperCloudCore's conservative LagrangianFrame the Lagrangian Sod is close to
            # the static one.
            p = merge(bgk_sod_params(200), Dict{Symbol, Any}(:Grid_mover => :lagrangian, :Grid_volumes => :mls, :Grid_volumes_degree => 2,
                                                             :Grid_randomness_factor => (0.2,), :snapshots => 2))
            err, sim = euler_l1(p)
            @test sim !== nothing
            if sim !== nothing
                @test err < 0.025                                              # measured 0.0129 (static 0.0153)
                xs = [x[1] for x in sim.x[end]]; pm = sortperm(xs); xs = xs[pm]; ρ = [u[1] for u in sim.u[end]][pm]
                hs = [i == 1 ? xs[2] - xs[1] : i == length(xs) ? xs[end] - xs[end - 1] : (xs[i + 1] - xs[i - 1]) / 2 for i in eachindex(xs)]
                x0 = [x[1] for x in sim.x[1]]; pm0 = sortperm(x0); x0 = x0[pm0]; ρ0 = [u[1] for u in sim.u[1]][pm0]
                h0 = [i == 1 ? x0[2] - x0[1] : i == length(x0) ? x0[end] - x0[end - 1] : (x0[i + 1] - x0[i - 1]) / 2 for i in eachindex(x0)]
                @test abs(sum(hs .* ρ) - sum(h0 .* ρ0)) < 5e-4             # measured 1.3e-4 (static 2.7e-4; frozen frame: +1e-3 to 5e-3, N-independent)
                @test all(>(0), ρ) && minimum(hs) > 0.2 / 200
                @test count(x -> 0.29 < x < 0.40, ρ) * (1 / 200) < 0.10        # measured 0.055 (static 0.070; generic frame flux: 0.135 at N = 800)                # the contact width
            end
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
            # ... also through the Kinetic namespace, as for run_kinetic_simulation (the BGK runner ignored it)
            s_kin2 = run_sim(merge(p, Dict{Symbol, Any}(:Kinetic_save_relax => true)))
            @test s_kin2 !== nothing && length(s_kin2.u[end][1]) == 16
        end

        @testset "τ = 0 (the Euler limit as a projection) runs end to end" begin
            p0 = bgk_wave_params(32; tau = 0.0, stepper = :ARS222, order = 2)
            s0 = run_sim(p0)
            @test s0 !== nothing && all(u -> all(isfinite, u), s0.u[end])
            if s0 !== nothing
                m(s, k) = sum(u -> u[k], s.u[end]) / length(s.u[end])
                @test all(k -> isapprox(m(s0, k), sum(u -> u[k], s0.u[1]) / length(s0.u[1]); atol = 1e-12), 1:3)   # conserved
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

    @testset "BGK mixture" begin
        @testset "Species densities, total momentum and energy are conserved; the temperatures equalise" begin
            sim = run_sim(mixture_wave_params(50; tmax = 0.2))
            @test sim !== nothing
            if sim !== nothing
                @test length(sim.u[end][1]) == 6                                 # the species' number moments
                t0, t1 = totals(sim, 1), totals(sim, length(sim.u))
                @test t1[1] ≈ t0[1] rtol = 1e-12
                @test t1[4] ≈ t0[4] rtol = 1e-12
                @test t1[2] + 10 * t1[5] ≈ t0[2] + 10 * t0[5] atol = 1e-11 * length(sim.u[1])
                @test t1[3] + 10 * t1[6] ≈ t0[3] + 10 * t0[6] rtol = 1e-12
                eqm = build_equation(Dict{Symbol, Any}(:name => :bgk_mixture, :D => 1, :masses => (1.0, 10.0), :lambda => MIXTURE_LAMBDA,
                                                       :epsilon => 1e-2, :Nv => (16, 12), :vmax => (8.0, 4.0)), Dict{Symbol, Any}(:T => Float64))
                spread(u) = (p = species_primitives(eqm, u); abs(p[1][3] - p[2][3]))
                @test maximum(spread, sim.u[1]) > 0.4                              # T_He = 1, T_Ar = 1.5 at the start
                @test maximum(spread, sim.u[end]) < 0.05                           # exchanged by t = 0.2 (κ = 1e-2)
                @test all(u -> all(p -> p[1] > 0 && p[3] > 0, species_primitives(eqm, u)), sim.u[end])
            end
        end

        @testset "Two identical species are the halves of the single gas (ν = λ n: τ = μ/ρ with μ = ε/λ, ω = 1)" begin
            ε, λ = 0.05, 2.0
            pm = mixture_wave_params(40; epsilon = ε, masses = (1.0, 1.0), lambda = fill(λ, 2, 2), Nv = (16, 16), vmax = (6.0, 6.0),
                                     gamma = 3.0, tmax = 0.1,
                                     a = (0.1, 0.05, 0.05, 0.1, 0.05, 0.05), c = (0.5, 0.0, 0.25, 0.5, 0.0, 0.25))
            ps = bgk_wave_params(40; Nv = 16, stepper = :ARS222, order = 2)
            delete!(ps, :PDE_tau)
            merge!(ps, Dict{Symbol, Any}(:PDE_viscosity => ε / λ, :PDE_omega => 1.0, :tmax => 0.1))
            simm, sims = run_sim(pm), run_sim(ps)
            @test simm !== nothing && sims !== nothing
            if simm !== nothing && sims !== nothing
                @test simm.t[end] == sims.t[end]
                err = maximum(maximum(abs.(um[1:3] + um[4:6] - us)) for (um, us) in zip(simm.u[end], sims.u[end]))
                @test err < 1e-11
                @test maximum(maximum(abs.(um[1:3] - um[4:6])) for um in simm.u[end]) < 1e-12
            end
        end
    end
end
