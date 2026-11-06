# Misc probes: thread-count reproducibility checksum, Euler vacuum, Float32 Burgers accuracy
using HyperCloud, StaticArrays, Test, Logging
enable_progress_bar!(false)
integrate(f, a, b; n = 20_000) = (h = (b - a) / n; h * sum(i -> f(a + (i - 0.5) * h), 1:n))

sod = euler_1d_sod()
p = merge(sod.shared, sod.methods[:RK2MUSCL2_U2_EPD1], Dict{Symbol, Any}(:Grid_Ns => (120,), :tmax => 0.1))
sim = with_logger(NullLogger()) do; run_direct_simulation(p); end
println("CHECKSUM threads=$(Threads.nthreads()) ", repr(sum(u -> sum(u), sim.u[end])), " steps=", sim.stats[:k_step][1])

@testset "misc" begin
    @testset "Euler exact solution with vacuum generation conserves mass" begin
        eq = EulerEquation(Val(1), Float64, 1.4, Conservative())
        ρ, u, p0 = 1.0, 6.0, 1.0
        UL = State{3, Float64}(ρ, -ρ * u, p0 / 0.4 + 0.5 * ρ * u^2)
        UR = State{3, Float64}(ρ,  ρ * u, p0 / 0.4 + 0.5 * ρ * u^2)
        geom = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-5.0,), :maxs => (5.0,), :periodic => false),
                            Dict{Symbol, Any}(:T => Float64, :D => 1))
        ex = HyperCloud.analytic_closure(eq, Riemann(UL, UR, Space{1}(0.0), Space{1}(1.0)), geom)
        t = 0.2
        m = integrate(x -> ex(SVector(x, t))[1], -3.0, 3.0)
        mref = 6.0 - 2 * u * t * ρ     # mass leaving through x = ±3 with the undisturbed states
        mid = ex(SVector(0.0, t))
        @info "vacuum case" m mref mid all(isfinite, mid)
        @test isapprox(m, mref; rtol = 1e-3)
        @test all(isfinite, mid) && mid[1] < 1e-6
    end

    @testset "Float32 Burgers Gauss values despite the warnings" begin
        ex32 = HyperCloud.analytic_closure(BurgersEquation{1, Float32}(), Gauss(State{1, Float32}(1), Space{1, Float32}(0), 1f0),
                   build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-5.0,), :maxs => (5.0,), :periodic => true),
                                Dict{Symbol, Any}(:T => Float32, :D => 1)))
        ex64 = HyperCloud.analytic_closure(BurgersEquation{1, Float64}(), Gauss(State{1, Float64}(1), Space{1, Float64}(0), 1.0),
                   build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-5.0,), :maxs => (5.0,), :periodic => true),
                                Dict{Symbol, Any}(:T => Float64, :D => 1)))
        d = with_logger(NullLogger()) do
            maximum(x -> abs(ex32(SVector(Float32(x), 0.5f0))[1] - ex64(SVector(x, 0.5))[1]), range(-2, 2; length = 41))
        end
        @info "Float32 vs Float64 Burgers Gauss at t = 0.5" d
        @test d < 1e-5
    end
end
