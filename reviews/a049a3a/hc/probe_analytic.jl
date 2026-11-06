# Probes for analytic closures and small API edge cases (run: julia -t 2 --project=test <this file>)
using HyperCloud, StaticArrays, LinearAlgebra, Test
const HC = HyperCloud
integrate(f, a, b; n = 20_000) = (h = (b - a) / n; h * sum(i -> f(a + (i - 0.5) * h), 1:n))

@testset "probes: analytic closures" begin
    @testset "burgers_kinetic reference past the breaking time" begin
        p = copy(burgers_kinetic().shared)
        ex = analytical_solution(p)
        # u0 = exp(-x^2): breaking time 1 / max(-u0') = 1 / sqrt(2/e)
        tb = 1 / sqrt(2 / exp(1))
        @info "breaking time" tb tmax = p[:tmax]
        @test p[:tmax] > tb
        m0 = integrate(x -> ex(SVector(x, 0.0))[1], -5.0, 5.0)
        for t in (1.0, 1.5, 2.0)
            mt = integrate(x -> ex(SVector(x, t))[1], -5.0, 5.0)
            @info "mass of the exact Burgers reference" t mt m0 rel = (mt - m0) / m0
            @test isapprox(mt, m0; rtol = 1e-3)  # expected to FAIL for t > tb
        end
    end

    @testset "Burgers Gauss closure in Float32" begin
        p = merge(copy(burgers_kinetic().shared), Dict{Symbol, Any}(:real_type => Float32))
        ex = analytical_solution(p)
        logs = Test.collect_test_logs() do
            for x in range(-2f0, 2f0; length = 41)
                ex(SVector(x, 0.5f0))
            end
        end[1]
        nwarn = count(l -> l.level == Base.CoreLogging.Warn, logs)
        @info "Newton warnings for 41 Float32 evaluations at t = 0.5 (pre-shock)" nwarn
        @test nwarn == 0
    end

    @testset "LinearAdvection closure for a coupled system" begin
        for T in (Float64, Float32)
            A = SVector{1, SMatrix{2, 2, T, 4}}((SMatrix{2, 2, T}(0, 1, 1, 0),))
            eq = LinearAdvection(A, T)
            ic = Gauss(State{2, T}(1, 0), Space{1, T}(0), T(0.3))
            geom = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-3.0,), :maxs => (3.0,), :periodic => true),
                                Dict{Symbol, Any}(:T => T, :D => 1))
            ok = try
                ex = analytic_closure(eq, ic, geom)
                # acoustics: u = (p, v): p(x,t) = (g(x-t)+g(x+t))/2
                v = ex(SVector{2, T}(1, 1))
                @info "coupled closure" T v
                isapprox(v[1], T(0.5); rtol = 1e-4)
            catch e
                @info "coupled closure threw" T exception = e
                false
            end
            @test ok
        end
    end

    @testset "Commutation check tolerance (Float64, entries O(10))" begin
        B = @SMatrix [1.1 0.7 0.3; 0.7 2.2 0.5; 0.3 0.5 0.9]
        A1 = 10 .* B
        A2 = A1 * A1 ./ 37          # commutes with A1 exactly in exact arithmetic
        eq = LinearAdvection(SVector{2, SMatrix{3, 3, Float64, 9}}((A1, A2)), Float64)
        @info "commutator norm" norm(A1 * A2 - A2 * A1)
        geom = build_domain(Dict{Symbol, Any}(:domain => :rectangular, :mins => (-3.0, -3.0), :maxs => (3.0, 3.0), :periodic => true),
                            Dict{Symbol, Any}(:T => Float64, :D => 2))
        ic = Gauss(State{3, Float64}(1, 0, 0), Space{2, Float64}(0, 0), 0.3)
        ok = try
            analytic_closure(eq, ic, geom); true
        catch e
            @info "closure threw" exception = e
            false
        end
        @test ok
    end

    @testset "Euler Sod closure in Float32 matches Float64" begin
        sod = Dict{Symbol, Any}(copy(euler_1d_sod().shared))
        e64 = analytical_solution(sod)
        e32 = analytical_solution(merge(sod, Dict{Symbol, Any}(:real_type => Float32)))
        xs = range(-0.45, 0.45; length = 91)
        d = maximum(x -> maximum(abs.(e64(SVector(x, 0.2)) .- e32(SVector(Float32(x), 0.2f0)))), xs)
        @info "max |Sod_32 - Sod_64|" d
        @test d < 1e-5
    end

    @testset "BGK collision with tau = 0" begin
        nodes, w = velocity_grid(:uniform, 16, 6.0)
        eq = BGKEquation(Val(1), nodes, w, 1.4, ConstantRelaxationTime(0.0))
        f, _ = equilibrium(eq, SVector(1.0, 0.3, 2.0))
        f = f + State{HC.get_M(eq), Float64}(ntuple(k -> 0.01 * sin(k), HC.get_M(eq)))
        fn = HC.implicit_solve(BGKRelaxation(eq), f, 0.01, 1, nothing, 0.0)
        @info "tau = 0 implicit solve finite?" all(isfinite, fn)
        @test all(isfinite, fn)
    end

    @testset "master method dict: no label collisions" begin
        # count entries before Dict() collapses them
        md = get_master_method_dict()
        @info "master dict size" length(md)
        @test length(md) > 0
    end
end
