using HyperCloud, StaticArrays
const HC = HyperCloud
const HITS = Threads.Atomic{Int}(0); const CALLS = Threads.Atomic{Int}(0)
const EX = Ref{Any}(nothing)
@eval HC @inline function cons2prim(eq::EulerEquation{D, M, T}, U::State{M, T}) where {D, M, T}
    rho = max(U[1], _euler_floor(T))
    m = Space{D, T}(ntuple(d -> U[1+d], Val(D)))
    E = U[M]
    u = m ./ rho
    p = (eq.gamma - T(1)) * (E - T(0.5) * sum(abs2, m) / rho)
    Threads.atomic_add!(Main.CALLS, 1)
    if U[1] < 1e-7 || p < 1e-7
        Threads.atomic_add!(Main.HITS, 1); Main.EX[] === nothing && (Main.EX[] = (U, p))
    end
    return State{M, T}(rho, u..., max(p, _euler_floor(T)))
end
sod = euler_1d_sod()
for m in (:RK2MUSCL2_U2_EPD1, :RK2MUSCL2_VK), T in (Float64, Float32)
    HITS[] = 0; CALLS[] = 0; EX[] = nothing
    sim = run_direct_simulation(merge(sod.shared, sod.methods[m], Dict{Symbol, Any}(:snapshots => 1, :T => T)))
    println("$m $T: cons2prim calls $(CALLS[]), below 1e-7: $(HITS[]), first: $(EX[])")
end
