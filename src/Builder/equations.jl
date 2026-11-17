export build_equation

# --- Main Equation Builder ---
"""
    build_equation(pde_conf::Dict, context::Dict)

Constructs the physical equation system based on the provided configuration dictionary.
Strictly requires a `:name` Symbol (e.g., `:linear`, `:burgers`, `:euler`) in the configuration to dynamically dispatch to the correct specific constructor using `Val`.
"""
function build_equation(pde_conf::Dict, context::Dict)
    if !haskey(pde_conf, :name)
        error("PDE configuration must include a strictly typed :name Symbol (e.g., :linear, :burgers).")
    end
    
    eq_name = pde_conf[:name]::Symbol
    return build_equation(Val(eq_name), pde_conf, context)
end


"""
    build_equation(eq_name::Val, pde_conf::Dict, context::Dict)

Constructs the physical equation system based on the provided configuration dictionary.
Strictly requires a `:name` Symbol (e.g., `:linear`, `:burgers`, `:euler`) in the configuration to dynamically dispatch to the correct specific constructor using `Val`.
"""
build_equation(eq_name::Val, pde_conf::Dict, context::Dict) = error("PDE '$(typeof(eq_name))' is not implemented.")

function build_equation(::Val{:burgers}, pde_conf::Dict, context::Dict)
    T = context[:T]::DataType
    D = pde_conf[:D]::Int
    
    return BurgersEquation{D, T}()
end

function build_equation(::Val{:euler}, pde_conf::Dict, context::Dict)
    T = context[:T]::DataType
    D = pde_conf[:D]::Int
    rep = parse_representation(pde_conf)
    
    gamma = T(pde_conf[:gamma])
    
    return EulerEquation(Val(D), T, gamma, rep)
end

function build_equation(::Val{:linear}, pde_conf::Dict, context::Dict)
    T = context[:T]::DataType
    
    # We pass T securely, and the smart constructor utilizes param2vel internally
    return LinearAdvection(pde_conf[:velocities], T) 
end

"""
    build_equation(::Val{:bgk}, pde_conf::Dict, context::Dict) -> BGKEquation

Discrete-velocity BGK model. Keys of the `PDE` namespace:

| Key | Meaning |
|:--|:--|
| `:D` | space dimension (the velocity grid has the same dimension) |
| `:Nv` | velocity nodes per direction (`Nv^D` discrete velocities) |
| `:vmax` | half-width of the velocity grid (outermost node for `:gauss_hermite`) |
| `:velocity_grid` | `:uniform` (default) or `:gauss_hermite`, see [`velocity_grid`](@ref) |
| `:vcenter` | centre of the velocity grid per direction (default `0`) |
| `:gamma` | ratio of specific heats (default `(D + 2) / D`, no internal degrees of freedom) |
| `:tau` | constant relaxation time ([`ConstantRelaxationTime`](@ref)); `Inf` is collisionless |
| `:viscosity`, `:omega` | instead of `:tau`: `τ = μ / p` with `μ = viscosity θ^omega` ([`ViscosityRelaxationTime`](@ref), `omega` defaults to `0.5`) |
"""
function build_equation(::Val{:bgk}, pde_conf::Dict, context::Dict)
    T = context[:T]::DataType
    D = pde_conf[:D]::Int
    haskey(pde_conf, :tau) && haskey(pde_conf, :viscosity) && error("BGK: give either :tau or :viscosity, not both.")
    relaxation = if haskey(pde_conf, :viscosity)
        ViscosityRelaxationTime(T(pde_conf[:viscosity]), T(get(pde_conf, :omega, 0.5)))
    elseif haskey(pde_conf, :tau)
        ConstantRelaxationTime(T(pde_conf[:tau]))
    else
        error("BGK: the PDE namespace needs a relaxation time (:tau, or :viscosity with :omega).")
    end
    nodes, weights = velocity_grid(get(pde_conf, :velocity_grid, :uniform)::Symbol, pde_conf[:Nv]::Int, pde_conf[:vmax], T)
    gamma = get(pde_conf, :gamma, (D + 2) / D)
    center = get(pde_conf, :vcenter, ntuple(_ -> 0.0, D))
    return BGKEquation(Val(D), nodes, weights, gamma, relaxation; center = center)
end

"""
    build_equation(::Val{:bgk_mixture}, pde_conf::Dict, context::Dict) -> BGKMixture

Discrete-velocity model of a gas mixture ([`BGKMixture`](@ref)). Keys of the `PDE` namespace:

| Key | Meaning |
|:--|:--|
| `:D` | space dimension |
| `:masses` | the species' masses, one per species (sets the number of species `L`) |
| `:lambda` | the symmetric `L × L` matrix of collision constants `λ_sr` (a matrix, or a tuple of rows) |
| `:epsilon` | intra-species Knudsen number `ε` (`Inf`: no intra-species collisions) |
| `:kappa` | inter-species Knudsen number `κ` (default `ε`) |
| `:kB` | Boltzmann constant in the problem's units (default `1`) |
| `:Nv` | velocity nodes per direction: one number for all species, or one per species |
| `:vmax` | half-width of the velocity grids: one number, or one per species |
| `:velocity_grid` | `:uniform` (default) or `:gauss_hermite`, see [`velocity_grid`](@ref) |
| `:vcenter` | centre of the velocity grids per direction (default `0`): one tuple for all species, or one per species |
| `:gamma` | ratio of specific heats of every species (default `5/3`, monatomic) |
"""
function build_equation(::Val{:bgk_mixture}, pde_conf::Dict, context::Dict)
    T = context[:T]::DataType
    D = pde_conf[:D]::Int
    masses = Tuple(pde_conf[:masses])
    L = length(masses)
    lam = pde_conf[:lambda]
    lambda = lam isa AbstractMatrix ? Matrix{Float64}(lam) : [Float64(lam[s][r]) for s in 1:L, r in 1:L]
    size(lambda) == (L, L) || error("BGK mixture: :lambda must be $L × $L for $L species, got $(size(lambda))")
    per_species(key, default = nothing) = begin
        v = default === nothing ? pde_conf[key] : get(pde_conf, key, default)
        v isa Number ? ntuple(_ -> v, L) : (length(v) == L ? Tuple(v) : error("BGK mixture: $key needs one value or one per species ($L)"))
    end
    Nvs   = per_species(:Nv)
    vmaxs = per_species(:vmax)
    kind  = get(pde_conf, :velocity_grid, :uniform)::Symbol
    grids = ntuple(s -> velocity_grid(kind, Int(Nvs[s]), vmaxs[s], T), L)
    vc = get(pde_conf, :vcenter, nothing)
    centers = vc === nothing ? nothing :
              (vc isa Number || (vc isa Tuple && !isempty(vc) && vc[1] isa Number)) ? ntuple(_ -> vc isa Number ? (vc,) : vc, L) : Tuple(vc)
    epsilon = pde_conf[:epsilon]
    return BGKMixture(Val(D), masses, lambda, grids; gamma = get(pde_conf, :gamma, 5 / 3), epsilon = epsilon,
                      kappa = get(pde_conf, :kappa, epsilon), kB = get(pde_conf, :kB, 1), centers = centers)
end
