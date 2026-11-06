export EquationRepresentation, NCRepresentation, Conservative, analytical_solution

"""
    EquationRepresentation

Abstract base type for categorizing the mathematical formulation of a PDE system.

# Subtypes
- `Conservative`: Concrete type indicating the system evaluates standard conservative fluxes and eigenvalues.
- `NCRepresentation`: Abstract type for non-conservative evaluations (e.g., primitive or Lagrangian representations).
"""
abstract type EquationRepresentation end
abstract type NCRepresentation <: EquationRepresentation end
struct Conservative <: EquationRepresentation end

# --- Representation Parsing ---
# Only the conservative form is implemented. `NCRepresentation` is the extension point for non-conservative
# (e.g. primitive or Lagrangian) forms, but no such representation exists yet, so `:primitive` and
# `:lagrangian` are rejected like any other unknown name.
function parse_representation(pde_conf::Dict)
    rep_sym = get(pde_conf, :representation, :conservative)::Symbol
    return _parse_representation(Val(rep_sym), pde_conf)
end

_parse_representation(::Val{:conservative}, pde_conf::Dict) = Conservative()
_parse_representation(::Val{R}, pde_conf::Dict) where {R} = throw(ArgumentError(
    "PDE representation :$R is not supported; only :conservative is implemented"))

include("burgers.jl")
include("linear_advection.jl")
include("euler.jl")
include("bgk.jl")