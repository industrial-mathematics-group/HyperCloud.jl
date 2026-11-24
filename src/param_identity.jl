# ==============================================================================
# --- What identifies HyperCloud's values in a run's parameters ---
# ==============================================================================
# PDEStudioCore (from 0.9) stores and hashes a value of a custom type in a run's parameters by what its
# `param_identity` names, and refuses values of types that don't declare it, and functions. The types that can appear
# in a configuration (`:Grid_bc`, `:Grid_shape`, `:Grid_domain`) declare it here; the engine's other types (grids,
# schemes, steppers, movers, equations, initial conditions) are built inside a run from symbols and numbers and never
# appear in parameters. Earlier PDEStudioCore versions (allowed by `[compat]`) have no `param_identity`: nothing is
# declared there, and the parameters are hashed as before.
if isdefined(PDEStudioCore, :param_identity)
    # Plain data: every field (stored as `TypeName(field=value, …)`, a type without fields by its type with its
    # parameters, e.g. `ExtrapolationBC{1}`). A `Tagged` tag or a `Moving` motion that is a function is refused by
    # PDEStudioCore's check: give such a shape as a string (`:Grid_shape`), which is data.
    for T in (SlipWall, DiffuseWall,
              (getfield(HyperCloudCore, n) for n in (:FixedDirichlet, :OutflowBC, :ExtrapolationBC, :Cuboid, :Ball, :HalfSpace,
                                                     :UnionShape, :IntersectionShape, :ComplementShape, :Translated, :Rotated,
                                                     :RigidMotion, :Tagged, :Moving))...)
        @eval PDEStudioCore.@param_struct $T
    end

    # The degree and boundedness of a `MirrorBC` are type parameters: listed when they differ from the defaults
    # (degree 1, bounded), so the stored form of a default condition stays `MirrorBC(op=…)`, as before 0.9
    PDEStudioCore.param_identity(bc::HyperCloudCore.MirrorBC{Q, F, B}) where {Q, F, B} =
        Q == 1 && B ? (; op = bc.op) : (; op = bc.op, degree = Q, bounded = B)

    # A domain by its inputs: not by its functions (`is_interior`, `get_tag`, which follow from the shape, or for a
    # domain without one from its label) nor by what is derived (`volume`, `pieces`, `piece_of_tag`). A domain built
    # from functions needs a label of its own for every region.
    PDEStudioCore.param_identity(d::HyperCloudCore.GeometricDomain) =
        (; label = d.label, mins = d.mins, maxs = d.maxs, is_periodic = d.is_periodic, bc_map = d.bc_map, shape = d.shape)
end
