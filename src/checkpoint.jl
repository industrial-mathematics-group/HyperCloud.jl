export Checkpointing, set_checkpointing!, get_checkpointing, remove_checkpoint

# =========================================================================
# CHECKPOINTS
#
# A run on a preemptible cluster partition can be killed at any time. `solve_equation` with a `Checkpointing` writes the
# state of the run (HyperCloudCore's `checkpoint_state` plus the loop's own state and the snapshots taken so far) to a
# JLD2 file every `every` seconds of wall time, and a run that finds its file resumes from it: rebuilt from its
# configuration as usual, the state is restored before the first step, and the run continues bit for bit as the run
# that wrote the file would have (same backend and device model). A machine setting like the memory limit (not a run
# parameter: it does not enter the run hash): `set_checkpointing!(dir)` makes every runner keep its checkpoint in
# `dir/<hash of the parameters>.jld2` and remove it when the run is done.
# =========================================================================

# The layout of the file (the keys below); a file of another layout is not resumed
const _CHECKPOINT_FILE_FORMAT = 1

"""
    Checkpointing(path; every = 1200, key = "")

Where and how often [`solve_equation`](@ref) saves the state of a run so that a stopped run (e.g. preempted on a cluster)
can resume: the JLD2 file `path`, written every `every` seconds of wall time (at the end of a step; to a temporary file
first, then renamed, so the file is always complete), and `key`, which names the run (the runners use the hash of the
parameters). A run that finds `path` resumes from it; a file with another `key` is refused (an `ArgumentError`), an
unreadable one (or one of another format) is moved aside to `path * ".unreadable"` with a warning and the run starts
from `t = 0`. `remove_checkpoint` deletes the file when the run is done. `restarts(ck)` is the number of times the run
was resumed (0 for a fresh run), set by `solve_equation`.
"""
struct Checkpointing
    path::String
    every::Float64
    key::String
    restarts::Base.RefValue{Int}
end
function Checkpointing(path::AbstractString; every::Real = 1200, key::AbstractString = "")
    every > 0 || throw(ArgumentError("Checkpointing: every must be a positive number of seconds, got $every"))
    isempty(path) && throw(ArgumentError("Checkpointing: the path is empty"))
    return Checkpointing(String(path), Float64(every), String(key), Ref(0))
end
restarts(ck::Checkpointing) = ck.restarts[]

const _CHECKPOINTING = Ref{Union{Nothing, NamedTuple}}(nothing)

"""
    set_checkpointing!(dir; every = 1200)
    set_checkpointing!(nothing)

Makes every following run of the runners (`run_direct_simulation`, `run_kinetic_simulation`, `run_bgk_simulation`)
checkpoint itself: the state is saved to `dir/<hash>.jld2` (`calculate_hash` of the run's parameters, the name of its
result file) every `every` seconds of wall time, a run that finds its file resumes from it, and the file is removed when
the run is done. A machine setting (it does not enter the run hash), like [`set_memory_limit!`](@ref); `nothing` turns it
off (the default). See [`Checkpointing`](@ref).
"""
function set_checkpointing!(dir::AbstractString; every::Real = 1200)
    every > 0 || throw(ArgumentError("set_checkpointing!: every must be a positive number of seconds, got $every"))
    _CHECKPOINTING[] = (; dir = String(dir), every = Float64(every))
    return nothing
end
set_checkpointing!(::Nothing) = (_CHECKPOINTING[] = nothing; nothing)

"The checkpoint setting of [`set_checkpointing!`](@ref) as `(; dir, every)`, or `nothing`."
get_checkpointing() = _CHECKPOINTING[]

# The checkpoint of a runner's run (`nothing` without `set_checkpointing!`)
function run_checkpoint(params::AbstractDict)
    s = _CHECKPOINTING[]
    s === nothing && return nothing
    key = calculate_hash(params)
    return Checkpointing(joinpath(s.dir, key * ".jld2"); every = s.every, key = key)
end

"""
    remove_checkpoint(ck::Union{Checkpointing, Nothing})

Deletes the checkpoint file of a finished run (and a temporary file a killed write left); nothing for `nothing`.
"""
remove_checkpoint(::Nothing) = nothing
function remove_checkpoint(ck::Checkpointing)
    rm(ck.path; force = true)
    rm(ck.path * ".tmp"; force = true)
    return nothing
end

# Writes the checkpoint: the solver state (`checkpoint_state`), the loop state and the snapshots stored so far (only the
# assigned slots), to a temporary file that then replaces the old checkpoint
function _write_checkpoint(ck::Checkpointing, state, loop, xs, us, ts)
    dir = dirname(ck.path)
    isempty(dir) || mkpath(dir)
    taken = findall(k -> isassigned(us, k), eachindex(us))
    tmp = ck.path * ".tmp"
    JLD2.jldsave(tmp; format = _CHECKPOINT_FILE_FORMAT, key = ck.key, state, loop,
                 snapshot_slots = taken, xs = xs[taken], us = us[taken], ts = ts[taken],
                 hypercloud = string(pkgversion(@__MODULE__)), hypercloudcore = string(pkgversion(HyperCloudCore)),
                 backend = _backend_name(get_backend()), written = string(Dates.now()))
    mv(tmp, ck.path; force = true)
    return nothing
end

# Reads a checkpoint: `nothing` if there is none (or it is unreadable: moved aside), else the stored data. A file with
# another key is refused.
function _read_checkpoint(ck::Checkpointing)
    isfile(ck.path) || return nothing
    data = try
        JLD2.load(ck.path)
    catch e
        _set_aside(ck, "it cannot be read ($(sprint(showerror, e)))")
        return nothing
    end
    if get(data, "format", nothing) != _CHECKPOINT_FILE_FORMAT
        _set_aside(ck, "it has the format $(get(data, "format", "unknown")), this version reads $_CHECKPOINT_FILE_FORMAT")
        return nothing
    end
    data["key"] == ck.key ||
        throw(ArgumentError("the checkpoint $(ck.path) belongs to another run (key $(repr(data["key"])), this run's is $(repr(ck.key)))"))
    for (name, mine) in (("hypercloud", string(pkgversion(@__MODULE__))), ("hypercloudcore", string(pkgversion(HyperCloudCore))),
                         ("backend", _backend_name(get_backend())))
        data[name] == mine || @warn "Resuming from $(ck.path), written with $name $(data[name]), now $mine: the run continues, " *
                                    "but not necessarily bit for bit as it would have"
    end
    return data
end

function _set_aside(ck::Checkpointing, why::String)
    @warn "The checkpoint $(ck.path) is not resumed: $why. It is moved to $(ck.path).unreadable, the run starts from t = 0."
    mv(ck.path, ck.path * ".unreadable"; force = true)
    return nothing
end

# The resumed part of `solve_equation`: restores the solver state with `restore!(state)` and the stored snapshots into
# `xs`, `us`, `ts`; returns the loop state to resume with, or `nothing` (no checkpoint: the run starts at t = 0)
function _resume!(restore!, ck::Union{Nothing, Checkpointing}, xs, us, ts)
    ck === nothing && return nothing
    data = _read_checkpoint(ck)
    data === nothing && return nothing
    restore!(data["state"])
    for (j, k) in enumerate(data["snapshot_slots"])
        k <= length(us) || throw(ArgumentError("the checkpoint $(ck.path) holds snapshot $k, the run stores $(length(us))"))
        xs[k], us[k], ts[k] = data["xs"][j], data["us"][j], data["ts"][j]
    end
    loop = data["loop"]
    ck.restarts[] = loop.restarts + 1
    @info "Resuming from the checkpoint $(ck.path) (written $(data["written"])): t = $(loop.t), step $(loop.k_step), " *
          "restart $(ck.restarts[])"
    return loop
end
