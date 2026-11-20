# Time Integration

[`solve_equation`](@ref) runs the time loop: it computes the step size, advances the particle
grid with the chosen Runge–Kutta or IMEX stepper (HyperCloudCore's
[Time Stepper](@extref HyperCloudCore Time-Stepper)), applies particle adaptivity between steps if
it is enabled, reports progress, and stores snapshots.

## Time step

- **Fixed:** with `is_cfl = false`, `dt` is the physical time step.
- **CFL:** with `is_cfl = true` (`:Time_CFL` in a configuration), `dt` is a CFL number and the
  physical step is recomputed every iteration as `dt * get_time_step(pg, eq, div_interp)`. This
  uses the current wave speeds of the equation and the MLS derivative stencils of the grid; see
  HyperCloudCore's [CFL time step](@extref HyperCloudCore CFL-time-step) for the formula and
  measured limits.

The last step is shortened so the run ends exactly at `tmax`.

If the step collapses (below ``10^{-12}\,t_{max}``, relative, so no time scale is assumed) or is not finite (NaN states or unbounded wave speeds after a
blow-up, or an infinite CFL number or fixed step), or a stored intermediate snapshot is not finite, the loop stops
before `tmax` with a warning that gives the time reached; a non-finite final snapshot (at `tmax`) gives a warning
too. Each blow-up is reported once. The snapshot check also catches blow-ups the step doesn't see: fixed time steps,
equations with constant wave speeds (linear advection, BGK, kinetic runs) and CFL estimates that skip NaN
particles. The final slot then holds the state at that time, snapshots that were not reached are dropped
by the runners, and no NaN time is recorded. The runners still return the data, so check the warning or
`t[end]`. A snapshot more than ``10^8`` times larger than the initial state gives a warning (once) without
stopping the run. If no wave moves at all (an infinite CFL step with a finite state), the run advances straight to `tmax`.

## Snapshots

`snapshots + 1` states are stored, at the target times `range(0, tmax, length = snapshots + 1)`;
`snapshots` must be at least 1 (an `ArgumentError` otherwise).

- Snapshot 1 is the initial state at `t = 0`.
- Each further snapshot is taken at the end of the first step that reaches its target time,
  so `t[k]` can lie slightly after the target.
- The final state is always stored in the last slot.

[`saveData!`](@ref) copies positions and states (and the snapshot fields, see below) into new
arrays, so later steps do not modify stored snapshots. With `remove_ghosts = true` only non-boundary particles are kept. The runners
pass `:remove_ghosts` from the configuration (default `true`); `solve_equation` itself defaults
to `false`.

## Progress output

- `show_progress = true`: a progress bar.
- `show_progress = false`: an `@info` line with the percentage and an ETA (HH:MM:SS) every
  `progress_interval` seconds, which is better suited for log files.

The runners take both settings from global switches: `enable_progress_bar!(false)` and
`set_progress_interval!(seconds)`.

The loop also logs the execution backend of the parallel loops (see
[`set_backend!`](@extref HyperCloudCore HyperCloudCore.set_backend!)).

## Return value

`solve_equation` returns the run as a PDEStudioCore `LSimData` (`create_sim_data`), ready to be saved or plotted:

- `sim.x[k]`, `sim.u[k]`, `sim.t[k]`: the stored snapshots. These are the snapshots the run reached, so a run that stopped
  early has fewer than `snapshots + 1`.
- `sim.u[k]` holds `snapshot_state(u)` of every particle's state: the state itself by default, the macroscopic moments
  for the kinetic and BGK runners.
- `sim.params`: the keyword `params`, the run's parameters. They name the dataset's file in PDEStudioCore's pipeline.
- `sim.metadata`: the run's metadata (`run_metadata`):
  - the HyperCloudCore version, the backend and the float type;
  - the steps and the time reached;
  - `:stopped_early`, `:finite`;
  - `:restarts` with a checkpoint.
- `sim.stats`: the statistics of the run, each registered with the dimensions it keeps:

| statistic | kept | value |
|:----------|:-----|:------|
| `:Runtime` | `Symbol[]` (a number) | wall-clock time of the loop in seconds (all attempts of a resumed run) |
| `:k_step` | `Symbol[]` | time steps taken (all attempts) |
| `:final_particle_count` | `Symbol[]` | particles of the grid at the end, ghosts included |
| `:particle_count` | `:time` | particles stored in each snapshot |
| each of `snapshot_fields` | `:all` | that per-particle array of the grid at each snapshot, aligned with `sim.x[k]` |

`snapshot_fields` (keyword, default [`DEFAULT_SNAPSHOT_FIELDS`](@ref) `= (:volumes, :particle_orders)`) names the
per-particle arrays of the grid stored with every snapshot: `:volumes`, `:particle_orders`, `:neighbor_counts`,
`:is_boundary`, `:tags`, `:wall_distances`. A method of [`snapshot_field`](@ref) for another name adds a field.

- The fields are copied with the positions (on a device: host copies). With `remove_ghosts` they are copied for the same
  particles.
- A checkpoint stores them with the snapshots, so a resumed run returns them complete.
- More statistics can be added to the returned dataset with PDEStudioCore's `add_stat!`.

PDEStudioCore stores a statistic with the dataset's component count and float type: a per-particle field holds `M` copies
of each value. The runners therefore store no fields for the full kinetic states (`:save_relax`) unless
`:snapshot_fields` is given.

## API

```@docs
solve_equation
saveData!
snapshot_field
DEFAULT_SNAPSHOT_FIELDS
```
