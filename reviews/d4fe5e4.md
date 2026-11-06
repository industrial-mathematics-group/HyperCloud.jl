# Review: HyperCloud.jl — whole repository at d4fe5e4 (round 2)
Reviewer: Claude Opus 5.5 (Anthropic), new session without the generating conversation

Second review (round 2) of the state after the fixes for reviews/a049a3a.md. Probes and outputs referenced below (`round2/hc/...`) are bundled in `reviews/d4fe5e4/`.

## Scope

Environment: Julia 1.13 (container), test environment `test/Manifest.toml` with the **registered HyperCloudCore 0.7.9**
(`~/.julia/packages/HyperCloudCore/YIxkg`); the local HyperCloudCore checkout was not used.

Read completely (final state at d4fe5e4, before reading the round-1 report):
- `src/HyperCloud.jl`, `src/device.jl`, `src/time_integration.jl`, `src/inbuilt_methods.jl`, `src/macros.jl`
- `src/PDEs/{_main,linear_advection,burgers,euler,bgk}.jl` (closures re-derived: Burgers Gauss breaking time
  `t_b = w e^{1/2}/(|a|√(2D))` and the bisection bracket `[min(0,a), max(0,a)]`; Burgers Box shock/fan interaction
  incl. the `C √t` shock after interaction; Toro's star region, fans and the new vacuum branch; BGK moments, Chu
  reduction, dual Hessian)
- `src/ICs/*.jl`, `src/SourceTerms/{kinetic,bgk}.jl`, `src/Builder/*.jl`, `src/SimulationFunctions/*.jl`, `src/Examples/*.jl`
- `test/runtests.jl`, `test/units.jl`, `test/sims.jl`, `test/sims_bgk.jl`, `test/sims_sweep.jl`, `test/examples.jl`
- `test/gpu/runtests.jl` (reading only), `benchmark/runners.jl`
- `docs/src/{configuration,simulation,wrappers,examples}.md`; the diff of every changed doc page
- Round-1 report `reviews/a049a3a.md` and the full diff `a049a3a..d4fe5e4` of `src/`, `docs/`, `test/` (after my own review)

Cross-checked in HyperCloudCore 0.7.9: `Interpolation/FluxFunctions.jl` (Rusanov/Upwind/AlignedRusanov fallbacks used by
the BGK flux override), `TimeStepping/TimeSteppers.jl:330-405` (how stages with `a_ii = 0` evaluate implicit sources),
`TimeStepping/ButcherTableaus.jl` (IMEX tableaus), `Grid/ParticleGrids.jl:356-389` (`get_time_step`),
`Builders/domains.jl:9-17` (`:bc` / `:bc_map`).

## Tests run

| Command | Result | Wall time |
|:--|:--|:--|
| `flock … timeout 60m julia -t 8 --project=test test/runtests.jl` (polyester, GUI tests on) | **PASS**, exit 0: units 746, sims-bgk 25, sweep-a 50, sweep-b 48, examples 78, sims 114 = **1061 passed**, 0 failed/errored; top-level 6/6 groups | 70 s under the lock (6 groups in parallel; group times 25–65 s, peak 1.4–1.9 GiB each) |
| `flock … HYPERCLOUD_BACKEND=ka_cpu HYPERCLOUD_TEST_GROUP=sims-bgk … test/runtests.jl` | **FAIL** (expected, round-1 Finding 1 deferred): 21 pass, 3 fail in "2D: collisionless convergence and conservation" (`test/sims_bgk.jl:77,78,80`), `ArgumentError: Work-group size (256, 1, 1) has more than 231 work-items` (log `ka_cpu_sims_bgk.log`) | 67 s under the lock |

The `examples` group now has 78 assertions and runs the examples in a temporary save path (it took 59 s, i.e. the
examples really run; at a049a3a it was 0 tests in 9.8 s). The runner prints only the test summary of a passing
group, so warnings inside passing groups are not visible in the log.

Probe scripts (in `/home/agent/.claude/jobs/b26d185e/tmp/reviews/round2/hc/`, run from the repo root with
`HYPERCLOUD_GUI_TESTS=false julia -t 2 --project=test <file>`; outputs in `*.out`):

| Probe | What it checks | Result |
|:--|:--|:--|
| `probe_closures.jl` | exact Euler Riemann solver against Toro's Table 4.2 (Tests 1–5: p*, u* at the contact) in Float64 **and Float32**; Burgers Gauss closure at `0.97 t_b` for D = 1, 2, `a = 1, −0.7`, Float64/Float32 (residual of `u = u₀(x − u t 1)`, no warnings); master-dictionary label collisions | **all 63 pass**. Toro values reproduced to < 2e-4 (Test 2 p* = 0.0018939 vs the table's 3-digit 0.00189) in both precisions; Gauss residual 2.2e-16 (F64) / 1.2e-7 (F32), 0 warnings; 50 entries = 1 reference + 49 `create_method` calls counted in the source, no collision |
| `probe_runs.jl` | (a) blow-up of a run with constant wave speeds (1-D linear advection, RK2 Upwind1 at CFL 5, and at a fixed `dt` of 5 Δx, `tmax = 20`); (b) runs with zero wave speeds (linear advection with velocity 0; Burgers with `u ≡ 0`); (c) BGK `τ = 0` Sod end to end with all five IMEX tableaus and with zero viscosity | 27 pass, **5 fail**: (a) both runs blow up (max \|u\| = 2.4e79 resp. 9.6e21) and are returned as normal results without the early-stop warning (**N1**); (b) both stationary runs stop at t = 0 with the "blown up" warning instead of reaching `tmax` (**N2**); (c) passes: finite, L1(ρ) vs exact Euler 0.0137–0.0143 for ARS222/ARS233/IMEXEuler/SSP332/PRSSP3 |

## Findings

### N1. The new early-stop warning misses blow-ups whenever the CFL step stays finite — medium — CONFIRMED
- **Where:** `src/time_integration.jl:127-136` (the only detection is `!(isfinite(dt_step) && dt_step > 1e-12)`);
  HyperCloudCore 0.7.9 `Grid/ParticleGrids.jl:384-387` (`sum_c > tol && (dt_buffer[i] = …)`, so a particle whose
  wave speed is NaN is *skipped* by the CFL estimate instead of producing a NaN step).
- **What:** the round-1 fix only reacts when the step itself collapses or is non-finite. That never happens for
  (i) every fixed-`dt` run (`:Time_dt`), (ii) every equation with constant wave speeds — `LinearAdvection`
  (`max_eigenvalue = eq.max_eigs[d]`), `BGKEquation` (`max_speed[d]`) and the kinetic relaxation runs, whose
  equation is a `LinearAdvection` — and (iii) partially blown-up nonlinear runs: NaN particles are ignored by the CFL
  estimate, so the step stays finite until *every* interior particle is NaN.
- **Scenario:** `probe_runs.jl` testset 1: 1-D linear advection, RK2 Upwind1, `:Time_CFL => 5.0`, `tmax = 20` →
  `sim.t[end] = 20.0`, max |u| = 2.4e79, no warning; same with `:Time_dt => 0.1` (max |u| = 9.6e21). Both return a
  normal `SimData`; stats against a reference are then simply huge/NaN, with nothing in the log.
  Docs (`docs/src/time_integration.md:19-24`, `solve_equation` docstring) promise a warning "after a blow-up".
- **Fix:** check the state, not only the step: e.g. at each snapshot (cheap, host copy already made in `saveData!`)
  test `all(u -> all(isfinite, u), us[snap])` and stop/warn at the first non-finite snapshot; or let
  `get_time_step` return NaN for NaN wave speeds (HyperCloudCore: `!(sum_c <= tol)` instead of `sum_c > tol`).
  Add the probe's linear-advection case as a regression test.

### N2. Regression from the round-1 fix 6: runs with zero wave speeds now stop at t = 0 with a "blown up" warning — low — CONFIRMED
- **Where:** `src/time_integration.jl:131` (`isfinite(dt_step)` rejects `dt_step = Inf`).
- **What:** `get_time_step` returns `Inf` when no particle has a positive wave speed (`ParticleGrids.jl:365,387`).
  At a049a3a the loop then took one step of `min(Inf, tmax − t) = tmax`, which is exact for stationary data (zero
  flux). Now `Inf` is treated like NaN: the run stops at `t = 0` with "the time step is not finite … the state has
  probably blown up", and the returned data end at `t = 0` (`sim.t == [0.0, 0.0]`).
- **Scenario:** `probe_runs.jl` testset 2: linear advection with `:PDE_velocities => ((0.0,),)` and CFL → `t = [0.0, 0.0]`
  plus the warning; Burgers with `:IC_a => (0.0,)` (`u ≡ 0`) → the same. Rare in practice (steady-state or
  zero-velocity sanity checks, a Burgers background at rest), but the message is wrong and the run is lost.
- **Fix:** distinguish `isnan(dt_step)` (blow-up) from `dt_step == Inf` (no wave moves: take `tmax − t`, or a
  `dt` per snapshot interval). Note that with N1's fix (state check) the `Inf` case no longer needs to be an error.

### N3. Docs, repo CLAUDE.md and a test comment say `:Grid_bc_map` is ignored, but HyperCloudCore 0.7.9 reads it — low — CONFIRMED (by reading)
- **Where:** `docs/src/configuration.md:58-59` (warning box "A key such as `:Grid_bc_map` is silently ignored"),
  `CLAUDE.md:49`, `test/runtests.jl:150`; HyperCloudCore 0.7.9 `Builders/domains.jl:17`
  (`get(domain_conf, :bc, get(domain_conf, :bc_map, …))`). `src/Examples/euler_1d_sod.jl:47` uses `:Grid_bc_map` and
  its boundary conditions are applied (the Sod tests and the examples group pass with the Dirichlet/outflow BCs).
- **What:** a user following the warning box would "fix" working configurations; the example used for "misspelled
  keys are silently ignored" is wrong. Round-1 listed this under minor issues; it is not in the fix list and still
  present.
- **Fix:** drop the warning box (or document `:bc_map` as an alias) and pick another example of an ignored key in
  `CLAUDE.md` and the test comment; optionally switch `euler_1d_sod` to `:Grid_bc` for consistency with
  `bgk_1d_sod` (that changes the parameter hash of its cached results, so call it out).

### N4. Implicit sources of BGK with τ = 0 still produce Inf/NaN in the stages with `a_ii = 0` (harmless today) — low — SUSPECTED (fragile, not a current failure)
- **Where:** `src/SourceTerms/bgk.jl:174-179` (`evaluate_source` divides by `τ`); HyperCloudCore
  `TimeSteppers.jl:353-356,401-403` (`_stage_implicit_rhs!` evaluates the implicit source at stages with `a_ii = 0`).
- **What:** for IMEX tableaus with an explicit first implicit stage (`IMEXEuler`, `ARS233`), `K_I[1] = (f_eq − f)/0`
  is ±Inf or NaN. It is never used because all shipped tableaus have a zero first column in `A` and `b[1] = 0`
  (`_stage_predictor!` skips zero coefficients), which my probe confirms (τ = 0 runs are finite and accurate for all
  five tableaus). A tableau with `a_i1 ≠ 0` after an explicit first stage (e.g. Kennedy–Carpenter ARK types) would
  turn every particle into NaN.
- **Fix:** in `bgk_sources`, or in `evaluate_source`, return `zero(f)` (or `f_eq − f` scaled by a large finite rate)
  when `τ = 0`, or reject τ = 0 for tableaus with `a_ii = 0` and a non-zero column; at least add a comment.

### N5. Euler exact solver: absolute tolerances make it scale dependent — low — SUSPECTED
- **Where:** `src/PDEs/euler.jl:151-157` (`abs(f_p) < 1e-9` absolute in velocity units, `p_star` floored at `1e-9`
  absolute, `dfdp + 1e-9`), `:166` (`t <= 1e-9` returns the IC).
- **What:** the closure is accurate for O(1) data (Toro Tests 1–5 to < 2e-4 in Float64 and Float32, `probe_closures.jl`),
  but for data with small pressure scales (e.g. `p ~ 1e-8`, a non-dimensionalisation with small numbers) the floor
  1e-9 is no longer negligible and the near-vacuum star pressure is wrong by O(1); in Float32 the 1e-9 velocity
  tolerance is never reached, so the iteration always runs 100 times (only at construction, so harmless).
  The same pattern (absolute tolerance) was fixed for the Burgers closure in round 1.
- **Fix:** relative tolerances (`|f| ≤ √eps(T) · (|u_L| + |u_R| + c_L + c_R)`, floor `eps(T) · min(p_L, p_R)`).

## Round-1 findings

| # | Handling | Verdict |
|:--|:--|:--|
| 1 | deferred (HyperCloudCore work-group size) | **OK** as a deferral: the cause is the fixed work-group size of HyperCloudCore 0.7.9's `KABackend`, which HyperCloud cannot change; the reason is sound. Reproduced at d4fe5e4 (`ka_cpu_sims_bgk.log`: same 3 failures, same work-group error). Nothing in the suite marks the testset as broken under `ka_cpu` or documents the limitation, so the `ka_cpu` variant the repo's CLAUDE.md asks for stays red until the HyperCloudCore release; a `@test_broken` guarded by `TEST_BACKEND == "ka_cpu"` (or a docs note) would make that explicit. |
| 2 | fixed (tmax 1.0; warn once past breaking / on Newton failure) | **OK.** `burgers_kinetic` stops at 1.0 < t_b = 1.166; the closure warns once per closure (atomic flags); `t_b` formula re-derived and correct for any D and sign of `a`. The regression tests (mass of the `burgers_kinetic` reference conserved at `tmax`, exactly one "breaking time" warning) fail without the fix. |
| 3 | fixed (ARS222MUSCL2 in the master dict; docs) | **OK.** Entry added, sweep runs it, the examples group asserts that every default method exists; `docs/src/wrappers.md` now matches the dictionary (checked entry by entry; 50 entries, no label collision). |
| 4 | fixed (examples group; driver test) | **OK.** Temporary save path, `force_overwrite = true`, every task must exist, reach `tmax` and stay below a measured error bound; the defaults check would have caught Finding 3; the driver test now asserts. The examples run with a shortened `tmax`, which is fine for the purpose. |
| 5 | fixed (ArgumentError; docs; dead helpers removed) | **OK.** `_parse_representation(::Val{R})` throws a clear `ArgumentError` for every non-conservative name; docs (`configuration.md`, `PDEs/main.md`) agree; test covers `:primitive`, `:lagrangian`, `:bogus`. The kept `NCRepresentation` methods in `euler.jl` are unreachable but documented as the extension point. |
| 6 | fixed (warning when the step collapses / is not finite; no NaN times) | **NOT OK (partly).** The two cases it targets work (Sod at CFL 4 warns; NaN `dt` no longer records a NaN time; tests fail without the fix). But the detection only looks at `dt`: blow-ups with constant wave speeds (linear advection, BGK, kinetic runs), with a fixed `dt`, or with only some NaN particles (the CFL kernel skips NaN) still end "successfully" without a warning (**N1**), and the change introduced a regression for stationary data (`dt = Inf` → stop at t = 0, **N2**). Minor: the NaN-`dt` test enshrines `sim.t == [0.0, 0.0]` (two snapshots at the same time). |
| 7 | fixed (snapshots ≥ 1) | **OK.** Clear `ArgumentError`, docs and docstring updated, test checks the logged exception. |
| 8 | fixed (τ = 0 → f_eq; explicit τ = 0 and τ < 0 rejected) | **OK.** `isinf(λ)` branch correct; `bgk_sources` rejects τ = 0 / μ_ref = 0 for explicit tableaus; τ < 0 and NaN rejected at construction; unit test fails without the fix. My probe adds the end-to-end check: τ = 0 Sod runs with all five IMEX tableaus are finite and match the Euler solution (L1(ρ) 0.014). Residual fragility for stages with `a_ii = 0`: **N4**. |
| 9 | fixed (relative tolerance + bisection; warn once) | **OK.** Bracket and bisection fallback correct (F increasing before breaking); Float32 converges without warnings up to 0.97 t_b in 1-D and 2-D for positive and negative amplitudes (`probe_closures.jl`, residual 1.2e-7). |
| 10 | fixed (vacuum generation) | **OK.** Pressure-positivity test and the vacuum branch (fans, vacuum, mirrored frame) re-derived; the conservation tests (mass/energy fluxes through the ends) and the mirror test are meaningful and fail without the fix (old code threw `DomainError`). Docs note that vacuum *initial* states are not handled. The Newton's absolute tolerances remain (**N5**). |
| 11a | fixed (relative tolerances in the closure checks) | **OK.** Commutation check `‖AB − BA‖ ≤ √eps ‖A‖‖B‖` and the eigenvector check `≤ √eps max(1, max|Λ|)`; test with O(100) entries (F64) and Float32 fails with the old absolute tolerances; non-commuting matrices still rejected. The eigenvector tolerance was loosened from 1e-10 to 1.5e-8 in Float64, which is harmless. |
| 11b | deferred (Burgers closures on periodic domains) | **OK** as a deferral: no shipped example or test evaluates a Burgers closure where the wave reaches a periodic boundary (`burgers_kinetic`: width 1 on [−5, 5] to t = 1; tests: width 0.2 on [−1, 1] to t = 0.15). It is not mentioned in `docs/src/PDEs/burgers.md`, though; a one-line note ("periodic wrap is ignored") would prevent silent misuse. |

Round-1 minor issues: the `:Grid_bc_map` statement is unchanged (**N3**); the other minor items (floors 1e-7 vs 1e-9,
`q_riemann` length check, `:indices::Vector{Int}`, `:Kinetic_save_relax` not read by the BGK runner, labels
omitting `limiter_mode`/`delta_relax`, duplicate snapshot times) are still present and still minor.

## Test coverage

Well tested now: everything listed by round 1, plus representation errors, `snapshots ≥ 1`, the early-stop warning
for the targeted cases, τ = 0 / τ < 0 handling of the BGK collision, Euler vacuum generation (conservation, mirror),
relative closure tolerances (large entries, Float32), Burgers breaking-time warning, and real assertions for every
exported example and the `run_all_simulations` driver.

Missing:
- A blow-up that keeps a finite step (linear advection / BGK / fixed `dt`) — `probe_runs.jl` testset 1 (fails today, N1).
- Stationary data (zero wave speeds) reaching `tmax` — `probe_runs.jl` testset 2 (fails today, N2).
- BGK τ = 0 end to end (only unit-tested in the suite) — `probe_runs.jl` testsets 3–4, ~10 s, worth adopting.
- Euler exact solver against Toro's Tests 2–5 (only Sod and vacuum are tested), incl. Float32 — `probe_closures.jl`
  testset 1, < 1 s, worth adopting.
- `ka_cpu` coverage of 2-D BGK remains red until the HyperCloudCore release (round-1 Finding 1).

## Minor issues

- `src/time_integration.jl:146-149`: when one step crosses several snapshot times, the same state is stored several
  times with the same `t` (also after an early stop at t = 0: `t == [0.0, 0.0]`).
- `test/sims.jl` "Early stop": the Sod case only asserts `sim.t[end] < 0.2`; it does not check that the stored data
  are finite (they are not necessarily, the last state can be the blown-up one, as the docstring says).
- `generate_relax_params(v_mag::Float64, …)` rejects an integer speed (`generate_relax_params(2, 1, 1)` is a `MethodError`).
- `test/gpu/runtests.jl` "second reserved device run": `first(l.kwargs[:exception])` throws a `KeyError` for an
  error log without an `:exception` keyword (only matters if such a log appears).
- `benchmark/runners.jl`: reads correctly; with `BENCH_GPU=0` the CPU-only table is complete (GPU rows are skipped).

## Verdict

The round-1 fixes are, with one exception, complete and well tested: the examples group now really runs and checks
the examples, the Burgers and Euler closures handle breaking and vacuum correctly (I reproduced Toro's Tests 1–5 in
both precisions), τ = 0 BGK works end to end with every IMEX tableau, and the docs match the code. The default suite
passes (1061 tests, 70 s). The exception is the early-stop handling (round-1 Finding 6): it detects only a collapsed
or non-finite *time step*, so blow-ups of linear advection, BGK and kinetic runs and of any fixed-`dt` run still
finish silently (N1, medium), and it now aborts stationary runs at t = 0 (N2, low). The remaining items (stale
`:Grid_bc_map` warning, a latent τ = 0 hazard for other tableaus, absolute tolerances in the Euler Newton) are low.
