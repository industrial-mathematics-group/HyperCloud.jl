# Review: HyperCloud.jl — branch `claude/review-followup` (origin/main_dev c276d39 .. fb93caf, 2 commits)
Reviewer: Claude Opus 5.5 (Anthropic), new session without the generating conversation (A2)

Independent review of b768f08 ("Time loop: report each stop once, stop on an infinite CFL number") and fb93caf ("Euler,
builders, BGK: review minors"), which address reviews/d3a0415.md findings 1–2, reviews/d4fe5e4.md N5 and minors, and
reviews/a049a3a.md minors. Test environment: registered HyperCloudCore 0.7.9 (`~/.julia/packages/HyperCloudCore/YIxkg`).
Probes and outputs: `/tmp/review_pr5/probes/`. Nothing in the worktree was edited; the scratch worktree of origin/main_dev
(`/tmp/review_pr5/old`, test manifest copied from the branch) has been removed.

## Scope

Read completely: the whole diff (13 files, +140/−37); `src/time_integration.jl` (all 245 lines, every exit path);
`src/PDEs/euler.jl` (floors, closure `:122-237`); `src/SourceTerms/bgk.jl`; `src/SimulationFunctions/bgk.jl`;
`src/inbuilt_methods.jl`; `src/Builder/kinetic.jl:15-30`; `src/ICs/quadrant_riemann.jl`; the new and changed tests
(`test/sims.jl:56-110`, `test/sims_bgk.jl:45-70`, `test/units.jl` at the changed lines and `:930-952`, `test/gpu/runtests.jl:170-182`);
reviews/d3a0415.md, d4fe5e4.md (N5, minors), a049a3a.md (minors).
Cross-checked: `src/SimulationFunctions/direct.jl:50-95` and `kinetic.jl:85-100` (is_cfl, `:save_relax`), `src/device.jl:48-65`
(`host_view`, `set_states!`), `src/PDEs/bgk.jl:310-370` (`equilibrium`, `_scaled_residual`); HyperCloudCore 0.7.9
`Builders/domains.jl:7-40` (`:bc_map` alias), `Builders/mood.jl:20-40`, `Builders/tableaus.jl:20-30`,
`Interpolation/Limiter.jl` (modes), `API.jl` (`cons2prim` is used only by HyperCloud's flux and the ALE frame);
docs `time_integration.md`, `configuration.md`, `wrappers.md`; Simulations (`/workspace/julia/Simulations`, worktrees excluded):
every `create_method` call and every label list.

## Tests run

| Command | Result | Wall time |
|:--|:--|:--|
| `flock /tmp/hcc_test.lock timeout 60m julia -t 8 --project=test test/runtests.jl` (polyester, GUI tests on, parallel groups) | **PASS**, exit 0: units 835, sims-bgk 28, sweep-a 50, examples 78, sims 127, sweep-b 48 = **1166 passed**, 0 failed/errored; top level 6/6 (`suite_new.log`) | 160 s (groups 26–70 s, peak 1.4–1.9 GiB each) |
| `HYPERCLOUD_TEST_GROUP={sims-bgk,examples,sims} julia -t 2 --project=test test/runtests.jl` (one process each, to see the logs) | PASS; **no `Warning` line at all** in the three logs, in particular no "BGK initial state" warning from any shipped BGK example or test (`newgroup_*.log`) | ~3 min |
| The branch's `test/units.jl`, `sims.jl`, `sims_bgk.jl` run against origin/main_dev code (groups units, sims, sims-bgk, `-t 2`) | units: 806 pass, **11 fail, 4 error**; sims: 123 pass, **4 fail**; sims-bgk: 27 pass, **1 fail** (`oldcode_newtests_*.log`, details under Test coverage) | ~2.5 min |

Probes (`-t 2`, run on the branch and on origin/main_dev unless noted):

| Probe | Result |
|:--|:--|
| `loop.jl`: 40 exit paths of `solve_equation` with warning counts (`loop_{old,new}.out`) | see Addressed items 1–2; every stop now gives exactly one stop warning (plus at most the growth warning); all times finite; data end at the stop time |
| `riemann.jl`: Toro 1–5 + near vacuum, 1e10 pressure ratio, colliding streams, density ratio 1e6, equal states; scales p × {1, 1e-8, 1e8, 1e-20}, u × {1, 1e-4, 1e4, 1e-10}; Float64/Float32; instrumented copies of the old and new Newton loops; t ∈ {0, 1e-300, 1e-40, 1e-12, 1e-9} (`riemann_{old,new}.out`) | new: converges in all 96 cases (Float64 ≤ 15 iterations, Float32 ≤ 9), residual ≤ 16 eps·u_scale; old: 100 iterations without convergence in 20 cases, wrong p* by O(1)–O(1e7) at the small scales |
| `riemann_zero.jl`: p = 0 on one/both sides, p = 1e-300 | same behaviour as origin/main_dev (finite except the already non-finite "p_L = 0 colliding" case); no regression |
| `euler_cmp.jl` + `euler_diff.jl`: Sod (all 6 example methods), Lagrangian+adaptivity Sod, Toro 2, vacuum-generating u = ∓6 / ∓20, pressures 1e-6/1e-8, 2-D quadrant Riemann (Lax–Liu 3, 50², MOOD U2 EPD1 and VK), 2-D double rarefaction; Float64/Float32; old vs new, new vs new (determinism), new on polyester vs new on `ka_cpu` (`set_backend!(HyperCloudCore.KA.CPU())`, as `HYPERCLOUD_BACKEND=ka_cpu`) | Finding 1; non-MOOD runs bit-identical; runs deterministic across processes; `ka_cpu` runs all 38 cases (no failures) |
| `floor_hits.jl` (redefines `cons2prim` in the probe process to count floor hits) | Sod RK2MUSCL2_U2_EPD1: 76 of 3.58 M `cons2prim` calls see ρ or p < 1e-7 (first: ρ = −0.124, p = −0.156); RK2MUSCL2_VK: 0 |
| `sod_l1.jl`: L1(ρ) vs the exact solution, old and new | unchanged to ≤ 0.13 % (largest: RK4MUSCL4_U2_EPD1 Float32 0.009015 → 0.009027) |
| `labels.jl`: `get_master_method_dict()` keys and hash of every parameter dict | **identical** on both (50 entries) |
| `bgk_warn.jl`, `bgk_conserve.jl`: BGK initial-state warning, 17 states × Float64/Float32, cost at 100² particles, M = 256 | Addressed item 10, Minor issues |

## Findings

### 1. The floor change 1e-7 → 1e-9 changes the results of MOOD runs; not stated anywhere — low — CONFIRMED
- **Where:** `src/PDEs/euler.jl:38,41,48` (`cons2prim`, used by `flux`).
- **What:** the commit message and the PR text present the single floor as a consistency fix. It only acts when a flux is
  evaluated at ρ or p below the floor, which does not happen for admissible states — but it does happen in MOOD runs: the unlimited
  high-order candidate reconstructs non-admissible face states (negative density and pressure), whose fluxes enter the candidate
  that MOOD then accepts or rejects. With a different floor the candidate values and, in Float32, some accept/reject decisions change.
  Runs without MOOD (VK, Upwind; Sod, Toro 2, vacuum u = ∓6/∓20) are bit-identical.
- **Evidence:** `floor_hits.out`: 76 floor hits in Sod RK2MUSCL2_U2_EPD1 (ρ = −0.124, p = −0.156), 0 for VK. `euler_diff` old vs new
  (max-norm of the final state): Float64 Sod MOOD 1.9e-10 … 8.4e-9, 2-D Lax–Liu 3 U2 EPD1 3.5e-8; Float32 Sod U2 EPD1 3.3e-4,
  RK4MUSCL4_U2_EPD1 8.6e-3 (rel 3.4e-3), 2-D U2 EPD1 5.3e-3; pressures below 1e-7 (p_R = 1e-8 Sod-like) 1.4e-6 rel — there the
  old floor clipped *physical* pressures, so the new value is better. The Float32 spread is of the same size as polyester vs `ka_cpu`
  on the same code (1.3e-2 for RK4MUSCL4_U2_EPD1, 4.2e-2 for the 2-D double rarefaction, `euler_new_ka.out`), and L1 errors move
  by ≤ 0.13 %. New vs new in a second process: identical, so the changes are not noise.
  The parameter dicts (and hence run hashes, e.g. Simulations' `_param_hash`) are unchanged, so cached MOOD results from
  origin/main_dev are no longer bit-reproducible under the same hash.
- **Fix:** none needed in the code. State it in the PR description / changelog ("results of MOOD runs change at round-off level
  in Float64 and up to ~1e-2 pointwise in Float32 at shocks; non-MOOD runs unchanged"), so cached data or regression baselines
  are not mistaken for a regression.

### 2. Docs and docstrings still describe the old behaviour of four changed items — low — CONFIRMED (by reading)
- **Where / What:**
  - `docs/src/wrappers.md:21-24` and the `create_method` docstring (`src/inbuilt_methods.jl:7-11`): the label rule does not mention
    the new `"<limiter> <mode>"` (non-`:hard` mode) and `"δ=<value>"` (non-zero `MOOD_delta_relax`) parts.
  - `docs/src/configuration.md:161`: `:indices` is still "`Vector{Int}` of edges"; a tuple is accepted now.
  - `docs/src/configuration.md:155` ("Only read by `run_kinetic_simulation`") and `src/SimulationFunctions/bgk.jl:16`
    ("with `:save_relax => true`"): the BGK runner now also reads `:Kinetic_save_relax`. Repo CLAUDE.md asks to update
    `configuration.md` when a key changes.
  - `src/inbuilt_methods.jl:137`: the docstring signature is still `generate_relax_params(v_mag::Float64, dim::Int, NM::Int)`.
- **Fix:** one line each (e.g. wrappers.md: "…the limiter (with its mode if not `:hard`), and for MOOD the strategy (plus the
  criterion if it is not `:U2`, and `δ=…` for a non-zero `MOOD_delta_relax`). Examples: … `"RK2MUSCL2(VK soft)"`,
  `"RK2MUSCL2(EPD1,δ=0.0001)"`").

No defects found in the changed logic itself; details per item below.

## Addressed items

1. **d3a0415 #1 (stop reported twice): OK.** `stopped` is set on all three stop paths (`time_integration.jl:171`, `:180`, `:199`)
   and not on the round-off end (`:184`) or a normal end, so the final slot is checked exactly when the loop reached tmax.
   Warning counts (`loop_new.out` vs `loop_old.out`): linear advection CFL 5 snaps 10: 3 → 2 (growth + one "stopped before tmax");
   Float32: 2 → 1; ghosts kept: 3 → 2; Burgers NaN IC snaps 1/4: 2 → 1; BGK wave CFL 5: 3 → 2. Unchanged where correct: snaps 1 / 2
   blow-ups reaching tmax keep their final "not finite" warning (1 resp. growth + final); fixed-dt Sod snaps 1/4 (1 resp. 2);
   legitimate runs (Sod VK snaps 1 and 51, Lagrangian + adaptivity, zero velocity, fixed dt 0.1/0.03, round-off end tmax = 0.3 in
   Float64 and Float32) give no warning. Data: all times finite, final slot = state at the stop, unreached slots dropped.
   One side effect (intended by "each blow-up once"): after a step-collapse stop the final growth check is skipped too — Sod CFL 4
   snaps 1 now gives only the stop warning (old: stop + "4.6e9 times larger"); the stop warning already says "probably blown up".
2. **d3a0415 #2 (`:Time_CFL => Inf` silent): OK.** `is_cfl && isfinite(dt_inp) && dt_step == Inf` (`:167`): CFL ±Inf / NaN stop at
   t = 0 with "from the CFL number …, which must be positive and finite", also for zero velocities (Inf·Inf, 0·Inf = NaN, −0.2·Inf
   all land in `:174`); CFL −0.2 / 0 and fixed dt ±Inf / NaN / −0.01 / 0 name the configured value; zero velocity with CFL 0.2 still
   advances to tmax silently (N2 behaviour kept); BGK and adaptivity paths behave the same. The round-3 minor (configuration error
   reported as blow-up) is fixed for non-positive and non-finite inputs.
3. **N5 (Riemann solver tolerances): OK.** Relative tolerance `16 eps(T)·(|u_L|+|u_R|+c_L+c_R)`, floor `eps(T)·min(p_L, p_R)`,
   no `+1e-9` in the slope, `dfdp > 0 || break` (also stops on NaN). `riemann_new.out`: correct p* at every scale (Test 2 at p×1e-8:
   1.894e-11, old 1e-9 = 52× too large; at p×1e-20 the old solver returned the initial guess for most tests); Float32 converges in
   2–9 iterations (old: 100 without convergence in 9 of 12 O(1) cases, harmless but as reported). Accuracy vs Toro is unchanged in
   Float64 (≤ 4e-6, Toro's digits) and ≤ 2.4e-6 in Float32 (slightly looser than the old 100-iteration result, 4.4e-7, well inside
   the test's 2e-4). Removed `t <= 1e-9` cut-off: t ∈ {1e-300, 1e-40, 1e-12, 1e-9} give the self-similar solution (star state at
   x = x0), finite in both precisions; t = 0 (or a Float32 underflow to 0) returns the IC as before. Zero-pressure data behave as on
   origin/main_dev.
4. **Floor consistency (a049a3a minor): OK** (`_euler_floor`, 1e-9 everywhere; `T(1e-9)` is a normal Float32, kernel-safe: all
   38 Euler cases run under `ka_cpu` and agree with polyester to round-off / MOOD-decision level). Result changes: Finding 1.
5. **q_riemann length check: OK** (`quadrant_riemann.jl:41-42`, `ArgumentError` at build time, 2^D from the context).
6. **Tuple `:Kinetic_indices`: OK** (`collect(Int, …)`; `Kin2Macro` gets a `Vector{Int}`; the tuple test gives the same map).
7. **`generate_relax_params(::Real, ::Integer, ::Integer)`: OK** (forwards to the `Float64/Int` method; no ambiguity or recursion).
8. **BGK runner `:Kinetic_save_relax`: OK** (`SimulationFunctions/bgk.jl:103`, same semantics as `kinetic.jl:96`; also reached
   through `run_kinetic_simulation`'s forwarding since the full params are passed). Docs: Finding 2.
9. **Labels: OK.** Default mode `:hard` and δ = 0 leave labels unchanged; `get_master_method_dict()` keys and parameter-dict hashes
   are identical on both commits (`labels_{old,new}.out`, 50 entries). Simulations: the only calls with a non-zero
   `MOOD_delta_relax` (`tools/mood_u2/*`, `tools/ssp_mood/ssp_study.jl`) discard the label (`_, method = …`) and
   `experiments/methods_review.jl`, `Mach3Step.jl:169`, `DMR.jl:154`, `clainshock.jl:158` pass `custom_label`; no call uses
   `limiter_mode = :soft`; the label lists in `experiments/` name master-dict or custom labels only. Parameter dicts are not touched
   by the change, so no run hash changes.
10. **BGK initial-state warning: OK.** True positives: u = 5.5 with vmax 6 (r = 0.36), unresolved θ = 0.01 (r = 0.99, 1.99), in
    both precisions; no false positives for 13 representable states per precision (rest, the test wave, u up to 4 in 1-D / 4.5 in
    2-D, θ = 10, ρ = 1e-6 … 1e6, tiny momentum, γ = 1.4 Chu pairs): r ≤ 1.6e-15 (Float64) and ≤ 4.5e-7 (Float32) against
    thresholds 1.5e-8 / 3.5e-4. None of the shipped BGK examples/tests warns (suite logs). Cost: init-only; `moments` costs about as
    much as `equilibrium` (10⁴ particles, M = 256: +0.066 s), negligible against a run. One warning per run.
11. **gpu KeyError: OK** (`haskey(l.kwargs, :exception)` works on the `Base.Pairs` of `Test.LogRecord`, checked).
12. **Comments/docs:** `docs/src/time_integration.md:19-24` and the `solve_equation` docstring now match the behaviour (final slot
    warns without stopping; each blow-up once; infinite CFL number stops). `CLAUDE.md` (`:Grid_bc_map` is an alias, confirmed in
    Core `parse_bc`; `:Grid_boundary` is indeed ignored) and the BGK τ = 0 comment are correct.

## Test coverage

The new tests are meaningful; on origin/main_dev code they fail as follows (`oldcode_newtests_*.log`):
- units: q_riemann short tuple (`:137`), labels `VK soft` and `δ=0.0001` (`:267`, `:269`), the 8 scaled Toro assertions
  (Test 2 at p×1e-8, Float64 and Float32: p 5.28e-12 vs 1.89e-11, u* −2.9e-5 vs 0) = 11 failures; `generate_relax_params(2, 1, 1)`
  (`:310`), `_euler_floor` (`:365`, undefined), tuple indices D = 1, 2 (`:400`) = 4 errors.
- sims: exact warning count of the CFL-5 stop (`:84`), one warning for the NaN IC (`:91`), CFL = Inf warning and `t == [0, 0]`
  (`:96-97`) = 4 failures.
- sims-bgk: `:Kinetic_save_relax` (`:58`) = 1 failure.
Pass on origin/main_dev too (coverage additions, as suggested by d3a0415/d4fe5e4): the fixed-dt Sod blow-up with snapshots = 1
(`sims.jl:99-103`), τ = 0 BGK end to end with conservation (`sims_bgk.jl:61-69`), Toro 1–5 at scale 1, the positive q_riemann case,
the default-label assertions (`:268`, `:270`).

Missing (small, worth adding):
- the BGK initial-state warning: e.g. `bgk_wave_params(16)` with `:IC_c_offset => (1.0, 5.5, 16.0)` → `@test_logs (:warn, r"cannot
  represent")`, and `@test_logs min_level = Logging.Warn run_sim(bgk_wave_params(16))` for no false positive (< 2 s);
- a fixed-step configuration error with a non-positive value (`:Time_dt => -0.01` or `0.0`) asserting the new "from the fixed step …"
  wording (only NaN/Inf fixed steps and the CFL Inf case are pinned).

## Minor issues

- `src/time_integration.jl:174,184`: the absolute thresholds `1e-12` (collapse) and the round-off end make the loop scale dependent,
  the same class of issue N5 fixed for the Riemann solver: a positive configured step below 1e-12 (`:Time_dt => 1e-13`, or
  `:Time_CFL => 1e-14`) is reported as "collapsed …; the state has probably blown up", and a run with `tmax ≤ 1e-12` returns the IC
  at t = 0 silently. Pre-existing; a relative test (`dt_step > 1e-12 * tmax_val`, `dt_val <= eps(T) * tmax_val`) would remove it.
- `src/SourceTerms/bgk.jl:92`: the residual is relative to `maximum(abs, U)`, so an error in a component much smaller than E
  (momentum at small Mach numbers) is hidden behind the energy; `_scaled_residual(moments(eq, f) - Uv, scale)` with the
  `scale = (ρ, ρ·vscale…, E)` of `equilibrium` would be per component. Not observed to matter in the probes.
- `src/SourceTerms/bgk.jl:81` docstring: "the collision term then does not conserve what the run starts from" — the primary effect is
  that the run starts from `moments(f) ≠ U` (u = 5.5: absolute moment error 5.6); the collision at that `f` is conservative to
  4e-15 (`bgk_conserve.out`), it fails only in the unresolved cold case. Suggest "…the run starts from a different macroscopic state
  (and the collision may not conserve it)".
- `src/SourceTerms/bgk.jl:37-38`: the function names are `IMEX_Euler_Tableau`, `IMEX_ARS233_Tableau`, … (with `_Tableau`); cosmetic.
- `test/sims.jl:97`: `simcfl.t == [0.0, 0.0]` enshrines the duplicated time of a stop at t = 0 (as the existing NaN-dt test does);
  fine, but it pins the bookkeeping rather than the behaviour.

## Verdict

The change set does what it claims. Every stop of `solve_equation` is now reported exactly once (40 exit paths probed, counts as
intended, data consistent), an infinite, NaN or non-positive CFL number or fixed step is a configuration error with a message that
names the value, and the zero-velocity shortcut is kept. The exact Riemann solver is now scale invariant and converges in ≤ 15
(Float64) / ≤ 9 (Float32) iterations on all 96 probe cases, the removed time cut-off is safe, and the remaining minors (q_riemann
check, tuple indices, integer speed, `:Kinetic_save_relax`, labels, BGK warning, gpu KeyError) are correct. Master-dictionary labels
and parameter hashes are unchanged, and Simulations is unaffected. The full suite passes (1166/1166, 160 s) with no warnings in the
logs, and the new tests give 20 failures or errors on origin/main_dev code. Findings: **0 high, 0 medium, 2 low** (1: the floor change alters MOOD
results, at round-off in Float64 and up to ~1e-2 pointwise in Float32, which should be said in the PR; 2: four doc/docstring spots
still describe the old behaviour). Neither blocks merging.
