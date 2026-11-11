# Re-review: HyperCloud.jl — branch `claude/walls-review` (origin/main_dev 81b8f0f .. 1b9ace6, 4 commits)
Reviewer: Claude Opus 5.5 (Anthropic), new session without the generating conversation (A2)

Independent re-review of the fixes for `/tmp/brev3/r3/review.md` (PR #18 walls, PR #17 run metadata): d0bdde7 ("BCs:
mass-conserving BGK specular reflection off the node grid"), 474036b ("Runners: :finite flag in the run metadata"), ad18783
("Tests: discriminating BGK wall checks, trapezoid mass drift"), 1b9ace6 ("Walls: hashed piston shape, operator checks before
the run, minors"). Worktree `HyperCloud.jl/.claude/worktrees/rr-walls` (1b9ace6, clean), HyperCloudCore
`HyperCloudCore.jl/.claude/worktrees/rr-moving` (405f797, branch `claude/moving-review`) via `/tmp/brev3/env_rr`; the parents
(HyperCloud `review-walls` = 81b8f0f, HyperCloudCore `review-mb` = 1fcc2ce) via `/tmp/brev3/env` for before/after runs.
Probes and outputs: `/tmp/brev3/rr3/probes/`. Nothing in the worktrees or environments was edited, no git state was changed.

## Scope

Read completely: the whole diff (14 files, +255/−45); `src/BoundaryConditions/_main.jl` (all 255 lines);
`src/Examples/euler_1d_piston.jl`; `src/device.jl:130-165`; `src/macros.jl:140-180`; the runner hunks
(`src/SimulationFunctions/{direct,bgk,kinetic}.jl`, packaging and `_check_wall_ops` call sites); `test/sims_walls.jl`
(whole file), `test/sims.jl:80-110`, `test/units.jl:40-50`; `docs/src/walls.md`, the touched parts of `configuration.md`,
`examples.md`, `simulation.md`; repo `CLAUDE.md`; the earlier review and its probes; reviews/fb93caf.md (format).
Cross-checked: `src/PDEs/bgk.jl:119-410` (grid, `center`, `moments`, `equilibrium`: entropic, `ok = false` only for
non-finite/ρ ≤ 0/θ ≤ 0 input, otherwise returns the best Newton iterate; `is_admissible` allows `f_k < 0`); HyperCloudCore
`src/Grid/BoundaryConditions.jl:140-390` (`WallPoint`, `MirrorBC`, `_WallOp`, clamp of the fitted value *before* the operator,
operator call `op(eq, U, wall)`), `src/Builders/domains.jl:68-128` (string shapes, `let T = …` only on rr-moving 405f797);
PDEStudioCore 0.5 `IOUtils.jl:145` (`_param_hash`).

## Tests run

| Command | Result | Wall time |
|:--|:--|:--|
| `HYPERCLOUD_GUI_TESTS=false flock /tmp/hcc_test.lock timeout 60m julia -t 6 --project=/tmp/brev3/env_rr test/runtests.jl` | **PASS**, exit 0: units 841, walls 57, sims-bgk 28, sweep-a 50, sweep-b 48, examples 92, sims 136 = **1252 passed**, 0 failed/errored (`fulltests.log`) | groups 29–77 s, peak 1.0–1.3 GiB each |
| `HYPERCLOUD_BACKEND=ka_cpu HYPERCLOUD_TEST_GROUP=walls … julia -t 6 …` (under the lock) | **PASS** 57/57 on `KABackend{POCLBackend}` (`walls_ka.log`) | 65 s |

Probes (all `-t 2`; each run in both environments where a comparison makes sense):

| Probe | What |
|:--|:--|
| `ops.jl` → `ops.out` / `ops_parent.out` | `SlipWall`/`DiffuseWall` on single states: drifting equilibria (1-D moving walls, `n = ±1`, `vcenter` 0.3/0.5, Nv 16/64, Gauss–Hermite; 2-D Nv 12/24, θ ∈ {0, π/8, π/6, 0.3, π/4, 1}, moving oblique wall; 3-D Nv 8; Float32; a fast drift u = 3.5), non-equilibrium states (1-D/2-D two-beam, hot/cold half-Maxwellians, a single-node f, f = −f_eq), DiffuseWall α ∈ {0, 0.5, 1}; half-range fluxes of mass, tangential momentum and energy, the "central" mass flux Σ w c (f + g)/2, ghost moments against the exact reflection, `min g / max f`; continuity of the flip switch; allocation and time per ghost |
| `runs.jl disk/vcenter/bgkpiston/hot/pos` → `runs_b.out`, `runs_a.out` (piston), `hot.out`, `runs_pos.out` | 2-D disk (24², Nv 8): gas at rest (slip/diffuse/α = 0.5), drifting gas, Riemann problem at τ = 1 and 1e-3; 1-D `vcenter` runs; BGK piston with a slip face (also on a grid centred at the piston velocity); hot wall and slip control (trapezoid); positivity of f over a run |
| `ka.jl` → `ka.out` | the general specular path inside a KA kernel (POCL): BGK piston with a slip face, 2-D Float32 disk; `:backend`/`:finite` metadata |
| `misc.jl` → `misc.out` | piston example: hashes (speed 0.5/0.4/`1//2`/`0.5f0`), Float32 run, speed/face edge cases, speed 0.8 / face 1.2 convergence to `exact_piston`, `piston_state` bracket and an independent shock-Mach check |
| `check.jl`, `kin.jl` → `check.out`, `kin.out` | `_check_wall_ops` with user operators (function, struct), `MirrorBC()`, Float32, DiffuseWall at θ_w ∈ {1e-4, 1e-2, 50, Inf}, a kinetic (Burgers relaxation) run with slip walls (both envs); `@sim` duplicate message |

## Status of the earlier findings

| # | Earlier finding | Status | Evidence (measured here) |
|:--|:--|:--|:--|
| 1 | specular flip reflects about the grid centre with `vcenter ≠ 0` | **fixed** | flip now only if `ax > 0 && eq.center[ax] == w[ax]` (`_main.jl:94`). Gas at rest, `vcenter = 0.5`, static wall: mass flux −1.3e-16 (parent 0.316), ghost u error 3e-16 (parent 1.0). Run (1-D, Nv 16, slip walls, t = 0.5): max deviation **1.2e-14** (parent 2.09, mean ρ +20.7 %). A drifting gas (u0 = 0.3) on a grid centred at 0.3 now behaves as on the centred grid (mean ρu −0.0554 vs −0.0553 at t = 0.5; parent: 0.300, i.e. the walls let the gas through). A wall moving at the grid centre takes the exact flip (rel. error 1e-16) |
| 2 | off-axis/moving specular BGK not conservative, not equilibrium-keeping (also α < 1) | **fixed for mass and equilibria; positivity regressed (new Finding 1)** | half-range mass flux ≤ 6e-16 in every case (parent up to 0.47). Drifting equilibrium, 2-D Nv 12: ghost moments vs exact reflection dρ ≤ 1.7e-4, du ≤ 1.2e-4 (θ = 0.3; parent 1.6e-2 / 3.5e-3 / e_int 3.7 %), Nv 24: 1.2e-6, 3-D Nv 8: 8.6e-4. 1-D moving wall w = 0.5, Nv 16: 1.9e-3 (parent: du 0.25). Runs: 2-D disk at rest with slip walls **1.8e-15** (parent 9.1e-2), α = 0.5 1.8e-15 (parent 5.2e-2); drifting gas t = 0.5: Δmean ρ 3.9e-4, ΔE 3.5e-4 (parent 1.0e-2 / 7.7e-2); **BGK piston with a slip face: ρ 1.4897 (−0.01 %), u −0.4999, p 1.7601 (−0.01 %)** (parent −9.1 % / −0.374 / −12.8 %), identical to the exact flip on a grid centred at −0.5. DiffuseWall α = 0/0.5 on a moving or oblique wall: mass flux ≤ 1.2e-16 (parent −0.118 / −0.059 / −2.2e-2). Same results in a KA kernel (`ka.out`) |
| 3 | compat `HyperCloudCore = "0.8"` vs registered Core 0.8.0 | not fixed (user's release decision, not a finding here) | additionally, 1b9ace6 now needs Core 405f797 (`claude/moving-review`, unmerged): Core main_dev evaluates a string `:shape` without binding `T` (`domains.jl:106` on review-mb), so `euler_1d_piston()` fails there (`UndefVarError: T`) unless `Main` happens to define a global `T`. The release must include that commit (by reading) |
| 4 | `:stopped_early = false` for a blow-up that reaches `tmax` | **fixed** | `:finite` (`device.jl:155`), fed by all three runners with the last stored state; `pb1`/`pf` tests assert `!finite` (`test/sims.jl:90, 107`). Normal runs: `finite => true` (Float32 example, KA BGK piston). Docs and docstring updated |
| 5 | run hash ignores the `Moving` motion; `:piston_speed` decoupled from the shape | **fixed for the example, documented for users** | `euler_1d_piston(; speed, face)` builds `:Grid_shape` as a string from the same values as `:piston_speed`/`:piston_face`; hash 0.5 vs 0.4 differ, 0.5 vs `0.5f0` equal. `walls.md`/`configuration.md` explain that a shape *instance* is still hashed as `"Closure"` and recommend strings. (Core-side named motions remain open; acceptable) |
| 6 | mass-drift metric biased (mean of point values) | **fixed** | test uses the trapezoid rule and has a slip control. Re-measured: hot wall Δmass 1.152e-2 / 7.105e-3 / 4.209e-3 (N = 50/100/200), θ at the wall 1.8318 / 1.8556 / 1.8714; slip control −1.50e-4 / −3.92e-5 / −1.02e-5 (second order) — all match `walls.md:42` |
| 7 | piston example not Float32 | **fixed** | `:T => Float32`: eltype Float32, plateau (1.4897, −0.4999, 1.7602) = Float64 result; covered by a test (`sims_walls.jl:167-169`) |

Earlier minor issues: `ok || return f` (DiffuseWall without Maxwellian) — **runner check added, but it almost never fires** (new
Finding 2); wrong-length sliding velocity — **fixed** (`ArgumentError` before the run, tested); nearest-node cost — **gone** (2-D Nv 12:
9.8 µs per ghost vs 24 µs before; Nv 24: 41 µs vs 393 µs; 3-D Nv 8: 60 µs vs 192 µs; 1-D moving 1.0 µs; 0 bytes everywhere);
unsupported equation — **fixed** (`ArgumentError` naming tag and equation, e.g. linear advection; Burgers relaxation system with
`slip_wall()`: "SlipWall … has no method for LinearAdvection" instead of the kernel `MethodError`); bisection bracket — **fixed**
(w = 50: p₂ = 3002, w = 1e3: 1.2e6; w → 0: S → c₁ = 1.18322; checked against the shock-Mach relations to 1e-6);
"to 1e-3" wording — **fixed** (`walls.md:38`); `:backend` for KA — **fixed** (`"KABackend(POCLBackend)"`); Float32 `:t_final`
digits — not changed (cosmetic); `@sim` duplicate options — **fixed** ("option(s) given twice: dry_run, calculate_stats", tested);
closure type names in the hash — moot for the example (string), unchanged for shape instances (documented).

## Findings

### 1. The new specular ghost is not positivity preserving for non-equilibrium f — low — CONFIRMED
- **Where:** `src/BoundaryConditions/_main.jl:111-131` (`raw = Mr + interp(f − Mf)` at the reflected velocity, then the emitted
  half scaled by `s`); used by `SlipWall` off the flip and by `DiffuseWall(α < 1)`.
- **What:** the non-equilibrium part `f − M[f]` is negative wherever `f` is below its Maxwellian; interpolated at a velocity
  where `M[ur]` is smaller than `M[f]` was there, the sum goes negative. Nothing clips it (the Core ghost fit clamps the value
  *before* the operator). The old nearest-node map was a selection of `f` values and kept `f ≥ 0`.
- **Evidence:** `ops.out`, `min g / max f`: 1-D two-beam, wall moving at 0.2: −1.1e-3; two-beam on a grid with `vcenter = 0.5`:
  −5.2e-4; 1-D hot/cold half-Maxwellians, w = 0.2 / 0.5: −6.2e-3 / −6.9e-3; **2-D two-beam, oblique static wall (θ = 0.3 / π/6):
  −0.104 / −0.102**; 2-D hot/cold half: −1.5e-2; single-node f: −2.3e-2 (parent: ≥ 0 in all of these). Equilibria stay positive
  (`Mr > 0`). Run (`runs_pos.out`, 2-D disk, Riemann problem, τ = 1, unlimited MUSCL2): `min f / max f` at t = 0.5 **−1.43e-2**
  (parent −5.4e-3, diffuse wall −2.8e-3); at τ = 1e-2 equal to the parent (−1.4e-3). The interior scheme alone already dips to
  ~−6e-3 here, so the effect is moderate, but the wall makes it worse in rarefied flow, where specular walls matter most.
  `is_admissible` accepts negative `f_k`, so MOOD PAD does not react.
- **Fix:** clip before the flux balance, `raw_k = max(raw_k, 0)` (the scaling then restores zero mass flux; equilibria are
  unchanged since `Mr > 0`), or limit the non-equilibrium part per component (`max(neq_interp, −Mr_k)`). Add a non-equilibrium
  case (e.g. the 2-D two-beam above) with `minimum(g) ≥ 0` to the `walls` unit checks.

### 2. The "no wall Maxwellian" check only catches non-finite input; an under-resolved wall temperature silently becomes a sink — low — CONFIRMED
- **Where:** `src/BoundaryConditions/_main.jl:224-234` (`_check_wall_op(::DiffuseWall, …)`), `:180`; claims in the comment `:205-208`
  and in the 1b9ace6 message ("no wall Maxwellian (inside the kernel: … silently no wall)").
- **What:** `equilibrium` returns `ok = false` only for non-finite input, ρ ≤ 0 or θ ≤ 0; for a temperature the grid cannot
  represent it returns the best Newton iterate with `ok = true`. The check therefore fires only for `θ_w = Inf` or a NaN
  velocity (the constructor already refuses θ ≤ 0 and NaN). It also uses only the sliding velocity, not the velocity of a
  `Moving` piece that the kernel adds.
- **Evidence:** `check.out`/`kin.out`, Nv 8, vmax 6: `DiffuseWall(1e-4)` passes the check; its `Mw` underflows (moments NaN/0),
  so `Φout = 0`, `ρw = 0` and the emitted half of the ghost is **zero**: the wall absorbs every incident particle (a vacuum
  outlet, not a wall). `DiffuseWall(1e-2)` passes with `Mw` moments 100 % off. `DiffuseWall(Inf)` is refused (correct).
- **Fix:** in the check compare the moments of `Mw` with `Uw` (scaled residual, e.g. ≤ 1e-6) and require `Φout < 0` for a
  normal; include the piece velocities where known (or check in the kernel-free setup with the maximum wall speed); otherwise
  reword the comment/commit claim to "non-finite wall states".

### 3. `euler_1d_piston(; speed, face)` accepts speeds and faces for which the example is wrong or the run fails obscurely — low — CONFIRMED
- **Where:** `src/Examples/euler_1d_piston.jl:72-74` (only `0 < face < 1.5` is checked), `piston_state:22-35`, `exact_piston:41-54`.
- **What / Evidence (`misc.out`):** `speed = −0.5` (a receding piston) runs, but `piston_state` returns p₂ = p₁, S = −Inf, so
  `exact_piston` is the gas at rest everywhere while the run is a rarefaction: L1(ρ) 0.128 against its own reference.
  `speed = 0`: S = NaN (harmless, the reference is the rest state). `face = 0.1, speed = 0.5`: the face crosses x = 0 before
  `tmax = 0.3`; the run fails with `ErrorException("Remap produced no volume at [0.0]; increase samples_per_dx.")`.
  Valid cases are right: `speed = 0.8, face = 1.2`: plateau (1.8360, −0.8000, 2.4055) vs RH (1.8361, −0.8, 2.4055), L1(ρ)
  6.18e-3 / 3.23e-3 (N = 300/600), last particle one spacing behind the face.
- **Fix:** `speed > 0 || throw(ArgumentError(…))` (or a rarefaction reference) and require `face − speed·tmax` to stay a few
  spacings above 0 (and say so in the docstring).

## Other points checked (no finding)

- Flip condition: exact float equality is right for the cases it is meant for (static wall on a centred grid, a wall moving at
  `:PDE_vcenter`); near equality the general path is continuous (|g(w) − g(0)| = 1.05·w for w = 1e-14 … 1e-4), so no jump.
  `n = −e_ax` and tangential wall velocities are handled (`v_ax → 2 w_ax − v_ax` does not depend on them).
- Equilibrium split: `Ur` keeps ρ and the internal energy (incl. `h`), `Mr`/`Mf` are discrete entropic equilibria (positive);
  fallback when `equilibrium` fails (f = −f_eq): the whole f is interpolated and, with a non-negative emitted flux, `s = 1`, so
  the mass flux is not zeroed there (3.2e-2) — only for unphysical input. The scaling acts on the emitted half only, as in
  `DiffuseWall`; the ghost's incident half is the reflection, so the "central" mass flux Σ w c (f + g)/2 is not zero
  (moving wall 5.5e-4, two-beam 1.8e-2), but BGK's Rusanov flux is exact upwinding per velocity, where only the half-range flux
  enters. Energy flux at a static (adiabatic) wall is not enforced: 2–3e-4 (Nv 12 equilibrium), 2.3e-3 (3-D Nv 8), up to
  1.5e-2 for the 2-D two-beam (parent 0.07–0.54) — consistent with "mass only", see Minor issues for the docstring wording.
- `DiffuseWall(α < 1)`: diffuse part and specular part each have zero mass flux, so the mix has (≤ 1.2e-16 measured).
- Kernel rules: 0 bytes per call for every case, `isbits` unchanged, KA CPU run of the general path (piston slip face, 2-D
  Float32 disk) reproduces the Polyester numbers (dev 4.8e-7 in Float32). No kernel signatures changed (20-argument rule
  unaffected). Cost per ghost is dominated by two `equilibrium` solves (one ≈ 45 µs at 2-D Nv 24).
- `_check_wall_ops`: no false positives found — a user operator as a plain function or a struct for the equation passes,
  `MirrorBC()` (no operator) is skipped, Float32 domains pass; a user operator lacking a method for the equation is refused
  with a clear message. `applicable` with a zero state only checks dispatch, so it cannot misfire on the state. Kinetic runs:
  the check sees the kinetic system (`LinearAdvection` for Burgers relaxation), which has no wall operator anyway (parent:
  kernel `MethodError`). Lagrangian movers keep the `Conservative` representation, so the dispatch is the same.
- `:finite`: computed from the last *stored* state on the host after packaging (BGK: moments unless `save_relax`); default
  `true` when no state is given (no other callers).
- `@sim` duplicate check happens at macro expansion, before the `kw` tuple is built; message lists each repeated key once.
- `walls.md` numbers re-measured: disk at rest 1.8e-15 for both walls (claim 2e-15); oblique drifting equilibrium within 2e-4
  at Nv 12 (dρ 1.7e-4, du 1.2e-4); hot wall/control as in the table above.

## Test coverage

- The new operator checks (`sims_walls.jl:97-121`) discriminate: on the parent the shifted-grid case has mass flux 0.316 /
  ghost u error 1.0, the moving wall −0.118 / 0.25, θ = 0.3 −2.2e-2 (`ops_parent.out`), all far above the `1e-13`/`tol`
  bounds; θ = π/4 (node permutation) passes on both, as intended. The tolerances (5e-3 at Nv 16, 1e-3 at Nv 12) are 2.5–6× the
  measured errors: meaningful.
- The slip control (`< 2e-4`, measured −3.9e-5) and the trapezoid drift (`drift[2] < 0.8 drift[1]`, measured ratio 0.62) are
  meaningful; the piston string/Float32/bracket tests fail on the parent by construction (no keywords; p₂ capped at 1000).
- **Not covered:** no run in the suite takes the general specular path inside a kernel — every BGK slip-wall run is a static
  axis-aligned wall on a centred grid (flip), the BGK piston uses a diffuse face. `ka.jl` shows it works, but a short BGK piston
  with `slip_wall()` on the face (or the 24² disk) in the `walls` group would keep it under the `ka_cpu` check. Non-equilibrium
  inputs and positivity (Finding 1) are not tested. `:finite => true` for a normal run is not asserted (a constant `false`
  would pass). No 2-D/3-D wall run in the suite (unchanged from the earlier review).

## Minor issues

- `SlipWall` docstring (`_main.jl:33-34`) and the code comment (`:100-102`): "a gas in equilibrium (at rest next to a curved
  wall, or drifting along an oblique one) is reflected exactly". At rest yes (1.8e-15); drifting, the emitted-half scaling
  changes the moments by the quadrature asymmetry: 1.7e-4 (2-D Nv 12), 8.6e-4 (3-D Nv 8), 1.9e-3 (1-D Nv 16, moving wall).
  `walls.md:41` states it correctly ("within 2·10⁻⁴"). Also state that energy (adiabatic) is not enforced off the flip.
- `device.jl:159` comment / 1b9ace6 message say `KABackend(CPU)`; the stored string is `"KABackend(POCLBackend)"`.
- `euler_1d_piston(speed = 1//2)` and `speed = 0.5` give different shape strings and hashes for the same run (cosmetic;
  `T($(Float64(speed)))` would normalise).
- The piston example's parameter dicts changed (`:Grid_shape` is now a string), so existing cached piston runs get new hashes;
  CLAUDE.md asks to call such changes out in the PR. Low impact (the example is from PR #18).
- The `_check_wall_ops` message for kinetic runs names the internal `LinearAdvection` system, not the configured equation;
  "(kinetic relaxation systems have no wall operators)" would help.
- Docs were not built here (`checkdocs = :exports`; the `euler_1d_piston` docstring is new text).

## Verdict

All seven earlier findings are addressed except the compat item, which is the user's release decision (and now also needs
Core 405f797 for string shapes with `T`). The new BGK specular reflection is a real improvement: zero half-range mass flux
to round-off on every normal, wall velocity, shifted and Gauss–Hermite grid, 1–3-D and Float32; equilibria at rest kept to
1e-15 next to a curved wall (was 9 %), drifting equilibria within 2e-4; the BGK piston with a slip face now hits the
Rankine–Hugoniot plateau to 0.01 % (was 9–13 % off); `vcenter` no longer makes the wall penetrable; it is allocation free,
works in a KA kernel and is faster than the nearest-node search. `:finite`, the trapezoid metric, the hashed/Float32 piston
example, the bracket, the operator checks and the `@sim` duplicate check work as described, and the documented numbers
reproduce. The full suite passes (1252/1252) and the `walls` group passes on `ka_cpu`. Remaining issues are small: the
interpolated non-equilibrium part can make the ghost negative (−10 % of max f for a 2-D two-beam, run dips 2.6× larger at
τ = 1), the Maxwellian check misses under-resolved wall temperatures (which turn the wall into a sink), and the piston
keywords accept invalid speeds/faces. Findings: **0 high, 0 medium, 3 low** (1: specular ghost not positivity preserving;
2: DiffuseWall Maxwellian check nearly dead; 3: `euler_1d_piston` keyword validation).
