# EMerge Loaded Antenna Designer and Optimizer

Design, simulate, optimize, and verify smooth loaded monopole antennas with
[EMerge](https://github.com/FennisRobert/EMerge). The project supports an
unloaded radiator or any number of helical loading coils, constructs the model
as one continuously curved conductor, and provides a campaign optimizer for
gain, impedance match, pattern shape, lobe direction, and beamwidth.

The main workflow is:

1. Describe a physical antenna with immutable Python configuration objects.
2. Simulate its S11 and far field in an open-region finite-element model.
3. Search several coil topologies and random seeds with differential evolution.
4. Fine-tune the most promising design with independent coil dimensions.
5. Verify the winner with a finer mesh, denser sweep, and convergence study.

All library dimensions are in **metres** and all frequencies are in **hertz**.
Command-line options use MHz, millimetres, and degrees where stated.

## Table of contents

- [What the project provides](#what-the-project-provides)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Recommended design workflow](#recommended-design-workflow)
- [Antenna designer](#antenna-designer)
  - [Geometry model](#geometry-model)
  - [Design objects](#design-objects)
  - [Create and edit designs](#create-and-edit-designs)
  - [Save, load, and scale designs](#save-load-and-scale-designs)
  - [Fabrication drawings](#fabrication-drawings)
  - [Printable forming tools](#printable-forming-tools)
- [Simulation](#simulation)
  - [Run a simulation](#run-a-simulation)
  - [Simulation configuration](#simulation-configuration)
  - [Coordinates and far-field results](#coordinates-and-far-field-results)
  - [Interactive model example](#interactive-model-example)
- [Campaign optimizer](#campaign-optimizer)
  - [First optimization campaign](#first-optimization-campaign)
  - [Starting design and wire diameter](#starting-design-and-wire-diameter)
  - [Topology selection](#topology-selection)
  - [Coil parameterization and fine tuning](#coil-parameterization-and-fine-tuning)
  - [Automatic rough-to-verified pipeline](#automatic-rough-to-verified-pipeline)
  - [Pattern, lobe, and beamwidth goals](#pattern-lobe-and-beamwidth-goals)
  - [Matching and physical constraints](#matching-and-physical-constraints)
  - [Budgets, seeds, restarts, and confirmation](#budgets-seeds-restarts-and-confirmation)
  - [Automatic numerical preflight](#automatic-numerical-preflight)
  - [Campaign output files](#campaign-output-files)
  - [Optimizer option reference](#optimizer-option-reference)
- [Custom optimization API](#custom-optimization-api)
- [Verify a winning design](#verify-a-winning-design)
- [Open-region convergence](#open-region-convergence)
- [Solvers and performance](#solvers-and-performance)
- [Interpreting results](#interpreting-results)
- [Troubleshooting](#troubleshooting)
- [Development and project layout](#development-and-project-layout)
- [Modeling scope](#modeling-scope)

## What the project provides

- Zero, one, or many integer-turn helical loading coils.
- Exact straight segments, local cubic Bezier transitions, and segmented helix
  arcs assembled into one continuous centerline and swept once.
- Configurable wire diameter, radials, coil dimensions, handedness, port, mesh,
  frequency sweep, and open-region boundary.
- S11, global peak gain, requested-direction gain, horizon statistics, ripple,
  null depth, peak direction, and directional half-power beamwidth.
- Wavelength-scaled reference designs for frequencies other than 868 MHz.
- Multi-topology, multi-seed optimization campaigns with global and fine modes.
- Dimensioned PDF, SVG, or PNG fabrication sheets with optional S11 and
  horizon-pattern plots.
- STEP/STL winding formers, sizing mandrels, transition markers, and a radial
  angle-and-length gauge generated from the antenna geometry.
- Broadband matching penalties, height limits, horizon or directional pattern
  goals, optional beamwidth targeting, and repeat confirmation of incumbents.
- Atomic progress artifacts, CSV evaluation logs, topology rankings, and JSON
  designs that can be loaded directly back into the library.
- Separate final-verification and numerical-convergence tools.

## Installation

The package requires Python 3.10 or newer, EMerge 2.8.4 or newer in the 2.x
series, and NumPy 2.x. SciPy is needed for optimization and Matplotlib for the
verification plots. EMerge and its native solver dependencies may impose
additional platform or Python-version constraints.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[optimize,verify]"
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[optimize,verify]"
```

Install the test extra when developing:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[optimize,verify,test]"
```

Use `python` in the commands below if the virtual environment is activated.
Otherwise, on Windows, substitute `.\.venv\Scripts\python.exe` as shown.

## Quick start

Run a short campaign at 868 MHz, then verify its winner:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --frequency-mhz 868 `
    --coil-counts 0,1,2 `
    --maxiter 3 `
    --seeds 2 `
    --output optimization_results\quick_start

.\.venv\Scripts\python.exe -u .\examples\verify_best.py `
    optimization_results\quick_start\campaign_best.json
```

`--maxiter 3` is a pipeline check, not a serious engineering search. A useful
campaign normally needs a measured time budget, several seeds, and final
verification:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --frequency-mhz 868 `
    --coil-counts 0,1,2,3 `
    --hours 12 `
    --seeds 4 `
    --wire-diameter-mm 1.6 `
    --output optimization_results\868mhz_broad
```

The output directory must not already contain campaign files. This prevents an
old and a new campaign from being mistaken for one run.

For the complete unattended workflow, use `--automatic`:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --automatic `
    --frequency-mhz 869.5 `
    --wire-diameter-mm 1.6 `
    --target-theta 100 `
    --target-beamwidth-deg 50 `
    --match-bandwidth-mhz 20 `
    --hours 12 `
    --solver cudss
```

This searches all requested topologies, fine-tunes and coordinate-polishes the
winner, performs independent coarse/fine verification, and puts the drawing,
forming STEP, plots, and reports in one timestamped optimization-result folder.

## Recommended design workflow

1. **Choose the electrical goal.** Decide the operating frequency, required
   S11 band, useful direction or horizon coverage, acceptable height, wire
   diameter, and whether beamwidth is a real requirement.
2. **Inspect a starting model.** Use `main.py` or a small library script to view
   the geometry and confirm units, coil order, radial angle, and port position.
3. **Run a global campaign.** Compare likely coil counts or explicit turn cases.
   Coil pitch and radius are independent by default; use `--lock-coils` for a
   deliberately smaller shared-coil search.
4. **Fine-tune the winner.** Warm-start from `campaign_best.json` and use
   `--finetune` to add multiscale populations, restarts, and local refinement.
   Coil sharing is controlled separately by `--lock-coils`.
   Use `--automatic` to perform steps 3 through 5 without manual handoff.
5. **Verify independently.** Run `verify_best.py` for a denser frequency sweep,
   finer mesh, and finer angular sampling.
6. **Test numerical convergence on the actual winner.** Run
   `check_open_region.py campaign_best.json`. The optimizer's automatic
   preflight uses a numerical reference design, not the eventual winner.
7. **Prototype and measure.** Numerical convergence does not account for
   connectors, lossy materials, support structures, construction tolerance, or
   the real installation environment.

## Antenna designer

### Geometry model

The radiator begins above a lumped feed and alternates straight sections and
coils along +Z:

```text
          straight N
              |
         smooth exit
          / coil N \
         smooth entry
              |
             ...
              |
         straight 1
              |
         smooth exit
          / coil 1 \
         smooth entry
              |
         straight 0
              |
             feed
       \  \   |   /  /
          radials
```

For `N` coils, `straight_lengths` must contain exactly `N + 1` values. Index 0
is the bottom section nearest the feed. Coil index 0 is the lowest coil.

The builder uses exact line segments, local transition curves, and helix arcs
of at most 120 degrees. These pieces form one continuous OpenCASCADE wire,
which is swept once with a circular conductor cross-section. Local transitions
avoid the distant-shape distortion that a single global spline can introduce.

The transition chord offset is derived automatically as
`transition * 19 / 24`. It is not an independent design parameter. A transition
must be at least `1.25 * wire_radius`, and its derived offset must remain below
the coil diameter.

Each coil has an integer number of turns and returns to the radiator axis.
`radius` is measured to the wire centerline, so the approximate physical coil
outside diameter is:

```text
2 * (coil.radius + design.wire_radius)
```

`radial_angle_deg` is the downward angle of the **ground-plane radials**. It is
not the far-field lobe direction. Lobe direction is selected in the optimizer
with `--target-theta`; add `--target-phi` only for a single azimuth direction.

### Design objects

`AntennaDesign` describes the complete physical structure:

| Field | Default | Meaning |
|---|---:|---|
| `wire_radius` | `1e-3` m | Conductor radius; diameter is twice this value. |
| `radial_length` | `72e-3` m | Length of each ground-plane radial. |
| `radial_angle_deg` | `45` | Radial angle below horizontal; strictly between 0 and 90. |
| `radial_count` | `4` | Equally spaced radials; at least two. |
| `straight_lengths` | `(0.140, 0.221, 0.140)` m | Bottom-to-top radiator straight sections. |
| `coils` | two default coils | Bottom-to-top sequence of `CoilDesign` objects. |
| `port_height` | `2e-3` m | Height of the lumped feed region. |
| `port_impedance` | `50` ohm | Reference impedance. |

`CoilDesign` describes one loading coil:

| Field | Default | Meaning |
|---|---:|---|
| `radius` | `10e-3` m | Helix centerline radius. |
| `turns` | `1` | Positive integer turns. |
| `pitch` | `7e-3` m | Axial advance per turn. |
| `transition` | `6e-3` m | Axial room assigned to each smooth transition. |
| `transition_offset` | derived | Always normalized to `transition * 19 / 24`; retained in JSON for compatibility. |
| `handedness` | `"RH"` | `"RH"` or `"LH"`. |

Configuration dataclasses are frozen. This makes designs safe to reuse across
simulations and optimizer evaluations.

### Create and edit designs

```python
from dataclasses import replace

from emerge_loaded_antenna import AntennaDesign, CoilDesign

design = AntennaDesign(
    wire_radius=0.8e-3,       # 1.6 mm diameter
    radial_length=82e-3,
    radial_angle_deg=35.0,   # ground-plane radial angle
    radial_count=4,
    straight_lengths=(118e-3, 185e-3, 132e-3),
    coils=(
        CoilDesign(radius=12e-3, turns=1, pitch=8e-3),
        CoilDesign(radius=10e-3, turns=2, pitch=7e-3, handedness="LH"),
    ),
)
design.validate()

# Frozen dataclasses are edited by creating a replacement.
thicker = replace(design, wire_radius=1.0e-3)
```

An unloaded radiator has no coils and one straight section:

```python
unloaded = AntennaDesign(
    straight_lengths=(0.25,),
    coils=(),
)
unloaded.validate()
```

### Save, load, and scale designs

```python
from emerge_loaded_antenna import (
    load_design,
    load_reference_design,
    save_design,
    scale_design,
)

reference_868 = load_reference_design()
reference_915 = load_reference_design(915e6)
save_design(reference_915, "designs/reference_915.json")
restored = load_design("designs/reference_915.json")

# Scale an arbitrary design from 915 MHz to 433 MHz.
scaled = scale_design(restored, factor=915e6 / 433e6)
```

`load_design` accepts either a standalone design JSON file or an optimizer
result containing a top-level `design` object. Wavelength scaling changes
physical lengths, including wire radius, while retaining angles, counts,
handedness, and impedance.

### Fabrication drawings

Generate a dimensioned A3 landscape sheet from a raw design or optimizer result:

```powershell
.\.venv\Scripts\python.exe -m emerge_loaded_antenna.drawing `
    optimization_results\868mhz_horizon\campaign_best.json `
    antenna.pdf `
    --title "868 MHz Prototype"
```

PDF, SVG, and PNG output are supported. The sheet contains X-Z, Y-Z, and X-Y
orthographic views, radiator and coil dimensions, radial geometry, and a
fabrication table. Coil callouts include centerline radius, clear inside or
mandrel diameter, pitch, and transition bend radius. The command-line exporter
has no live solver result, so its RF panels are labeled as unavailable.

Pass a solved `SimulationResult` through the Python API to include the complete
S11 sweep and overlaid XY/horizon and XZ/elevation realized-gain lobes. Optional
target theta rings can be drawn from the same solved far field:

```python
from emerge_loaded_antenna.drawing import export_drawing

export_drawing(
    design,
    "antenna.pdf",
    result=result,
    title="868 MHz Prototype",
    target_ring_thetas_deg=(90.0, 130.0),
)
```

Drawing export requires Matplotlib. Install only that optional feature with
`pip install -e ".[drawing]"`; it is also installed by the `verify` extra.

### Printable forming tools

The easyradius forming tool creates exact CAD solids independently of the EM
model:

```python
from emerge_loaded_antenna.formers import export_coil_formers

export_coil_formers(design, "coil_formers.step")
export_coil_formers(design, "coil_formers.stl")
```

For every loading coil, the output contains a winding former whose blank
diameter equals the coil centerline diameter. The complete Hermite-transition
and helix path is swept and subtracted to form a half-round wire groove through
both end faces. The default groove clearance is 0.1 mm.

Each coil also receives an inside-diameter sizing mandrel with a shallow guide
for correcting spring-back. Four witness notches identify the start and end of
both transitions. A flat radial gauge carries the modeled radial angle, nominal
length mark, and ground-hub relief. All parts are separate solids in the STEP
file and are spaced for fabrication.

The standalone command accepts either a design JSON or an optimizer result:

```powershell
.\.venv\Scripts\python.exe -m emerge_loaded_antenna.formers `
    optimization_results\868mhz_horizon\campaign_best.json `
    coil_formers.step
```

Use `--no-sizing-mandrels` or `--no-radial-gauge` to omit those tools. The CLI
also exposes groove clearance, spacing, marker dimensions, gauge dimensions,
and STL mesh size; run it with `--help` for the full list. Former construction
uses an independent Gmsh model and never enters `build_model()` or `simulate()`.

## Simulation

### Run a simulation

```python
from emerge_loaded_antenna import (
    FrequencySweep,
    SimulationOptions,
    load_reference_design,
    simulate,
)

frequency = 868e6
design = load_reference_design(frequency)
options = SimulationOptions(
    sweep=FrequencySweep(center=frequency, span=20e6, points=5),
    compute_farfield=True,
    farfield_frequency=frequency,
    farfield_angular_step_deg=2.0,
    solver="auto",
)

result = simulate(design, options)
print(result.frequencies)
print(result.s11_db)
print(result.peak_gain_dbi)
print(result.farfield_metrics)
print(result.gain_db_at(theta_deg=90, phi_deg=0))
```

`simulate()` validates the design and options, builds the model, meshes it,
solves when requested, and returns a `SimulationResult`. Set `solve=False` to
build or preview geometry without solving. Far-field calculation requires a
solve.

### Simulation configuration

`FrequencySweep` uses a center frequency, total span, and number of points.
`FrequencySweep.single(frequency)` creates a one-point sweep.

Important `MeshSettings` controls:

| Field | Default | Effect |
|---|---:|---|
| `wire_sections` | `6` | Circumferential conductor resolution; minimum six. |
| `antenna_size_factor` | `3.0` | Antenna surface mesh size relative to wire radius. |
| `radial_size_factor` | `10.0` | Radial surface mesh factor. |
| `feed_size_factor` | `3.0` | Feed-region mesh factor. |
| `curved_boundary_segments` | `12` | Segmentation of curved boundaries. |
| `wavelength_resolution` | `0.33` | Free-space volumetric mesh resolution in wavelengths. |
| `air_margin_wavelengths` | `0.25` | Air between structure and Huygens box. |
| `preview_points_per_turn` | `20` | Sampling used by geometry previews. |

`OpenRegionSettings` defaults to an ordinary-air buffer of one wavelength and
a second-order type-B absorbing boundary on **all six outer faces**, including
the bottom. `mode="pml"` selects a perfectly matched layer instead. The default
ABC setup is the path exercised by the campaign convergence workflow.

`SimulationOptions` also controls the solver, far-field sampling, verbosity,
model name, and optional geometry, mesh, and coil-preview viewers.

### Coordinates and far-field results

The far-field convention is spherical:

- `theta = 0 degrees` points along +Z.
- `theta = 90 degrees` is the horizontal plane.
- `theta = 180 degrees` points along -Z.
- `phi = 0 degrees` points along +X.
- `phi = 90 degrees` points along +Y.

`SimulationResult` exposes complex S11, S11 in dB, peak gain, mesh counts, raw
solver data, and `FarFieldMetrics`. The latter includes peak direction and
horizon minimum, P10, mean, P90, peak, P90-P10 ripple, and peak-to-null range.

For an omnidirectional ring away from the horizon:

```python
ring = result.azimuth_ring_metrics(theta_deg=100)
print(ring.p10_gain_dbi, ring.min_gain_dbi, ring.ripple_p90_p10_db)
ring_hpbw = result.azimuth_ring_beamwidth_deg(theta_deg=100)
```

The ring beamwidth uses the P10-over-phi gain profile versus theta, so it
represents the robust elevation width of the full azimuth ring.

For a requested directional lobe, use:

```python
gain = result.gain_db_at(theta_deg=70, phi_deg=20)
elevation_hpbw, azimuth_hpbw = result.directional_beamwidths_deg(70, 20)
```

Beamwidth is the contiguous half-power lobe containing the requested direction,
measured on two orthogonal great-circle cuts. The threshold is 3 dB below gain
at that requested direction, not necessarily 3 dB below the global peak.

### Interactive model example

`main.py` is an editable example for geometry, mesh, and far-field inspection:

```powershell
.\.venv\Scripts\python.exe .\main.py --solver auto
```

The complete physical antenna is declared directly in `main.py` with static
`AntennaDesign` and `CoilDesign` values; it does not inherit dimensions from
the packaged optimizer reference. The simulation, viewer, and export switches
are also defined beside it so the file remains a compact experiment.

With `EXPORT_DESIGN_SHEET` and `EXPORT_FORMERS` enabled, the example writes
`example_outputs/design_sheet.pdf` and `example_outputs/coil_formers.step`.
The STEP file contains winding formers, sizing mandrels, and the radial gauge.
A solved run adds S11 and gain lobes to the sheet; a mesh-only run still creates
the dimensions, placeholders, and forming tools. Closing an EMerge/Gmsh viewer
continues execution.

## Campaign optimizer

### First optimization campaign

The campaign entry point is `examples/optimize_gain.py`. It evaluates every
requested topology and seed, writes progress continuously, confirms apparent
incumbents, and ranks the final feasible designs.

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --frequency-mhz 868 `
    --match-bandwidth-mhz 10 `
    --coil-counts 0,1,2,3 `
    --pattern horizon `
    --hours 12 `
    --seeds 4 `
    --output optimization_results\868mhz_horizon
```

Use unbuffered mode (`-u`) so long-running progress appears immediately.

### Starting design and wire diameter

Without `--warm-start`, the campaign loads the tracked 868 MHz reference and
scales all physical dimensions by wavelength to the requested frequency. A
warm start may be a raw design JSON or any optimizer JSON containing `design`:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --warm-start optimization_results\868mhz_horizon\campaign_best.json `
    --finetune `
    --hours 6 `
    --output optimization_results\868mhz_fine
```

Use `--random-start` when the synthesized or warm design should not occupy the
explicit `x0` slot in a broad-search population:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --random-start `
    --turn-cases none,1,1x1,1x2,1x1x1 `
    --hours 8
```

For every topology and seed, the optimizer samples uniformly across every
continuous variable's complete bounds until `AntennaDesign.validate()` accepts
the combined geometry. That validated design replaces the reference/warm
`x0`; the rest of SciPy's broad population remains globally distributed as
usual. The template still supplies fixed properties such as radial count,
handedness, impedance, and any dimensions not exposed as optimizer variables.
The selected starts are printed and saved in `random_starts.json` with their
seeds and complete designs. `--random-start` conflicts with manual
`--fine-tune`; with `--automatic`, it applies only to rough search and the fine
stage starts from the rough winner.

Set the fixed conductor diameter explicitly with `--wire-diameter-mm` (or its
alias `--wire-diameter`):

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --wire-diameter-mm 1.6 `
    --hours 8
```

The diameter flag overrides both the wavelength-scaled reference and a warm
start. The campaign converts it to `AntennaDesign.wire_radius`, keeps it fixed
during optimization, and uses it to derive collision-safe lower bounds for
straight lengths, radial length, coil pitch, and coil radius.

There is no optimizer flag that fixes `AntennaDesign.radial_angle_deg`; that
angle remains a design variable. It describes ground radials, not the desired
radiation direction.

### Topology selection

Topology means the number of coils and the integer turns in each coil. Turn
counts are discrete campaign cases; differential evolution optimizes only the
continuous dimensions inside each case.

| Option | Example | Meaning |
|---|---|---|
| no topology flag | | Keep the reference or warm-start turn pattern. |
| `--coil-count N` | `--coil-count 2` | Run one N-coil case, normally one turn per coil. |
| `--coil-counts LIST` | `--coil-counts 0,1,2,3` | Compare several one-turn-per-coil cases. |
| `--turn-cases LIST` | `--turn-cases none,1,1x2,1x1x1` | Compare explicit turn tuples. `none` is zero-coil. |

`--coil-counts` and `--turn-cases` cannot be combined. `--coil-count` may be
combined with `--turn-cases` only when every explicit case has that count.
Examples:

```powershell
# Unloaded, one 1-turn coil, two coils with 1 and 2 turns, and three 1-turn coils
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --turn-cases none,1,1x2,1x1x1 `
    --hours 10

# Only a two-coil topology
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --coil-count 2 `
    --hours 6
```

When topology differs from the starting design, a wavelength-based seed is
generated for that case. Existing valid dimensions are retained where the
topology permits it.

### Coil parameterization and fine tuning

By default, the optimizer searches every straight length, every coil pitch,
every coil radius, radial length, and ground-radial angle independently. A
loaded design with `N` coils therefore has `3N + 3` continuous variables; an
unloaded radiator has three.

Use `--lock-coils` when every coil should share one pitch and one radius. This
reduces an N-coil design to `N + 5` variables:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --coil-counts 1,2,3 `
    --lock-coils `
    --hours 8
```

Locking is independent of optimizer mode. It can make an initial global search
cheaper, but it also excludes antennas whose coils need different dimensions.
The result metadata records `coil_parameterization` as `independent` or
`shared`.

`--finetune` (alias `--fine-tune`) changes the search strategy, not coil
coupling. It uses a good warm start, mixed local/global populations,
deterministic restarts, and bounded coordinate search. Unless `--lock-coils` is
also present, all coil pitches and radii remain independent. Its initial
population contains:

- 50% near the warm start, within 3% of normalized range by default;
- 30% within a wider 10% neighborhood;
- 20% across the global bounds.

Fine mode's mutation, recombination, neighborhood radii, restart threshold, and
local-search budget are configurable. If a global campaign used
`--lock-coils`, omit that flag from the fine-tuning command to release the coil
dimensions.

Manual fine mode normally ends with its existing multi-elite bounded coordinate
search. `--polish` replaces that terminal refinement with a stricter
single-incumbent coordinate descent; the two local refiners are not run back to
back. Polish evaluates positive and negative changes to one normalized
parameter at a time, accepts feasibility before raw score, and halves its step
after a complete unsuccessful sweep. It stops after an unsuccessful sweep at
`--local-search-min-step` or when its candidate reserve is exhausted. This is a
local coordinate minimum at the configured numerical resolution—not a
mathematical proof of a perfect design. Automatic mode always selects this
stricter polish. The former SciPy behavior is available explicitly as
`--scipy-polish`; it is broad-search-only, mutually exclusive with
`--polish`, and not recommended for noisy or tightly budgeted EM objectives.

### Automatic rough-to-verified pipeline

`--automatic` is the hands-off production workflow:

1. Run the broad multi-topology, multi-seed campaign. Each rough DE run stops
   when `--restart-stagnation-generations` is reached, on SciPy convergence, or
   at its candidate cap.
2. Load the global rough winner and retain only its discrete turn topology.
3. Run multiscale fine DE from that winner. Fine stagnation transitions directly
   into `--polish` instead of spending the remainder on repeated DE restarts.
4. Polish one parameter at a time until the configured coordinate resolution or
   candidate cap is reached.
5. Launch `verify_best.py` in an isolated process with `--design-sheet` and
   `--jig-models`.

By default, 65% of the optimizer candidate estimate is assigned to rough search
and 35% to fine tuning plus polish. Change this with
`--automatic-rough-fraction`. For `--maxiter`, the split accounts for the
different rough/fine seed counts, topology counts, and population sizes instead
of simply dividing generations: its `maxiter + 1` population-batch budget is
shared across the pipeline. Automatic mode refuses a budget that cannot hold at
least one rough population plus fine DE and its coordinate-polish reserve. It
uses four rough seeds and two fine seeds unless `--seeds` is supplied
explicitly. Candidate capacity left unused when rough runs stagnate or converge
early is rolled forward into the fine/polish stage.

Omitted seeds are generated from system randomness at campaign startup and
recorded in result metadata. Automatic mode records both its rough and
fine/polish seed sets in `automatic_pipeline.json` before simulation begins.
Pass the recorded values back through `--seeds` to reproduce a population.

`--hours` is an estimate based on `--seconds-per-eval`, not a hard wall-clock
deadline. Incumbent confirmation simulations are counted separately and can be
material during polish. Fine verification and fabrication are also outside the
estimate because their runtime depends on the winning geometry and fine mesh.

The root result directory contains the canonical `campaign_best.json`,
`automatic_pipeline.json`, verification report, plots, `design_sheet.pdf`, and
`coil_formers.step` when the winner has coils. Detailed search logs live under
`rough_search/` and `fine_tune/`. The verifier checks fine-mesh worst-band S11
against the configured limit and checks coarse/fine metric drift against 0.5
dB. It also carries an uncertified or explicitly skipped open-region preflight
forward as a warning. A numerical or matching warning produces
`complete_with_warnings` in the pipeline manifest; a verifier/export failure records
`verification_failed`. In either case, the optimizer winner and both stage logs
remain intact.

The normal search bounds are wavelength-based and then enlarged when needed to
contain a valid warm start:

| Quantity | Normal range |
|---|---:|
| Loaded straight length | 0.15 to 0.72 wavelength |
| Unloaded straight length | 0.18 to 0.70 wavelength |
| Radial length | 0.15 to 0.40 wavelength |
| Ground-radial angle | 5 to 75 degrees |
| Coil pitch | 0.010 to 0.040 wavelength |
| Coil radius | 0.015 to 0.050 wavelength |

Wire-aware floors prevent obviously impossible geometry. In particular, the
campaign requires clearance proportional to conductor diameter and transition
offset. Bounds enclose valid warm-start values instead of silently clipping
them.

### Pattern, lobe, and beamwidth goals

Choose one of four pattern modes:

| Mode | Useful-gain term | Additional pattern behavior |
|---|---|---|
| `horizon` | Horizon P10 gain | Penalizes insufficient horizon minimum and excess P90-P10 ripple. |
| `ring` | Worst P10 gain among all requested theta rings | Applies the minimum/ripple objective to every conical azimuth ring. |
| `directional` | Gain at requested theta/phi | Optionally targets HPBW on two orthogonal cuts. |
| `peak` | Global peak gain | No directional or horizon-shape penalty. |

With no pattern flags, the optimizer uses the existing horizon ring at theta
90 degrees. Supplying one or more values to `--target-theta` (or its shorter
alias `--theta`) selects `ring` mode and optimizes every phi equally at every
requested theta. The useful-gain term is the weakest ring's P10, so a strong
ring cannot hide a weak one. Minimum-gain and ripple penalties are evaluated on
every ring and averaged. Supplying `--target-phi` selects a single
`directional` coordinate, so it cannot be combined with multiple theta values.
Explicit conflicting combinations are rejected.

Omnidirectional conical-ring example:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --target-theta 100 `
    --hours 8
```

This maximizes azimuthal P10 gain at theta 100 degrees and penalizes weak
azimuths and excess P90-P10 ripple around that entire ring. Phi is deliberately
unspecified.

Multiple omnidirectional rings use space-separated theta values:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --theta 90 130 `
    --hours 8
```

This maximizes the lower of the theta-90 and theta-130 ring P10 gains. Live
progress, CSV metrics, campaign JSON, and verification output report both rings
and identify the current bottleneck.

Directional example:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --pattern directional `
    --target-theta 70 `
    --target-phi 25 `
    --hours 8
```

`--target-theta` is the ring or lobe's polar angle. `--target-phi` is used only
for a single directional target. Defaults are theta 90 degrees and, when
needed, phi 0 degrees. These options are unrelated to the physical
ground-radial angle.

Add a half-power beamwidth goal with:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --pattern directional `
    --target-theta 70 `
    --target-phi 25 `
    --target-beamwidth-deg 60 `
    --beamwidth-weight 1.5 `
    --hours 8
```

For directional mode, the same target is applied to the two orthogonal
great-circle cuts through the requested lobe and their errors are combined as
RMS. For ring mode, beamwidth is measured in theta/elevation from the
azimuthal-P10 profile, so the entire ring must retain the requested width. With
multiple rings, the same HPBW goal is applied to every ring and their squared
errors are averaged. The penalty is:

```text
beamwidth_weight * mean((beamwidth_error_degrees / 10)^2)
```

Thus 10 degrees is only the error-normalization scale: it is **not a hard-coded
beamwidth goal**. With weight 1, a 10-degree RMS error adds 1 to the minimized
score. Beamwidth is a soft objective and does not by itself decide feasibility.
`--angular-step` controls far-field sampling resolution, not desired width.

### Matching and physical constraints

The robust campaign evaluates S11 at three frequencies spanning
`--match-bandwidth-mhz`. For every point above `--s11-limit-db`, it adds a
quadratic mismatch penalty. A small reward encourages margin beyond
`--s11-margin-target-db` instead of making every barely feasible match equal.

The minimized score is conceptually:

```text
mismatch penalty
+ pattern penalty
+ optional beamwidth penalty
+ height penalty
- match-margin reward
- useful gain
```

Main controls:

| Option | Default | Purpose |
|---|---:|---|
| `--match-bandwidth-mhz` | wavelength-scaled 10 MHz at 868 MHz | Three-point S11 span. |
| `--s11-limit-db` | `-10` | Maximum acceptable S11 at each match point. |
| `--mismatch-weight` | `2` | Weight of squared S11 violations. |
| `--s11-margin-target-db` | `-12` | Target for additional match margin. |
| `--s11-margin-weight` | `0.10` | Match-margin reward weight. |
| `--maximum-height-mm` | topology-dependent | Physical radiator-height limit. |
| `--height-weight` | `0.10` | Weight of squared height excess. |
| `--minimum-horizon-gain-dbi` | `2` | Minimum azimuth-ring gain in horizon/ring mode (legacy option name). |
| `--null-weight` | `0.25` | Weight of azimuth-ring minimum deficit. |
| `--maximum-ripple-db` | `1.5` | Allowed azimuth-ring P90-P10 ripple. |
| `--ripple-weight` | `0.15` | Weight of excess ripple. |

If height is not set, the campaign uses
`(0.70 + 0.50 * maximum_coil_count) * wavelength`. Constraint violations are
recorded separately so topology ranking can prefer feasible candidates.
Simulation or geometry failures receive a finite penalty and remain visible in
the evaluation log rather than aborting the campaign.

### Budgets, seeds, restarts, and confirmation

Choose exactly one campaign budget style:

- `--maxiter N` gives each differential-evolution run N generations.
- `--hours H` estimates an evaluation budget from `--seconds-per-eval`, divides
  it across topology cases and seeds, and accounts for population dimension.

The default population multiplier is 8. By default, four system-random seeds
are generated for a broad campaign and two for fine tuning; their exact values
are printed and recorded. Runs are sequential because EMerge and Gmsh maintain
process-global state. For independent external parallelism, use separate
processes and separate output directories, never threads.

An apparent new best is repeated `--confirmation-runs` times (default 3, an odd
number). The median score becomes the consensus; outliers beyond
`--confirmation-score-tolerance` are recorded and unreliable candidates can be
quarantined. Campaign metadata distinguishes optimizer candidate calls from
physical simulation calls, because confirmation performs extra simulations.

Ctrl-C is handled between evaluations. Atomic best-result files and completed
run summaries remain available, but the campaign does not resume an incomplete
differential-evolution population.

### Automatic numerical preflight

By default, optimization checks for a convergence certificate matching the
frequency, numerical configuration, angular sampling, and numerical-reference
design fingerprint. A valid passing certificate is reused. If none matches,
the campaign runs the open-region convergence study automatically.

Important distinctions:

- The preflight design is a wavelength-scaled numerical reference independent
  of the user's warm start and topology.
- A failed preflight warns and continues unless `--require-convergence` is set.
- `--no-auto-convergence` disables automatic certificate generation but still
  permits checking an explicitly supplied report.
- `--skip-convergence-check` bypasses the preflight completely; use it only for
  development or when convergence has been established another way.
- Always run convergence again on the final physical winner.

### Campaign output files

| File | Contents |
|---|---|
| `campaign_best.json` | Best campaign-wide design, metrics, objective terms, bounds, and provenance. |
| `turns_<case>_best.json` | Current best for one discrete turn topology. |
| `turns_<case>_seed_<seed>.json` | Completed run result for one topology and seed. |
| `topology_leaderboard.json` | Ranked topology results with feasibility information. |
| `run_summaries.json` | All completed optimizer-run summaries. |
| `evaluations.csv` | Candidate-by-candidate variables, scores, metrics, failures, and confirmation data. |
| `campaign_seeds.json` | Exact random or command-line seeds selected before candidate evaluation. |
| `random_starts.json` | Validated per-topology/per-seed starting parameters and designs when `--random-start` is active. |
| `convergence_reference_design.json` | Numerical reference used by an automatic preflight. |
| `automatic_pipeline.json` | Automatic stage status, selected winners, verification state, and artifact paths. |

Automatic runs place detailed optimizer files under `rough_search/` and
`fine_tune/`, then place the canonical best, verification JSON, plots, drawing,
and forming STEP at the run root.

The best-result JSON records the final physical design, search-space bounds,
initial vector, topology, seed, objective components, S11 and far-field
metrics, numerical settings, fixed wire diameter, and candidate/simulation
counts. It can be passed directly to `--warm-start`, `verify_best.py`,
`check_open_region.py`, or `load_design()`.

Live progress follows the selected objective: horizon P10, every requested
theta-ring P10 plus the worst value, requested theta/phi directional gain, or
peak gain. It also shows the S11 limit and any active beamwidth goal instead of
reporting horizon defaults for every mode.

### Optimizer option reference

Run `python examples/optimize_gain.py --help` for the authoritative CLI. The
options are grouped here by intent.

Campaign and topology:

| Option | Default | Meaning |
|---|---:|---|
| `--frequency-mhz` | `868` | Target frequency. |
| `--match-bandwidth-mhz` | scaled 10 MHz | Three-point match span. |
| `--wire-diameter-mm`, `--wire-diameter` | inherited | Fixed conductor diameter. |
| `--maxiter` / `--hours` | `20` iterations | Mutually exclusive generation or time budget. |
| `--seconds-per-eval` | `8` | Estimate used to translate hours into evaluations. |
| `--popsize` | `8` | Differential-evolution population multiplier. |
| `--seeds` | random: broad 4; fine 2 | Explicit comma-separated seeds for reproducibility; omitted values are randomized and recorded. |
| `--coil-count` | inherited | One fixed coil count. |
| `--coil-counts` | inherited | Comma-separated one-turn topologies. |
| `--turn-cases` | inherited | Explicit cases such as `none,1,1x2`. |
| `--lock-coils` | off | Share one pitch and radius across all coils. |
| `--warm-start` | reference | Raw design or optimizer-result JSON. |
| `--random-start` | off | Replace broad-search `x0` with a reproducible validated uniform sample across complete bounds. |
| `--finetune`, `--fine-tune` | off | Multiscale populations, restarts, and local refinement. |
| `--automatic` | off | Rough search, winner fine tune, coordinate polish, verification, drawing, and forming tools. |
| `--automatic-rough-fraction` | `0.65` | Automatic optimizer budget assigned to rough search. |
| `--output` | timestamped directory | New campaign output directory. |
| `--report-every` | `10` | Console reporting interval in candidate calls. |

Fine-tuning controls:

| Option | Default | Meaning |
|---|---:|---|
| `--finetune-near-radius` | `0.03` | Normalized near-start population radius. |
| `--finetune-wide-radius` | `0.10` | Normalized wider-start population radius. |
| `--finetune-mutation MIN MAX` | `0.20 0.60` | Differential-evolution mutation range. |
| `--finetune-recombination` | `0.30` | Differential-evolution recombination. |
| `--restart-stagnation-generations` | `10` | Generations without progress before manual-fine restart or automatic stage transition. |
| `--restart-min-improvement` | `0.05` | Score improvement required to reset stagnation. |
| `--local-search-evaluations` | `24` | Bounded local-search budget. |
| `--local-search-step` | `0.03` | Initial normalized coordinate step. |
| `--local-search-min-step` | `0.001` | Smallest normalized step. |
| `--local-search-elites` | `3` | Number of elite starts for local search. |
| `--polish` | off | Deterministic one-parameter-at-a-time coordinate polish. |
| `--polish-evaluations` | `12 * variables` | Polish reserve, capped by the candidate budget. |
| `--polish-min-improvement` | `0.001` | Smallest accepted polish score decrease. |

Pattern and objective:

| Option | Default | Meaning |
|---|---:|---|
| `--pattern` | `horizon` | `horizon`, `ring`, `directional`, or `peak`. |
| `--target-theta`, `--theta` | `90` degrees | One or more space-separated omnidirectional ring angles; one value can also be a directional polar angle when phi is supplied. |
| `--target-phi` | unspecified | Optional azimuth; supplying it selects a single directional target. |
| `--target-beamwidth-deg` | disabled | Ring elevation HPBW, or both directional cuts when phi is supplied. |
| `--beamwidth-weight` | `1` | Beamwidth-error penalty weight. |
| `--maximum-height-mm` | automatic | Maximum radiator height. |
| `--s11-limit-db` | `-10` | Match constraint at every sampled frequency. |
| `--mismatch-weight` | `2` | S11 violation weight. |
| `--s11-margin-target-db` | `-12` | Extra-match-margin target. |
| `--s11-margin-weight` | `0.10` | Match-margin reward weight. |
| `--confirmation-runs` | `3` | Odd number of incumbent repeat solves. |
| `--confirmation-score-tolerance` | `1` | Allowed confirmation-score spread. |
| `--minimum-horizon-gain-dbi` | `2` | Minimum azimuth-ring gain in horizon/ring mode (legacy option name). |
| `--null-weight` | `0.25` | Azimuth-ring minimum-deficit weight. |
| `--maximum-ripple-db` | `1.5` | Azimuth-ring P90-P10 ripple target. |
| `--ripple-weight` | `0.15` | Excess-ripple weight. |
| `--height-weight` | `0.10` | Height-excess weight. |

Numerics and convergence:

| Option | Default | Meaning |
|---|---:|---|
| `--angular-step` | `2` degrees | Far-field grid spacing; greater than 0 and at most 10. |
| `--air-margin-wavelengths` | `0.25` | Structure-to-Huygens air margin. |
| `--abc-buffer-wavelengths` | `1.0` | Huygens-to-ABC ordinary-air buffer. |
| `--wavelength-resolution` | `0.33` | Free-space mesh wavelength factor. |
| `--solver` | `auto` | EMerge linear solver backend. |
| `--convergence-report` | frequency-specific | Certificate to validate or create. |
| `--skip-convergence-check` | off | Bypass convergence preflight. |
| `--no-auto-convergence` | off | Do not generate a missing certificate. |
| `--require-convergence` | off | Abort unless a matching certificate passes. |
| `--scipy-polish` | off | Legacy SciPy DE polish; unbudgeted and separate from coordinate polish. |

## Custom optimization API

The library-level optimizer tools support smaller custom studies without the
campaign script. A `DesignVariable` targets a dataclass field using a dotted
path; tuple indices are supported. `linked_paths` changes several physical
fields with one optimizer coordinate.

```python
from emerge_loaded_antenna import (
    DesignSpace,
    DesignVariable,
    FrequencySweep,
    S11Objective,
    SimulationOptions,
    load_reference_design,
)

base = load_reference_design(868e6)
space = DesignSpace(
    base=base,
    variables=(
        DesignVariable("straight_lengths.0", 0.10, 0.30, label="bottom length"),
        DesignVariable("radial_length", 0.05, 0.14),
        DesignVariable(
            "coils.0.radius",
            0.006,
            0.025,
            linked_paths=("coils.1.radius",),
        ),
    ),
)

print(space.names)
print(space.bounds)
print(space.initial_vector)
candidate = space.decode(space.initial_vector)

objective = S11Objective(
    space,
    target_frequency=868e6,
    options=SimulationOptions(sweep=FrequencySweep.single(868e6)),
)
score = objective(space.initial_vector)
print(score, objective.records[-1])
```

`DesignSpace` can normalize and denormalize vectors and validates decoded
designs before simulation. `S11Objective` minimizes single-frequency S11.
`GainMatchObjective` combines peak gain with an S11 ceiling.
`RobustGainObjective` is the campaign-grade three-point matching, pattern,
height, beamwidth, and confirmation objective. Optional evaluation and
confirmation callbacks can stream records into another optimizer or database.

EMerge/Gmsh model construction is process-global. Evaluate one objective at a
time in a process. Use process-based parallelism only when each worker has an
independent process and output location.

## Verify a winning design

`verify_best.py` runs an optional coarse solve and a finer solve, compares the
metrics, saves JSON, and plots the impedance and patterns. It can also generate
the fabrication sheet and printable coil-winding jigs from the verified design:

```powershell
.\.venv\Scripts\python.exe -u .\examples\verify_best.py `
    optimization_results\868mhz_horizon\campaign_best.json `
    --frequency-points 13 `
    --angular-step 0.5 `
    --design-sheet `
    --jig-models
```

Use `--latest` to verify the most recently updated campaign winner without
copying its path:

```powershell
.\.venv\Scripts\python.exe -u .\examples\verify_best.py `
    --latest `
    --design-sheet `
    --jig-models
```

It searches recursively under `optimization_results` for
`campaign_best.json` and selects the newest file by modification time. This
also works with an in-progress campaign after it has checkpointed its first
winner. An explicit result path and `--latest` cannot be combined.

The solve frequency is inferred from optimizer metadata when possible. The
default sweep is the wavelength-scaled equivalent of 30 MHz at 868 MHz. The
fine mesh uses eight wire sections, a 2x antenna surface factor, 6x radial
factor, 2x feed factor, 20 curved-boundary segments, 0.33-wavelength volume
resolution, and a 0.30-wavelength air margin.

Verification output includes:

- `verification.json` with coarse/fine metrics and a `quality` verdict for the
  configured worst-band S11 limit, 0.5 dB coarse/fine agreement, and numerical
  preflight status;
- `s11_verified.png`;
- `horizon_gain.png`;
- `principal_plane_gain.png`.

For a ring objective, verification also writes `target_ring_gain.png` with all
requested theta rings overlaid.

With `--design-sheet`, it additionally writes `design_sheet.pdf` using the fine
verification result, so the sheet includes dimensions, the verified S11 sweep,
the XY/horizon and XZ/elevation gain lobes, and every configured optimizer
target ring when the source campaign uses ring mode. Each verified target-ring
legend entry reports its minimum, maximum, and power-averaged realized gain in
dBi.

With `--jig-models`, it writes `coil_formers.step`, containing the exact grooved
winding formers, sizing mandrels, transition witness marks, and radial gauge.
For a zero-coil design the flag reports that no forming tools are needed.

Use `--show-model`, `--show-mesh`, or `--show-3d` for interactive inspection,
and `--skip-coarse` when only the fine run is needed. Beamwidth remains a soft
optimization goal, so its verified value/error is reported as an observation
rather than a pass/fail threshold. Investigate any quality warning rather than
treating the optimizer score as final truth.

## Open-region convergence

Run the convergence study on the final result:

```powershell
.\.venv\Scripts\python.exe -u .\examples\check_open_region.py `
    optimization_results\868mhz_horizon\campaign_best.json
```

The tool independently varies air margin, ABC buffer, and wavelength mesh
resolution in isolated worker solves. Defaults are:

| Study axis | Probe values | Selected value |
|---|---|---:|
| Air margin | `0.20, 0.25, 0.35` wavelength | `0.25` |
| ABC buffer | `0.75, 1.00, 1.25` wavelength | `1.00` |
| Mesh resolution | `0.50, 0.33, 0.25` wavelength | `0.33` |

The default angular step is 4 degrees and per-sample timeout is 600 seconds.
The resulting schema-versioned certificate contains the design fingerprint,
frequency, numerical settings, samples, metric deltas, individual checks, and
overall pass/fail state. It checks reflection magnitude, peak and horizon gain,
ripple, and peak-direction stability.

Use `--air-margins`, `--abc-buffers`, and `--mesh-resolutions` to widen the
study. The value lists must include a stricter probe than each selected value.
Do not reuse a certificate for another design or numerical configuration; the
validation helpers reject mismatched metadata.

## Solvers and performance

Accepted solver names are `auto`, `superlu`, `umfpack`, `pardiso`, `cudss`,
`mumps`, `aasds`, and `cholmod`. Availability depends on the local EMerge
installation. Start with `auto`, then benchmark an installed sparse direct
solver on the actual mesh size.

For example, install EMerge's CuDSS integration when supported:

```powershell
.\.venv\Scripts\emerge.exe install-solver cudss
```

Optimization cost is dominated by full-wave solves. Before committing to a
long run:

- time several representative evaluations and set `--seconds-per-eval`;
- use global mode before fine mode;
- compare only plausible topology cases;
- keep the optimizer mesh fixed within one campaign;
- use multiple seeds rather than trusting one stochastic run;
- reserve a separate budget for verification and convergence.

A finer angular step increases far-field sampling cost and is especially
important for narrow directional lobes and beamwidth objectives.

## Interpreting results

- **Feasible beats merely high-gain.** Check every S11 point, height, pattern
  constraints, and the topology leaderboard's feasibility fields.
- **P10 is robust ring gain.** In horizon and ring modes, 90% of azimuth
  samples are at or above P10; it is less sensitive to one isolated numerical
  spike than peak gain.
- **Beamwidth is a tradeoff.** It is a weighted soft penalty. Inspect the ring
  elevation width or both directional cut widths in the JSON instead of only
  total score.
- **A best candidate is not yet a validated antenna.** Confirmation reduces
  solver noise; it does not replace mesh/open-region convergence or measurement.
- **Scores compare only like-for-like campaigns.** Changing weights, match
  span, far-field step, solver settings, or numerical domain changes the score.
- **Topology rankings include search quality.** A topology with a weak budget
  may rank poorly because it was underexplored, not because it is impossible.

## Troubleshooting

**The campaign rejects the output directory.** Choose a new or empty directory.
Campaigns intentionally do not merge with or resume old populations.

**A warm start is outside the normal bounds.** Valid warm-start dimensions
expand the wavelength-based bounds. Invalid geometry still fails validation;
the optimizer does not clip it into a different antenna.

**A thick wire makes the search space invalid.** Coil pitch, radius,
transition, straight, and radial bounds include diameter-dependent clearance.
Reduce the requested diameter or choose a frequency/topology with enough
physical room.

**A pattern target conflicts with `--pattern`.** Theta alone selects ring mode;
phi selects directional mode. Do not combine `--pattern ring` with
`--target-phi`, or `--pattern horizon`/`peak` with a ring or direction target.
A beamwidth target must be greater than zero and no more than 180 degrees for a
ring or 360 degrees for a directional point.

**The lobe points in the wrong direction.** Confirm the spherical convention:
theta is measured from +Z and phi from +X toward +Y. Do not confuse target
theta with `radial_angle_deg`, which controls the physical ground radials.
Directional mode maximizes gain at the requested coordinate; it does not force
the global peak to occur there. Verification reports requested-direction gain
and global peak direction separately.

**The beamwidth looks quantized.** Reduce `--angular-step` and verify with the
0.5-degree default used by `verify_best.py`. A grid cannot resolve features
much smaller than its sampling interval.

**Repeated evaluations differ.** Use incumbent confirmation, review the
recorded score spread, and run convergence checks. Solver or mesh sensitivity
can make tiny score differences meaningless.

**A solver name is accepted but fails.** The CLI lists supported EMerge backend
names, not what is installed locally. Select `auto`, install the desired native
backend, or consult EMerge's solver setup.

**A long run stops.** Inspect `campaign_best.json`, topology best files,
`evaluations.csv`, and completed run summaries. These are written incrementally
and atomically where appropriate, but an interrupted evolutionary population
cannot be resumed.

## Development and project layout

```text
emerge_loaded_antenna/
  config.py          physical and numerical configuration
  geometry.py        continuous radiator centerline construction
  drawing.py         fabrication dimensions and PDF/SVG/PNG export
  formers.py         STEP/STL winding, sizing, and radial forming tools
  simulation.py      EMerge model, solve, and far-field metrics
  optimize.py        reusable design spaces and objectives
  presets.py         tracked reference design and scaling
  serialization.py   design JSON helpers
  convergence.py     convergence-certificate validation
  data/              versioned reference design
examples/
  optimize_gain.py   campaign optimizer
  verify_best.py     independent fine verification and plots
  check_open_region.py numerical convergence study
main.py              editable interactive example
tests/               geometry, simulation, optimizer, and campaign tests
```

Run the checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The test suite mocks expensive EMerge paths where appropriate and also tests
geometry and drawing invariants, serialization, campaign scheduling, objective
accounting, beamwidth measurement, CLI validation, and convergence certificates.

## Modeling scope

The current model is an idealized free-space antenna study. It includes the
radiator, swept finite-radius conductor, feed region, radial ground system,
closed Huygens surface, and all-face ABC or PML termination. It does not by
itself model every practical installation detail, such as connector launch,
cable common-mode current, finite conductivity and surface finish, dielectric
supports, enclosure coupling, nearby structures, soil, weather, or fabrication
tolerance.

Treat optimization as a disciplined way to generate and compare candidates.
Treat fine-mesh verification, numerical convergence, tolerance analysis, and
physical measurement as separate required stages before relying on a design.
