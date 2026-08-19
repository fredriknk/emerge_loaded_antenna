# Smooth Variable-Coil Loaded Antenna in EMerge

This package generates and simulates a **smooth, continuously curved loaded
antenna with any number of coils, including zero**, directly in
[EMerge](https://github.com/FennisRobert/EMerge). It provides a reusable Python
API for scripts and optimizers, plus `main.py` as an interactive plotting
example. The default design has two coils.

The default antenna geometry consists of:

```text
        Top straight
             │
        ╭─────────╮
       ╱           ╲
      │   Coil 2    │
       ╲           ╱
        ╰─────────╯
             │
       Middle straight
             │
        ╭─────────╮
       ╱           ╲
      │   Coil 1    │
       ╲           ╱
        ╰─────────╯
             │
       Bottom straight
             │
            Feed
```

The important feature is that the straight sections and coils are **not joined
by sharp corners**. Exact lines and local cubic Bezier pieces are assembled
into one continuous OpenCASCADE wire, then swept once with a circular wire
cross-section. Each piece is independent, so a coil cannot bend a distant
straight section as it could with one global BSpline.

## Requirements

The package is intended for:

* Python 3.12–3.13
* EMerge 2.8.4
* NumPy

## Environment setup

Run these commands from the repository root. The `.venv` folder is local to
this project and is ignored by git.

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[optimize,verify]"
```

If PowerShell blocks activation scripts, you can still use the full
`.\.venv\Scripts\python.exe` commands above. To activate the environment for
the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[optimize,verify]"
```

If Python 3.13 is not installed but you have `uv`, let `uv` create the
environment:

```powershell
uv venv .venv --python 3.13 --seed
.\.venv\Scripts\python.exe -m pip install -e ".[optimize,verify]"
```

## Library API

All public dimensions are metres and all frequencies are hertz. A headless
single-frequency evaluation suitable for an optimizer is:

```python
from dataclasses import replace

from emerge_loaded_antenna import (
    AntennaDesign,
    FrequencySweep,
    SimulationOptions,
    simulate,
)

design = replace(
    AntennaDesign(),
    straight_lengths=(140e-3, 225e-3, 140e-3),
)
options = SimulationOptions(
    sweep=FrequencySweep.single(868e6),
    show_geometry=False,
    show_mesh=False,
    compute_farfield=False,
    verbose=False,
)
result = simulate(design, options)
print(result.s11_db_at(868e6))
print(result.as_dict())
```

`simulate()` returns structured frequencies, complex S11, S11 in dB, mesh
counts, optional peak gain, and the underlying EMerge artifacts when deeper
inspection is needed. `build_model()` performs geometry and meshing without a
solve.

### Optimizer Adapter

```python
from emerge_loaded_antenna import (
    AntennaDesign,
    DesignSpace,
    DesignVariable,
    S11Objective,
)

space = DesignSpace(
    AntennaDesign(),
    (
        DesignVariable("straight_lengths.0", 100e-3, 180e-3),
        DesignVariable("straight_lengths.1", 160e-3, 260e-3),
        DesignVariable("coils.0.pitch", 5e-3, 10e-3),
        DesignVariable("coils.1.pitch", 5e-3, 10e-3),
    ),
)
objective = S11Objective(space, target_frequency=868e6)

# Pass objective and space.bounds to scipy.optimize or another minimizer.
score = objective(space.initial_vector)
```

`GainMatchObjective` optimizes unrestricted peak-anywhere gain.
`RobustGainObjective` can instead maximize 10th-percentile horizon gain or gain
in a requested direction while penalizing worst-band mismatch, azimuth ripple,
deep nulls and excessive height. All objectives retain evaluation history and
convert geometry/solver failures into a finite penalty.

### Robust optimization campaign

The default remains 868 MHz, so the existing twelve-hour command still works:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py --hours 12
```

The target is not hard-coded. For example, a fresh 915 MHz search over a
20 MHz matching band is:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --frequency-mhz 915 `
    --match-bandwidth-mhz 20 `
    --hours 12
```

With no `--warm-start`, the script synthesizes an electrically equivalent
starting geometry by scaling every length with wavelength. The dimensional
search bounds and default maximum-height penalty scale the same way. This is
only an initial point for differential evolution, not a required pre-optimized
antenna. Supply any current-schema design or campaign result with
`--warm-start`; its physical dimensions, coil count and turn counts are used
as written unless explicitly overridden by `--coil-count`, `--coil-counts`, or
`--turn-cases`.

Before the optimization timer starts, the script checks a frequency-specific
certificate such as
`optimization_results/open_region_convergence_915000000hz.json`. If missing or
mismatched, it runs seven isolated solves that vary Huygens clearance, ABC
distance, and air-mesh resolution. This preflight uses a wavelength-scaled
numerical reference problem, independently of the user's starting antenna, so
poor initial S11 or gain cannot block a new search. A passing certificate is
reused for later runs with matching frequency and numerical settings.

If the automatic comparison fails, the default is to print a prominent
warning, record `convergence_status: "warning"` in every result, and continue
the optimization. Use strict mode when an uncertified campaign must not start:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --frequency-mhz 915 `
    --hours 12 `
    --require-convergence
```

The convergence campaign can also be run explicitly:

```powershell
.\.venv\Scripts\python.exe -u .\examples\check_open_region.py `
    --frequency-mhz 915
```

`--no-auto-convergence` prevents generation and reuses a matching report when
available; otherwise it warns and continues, or aborts when combined with
`--require-convergence`. `--skip-convergence-check` bypasses even certificate
validation and cannot be combined with strict mode.

After preflight, the recommended campaign runs four independent
differential-evolution populations and divides the requested time budget
between them. To continue from one of your own saved winners, pass it
explicitly:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py --hours 12 `
    --warm-start `
    .\optimization_results\868mhz_YYYYMMDD_HHMMSS\campaign_best.json
```

The wall-time conversion assumes roughly eight seconds per robust evaluation;
override it with `--seconds-per-eval` if the live ETA on your machine settles
substantially higher or lower.

The broad search optimizes every straight length, one pitch and radius shared
by all coils, radial length, and radial angle. Thus a design with N >= 1 coils
has only N+5 continuous variables; the zero-coil case has three. After the first
coil, each added coil introduces only one additional straight length. S11 is
constrained at the lower edge, center and upper edge of the requested matching
band. Every run receives the synthesized or supplied design as `x0`; each
candidate is flushed to CSV and every new global best is atomically
checkpointed as `campaign_best.json`. Output goes into a new frequency-labelled
timestamped directory so previous campaigns are never overwritten.

Choose the coil count explicitly. A zero-coil campaign optimizes a conventional
straight radiator:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py --coil-count 0
```

To compare coil counts automatically in one campaign, pass a comma-separated
list. Each count starts with one turn per coil, so this searches `none`, `1`,
`1x1`, and `1x1x1` with the same objective and one shared leaderboard:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --hours 12 `
    --coil-counts 0,1,2,3
```

The time estimate is divided across every topology and seed. Lower-dimensional
topologies receive more generations so that each run gets approximately the
same number of expensive antenna evaluations. The CSV contains the union of
all topology parameters; inapplicable coil columns are left blank. The global
`campaign_best.json` records both `coil_count` and `turn_case`. Each topology
also gets an interruption-safe `turns_*_best.json` checkpoint, and
`topology_leaderboard.json` ranks their best results under the shared
objective.

Once the broad campaign has selected a topology and a good geometry, rerun its
winner with `--finetune`. This releases every coil pitch and radius as an
independent variable; with N coils the search then has 3N+3 variables:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --hours 12 `
    --warm-start `
    .\optimization_results\868mhz_YYYYMMDD_HHMMSS\campaign_best.json `
    --finetune
```

No topology flag is needed for that second stage: the coil and turn counts are
inferred from the saved winner. `--finetune` can also be applied directly to a
multi-topology campaign, but the larger populations make that substantially
more expensive.

Coil turns are discontinuous geometry choices and are therefore separate
searches rather than rounded continuous variables. Mixed coil counts and turn
counts can be selected directly in one list:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --hours 12 `
    --turn-cases none,1,1x1,1x2,2x1,1x1x1
```

To constrain all cases to two coils, the existing spelling remains available:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py `
    --hours 12 `
    --coil-count 2 `
    --turn-cases 1x1,1x2,2x1,2x2,3x3
```

When `--coil-count` is present, every turn case must contain that number of
entries. For example, three coils use
`--coil-count 3 --turn-cases 1x1x1,1x2x1`. `none` is the zero-coil case.
`--coil-counts` generates its own one-turn cases and therefore cannot be
combined with `--turn-cases`; use the mixed list form when explicit turns are
needed.

For best optimizer convergence, first use the shared-geometry broad campaign,
then fine-tune its winner. `--pattern directional --target-theta ...
--target-phi ...` selects a fixed beam direction; `--pattern peak` restores
unrestricted maximum-gain optimization. Run `--help` for all objective weights
and budget controls.

### Final verification

Never accept an optimizer's coarse-mesh number without convergence testing.
Verify a campaign winner with:

```powershell
.\.venv\Scripts\python.exe -u .\examples\verify_best.py `
    .\optimization_results\868mhz_YYYYMMDD_HHMMSS\campaign_best.json `
    --show-model `
    --show-mesh `
    --show-3d
```

This repeats the design on coarse and fine meshes, samples the 3D pattern at
0.5-degree resolution, infers the target frequency from the campaign result,
reports peak direction and horizon statistics, and saves S11, XY/horizon and
XZ/YZ/XY plots plus JSON convergence data. `--show-model` opens the final
geometry before meshing, `--show-mesh` opens its generated surface mesh, and
`--show-3d` displays the solved far-field lobe. The model and mesh viewers are
only enabled for the fine verification solve, not the preliminary coarse solve.
For a final seven-probe open-region certificate on the actual winner, run:

```powershell
.\.venv\Scripts\python.exe -u .\examples\check_open_region.py `
    .\optimization_results\915mhz_YYYYMMDD_HHMMSS\campaign_best.json
```

The initial reference preflight establishes sane campaign numerics; this final
design-specific check is the evidence to use when reporting the winning gain.

EMerge and Gmsh use process-global model state. Sequential evaluations in one
process are supported and tested. Use separate processes—not worker threads—
for parallel optimization.

The script imports:

```python
import numpy as np
import emerge as em

from emerge.plot import plot_sp, plot_ff, smith
```

## How the Geometry Is Constructed

The default two-coil antenna is generated as one continuous sequence:

```text
feed
 ↓
bottom straight
 ↓
smooth transition
 ↓
coil 1
 ↓
smooth transition
 ↓
middle straight
 ↓
smooth transition
 ↓
coil 2
 ↓
smooth transition
 ↓
top straight
```

The independent curve pieces are assembled into one OpenCASCADE wire:

```python
antenna_curve = CompositeCurve(path.segments)
```

EMerge then sweeps a circular cross-section along this centerline:

```python
antenna = antenna_curve.pipe(wire_section)
```

This produces the actual 3D copper conductor.

## Why the Coil Transitions Are Smooth

Simply joining a vertical line and a helix produces a discontinuous tangent:

```text
helix
   /
  /
 /
│
│ straight
```

That represents an infinitely sharp bend, which is not representative of a real bent wire.

This script instead uses compact cubic Hermite connectors. Each connector
starts vertically and arrives tangent to the constant-pitch helix. A small
chord offset controls how far it travels around the coil, independently of
pitch, so a 6 mm connector no longer creates a broad sweeping elbow.

In practical terms:

```text
Straight
   │
   │
   ╰────╮
        ╰──╮
           ╰──── helix
```

rather than:

```text
Straight
   │
   │
   └──────── helix
```

## Main Parameters

All physical dimensions live in immutable `AntennaDesign` and `CoilDesign`
objects. SI units are used throughout, so dimensions are specified in metres.

```python
from emerge_loaded_antenna import AntennaDesign, CoilDesign

design = AntennaDesign(
    wire_radius=1.0e-3,
    straight_lengths=(0.120, 0.221, 0.150),
    coils=(
        CoilDesign(
            radius=0.010,
            turns=1,
            pitch=0.007,
            transition=0.006,
            transition_offset=0.00475,
            handedness="RH",
        ),
        CoilDesign(
            radius=0.010,
            turns=1,
            pitch=0.007,
            transition=0.006,
            transition_offset=0.00475,
            handedness="RH",
        ),
    ),
)
```

The rule is simple: `N` coils require `N + 1` straight lengths. A zero-coil
design is therefore just:

```python
design = AntennaDesign(straight_lengths=(86e-3,), coils=())
```

Add entries to both tuples to create three or more coils.

The coil radius is measured to the wire centerline, so its approximate outside
diameter is `2 * (coil.radius + design.wire_radius)`. Pitch is the axial rise
per complete revolution. Transition length controls the local entry/exit bend;
transition offset sets the chord distance from the straight axis to the helix
join independently of pitch.

Coil turns must currently be positive integers. Handedness may be `"RH"` or
`"LH"`, independently for each coil.

Because designs are frozen and safe to reuse, change one parameter with
`dataclasses.replace`:

```python
from dataclasses import replace

lengths = list(design.straight_lengths)
lengths[1] = 0.225
coils = list(design.coils)
coils[0] = replace(coils[0], pitch=0.0065)
candidate = replace(
    design,
    straight_lengths=tuple(lengths),
    coils=tuple(coils),
)
```

## Geometry Resolution

### Preview Sampling

```python
POINTS_PER_TURN = 20
```

This controls only the sampled coordinates used for reporting dimensions and
computing the air-box bounds. It does not control the CAD or mesh complexity.
The actual helix uses three tangent-continuous cubic Bezier arcs per turn.

### Wire Cross-Section

```python
WIRE_SECTIONS = 6
```

The circular wire cross-section is represented by a polygon.

For example:

```text
6  = hexagonal approximation
8  = octagonal approximation
12 = smoother
16 = smoother again
```

Increasing this value increases mesh complexity.

For initial antenna optimization, 6–8 sections will usually be considerably faster than using a very finely segmented conductor.

## Simulation Frequency

The default sweep targets the 868 MHz ISM/SRD region:

```python
from emerge_loaded_antenna import FrequencySweep

sweep = FrequencySweep(center=868e6, span=100e6, points=5)
```

The center frequency is always included when `points` is odd. Use
`FrequencySweep.single(868e6)` for inexpensive optimizer evaluations.

The antenna dimensions supplied with the script are **examples only** and are not claimed to form a properly tuned 868 MHz antenna.

## Feed

Feed dimensions and reference impedance are part of the design:

```python
design = AntennaDesign(
    port_height=2e-3,
    port_impedance=50.0,
)
```

A short cylindrical lumped-port region is created from the grounded hub to
the antenna. The feed volume intentionally has no copper material assignment. Its side
surface carries the EMerge lumped-port boundary condition, while the solid
radial hub beneath it supplies the ground reference. Assigning copper to the
feed suppresses the port field and produces a flat 0 dB S11 response.

## Open Air Region and Far-Field Surface

The antenna is enclosed by a closed inner air box. Its six faces form the
Huygens integration surface used for every 2D and 3D far-field calculation.
The clearance defaults to a quarter wavelength:

```python
mesh = MeshSettings(air_margin_wavelengths=0.25)
```

where:

```text
wavelength = c / frequency
```

At 868 MHz, free-space wavelength is approximately:

```text
345 mm
```

so a quarter wavelength is approximately:

```text
86 mm
```

This is not the termination boundary. A separate ordinary-air shell extends a
further wavelength by default, and a second-order absorbing boundary condition
is applied to every exterior face:

```python
from emerge_loaded_antenna import OpenRegionSettings, SimulationOptions

options = SimulationOptions(
    open_region=OpenRegionSettings(
        mode="abc",
        abc_buffer_wavelengths=1.0,
    ),
)
```

The bottom is included. It is never left unassigned for EMerge to turn into a
PEC wall. Keeping the inner Huygens surface separate from the outer termination
also prevents boundary placement from changing the integration surface.

An all-sided PML is available for high-memory reference runs:

```python
options = SimulationOptions(
    open_region=OpenRegionSettings(
        mode="pml",
        pml_thickness_wavelengths=0.25,
        pml_mesh_layers=5,
    ),
)
```

The PML creates 26 edge/face/corner blocks around the inner box and is far more
expensive than the buffered ABC. It is intended for final cross-checks rather
than thousands of optimizer evaluations.

## Meshing

The antenna conductor remains one continuous swept volume. Its path is a
composite wire of exact straight lines, local transition curves, and three
Bezier arcs per helical turn. This avoids both coincident-volume PLC errors and
unnecessary triangulation along straight sections.

`MeshSettings(antenna_size_factor=3.0)` sets the antenna maximum element size
to three wire radii. This setting is tested with both one- and two-turn coils.

The equivalent global and curved-boundary settings are also configurable:

```python
mesh = MeshSettings(
    wavelength_resolution=0.33,
    curved_boundary_segments=12,
)
```

The curved-boundary factor is moderate because each curved segment is compact
and local. The script prints current node, total-element, and volume-element
counts after every mesh build.

Mesh settings strongly affect both simulation accuracy and run time.

It is usually best to use a relatively coarse mesh while exploring antenna dimensions, then rerun promising designs with progressively finer settings.

## Geometry Preview

Enable:

```python
SHOW_GEOMETRY = True
```

to open the EMerge geometry viewer before meshing.

This is strongly recommended while developing the antenna.

You should verify that:

* every configured coil has the expected number of turns,
* the straight sections are correctly aligned,
* the transitions are smooth,
* the conductor does not self-intersect,
* the wire diameter is correct,
* the feed joins the antenna correctly.

Set:

```python
SHOW_GEOMETRY = False
```

when performing automated parameter sweeps.

## Mesh Preview

To inspect the generated FEM mesh:

```python
SHOW_MESH = True
```

The preview renders boundary triangles in wireframe mode, so element edges are
visible instead of only the metallic material shading. Internal air-volume
tetrahedra are intentionally hidden; displaying them produces the dense web of
crossing magenta lines that can be mistaken for geometry artifacts.

For normal repeated simulations:

```python
SHOW_MESH = False
```

is more convenient.

## Running Without Solving

To generate and inspect geometry without performing the electromagnetic simulation:

```python
RUN_SOLVER = False
```

This is useful while adjusting antenna dimensions.

Once the geometry looks correct:

```python
RUN_SOLVER = True
```

## Choosing a Linear Solver

EMerge selects a linear solver automatically by default. Override it from the
interactive example with:

```powershell
.\.venv\Scripts\python.exe .\main.py --solver pardiso
```

Available names are `auto`, `superlu`, `umfpack`, `pardiso`, `cudss`, `mumps`,
`aasds`, and `cholmod`. Optional backends must be installed before use. On
Windows with an NVIDIA GPU, install EMerge's CUDA 12 CuDSS dependencies with:

```powershell
.\.venv\Scripts\emerge.exe install-solver cudss
```

Then run:

```powershell
.\.venv\Scripts\python.exe .\main.py --solver cudss
```

The optimization and verification scripts accept the same `--solver` flag.
An unavailable backend fails before meshing starts.

## S11 Results

After solving, the script displays S11 using:

```python
plot_sp(
    glob.freq,
    glob.S(1, 1),
)
```

and a Smith chart using:

```python
smith(
    glob.S(1, 1),
    f=glob.freq,
)
```

For antenna tuning, the main quantity of interest initially is usually:

```text
S11
```

around the desired operating frequency.

For a 50 Ω antenna system, a common first target might be:

```text
S11 < -10 dB
```

across the required operating band.

Lower is generally better at the target frequency, but impedance match alone does not guarantee good radiation efficiency.

## Far-Field Calculation

The script calculates complete total-gain cuts through all three principal
planes at the target frequency:

```text
X-Z elevation plane
Y-Z elevation plane
X-Y azimuth plane
```

It reports peak isotropic gain, peak direction, plane-average gain, approximate
3 dB beamwidth and front-to-back ratio for every cut.

When enabled, the interactive 3D total-gain lobe is shown with the antenna for
orientation:

```python
SHOW_3D_FARFIELD = True
FARFIELD_DB_FLOOR = -30
```

The 3D report includes peak gain plus spherical theta/phi and elevation.

This can be expanded later to investigate:

* antenna efficiency,
* directivity,
* polarization,
* gain across the full frequency sweep.

## Changing the Antenna

Construct a new design or use `dataclasses.replace` to derive one from a
known baseline. Nested coils can be replaced independently:

```python
from dataclasses import replace

candidate = replace(
    design,
    wire_radius=0.5e-3,
    straight_lengths=(90e-3, 120e-3, 80e-3),
    coils=(
        replace(design.coils[0], radius=8e-3, turns=8, pitch=2.5e-3),
        replace(design.coils[1], radius=8e-3, turns=5, pitch=2.5e-3),
    ),
)
```

The geometry and simulation regenerate from the candidate automatically.

## Automatic Optimization

The `DesignSpace` adapter exposes scalar and integer parameters using dotted
paths, including nested coil fields such as:

```text
straight_lengths.0
coils.0.radius
coils.0.turns
coils.0.pitch
straight_lengths.1
coils.1.radius
coils.1.turns
coils.1.pitch
straight_lengths.2
```

A variable can intentionally control several fields. This is how the campaign
shares coil geometry during its broad phase:

```python
DesignVariable(
    "coils.0.pitch",
    4e-3,
    12e-3,
    linked_paths=("coils.1.pitch", "coils.2.pitch"),
    label="shared_coil_pitch",
)
```

Those variables can then be passed to an objective such as:

```text
minimize |S11| at 868 MHz
```

or, more realistically:

```text
minimize worst-case S11 from 863–870 MHz
```

`S11Objective`, `GainMatchObjective`, and `RobustGainObjective` are directly
callable by SciPy and other minimizers. They decode vectors, validate and
simulate candidates, keep history, invoke optional per-evaluation callbacks,
and return a finite penalty when a candidate cannot be meshed or solved.

A typical workflow could become:

```text
Choose dimensions
       ↓
Generate smooth centerline
       ↓
Sweep copper conductor
       ↓
Generate mesh
       ↓
Run EMerge
       ↓
Read S11
       ↓
Optimizer changes dimensions
       ↓
Repeat
```

This is considerably easier than regenerating geometry in FreeCAD and repeatedly exporting STEP files.

## Why Use EMerge Directly Instead of FreeCAD?

FreeCAD is still useful when the simulation needs mechanically complex objects such as:

* antenna housings,
* brackets,
* PCB assemblies,
* mounting hardware,
* radomes,
* nearby metallic structures.

For the radiating wire itself, however, direct programmatic generation has several advantages:

* exact control over dimensions,
* one continuous 3D centerline,
* smooth coil transitions,
* easy parameter changes,
* automatic optimization,
* no STEP export/import loop,
* geometry and simulation remain in the same script.

A useful combined workflow may eventually be:

```text
Programmatic antenna wire
        ↓
      EMerge
        ↑
STEP geometry from FreeCAD
for housing / PCB / enclosure
```

## Important Modeling Notes

### The Coil Radius Is a Centerline Radius

If:

```python
design = AntennaDesign(
    wire_radius=0.75e-3,
    straight_lengths=(140e-3, 140e-3),
    coils=(CoilDesign(radius=10e-3),),
)
```

the physical outer radius of the coil is approximately:

```text
10.75 mm
```

and the outside diameter is approximately:

```text
21.5 mm
```

### Transition Length Is Independent of Pitch

The entrance and exit connector lengths are added around the constant-pitch
portion and may be longer than `turns * pitch`.

The two connectors use a small chord allowance set by
`CoilDesign.transition_offset`; the remaining requested rotation uses the
specified constant pitch. Consequently, connector length can be selected for
bend quality without imposing a minimum coil pitch. The exact axial height is:

```text
2 x transition + turns x pitch - join_angle x pitch / pi
```

where `join_angle = 2 x asin(offset / (2 x radius))` in radians.

Each long straight is one exact CAD line with only two endpoints. Its length
therefore does not increase centerline complexity, and no guard points are
needed to keep it straight at a coil junction.

### Integer Turns

The current implementation expects an integer number of turns.

This is intentional because after:

```text
N × 360°
```

the helix naturally returns to the same side of its axis, allowing the following straight section to lie on exactly the same vertical centerline.

Support for fractional turns could be added, but the outgoing straight would then need either:

* a shifted position,
* another smooth correction section, or
* a different coil geometry.

## Suggested Development Workflow

1. Set `RUN_SOLVER = False`.
2. Enter the approximate physical antenna dimensions.
3. Inspect the geometry with `SHOW_GEOMETRY = True`.
4. Confirm all smooth transitions visually.
5. Enable the solver.
6. Run a relatively coarse frequency sweep.
7. Adjust the antenna dimensions based on S11.
8. Automate the parameter search.
9. Refine the mesh around promising designs.
10. Evaluate S11, efficiency and radiation pattern together.

## Future Improvements

Useful additions to this script would include:

* automatic parameter optimization,
* CSV logging of each simulated design,
* automatic extraction of resonant frequency,
* bandwidth calculation,
* radiation efficiency calculation,
* realized gain optimization,
* multiple frequency-band objectives,
* support for fractional turns,
* arbitrary coil orientation,
* different coil radii for each transition,
* tapered coils,
* exporting generated antenna geometry,
* importing a PCB or enclosure from STEP,
* batch-running parameter studies without opening the viewer.

## Summary

The script models the antenna as:

```text
one mathematical centerline
          +
smooth straight/helix transitions
          +
circular conductor sweep
          =
one continuous 3D antenna
```

This approach is well suited to EMerge because the geometry can be generated, modified, meshed and simulated entirely from Python.

For antenna experimentation and automated tuning, this is generally more convenient than constructing the wire in FreeCAD and exporting a new STEP file after every dimensional change.
