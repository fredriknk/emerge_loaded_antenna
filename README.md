# Smooth Two-Coil Loaded Antenna in EMerge

This package generates and simulates a **smooth, continuously curved two-coil loaded antenna** directly in [EMerge](https://github.com/FennisRobert/EMerge). It provides a reusable Python API for scripts and optimizers, plus `main.py` as an interactive plotting example.

The antenna geometry consists of:

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

* Python 3.10–3.13
* EMerge 2.8.4
* NumPy

Install the project in editable mode inside the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
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

design = replace(AntennaDesign(), middle_length=225e-3)
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
        DesignVariable("bottom_length", 100e-3, 180e-3),
        DesignVariable("middle_length", 160e-3, 260e-3),
        DesignVariable("coil1.pitch", 5e-3, 10e-3),
        DesignVariable("coil2.pitch", 5e-3, 10e-3),
    ),
)
objective = S11Objective(space, target_frequency=868e6)

# Pass objective and space.bounds to scipy.optimize or another minimizer.
score = objective(space.initial_vector)
```

`GainMatchObjective` adds a configurable S11 constraint penalty and maximizes
3D peak isotropic gain. Both objectives retain evaluation history and convert
geometry/solver failures into a finite penalty. See
`examples/optimize_gain.py` for a SciPy differential-evolution example.

Run the comprehensive 868 MHz search with:

```powershell
.\.venv\Scripts\python.exe -u .\examples\optimize_gain.py
```

Routine EMerge INFO messages are suppressed during the search. A compact
progress line reports completion, elapsed time, ETA, failures, best S11 and
best gain every ten evaluations. Every candidate is flushed to
`optimization_results/evaluations.csv`, and the final design is written to
`optimization_results/best_result.json`. Use `--report-every`, `--maxiter`,
`--popsize`, `--seed`, and `--output` to control a run; `--help` lists them.

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

The antenna is generated as one continuous sequence:

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

<!-- Description of the superseded angular-velocity transition:
The transition uses the quintic smoothstep function:

```text
S(u) = 10u³ - 15u⁴ + 6u⁵
```

where:

```text
u = 0   start of transition
u = 1   end of transition
```

The function has zero first and second derivatives at its endpoints.

It is used to smoothly ramp the **angular velocity** of the antenna path from:

```text
0
```

for a straight wire to:

```text
2π / pitch
```

for the normal helical section.

This means that the generated mathematical path has smooth position, direction, and curvature through the transition.
-->

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
    bottom_length=0.120,
    coil1=CoilDesign(
        radius=0.010,
        turns=1,
        pitch=0.007,
        transition=0.006,
        transition_offset=0.00475,
        handedness="RH",
    ),
    middle_length=0.221,
    coil2=CoilDesign(
        radius=0.010,
        turns=1,
        pitch=0.007,
        transition=0.006,
        transition_offset=0.00475,
        handedness="RH",
    ),
    top_length=0.150,
)
```

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

candidate = replace(design, middle_length=0.225)
candidate = replace(
    candidate,
    coil1=replace(candidate.coil1, pitch=0.0065),
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

## Air Region

The antenna is placed inside an air box.

The margin around the antenna defaults to a quarter wavelength:

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

The absorbing boundary condition is applied to the outside surfaces of this region.

## Meshing

The antenna conductor remains one continuous swept volume. Its path is a
composite wire of exact straight lines, local transition curves, and three
Bezier arcs per helical turn. This avoids both coincident-volume PLC errors and
the excessive triangulation caused by the former global BSpline.

`MeshSettings(antenna_size_factor=3.0)` sets the antenna maximum element size
to three wire radii. This setting is tested with both one- and two-turn coils.

The equivalent global and curved-boundary settings are also configurable:

```python
mesh = MeshSettings(
    wavelength_resolution=0.5,
    curved_boundary_segments=12,
)
```

The curved-boundary factor is moderate because it no longer has to repair a
distorted global spline.

With the current example geometry, the old global-spline mesh used 7,595 nodes
and 54,925 total elements. The composite path uses 3,048 nodes and 22,734
elements for one-turn coils. Two-turn coils use 3,521 nodes and 26,586
elements. The script prints current node, total-element, and volume-element
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

* both coils have the expected number of turns,
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
    bottom_length=90e-3,
    middle_length=120e-3,
    top_length=80e-3,
    coil1=replace(design.coil1, radius=8e-3, turns=8, pitch=2.5e-3),
    coil2=replace(design.coil2, radius=8e-3, turns=5, pitch=2.5e-3),
)
```

The geometry and simulation regenerate from the candidate automatically.

## Automatic Optimization

The `DesignSpace` adapter exposes scalar and integer parameters using dotted
paths, including nested coil fields such as:

```text
bottom straight length
coil 1 radius
coil 1 turns
coil 1 pitch
middle length
coil 2 radius
coil 2 turns
coil 2 pitch
top length
```

and define an objective such as:

```text
minimize |S11| at 868 MHz
```

or, more realistically:

```text
minimize worst-case S11 from 863–870 MHz
```

`S11Objective` and `GainMatchObjective` are directly callable by SciPy and
other minimizers. They decode vectors, validate and simulate candidates, keep
history, and return a finite penalty when a candidate cannot be meshed or
solved.

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
    coil1=CoilDesign(radius=10e-3),
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
portion. They are no longer required to be shorter than `turns * pitch`.

<!-- Superseded total-height example:
For:

```text
6 turns
3 mm pitch
```

the total coil height remains:

```text
18 mm
```
-->

The two connectors use a small chord allowance set by
`COIL1_TRANSITION_OFFSET`; the remaining requested rotation uses the specified
constant pitch. Consequently, connector length can be selected for bend
quality without imposing a minimum coil pitch. The exact axial height is:

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
