# Smooth Two-Coil Loaded Antenna in EMerge

This script generates and simulates a **smooth, continuously curved two-coil loaded antenna** directly in [EMerge](https://github.com/FennisRobert/EMerge).

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

The important feature is that the straight sections and coils are **not joined by sharp corners**. Instead, the entire antenna is generated as one smooth 3D centerline and then swept with a circular wire cross-section.

## Requirements

The script is intended for:

* Python 3.10–3.13
* EMerge 2.8.4
* NumPy

Install EMerge using your normal EMerge installation procedure.

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

The resulting XYZ coordinates are passed to:

```python
antenna_curve = em.geo.Curve(
    xpts,
    ypts,
    zpts,
    ctype="Spline",
)
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

This script instead uses compact quintic Hermite connectors. Each connector
starts vertically and arrives tangent to the constant-pitch helix. Its angular
allowance is independent of pitch, so a 6 mm connector no longer consumes most
of a tightly pitched turn or creates a broad sweeping elbow.

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

Most antenna dimensions are located near the top of the script.

### Wire

```python
WIRE_RADIUS = 0.75 * mm
```

This is the physical radius of the conductor.

For example:

```text
0.75 mm radius
=
1.50 mm wire diameter
```

### Bottom Straight

```python
BOTTOM_LENGTH = 50 * mm
```

Length from the feed to the beginning of the first loading coil.

### Coil 1

```python
COIL1_RADIUS = 10 * mm
COIL1_TURNS = 6
COIL1_PITCH = 3.0 * mm
COIL1_TRANSITION = 6 * mm
COIL1_TRANSITION_ANGLE = 45.0
```

`COIL1_RADIUS` is measured from the helix axis to the **wire centerline**.

Approximate outside coil diameter is therefore:

```text
2 × (coil radius + wire radius)
```

For the default example:

```text
2 × (10 + 0.75)
=
21.5 mm
```

`COIL1_TURNS` must currently be an integer.

For example:

```text
6 turns = 2160°
```

Using an integer number of turns allows the outgoing straight wire to return to exactly the same XY position as the incoming wire.

`COIL1_PITCH` is the vertical rise per complete revolution.

For example:

```text
6 turns × 3 mm pitch
=
18 mm total axial coil height
```

`COIL1_TRANSITION` controls how gradually the wire enters and exits the helix.

A larger value produces a gentler bend.

`COIL1_TRANSITION_ANGLE` controls where each compact connector joins the
constant-pitch helix. It is independent of pitch; 45 degrees is the default.

### Middle Straight

```python
MIDDLE_LENGTH = 100 * mm
```

Distance between the two loading coils.

### Coil 2

```python
COIL2_RADIUS = 10 * mm
COIL2_TURNS = 6
COIL2_PITCH = 3.0 * mm
COIL2_TRANSITION = 6 * mm
```

These have the same meaning as the Coil 1 parameters.

### Top Straight

```python
TOP_LENGTH = 50 * mm
```

Length above the second loading coil.

## Coil Handedness

Each coil can currently be generated as either right-handed or left-handed.

For example:

```python
path_builder.coil(
    radius=COIL1_RADIUS,
    turns=COIL1_TURNS,
    pitch=COIL1_PITCH,
    transition=COIL1_TRANSITION,
    handedness="RH",
)
```

Available values are:

```text
"RH"    right-handed
"LH"    left-handed
```

Both coils can use the same handedness or different handedness.

## Geometry Resolution

### Centerline Points

```python
POINTS_PER_TURN = 20
```

This controls how densely the helix is sampled before creating the EMerge
spline. Two-turn coils need at least 20 samples per turn with the current
geometry to avoid Gmsh PLC errors.

Higher values provide a more accurate representation but increase geometry complexity.

A reasonable starting point is:

```text
24–48 points per turn
```

### Wire Cross-Section

```python
WIRE_SECTIONS = 8
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

The default example targets the 868 MHz ISM/SRD region:

```python
F0 = 868 * MHz

FREQ_START = 830 * MHz
FREQ_STOP = 906 * MHz
FREQ_POINTS = 39
```

This produces a sweep around 868 MHz.

With these values, 868 MHz is one of the actual simulated frequency points.

The antenna dimensions supplied with the script are **examples only** and are not claimed to form a properly tuned 868 MHz antenna.

## Feed

The antenna centerline starts at:

```python
z = PORT_HEIGHT
```

A short cylindrical lumped-port region is created from the grounded hub to
the antenna:

```python
PORT_HEIGHT = 2 * mm
```

The feed volume intentionally has no copper material assignment. Its side
surface carries the EMerge lumped-port boundary condition, while the solid
radial hub beneath it supplies the ground reference. Assigning copper to the
feed suppresses the port field and produces a flat 0 dB S11 response.

The default port impedance is:

```python
PORT_IMPEDANCE = 50.0
```

corresponding to a conventional 50 Ω RF system.

## Air Region

The antenna is placed inside an air box.

The margin around the antenna defaults to:

```python
AIR_MARGIN = 0.25 * WAVELENGTH
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

The antenna conductor is kept as one continuous swept volume. Splitting it
into touching coil and straight volumes causes Gmsh PLC errors for this thin
wire in the current EMerge version, so it uses one reliable local mesh size:

```python
ANTENNA_MESH_SIZE = 2.5 * WIRE_RADIUS
```

The 2.5-radius setting is required for the tested two-turn coils; the previous
3-radius setting remains adequate for the simpler one-turn geometry.

and the model also receives the global setting:

```python
model.set_resolution(0.33)
```

For the thin curved wire, the script also increases Gmsh's curved-boundary
meshing factor to reduce PLC errors at coil transitions:

```python
model.mesher.set_curved_boundary_meshing(20)
```

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

The preview is rendered in mesh wireframe mode, so element edges are visible
instead of only the metallic material shading.

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

The simplest workflow is to modify only the parameter section.

For example:

```python
WIRE_RADIUS = 0.5 * mm

BOTTOM_LENGTH = 90 * mm

COIL1_RADIUS = 8 * mm
COIL1_TURNS = 8
COIL1_PITCH = 2.5 * mm
COIL1_TRANSITION = 5 * mm

MIDDLE_LENGTH = 120 * mm

COIL2_RADIUS = 8 * mm
COIL2_TURNS = 5
COIL2_PITCH = 2.5 * mm
COIL2_TRANSITION = 5 * mm

TOP_LENGTH = 80 * mm
```

The rest of the geometry and simulation regenerate automatically.

## Automatic Optimization

One major advantage of generating the antenna directly in EMerge instead of FreeCAD is that all antenna dimensions are Python variables.

This makes automatic optimization possible.

For example, a future version can expose parameters such as:

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

An optimizer can then automatically generate and simulate many antenna geometries.

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
COIL1_RADIUS = 10 * mm
WIRE_RADIUS = 0.75 * mm
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

The two connectors use a fixed angular allowance set by
`COIL1_TRANSITION_ANGLE`; the remaining requested rotation uses the specified
constant pitch. Consequently, connector length can be selected for bend
quality without imposing a minimum coil pitch.

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
