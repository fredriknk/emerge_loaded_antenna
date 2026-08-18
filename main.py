"""
Smooth two-coil loaded antenna with four angled radials in EMerge
=========================================

Geometry:

        top straight
             |
        ~~~~~~~~~~~       coil 2
             |
        middle straight
             |
        ~~~~~~~~~~~       coil 1
             |
       bottom straight
             |
            feed

The ENTIRE antenna conductor above the feed is generated as one continuous
3D spline and swept with a circular wire cross-section.

The straight -> helix and helix -> straight transitions use a quintic
smoothstep on angular velocity, giving a smooth tangent and curvature
transition instead of a sharp bend.

Tested against the EMerge 2.8.4 API layout.
"""

import os
from pathlib import Path

# Prefer the MKL runtime installed in this venv.  On Windows, EMerge's
# automatic search can otherwise pick an inaccessible DLL from a base Conda
# installation, which makes Pardiso fail with 0xc06d007e at solve time.
_venv_mkl_dir = Path(__file__).resolve().parent / ".venv" / "Library" / "bin"
_venv_mkl = next(iter(sorted(_venv_mkl_dir.glob("mkl_rt*.dll"))), None)
if _venv_mkl is not None:
    os.environ["EMERGE_PARDISO_PATH"] = str(_venv_mkl)

import numpy as np
import emerge as em

from emerge.plot import plot_sp, plot_ff, smith


# ============================================================================
# UNITS
# ============================================================================

mm = 1e-3
MHz = 1e6


# ============================================================================
# ANTENNA PARAMETERS
#
# These are example values, NOT a tuned 868 MHz antenna.
# Change these first.
# ============================================================================

WIRE_RADIUS = 1.0 * mm          # 1.5 mm diameter wire

RADIAL_LENGTH = 86 * mm
RADIAL_ANGLE = 45.0              # degrees below the horizontal
RADIAL_COUNT = 4

BOTTOM_LENGTH = 170 * mm

COIL1_RADIUS = 10 * mm          # centerline radius
COIL1_TURNS = 1                 # integer turns
COIL1_PITCH = 7.0 * mm          # axial rise per full turn
COIL1_TRANSITION = 6 * mm       # smooth entrance/exit distance

MIDDLE_LENGTH = BOTTOM_LENGTH

COIL2_RADIUS = COIL1_RADIUS
COIL2_TURNS = COIL1_TURNS
COIL2_PITCH = COIL1_PITCH
COIL2_TRANSITION = COIL1_TRANSITION

TOP_LENGTH = BOTTOM_LENGTH


# ============================================================================
# GEOMETRY RESOLUTION
# ============================================================================

# Number of centerline samples for every complete revolution.
# 32 is a reasonable starting point.
POINTS_PER_TURN = 16

# Number of polygon sides approximating the circular wire.
# Higher = rounder but heavier mesh.
WIRE_SECTIONS = 6

# Mesh size for the single continuous swept conductor.  Keeping one sweep is
# important here: separate touching sweeps trigger Gmsh PLC errors.
ANTENNA_MESH_SIZE = 3 * WIRE_RADIUS


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

F0 = 868 * MHz
FSPAN = 300 * MHz
FREQ_START = F0 - FSPAN / 2
FREQ_STOP = F0 + FSPAN / 2     
FREQ_POINTS = 11

PORT_HEIGHT = 2 * mm
PORT_IMPEDANCE = 50.0

# Distance from antenna to absorbing boundary.
C0 = 299_792_458.0
WAVELENGTH = C0 / F0
AIR_MARGIN = 0.25 * WAVELENGTH

SHOW_GEOMETRY = True
SHOW_COIL_PREVIEW = True
SHOW_MESH = True
RUN_SOLVER = True


# ============================================================================
# SMOOTH TRANSITION FUNCTIONS
# ============================================================================

def smoothstep5(u):
    """
    Quintic smoothstep:

        S(0) = 0
        S(1) = 1

    and both first and second derivatives behave smoothly at the ends.

    Used here as the normalized ANGULAR VELOCITY of the wire.
    """
    return 10*u**3 - 15*u**4 + 6*u**5


def smoothstep5_integral(u):
    """
    Integral of smoothstep5 from 0 -> u.

    F(u) = integral(S(u) du)

    F(0) = 0
    F(1) = 0.5
    """
    return 2.5*u**4 - 3*u**5 + u**6


# ============================================================================
# CENTERLINE BUILDER
# ============================================================================

class AntennaPath:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = []
        self.y = []
        self.z = []
        self.current_x = x
        self.current_y = y
        self.current_z = z

        self._append(
            np.array([x]),
            np.array([y]),
            np.array([z]),
        )

    def _append(self, x, y, z):
        """
        Append coordinates while avoiding duplicate points at section joins.
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        if len(self.x) > 0:
            # Remove first point if it is identical to previous endpoint.
            dx = x[0] - self.x[-1]
            dy = y[0] - self.y[-1]
            dz = z[0] - self.z[-1]

            if np.sqrt(dx*dx + dy*dy + dz*dz) < 1e-12:
                x = x[1:]
                y = y[1:]
                z = z[1:]

        self.x.extend(x.tolist())
        self.y.extend(y.tolist())
        self.z.extend(z.tolist())

        if len(x):
            self.current_x = float(x[-1])
            self.current_y = float(y[-1])
            self.current_z = float(z[-1])

    def straight(self, length, points=8):
        """
        Add a vertical straight section.
        """

        if length <= 0:
            return

        z = np.linspace(
            self.current_z,
            self.current_z + length,
            points,
        )

        x = np.full_like(z, self.current_x)
        y = np.full_like(z, self.current_y)

        self._append(x, y, z)
    def coil(
        self,
        radius,
        turns,
        pitch,
        transition,
        handedness="RH",
        points_per_turn=32,
    ):
        """
        Add one smooth helical loading coil.

        The straight wire enters vertically.

        Angular velocity smoothly changes:

            0
            ↓
            helix angular velocity
            ↓
            0

        Therefore there is NO sharp straight-to-helix corner.

        The coil center is offset horizontally so that an integer number
        of turns returns exactly to the incoming straight-wire position.

        Parameters
        ----------
        radius:
            Helix centerline radius.

        turns:
            Number of full 360-degree turns. Must be a positive integer
            if the outgoing straight is to return to the same line.

        pitch:
            Axial rise per turn in the constant-pitch part.

        transition:
            Length of each smooth transition zone.

        handedness:
            "RH" or "LH".
        """

        if radius <= 0:
            raise ValueError("Coil radius must be > 0.")

        if pitch <= 0:
            raise ValueError("Coil pitch must be > 0.")

        if transition <= 0:
            raise ValueError(
                "Transition must be > 0. "
                "A zero transition would produce the sharp bend "
                "we are deliberately avoiding."
            )

        if int(turns) != turns or turns <= 0:
            raise ValueError(
                "turns must be a positive integer so the coil returns "
                "to the same straight-line location."
            )

        turns = int(turns)

        nominal_coil_length = turns * pitch

        if transition >= nominal_coil_length:
            raise ValueError(
                f"Transition ({transition/mm:.2f} mm) is too long for "
                f"{turns} turns × {pitch/mm:.2f} mm pitch."
            )

        if handedness.upper() == "RH":
            sign = 1.0
        elif handedness.upper() == "LH":
            sign = -1.0
        else:
            raise ValueError("handedness must be 'RH' or 'LH'.")

        omega = sign * 2*np.pi / pitch

        # ---------------------------------------------------------------
        # Geometry concept
        #
        # Incoming straight wire is at:
        #
        #       x = x0
        #       y = y0
        #
        # Put the coil's rotation axis one radius to the -X side:
        #
        #       center_x = x0 - radius
        #
        # Therefore theta = 0 corresponds exactly to x0,y0.
        # After an integer number of revolutions it returns to x0,y0.
        # ---------------------------------------------------------------

        x0 = self.current_x
        y0 = self.current_y
        z0 = self.current_z

        center_x = x0 - radius
        center_y = y0

        # ---------------------------------------------------------------
        # SECTION 1:
        # Smoothly accelerate from vertical line into full helix.
        # ---------------------------------------------------------------

        n_transition = max(
            16,
            int(
                np.ceil(
                    points_per_turn * transition / pitch
                )
            ) + 1,
        )

        u = np.linspace(0.0, 1.0, n_transition)

        z_in = z0 + transition*u

        theta_in = (
            omega
            * transition
            * smoothstep5_integral(u)
        )

        x_in = center_x + radius*np.cos(theta_in)
        y_in = center_y + radius*np.sin(theta_in)

        self._append(x_in, y_in, z_in)

        theta = theta_in[-1]
        zpos = z_in[-1]

        # ---------------------------------------------------------------
        # SECTION 2:
        # Constant-pitch helix.
        #
        # Each transition contributes half of its length in angular
        # rotation. Together they contribute one transition-length.
        #
        # So this middle section is:
        #
        #       turns*pitch - transition
        #
        # This ensures TOTAL rotation remains exactly N × 360 degrees.
        # ---------------------------------------------------------------

        middle_length = nominal_coil_length - transition

        n_middle = max(
            8,
            int(
                np.ceil(
                    points_per_turn * middle_length / pitch
                )
            ) + 1,
        )

        s = np.linspace(0.0, middle_length, n_middle)

        theta_mid = theta + omega*s
        z_mid = zpos + s

        x_mid = center_x + radius*np.cos(theta_mid)
        y_mid = center_y + radius*np.sin(theta_mid)

        self._append(x_mid, y_mid, z_mid)

        theta = theta_mid[-1]
        zpos = z_mid[-1]

        # ---------------------------------------------------------------
        # SECTION 3:
        # Smoothly decelerate helix rotation back to vertical.
        #
        # Angular velocity is:
        #
        #       omega * (1 - smoothstep5(u))
        #
        # ---------------------------------------------------------------

        u = np.linspace(0.0, 1.0, n_transition)

        z_out = zpos + transition*u

        theta_out = (
            theta
            + omega
            * transition
            * (
                u - smoothstep5_integral(u)
            )
        )

        x_out = center_x + radius*np.cos(theta_out)
        y_out = center_y + radius*np.sin(theta_out)

        self._append(x_out, y_out, z_out)
        # Because turns is integer, numerical noise aside, the final point
        # should equal x0,y0. Force the exact location to avoid tiny drift.
        self.x[-1] = x0
        self.y[-1] = y0

        self.current_x = x0
        self.current_y = y0
        self.current_z = float(z_out[-1])

    def arrays(self):
        return (
            np.asarray(self.x),
            np.asarray(self.y),
            np.asarray(self.z),
        )


# ============================================================================
# BUILD THE ANTENNA CENTERLINE
# ============================================================================

path_builder = AntennaPath(
    x=0.0,
    y=0.0,
    z=PORT_HEIGHT,
)

path_builder.straight(
    BOTTOM_LENGTH
)

path_builder.coil(
    radius=COIL1_RADIUS,
    turns=COIL1_TURNS,
    pitch=COIL1_PITCH,
    transition=COIL1_TRANSITION,
    handedness="RH",
    points_per_turn=POINTS_PER_TURN,
)

path_builder.straight(
    MIDDLE_LENGTH
)

path_builder.coil(
    radius=COIL2_RADIUS,
    turns=COIL2_TURNS,
    pitch=COIL2_PITCH,
    transition=COIL2_TRANSITION,
    handedness="RH",
    points_per_turn=POINTS_PER_TURN,
)

path_builder.straight(
    TOP_LENGTH
)

xpts, ypts, zpts = path_builder.arrays()


# ============================================================================
# PRINT USEFUL DIMENSIONS
# ============================================================================

antenna_height = zpts[-1] - PORT_HEIGHT

print()
print("----------------------------------------------------")
print("ANTENNA")
print("----------------------------------------------------")
print(f"Centerline samples : {len(xpts)}")
print(f"Wire diameter      : {2*WIRE_RADIUS/mm:.2f} mm")
print(f"Antenna height     : {antenna_height/mm:.2f} mm")
print(f"Overall height     : {zpts[-1]/mm:.2f} mm")
print(f"Radials            : {RADIAL_COUNT} x {RADIAL_LENGTH/mm:.1f} mm")
print(f"Radial angle       : {RADIAL_ANGLE:.1f} deg below horizontal")
print()
print(
    f"Coil 1 OD approx   : "
    f"{2*(COIL1_RADIUS + WIRE_RADIUS)/mm:.2f} mm"
)
print(
    f"Coil 2 OD approx   : "
    f"{2*(COIL2_RADIUS + WIRE_RADIUS)/mm:.2f} mm"
)
print("----------------------------------------------------")
print()


# ============================================================================
# CREATE EMERGE MODEL
# ============================================================================

model = em.Simulation(
    "SmoothLoadedAntenna"
)

model.check_version(
    "2.8.4"
)


# ============================================================================
# CREATE ONE CONTINUOUS XYZ SPLINE AND SWEEP THE WIRE
#
# Keeping this as one swept volume avoids coincident end faces at the
# coil/straight joins, which can trigger Gmsh PLC errors for thin wires.
# ============================================================================

antenna_curve = em.geo.Curve(
    xpts,
    ypts,
    zpts,
    ctype="BSpline",
    name="AntennaCenterline",
)

wire_section = em.geo.XYPolygon.circle(
    WIRE_RADIUS,
    Nsections=WIRE_SECTIONS,
)

antenna = (
    antenna_curve
    .pipe(
        wire_section,
        max_mesh_size=ANTENNA_MESH_SIZE,
        name="Antenna",
    )
    .set_material(em.lib.MET_COPPER)
    .foreground()
)

# EMerge 2.8.x requires the volume property to be set explicitly.
antenna.max_meshsize = ANTENNA_MESH_SIZE

print(
    f"Conductor mesh size: {ANTENNA_MESH_SIZE/mm:.1f} mm "
    "(single continuous sweep)"
)


# ============================================================================
# OPTIONAL COIL-ONLY PREVIEW
# ============================================================================

# This is deliberately before the feed and airbox are created.  Using the
# Gmsh viewer here previews the CAD geometry without triggering a mesh.
if SHOW_COIL_PREVIEW:
    print("Opening coil preview...")
    print("Close the Gmsh window to continue.")
    model.view(use_gmsh=True)


# ============================================================================
# FEED
#
# This follows the feed approach from EMerge's helix antenna example.
# The antenna spline starts at z = PORT_HEIGHT, so the feed extrusion
# reaches directly from z=0 to the antenna.
# ============================================================================

feed_poly = em.geo.XYPolygon.circle(
    WIRE_RADIUS,
    Nsections=WIRE_SECTIONS,
)

feed = (
    feed_poly
    .extrude(
        PORT_HEIGHT,
        em.GCS.displace(
            xpts[0],
            ypts[0],
            0,
        ),
        name="Feed",
    )
    .set_material(
        em.lib.MET_COPPER
    )
    .foreground()
)

feed.max_meshsize = 3 * WIRE_RADIUS


# Annular copper hub for the radials.  The center hole keeps the signal feed
# electrically separate from the radial ground assembly.
GROUND_HUB_OUTER_RADIUS = 4 * WIRE_RADIUS
GROUND_HUB_INNER_RADIUS = 1.5 * WIRE_RADIUS
GROUND_HUB_HEIGHT = 3 * WIRE_RADIUS
GROUND_HUB_Z = -GROUND_HUB_HEIGHT
GROUND_HUB_SECTIONS = max(24, WIRE_SECTIONS)

ground_hub_outer = em.geo.Cylinder(
    GROUND_HUB_OUTER_RADIUS,
    GROUND_HUB_HEIGHT,
    cs=em.GCS.displace(0.0, 0.0, GROUND_HUB_Z),
    Nsections=GROUND_HUB_SECTIONS,
    name="GroundHubOuter",
)
ground_hub_inner = em.geo.Cylinder(
    GROUND_HUB_INNER_RADIUS,
    GROUND_HUB_HEIGHT,
    cs=em.GCS.displace(0.0, 0.0, GROUND_HUB_Z),
    Nsections=GROUND_HUB_SECTIONS,
    name="GroundHubHole",
)
ground_hub = em.geo.subtract(
    ground_hub_outer,
    ground_hub_inner,
).set_material(
    em.lib.MET_COPPER
).foreground()
ground_hub.max_meshsize = 3 * WIRE_RADIUS


# ============================================================================
# FOUR- RADIAL ANGLED GROUND PLANE
#
# Each radial starts at the annular hub and slopes downward.  The four
# cylinders are arranged at 90-degree azimuth spacing, making a symmetric
# ground plane around the vertical radiator while leaving the feed hole clear.
# ============================================================================

radials = []
radial_tilt = 90.0 + RADIAL_ANGLE
radial_start = 3 * WIRE_RADIUS

for index in range(RADIAL_COUNT):
    radial = em.geo.Cylinder(
        WIRE_RADIUS,
        RADIAL_LENGTH - radial_start,
        cs=em.GCS.displace(0.0, 0.0, radial_start),
        Nsections=WIRE_SECTIONS,
        name=f"Radial{index + 1}",
    )
    radial = em.geo.rotate(
        radial,
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        radial_tilt,
    )
    radial = em.geo.rotate(
        radial,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        index * 360.0 / RADIAL_COUNT,
    )
    radial = (
        radial
        .set_material(em.lib.MET_COPPER)
        .foreground()
    )
    radial.max_meshsize = 10 * WIRE_RADIUS
    radials.append(radial)

# The radial cylinders overlap one another and the hub at the origin. Fuse
# them into one copper body so the preview and mesh contain a single ground
# assembly instead of several intersecting solids.
ground_system = em.geo.unite(
    ground_hub,
    *radials,
).set_material(
    em.lib.MET_COPPER
).foreground()
ground_system.max_meshsize = 10 * WIRE_RADIUS


# ============================================================================
# AIR REGION
# ============================================================================

radial_horizontal_span = RADIAL_LENGTH * np.cos(np.deg2rad(RADIAL_ANGLE))

xmin = min(np.min(xpts), -radial_horizontal_span) - AIR_MARGIN
xmax = max(np.max(xpts), radial_horizontal_span) + AIR_MARGIN

ymin = min(np.min(ypts), -radial_horizontal_span) - AIR_MARGIN
ymax = max(np.max(ypts), radial_horizontal_span) + AIR_MARGIN

# Leave clearance below the downward-sloping radial tips.
radial_lowest_z = -RADIAL_LENGTH * np.sin(np.deg2rad(RADIAL_ANGLE))
zmin = radial_lowest_z - AIR_MARGIN
zmax = np.max(zpts) + AIR_MARGIN

airbox = em.geo.Box(
    width=xmax - xmin,
    depth=ymax - ymin,
    height=zmax - zmin,
    position=(
        xmin,
        ymin,
        zmin,
    ),
    name="Airbox",
).background()


# ============================================================================
# OPTIONAL GEOMETRY PREVIEW
# ============================================================================

if SHOW_GEOMETRY:
    print("Opening geometry preview...")
    print("Close the EMerge window to continue.")
    model.view(use_gmsh=True)


# ============================================================================
# FREQUENCY SWEEP
# ============================================================================

model.mw.set_frequency_range(
    FREQ_START,
    FREQ_STOP,
    FREQ_POINTS,
)

# Global wavelength-based mesh resolution.
model.set_resolution(
    0.5
)

# The antenna is a thin, tightly curved swept volume.  Give Gmsh extra
# resolution on curved boundary edges to avoid PLC self-intersections at the
# coil/straight transitions.
model.mesher.set_curved_boundary_meshing(20)


# ============================================================================
# GENERATE MESH
# ============================================================================

print("Generating mesh...")

model.generate_mesh()

if SHOW_MESH:
    print("Opening mesh preview...")
    print("Close the EMerge window to continue.")
    # Mesh mode uses a wireframe/edge display instead of the metallic
    # material rendering, making the element layout visible on the coils and
    # the coarser straight sections.
    model.view(
        plot_mesh=True,
        volume_mesh=True,
    )


# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================

# Absorbing boundary around the airbox.
#
# Bottom is excluded, matching EMerge's official helix-antenna example.

abc_sel = airbox.boundary(
    exclude=("bottom",)
)

abc = model.mw.bc.AbsorbingBoundary(
    abc_sel
)


# ============================================================================
# LUMPED PORT
# ============================================================================

port_sel = feed.boundary(
    exclude=(
        "front",
        "back",
    )
)

port = model.mw.bc.LumpedPort(
    port_sel,
    1,
    feed_poly.length,
    PORT_HEIGHT,
    em.ZAX,
    Z0=PORT_IMPEDANCE,
)


# ============================================================================
# RUN SOLVER
# ============================================================================

if RUN_SOLVER:

    print()
    print(
        f"Running {FREQ_START/MHz:.1f}–"
        f"{FREQ_STOP/MHz:.1f} MHz sweep..."
    )

    data = model.mw.run_sweep()

    print("Sweep complete.")


    # ========================================================================
    # S11
    # ========================================================================

    glob = data.scalar.grid
    s11 = np.asarray(glob.S(1, 1))
    s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-12))
    finite_s11_db = s11_db[np.isfinite(s11_db)]

    print()
    print("S11 RESULTS")
    print("----------------------------------------------------")
    print("Frequency (MHz)   S11 (dB)")
    print("----------------------------------------------------")
    for frequency, value in zip(glob.freq, s11_db):
        print(f"{frequency/MHz:10.3f}       {value:8.3f}")

    target_index = int(np.argmin(np.abs(glob.freq - F0)))
    minimum_index = int(np.nanargmin(s11_db))
    print("----------------------------------------------------")
    print(
        f"Nearest to 868   : {glob.freq[target_index]/MHz:.3f} MHz, "
        f"{s11_db[target_index]:.3f} dB "
        f"(|S11|={abs(s11[target_index]):.5f})"
    )
    print(
        f"Best in sweep    : {glob.freq[minimum_index]/MHz:.3f} MHz, "
        f"{s11_db[minimum_index]:.3f} dB"
    )
    print("----------------------------------------------------")

    if finite_s11_db.size:
        s11_dblim = [
            float(np.floor(np.min(finite_s11_db) - 3)),
            float(np.ceil(np.max(finite_s11_db) + 1)),
        ]
    else:
        s11_dblim = [-40, 5]

    plot_sp(
        glob.freq,
        s11,
        dblim=s11_dblim,
    )

    smith(
        s11,
        f=glob.freq,
    )


    # ========================================================================
    # FAR FIELD @ F0
    # ========================================================================

    field = data.field.find(
        freq=F0
    )

    # Y-Z cut
    ff_yz = field.farfield_2d(
        (0, 0, 1),
        (0, 1, 0),
        abc_sel,
        (-90, 90),
    )

    # X-Z cut
    ff_xz = field.farfield_2d(
        (0, 0, 1),
        (1, 0, 0),
        abc_sel,
        (-90, 90),
    )

    EISO = em.lib.EISO

    plot_ff(
        ff_yz.ang * 180 / np.pi,
        [
            ff_yz.Elhcp / EISO,
            ff_yz.Erhcp / EISO,
            ff_xz.Elhcp / EISO,
            ff_xz.Erhcp / EISO,
        ],
        dB=True,
        labels=[
            "LHCP YZ",
            "RHCP YZ",
            "LHCP XZ",
            "RHCP XZ",
        ],
    )


print("Done.")
