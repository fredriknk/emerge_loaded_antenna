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
composite wire and swept once with a circular wire cross-section. Exact lines
and local Bezier pieces replace the former artifact-prone global BSpline.

The straight -> helix and helix -> straight transitions use compact cubic
Hermite connectors with a small local sideways offset. The offset is
independent of helix pitch, avoiding a broad entry sweep.

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
import gmsh

from emerge._emerge.geometry import GeoEdge
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

WIRE_RADIUS = 1.0 * mm          # 2.0 mm diameter wire

RADIAL_LENGTH = 72 * mm
RADIAL_ANGLE = 45.0              # degrees below the horizontal
RADIAL_COUNT = 4

BOTTOM_LENGTH = 140 * mm

COIL1_RADIUS = 10 * mm          # centerline radius
COIL1_TURNS = 1                 # integer turns
COIL1_PITCH = 7.0 * mm          # axial rise per full turn
COIL1_TRANSITION = 6 * mm       # smooth entrance/exit distance
COIL1_TRANSITION_OFFSET = 4.75 * mm  # compact, two-turn sweep-safe lead-in

MIDDLE_LENGTH = 221 * mm    

COIL2_RADIUS = COIL1_RADIUS
COIL2_TURNS = COIL1_TURNS
COIL2_PITCH = COIL1_PITCH
COIL2_TRANSITION = COIL1_TRANSITION
COIL2_TRANSITION_OFFSET = COIL1_TRANSITION_OFFSET

TOP_LENGTH = BOTTOM_LENGTH


# ============================================================================
# GEOMETRY RESOLUTION
# ============================================================================

# Preview samples per turn. The actual CAD helix uses three independent cubic
# Bezier arcs per turn, so this no longer increases CAD or mesh complexity.
POINTS_PER_TURN = 20

# Number of polygon sides approximating the circular wire.
# Higher = rounder but heavier mesh.
WIRE_SECTIONS = 6

# Longitudinal mesh target for the single continuous swept conductor. The
# composite path remains robust at 3 radii for both one- and two-turn coils.
ANTENNA_MESH_SIZE = 3.0 * WIRE_RADIUS


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

F0 = 868 * MHz
FSPAN = 100 * MHz
FREQ_START = F0 - FSPAN / 2
FREQ_STOP = F0 + FSPAN / 2     
FREQ_POINTS = 5

PORT_HEIGHT = 2 * mm
PORT_IMPEDANCE = 50.0

# Distance from antenna to absorbing boundary.
C0 = 299_792_458.0
WAVELENGTH = C0 / F0
AIR_MARGIN = 0.25 * WAVELENGTH

SHOW_GEOMETRY = False
SHOW_COIL_PREVIEW = False
SHOW_MESH = True
RUN_SOLVER = True
SHOW_3D_FARFIELD = True


FARFIELD_DB_FLOOR = -30


# ============================================================================
# LOCAL CURVE HELPERS
# ============================================================================


def cubic_hermite(p0, p1, v0, v1, u):
    """Tangent-continuous vector blend with independently sized handles."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    u = np.asarray(u, dtype=float)[:, None]

    return (
        (2*u**3 - 3*u**2 + 1)*p0
        + (-2*u**3 + 3*u**2)*p1
        + (u**3 - 2*u**2 + u)*v0
        + (u**3 - u**2)*v1
    )


def cubic_bezier_controls(p0, p1, v0, v1):
    """Convert cubic Hermite endpoints/derivatives to four Bezier controls."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    return np.vstack((p0, p0 + v0/3, p1 - v1/3, p1))


class CompositeCurve(em.geo.Curve):
    """One OpenCASCADE wire assembled from independent local curve pieces."""

    def __init__(self, segments, name="CompositeCurve"):
        if not segments:
            raise ValueError("A composite curve needs at least one segment.")

        edge_tags = []
        point_tags = []
        last_point_tag = None
        last_point = None

        for kind, coordinates in segments:
            coordinates = np.asarray(coordinates, dtype=float)
            if last_point is not None:
                if np.linalg.norm(coordinates[0] - last_point) > 1e-9:
                    raise ValueError("Composite curve segments are not connected.")
                coordinates = coordinates.copy()
                coordinates[0] = last_point

            if last_point_tag is None:
                last_point_tag = gmsh.model.occ.addPoint(*coordinates[0])
                point_tags.append(last_point_tag)

            local_tags = [last_point_tag]
            for point in coordinates[1:]:
                tag = gmsh.model.occ.addPoint(*point)
                point_tags.append(tag)
                local_tags.append(tag)

            if kind == "line":
                if len(local_tags) != 2:
                    raise ValueError("A line segment needs exactly two points.")
                edge_tag = gmsh.model.occ.addLine(*local_tags)
            elif kind == "bezier":
                if len(local_tags) != 4:
                    raise ValueError("A cubic Bezier needs exactly four controls.")
                edge_tag = gmsh.model.occ.addBezier(local_tags)
            else:
                raise ValueError(f"Unknown composite segment type: {kind}")

            edge_tags.append(edge_tag)
            last_point_tag = local_tags[-1]
            last_point = coordinates[-1]

        wire_tag = gmsh.model.occ.addWire(edge_tags, checkClosed=False)
        gmsh.model.occ.remove([(0, tag) for tag in point_tags])

        first = np.asarray(segments[0][1][0], dtype=float)
        last = np.asarray(segments[-1][1][-1], dtype=float)
        self.xpts = np.array((first[0], last[0]))
        self.ypts = np.array((first[1], last[1]))
        self.zpts = np.array((first[2], last[2]))
        self.dstart = (0.0, 0.0, 1.0)
        GeoEdge.__init__(self, wire_tag, name=name)
        gmsh.model.occ.synchronize()


# ============================================================================
# CENTERLINE BUILDER
# ============================================================================

class AntennaPath:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = []
        self.y = []
        self.z = []
        self.segments = []
        self.current_x = x
        self.current_y = y
        self.current_z = z

        self._append(
            np.array([x]),
            np.array([y]),
            np.array([z]),
        )

    def _add_segment(self, kind, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Segment coordinates must have shape (N, 3).")
        if self.segments:
            previous_end = self.segments[-1][1][-1]
            if np.linalg.norm(coordinates[0] - previous_end) > 1e-9:
                raise ValueError("Centerline segments must meet exactly.")
            coordinates = coordinates.copy()
            coordinates[0] = previous_end
        self.segments.append((kind, coordinates))

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

    def straight(self, length):
        """Add one exact line; its length does not increase CAD complexity."""

        if length <= 0:
            return

        coordinates = np.array(
            (
                (self.current_x, self.current_y, self.current_z),
                (self.current_x, self.current_y, self.current_z + length),
            )
        )
        self._add_segment("line", coordinates)
        self._append(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
        )
    def coil(
        self,
        radius,
        turns,
        pitch,
        transition,
        transition_offset=None,
        handedness="RH",
        points_per_turn=32,
    ):
        """Add a helix with compact tangent-continuous connectors.

        ``transition`` controls each connector's axial length, while
        ``transition_offset`` limits how far around the coil circumference a
        connector travels. The complete path still makes the requested
        integer number of turns, independent of helix pitch.
        """
        if radius <= 0 or pitch <= 0 or transition <= 0:
            raise ValueError("radius, pitch and transition must all be > 0.")
        if int(turns) != turns or turns <= 0:
            raise ValueError("turns must be a positive integer.")
        if transition_offset is None:
            transition_offset = min(0.5*radius, 0.8*transition)
        if not 0 < transition_offset < 2*radius:
            raise ValueError(
                "transition_offset must be > 0 and smaller than diameter."
            )

        turns = int(turns)
        if handedness.upper() == "RH":
            sign = 1.0
        elif handedness.upper() == "LH":
            sign = -1.0
        else:
            raise ValueError("handedness must be 'RH' or 'LH'.")

        omega = sign * 2*np.pi / pitch
        # Convert the requested local chord displacement to a join angle.
        # This keeps the visible elbow compact and, unlike the old transition,
        # does not make its angular span depend on pitch.
        alpha = sign * 2*np.arcsin(transition_offset / (2*radius))
        middle_rotation = sign * 2*np.pi*turns - 2*alpha
        if sign * middle_rotation <= 0:
            raise ValueError("transition_offset is too large for the turn count.")
        middle_length = abs(middle_rotation / omega)

        x0 = self.current_x
        y0 = self.current_y
        z0 = self.current_z
        center_x = x0 - radius
        center_y = y0

        n_transition = max(12, int(np.ceil(points_per_turn / 4)) + 1)
        u = np.linspace(0.0, 1.0, n_transition)

        # Entrance connector: vertical line -> constant-pitch helix. Only the
        # tangent direction must match; matching the helix's z-parameter speed
        # would create huge control handles for tightly pitched coils.
        p_in_0 = np.array((x0, y0, z0))
        p_in_1 = np.array(
            (
                center_x + radius*np.cos(alpha),
                center_y + radius*np.sin(alpha),
                z0 + transition,
            )
        )
        d_in_0 = np.array((0.0, 0.0, 1.0))
        d_in_1 = np.array(
            (
                -radius*omega*np.sin(alpha),
                radius*omega*np.cos(alpha),
                1.0,
            )
        )
        d_in_1 /= np.linalg.norm(d_in_1)
        # These handle lengths distribute curvature across the connector
        # instead of concentrating a hook near either endpoint. For the
        # default geometry the minimum centerline bend radius is ~4.6 mm.
        vertical_handle = 1.60*transition
        helix_handle = 1.45*transition
        controls_in = cubic_bezier_controls(
            p_in_0,
            p_in_1,
            vertical_handle*d_in_0,
            helix_handle*d_in_1,
        )
        self._add_segment("bezier", controls_in)
        path_in = cubic_hermite(
            p_in_0,
            p_in_1,
            vertical_handle*d_in_0,
            helix_handle*d_in_1,
            u,
        )
        self._append(path_in[:, 0], path_in[:, 1], path_in[:, 2])

        # Constant-pitch middle section. The two compact connectors consume
        # the small remaining angular allowances at the ends.
        n_middle = max(
            8,
            int(np.ceil(points_per_turn * middle_length / pitch)) + 1,
        )
        s = np.linspace(0.0, middle_length, n_middle)
        theta_mid = alpha + omega*s
        z_mid = p_in_1[2] + s
        x_mid = center_x + radius*np.cos(theta_mid)
        y_mid = center_y + radius*np.sin(theta_mid)
        self._append(x_mid, y_mid, z_mid)

        # Approximate the constant-pitch helix with independent Bezier arcs no
        # longer than 120 degrees. Each arc has exact endpoints and helix
        # tangents; unlike a global BSpline, it cannot deform a straight or a
        # neighbouring transition.
        arc_count = max(
            1,
            int(np.ceil(abs(middle_rotation) / (2*np.pi/3))),
        )
        theta_edges = np.linspace(
            alpha,
            alpha + middle_rotation,
            arc_count + 1,
        )
        for theta_a, theta_b in zip(theta_edges[:-1], theta_edges[1:]):
            z_a = p_in_1[2] + (theta_a - alpha)/omega
            z_b = p_in_1[2] + (theta_b - alpha)/omega
            point_a = np.array(
                (
                    center_x + radius*np.cos(theta_a),
                    center_y + radius*np.sin(theta_a),
                    z_a,
                )
            )
            point_b = np.array(
                (
                    center_x + radius*np.cos(theta_b),
                    center_y + radius*np.sin(theta_b),
                    z_b,
                )
            )
            bezier_factor = 4/3*np.tan((theta_b - theta_a)/4)
            tangent_a = np.array(
                (
                    -radius*np.sin(theta_a),
                    radius*np.cos(theta_a),
                    1/omega,
                )
            )
            tangent_b = np.array(
                (
                    -radius*np.sin(theta_b),
                    radius*np.cos(theta_b),
                    1/omega,
                )
            )
            helix_controls = np.vstack(
                (
                    point_a,
                    point_a + bezier_factor*tangent_a,
                    point_b - bezier_factor*tangent_b,
                    point_b,
                )
            )
            self._add_segment("bezier", helix_controls)

        # Exit connector: helix -> outgoing vertical line.
        beta = alpha + middle_rotation
        z_out_start = float(z_mid[-1])
        p_out_0 = np.array(
            (
                center_x + radius*np.cos(beta),
                center_y + radius*np.sin(beta),
                z_out_start,
            )
        )
        p_out_1 = np.array((x0, y0, z_out_start + transition))
        d_out_0 = np.array(
            (
                -radius*omega*np.sin(beta),
                radius*omega*np.cos(beta),
                1.0,
            )
        )
        d_out_0 /= np.linalg.norm(d_out_0)
        d_out_1 = np.array((0.0, 0.0, 1.0))
        controls_out = cubic_bezier_controls(
            p_out_0,
            p_out_1,
            helix_handle*d_out_0,
            vertical_handle*d_out_1,
        )
        self._add_segment("bezier", controls_out)
        path_out = cubic_hermite(
            p_out_0,
            p_out_1,
            helix_handle*d_out_0,
            vertical_handle*d_out_1,
            u,
        )
        self._append(path_out[:, 0], path_out[:, 1], path_out[:, 2])

        self.x[-1] = x0
        self.y[-1] = y0
        self.current_x = x0
        self.current_y = y0
        self.current_z = float(path_out[-1, 2])

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
    transition_offset=COIL1_TRANSITION_OFFSET,
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
    transition_offset=COIL2_TRANSITION_OFFSET,
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
print(f"CAD path segments  : {len(path_builder.segments)}")
print(f"Preview samples    : {len(xpts)}")
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
# CREATE ONE CONTINUOUS COMPOSITE WIRE AND SWEEP THE WIRE
#
# Exact line and local Bezier pieces prevent one global spline from deforming
# neighbouring sections. They are joined into one wire before the single pipe
# operation, avoiding coincident end faces and PLC errors.
# ============================================================================

antenna_curve = CompositeCurve(
    path_builder.segments,
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
# This is the non-metal lumped-port region, following EMerge's helix antenna
# example. It spans from the grounded hub at z=0 to the radiator at
# z=PORT_HEIGHT. Making this volume copper suppresses the imposed port field
# and produces |S11| ~= 1 at every frequency.
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
    .foreground()
)

feed.max_meshsize = 3 * WIRE_RADIUS


# Solid copper hub beneath the non-metal feed region. Its top face at z=0 is
# the lumped port's ground reference; the radiator starts at z=PORT_HEIGHT.
GROUND_HUB_OUTER_RADIUS = 4 * WIRE_RADIUS
GROUND_HUB_HEIGHT = 3 * WIRE_RADIUS
GROUND_HUB_Z = -GROUND_HUB_HEIGHT
GROUND_HUB_SECTIONS = max(24, WIRE_SECTIONS)

ground_hub = em.geo.Cylinder(
    GROUND_HUB_OUTER_RADIUS,
    GROUND_HUB_HEIGHT,
    cs=em.GCS.displace(0.0, 0.0, GROUND_HUB_Z),
    Nsections=GROUND_HUB_SECTIONS,
    name="GroundHub",
)
ground_hub.max_meshsize = 3 * WIRE_RADIUS


# ============================================================================
# FOUR- RADIAL ANGLED GROUND PLANE
#
# Each radial starts inside the solid hub and slopes downward.  The four
# cylinders are arranged at 90-degree azimuth spacing, making a symmetric
# ground plane around the vertical radiator.
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

# Twelve segments per full curved-boundary revolution preserves the coil shape
# without the heavy factor-20 workaround required by the former global spline.
model.mesher.set_curved_boundary_meshing(12)


# ============================================================================
# GENERATE MESH
# ============================================================================

print("Generating mesh...")

model.generate_mesh()

mesh_node_count = len(gmsh.model.mesh.getNodes()[0])
mesh_element_count = sum(
    len(block) for block in gmsh.model.mesh.getElements()[1]
)
mesh_volume_count = sum(
    len(block) for block in gmsh.model.mesh.getElements(3)[1]
)
print(f"Mesh nodes          : {mesh_node_count}")
print(f"Mesh elements       : {mesh_element_count}")
print(f"Volume elements     : {mesh_volume_count}")

if SHOW_MESH:
    print("Opening mesh preview...")
    print("Close the EMerge window to continue.")
    # Show boundary triangles only. Internal air-volume tetrahedra create a
    # dense web of crossing lines that looks like geometry artifacts.
    model.view(
        plot_mesh=True,
        volume_mesh=False,
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

    # Complete principal-plane cuts. In EMerge the first vector is the
    # zero-degree reference and the second vector is the plane normal.
    ff_xz = field.farfield_2d(
        (0, 0, 1),
        (0, 1, 0),
        abc_sel,
        (-180, 180),
    )
    ff_yz = field.farfield_2d(
        (0, 0, 1),
        (-1, 0, 0),
        abc_sel,
        (-180, 180),
    )
    ff_xy = field.farfield_2d(
        (1, 0, 0),
        (0, 0, 1),
        abc_sel,
        (-180, 180),
    )

    EISO = em.lib.EISO

    def gain_amplitude(ff):
        """EMerge isotropic-gain amplitude for a far-field result."""
        return np.abs(np.asarray(ff.normE) / EISO)

    def plane_metrics(name, ff, orientation):
        """Print peak gain, azimuthal average, HPBW and front/back ratio."""
        angles = np.asarray(ff.ang, dtype=float) * 180 / np.pi
        gain_amp = gain_amplitude(ff)
        gain_db = 20 * np.log10(np.maximum(gain_amp, 1e-12))

        order = np.argsort(angles)
        angles = angles[order]
        gain_db = gain_db[order]
        gain_amp = gain_amp[order]

        # -180 and +180 describe the same direction. Remove one duplicate so
        # circular beamwidth and average calculations do not double-count it.
        if len(angles) > 1 and np.isclose(angles[-1] - angles[0], 360.0):
            angles = angles[:-1]
            gain_db = gain_db[:-1]
            gain_amp = gain_amp[:-1]

        peak_index = int(np.nanargmax(gain_db))
        peak_angle = float(angles[peak_index])
        peak_gain_db = float(gain_db[peak_index])
        average_gain_db = float(
            10 * np.log10(np.maximum(np.mean(gain_amp**2), 1e-24))
        )

        opposite_angle = ((peak_angle + 180.0 + 180.0) % 360.0) - 180.0
        opposite_index = int(np.argmin(np.abs(angles - opposite_angle)))
        front_to_back_db = peak_gain_db - float(gain_db[opposite_index])

        threshold = peak_gain_db - 3.0
        above = gain_db >= threshold
        if np.all(above):
            beamwidth = 360.0
        else:
            left = peak_index
            right = peak_index
            while above[(left - 1) % len(above)] and (left - 1) % len(above) != right:
                left = (left - 1) % len(above)
            while above[(right + 1) % len(above)] and (right + 1) % len(above) != left:
                right = (right + 1) % len(above)
            angular_step = float(np.median(np.abs(np.diff(angles))))
            beamwidth = ((right - left) % len(above) + 1) * angular_step

        print(
            f"{name:4s}  peak {peak_gain_db:7.2f} dBi at {peak_angle:7.1f} deg  "
            f"avg {average_gain_db:7.2f} dBi  "
            f"HPBW {beamwidth:6.1f} deg  F/B {front_to_back_db:6.2f} dB"
        )
        print(f"      angle reference: {orientation}")

    print()
    print(f"FAR-FIELD GAIN @ {F0/MHz:.1f} MHz")
    print("--------------------------------------------------------------------------")
    plane_metrics("X-Z", ff_xz, "0 deg = +Z, +90 deg = +X")
    plane_metrics("Y-Z", ff_yz, "0 deg = +Z, +90 deg = +Y")
    plane_metrics("X-Y", ff_xy, "0 deg = +X, +90 deg = +Y")

    # Combined total-gain lobe plot for the three principal planes.
    plane_peak_db = max(
        float(np.nanmax(20 * np.log10(np.maximum(gain_amplitude(ff), 1e-12))))
        for ff in (ff_xz, ff_yz, ff_xy)
    )
    plot_ff(
        ff_xz.ang * 180 / np.pi,
        [
            ff_xz.normE / EISO,
            ff_yz.normE / EISO,
            ff_xy.normE / EISO,
        ],
        dB=True,
        labels=[
            "Total gain X-Z",
            "Total gain Y-Z",
            "Total gain X-Y",
        ],
        xlabel="Cut angle (degrees)",
        ylabel="Isotropic gain (dBi)",
        ylim=(FARFIELD_DB_FLOOR, max(5.0, np.ceil(plane_peak_db + 1))),
        title=f"Principal-plane gain at {F0/MHz:.1f} MHz",
    )

    # Interactive 3D total-gain lobe with the antenna shown for orientation.
    if SHOW_3D_FARFIELD:
        ff_3d = field.farfield_3d(abc_sel)
        gain_3d_db = 20 * np.log10(
            np.maximum(gain_amplitude(ff_3d), 1e-12)
        )
        peak_3d_index = np.unravel_index(
            int(np.nanargmax(gain_3d_db)),
            gain_3d_db.shape,
        )
        peak_3d_theta = float(ff_3d.theta[peak_3d_index] * 180 / np.pi)
        peak_3d_phi = float(ff_3d.phi[peak_3d_index] * 180 / np.pi)
        print(
            f"3D peak isotropic gain: {np.nanmax(gain_3d_db):.2f} dBi  "
            f"theta={peak_3d_theta:.1f} deg, phi={peak_3d_phi:.1f} deg "
            f"(elevation={90.0 - peak_3d_theta:.1f} deg)"
        )

        model.display.add_object(antenna, opacity=0.85)
        model.display.add_object(ground_system, opacity=0.85)
        model.display.add_farfield3d(
            ff_3d,
            component="normE",
            quantity="abs",
            dB=True,
            dBfloor=FARFIELD_DB_FLOOR,
            rmax=0.45 * antenna_height,
            offset=(0.0, 0.0, PORT_HEIGHT + antenna_height / 2),
            opacity=0.7,
        )
        model.display.show()


print("Done.")
