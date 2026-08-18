"""Low-complexity composite centerline geometry."""

from __future__ import annotations

from collections.abc import Sequence

import emerge as em
from emerge._emerge.geometry import GeoEdge
import gmsh
import numpy as np

from .config import AntennaDesign, CoilDesign, MeshSettings

Segment = tuple[str, np.ndarray]


def cubic_hermite(p0, p1, v0, v1, u):
    """Evaluate a cubic Hermite curve."""
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
    """Convert cubic Hermite endpoints and derivatives to Bezier controls."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    return np.vstack((p0, p0 + v0/3, p1 - v1/3, p1))


class CompositeCurve(em.geo.Curve):
    """One OpenCASCADE wire assembled from independent local pieces."""

    def __init__(self, segments: Sequence[Segment], name="CompositeCurve"):
        if not segments:
            raise ValueError("a composite curve needs at least one segment")

        edge_tags: list[int] = []
        point_tags: list[int] = []
        last_point_tag: int | None = None
        last_point: np.ndarray | None = None

        for kind, coordinates in segments:
            coordinates = np.asarray(coordinates, dtype=float)
            if last_point is not None:
                if np.linalg.norm(coordinates[0] - last_point) > 1e-9:
                    raise ValueError("composite curve segments are not connected")
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
                    raise ValueError("a line segment needs exactly two points")
                edge_tag = gmsh.model.occ.addLine(*local_tags)
            elif kind == "bezier":
                if len(local_tags) != 4:
                    raise ValueError("a cubic Bezier needs exactly four controls")
                edge_tag = gmsh.model.occ.addBezier(local_tags)
            else:
                raise ValueError(f"unknown composite segment type: {kind}")

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


class AntennaPath:
    """Build a preview polyline and an independent-segment CAD path."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = [float(x)]
        self.y = [float(y)]
        self.z = [float(z)]
        self.segments: list[Segment] = []
        self.current_x = float(x)
        self.current_y = float(y)
        self.current_z = float(z)

    def _add_segment(self, kind: str, coordinates) -> None:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("segment coordinates must have shape (N, 3)")
        if self.segments:
            previous_end = self.segments[-1][1][-1]
            if np.linalg.norm(coordinates[0] - previous_end) > 1e-9:
                raise ValueError("centerline segments must meet exactly")
            coordinates = coordinates.copy()
            coordinates[0] = previous_end
        self.segments.append((kind, coordinates))

    def _append_preview(self, coordinates) -> None:
        coordinates = np.asarray(coordinates, dtype=float)
        if np.linalg.norm(coordinates[0] - np.array(
            (self.x[-1], self.y[-1], self.z[-1])
        )) < 1e-12:
            coordinates = coordinates[1:]
        if len(coordinates):
            self.x.extend(coordinates[:, 0].tolist())
            self.y.extend(coordinates[:, 1].tolist())
            self.z.extend(coordinates[:, 2].tolist())
            self.current_x, self.current_y, self.current_z = map(
                float, coordinates[-1]
            )

    def straight(self, length: float) -> None:
        if length <= 0:
            raise ValueError("straight length must be positive")
        coordinates = np.array(
            (
                (self.current_x, self.current_y, self.current_z),
                (self.current_x, self.current_y, self.current_z + length),
            )
        )
        self._add_segment("line", coordinates)
        self._append_preview(coordinates)

    def coil(self, coil: CoilDesign, points_per_turn: int = 20) -> None:
        coil.validate()
        radius = coil.radius
        turns = int(coil.turns)
        pitch = coil.pitch
        transition = coil.transition
        transition_offset = coil.transition_offset
        sign = 1.0 if coil.handedness.upper() == "RH" else -1.0
        omega = sign*2*np.pi/pitch
        alpha = sign*2*np.arcsin(transition_offset/(2*radius))
        middle_rotation = sign*2*np.pi*turns - 2*alpha
        if sign*middle_rotation <= 0:
            raise ValueError("transition_offset is too large for the turn count")
        middle_length = abs(middle_rotation/omega)

        x0, y0, z0 = self.current_x, self.current_y, self.current_z
        center_x = x0 - radius
        center_y = y0
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

        vertical_handle = 1.60*transition
        helix_handle = 1.45*transition
        controls_in = cubic_bezier_controls(
            p_in_0,
            p_in_1,
            vertical_handle*d_in_0,
            helix_handle*d_in_1,
        )
        self._add_segment("bezier", controls_in)
        transition_samples = max(8, int(np.ceil(points_per_turn/4)) + 1)
        u = np.linspace(0.0, 1.0, transition_samples)
        self._append_preview(
            cubic_hermite(
                p_in_0,
                p_in_1,
                vertical_handle*d_in_0,
                helix_handle*d_in_1,
                u,
            )
        )

        middle_samples = max(
            8,
            int(np.ceil(points_per_turn*middle_length/pitch)) + 1,
        )
        s = np.linspace(0.0, middle_length, middle_samples)
        theta_mid = alpha + omega*s
        z_mid = p_in_1[2] + s
        middle_preview = np.column_stack(
            (
                center_x + radius*np.cos(theta_mid),
                center_y + radius*np.sin(theta_mid),
                z_mid,
            )
        )
        self._append_preview(middle_preview)

        arc_count = max(
            1,
            int(np.ceil(abs(middle_rotation)/(2*np.pi/3))),
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
            factor = 4/3*np.tan((theta_b - theta_a)/4)
            tangent_a = np.array(
                (-radius*np.sin(theta_a), radius*np.cos(theta_a), 1/omega)
            )
            tangent_b = np.array(
                (-radius*np.sin(theta_b), radius*np.cos(theta_b), 1/omega)
            )
            controls = np.vstack(
                (
                    point_a,
                    point_a + factor*tangent_a,
                    point_b - factor*tangent_b,
                    point_b,
                )
            )
            self._add_segment("bezier", controls)

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
        self._append_preview(
            cubic_hermite(
                p_out_0,
                p_out_1,
                helix_handle*d_out_0,
                vertical_handle*d_out_1,
                u,
            )
        )
        self.x[-1] = x0
        self.y[-1] = y0
        self.current_x = x0
        self.current_y = y0
        self.current_z = float(p_out_1[2])

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.x), np.asarray(self.y), np.asarray(self.z)


def build_centerline(
    design: AntennaDesign,
    mesh: MeshSettings | None = None,
) -> AntennaPath:
    """Create the reusable path representation for a design."""
    design.validate()
    mesh = mesh or MeshSettings()
    mesh.validate()
    path = AntennaPath(z=design.port_height)
    path.straight(design.bottom_length)
    path.coil(design.coil1, mesh.preview_points_per_turn)
    path.straight(design.middle_length)
    path.coil(design.coil2, mesh.preview_points_per_turn)
    path.straight(design.top_length)
    return path
