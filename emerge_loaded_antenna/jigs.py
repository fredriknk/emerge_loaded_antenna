"""Printable winding-jig models derived from antenna coil geometry."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from .config import AntennaDesign, CoilDesign

Vector3 = tuple[float, float, float]
Triangle = tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True)
class JigModelSettings:
    """Meshing and fabrication allowances for coil winding mandrels."""

    radial_clearance: float = 0.15e-3
    end_margin: float = 3.0e-3
    groove_depth_wire_diameters: float = 0.30
    groove_width_wire_diameters: float = 1.25
    circumferential_segments: int = 128
    axial_segments_per_turn: int = 64
    end_margin_segments: int = 16

    def validate(self) -> None:
        positive_values = (
            self.radial_clearance,
            self.end_margin,
            self.groove_depth_wire_diameters,
            self.groove_width_wire_diameters,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError("jig dimensions and ratios must be finite and positive")
        if self.circumferential_segments < 24:
            raise ValueError("circumferential_segments must be at least 24")
        if self.axial_segments_per_turn < 12:
            raise ValueError("axial_segments_per_turn must be at least 12")
        if self.end_margin_segments < 2:
            raise ValueError("end_margin_segments must be at least two")


def _normal(triangle: Triangle) -> Vector3:
    a, b, c = triangle
    ab = tuple(b[index] - a[index] for index in range(3))
    ac = tuple(c[index] - a[index] for index in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    magnitude = math.sqrt(sum(value * value for value in cross))
    if magnitude == 0:
        return (0.0, 0.0, 0.0)
    return (
        cross[0] / magnitude,
        cross[1] / magnitude,
        cross[2] / magnitude,
    )


def _write_binary_stl(path: Path, triangles: Iterator[Triangle], count: int) -> None:
    header = b"EMerge loaded antenna coil winding jig".ljust(80, b" ")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", count))
        written = 0
        for triangle in triangles:
            normal = _normal(triangle)
            values = (*normal, *(value for point in triangle for value in point))
            stream.write(struct.pack("<12fH", *values, 0))
            written += 1
    if written != count:
        raise RuntimeError(f"expected {count} STL facets, wrote {written}")


def _jig_geometry(
    design: AntennaDesign,
    coil: CoilDesign,
    settings: JigModelSettings,
) -> tuple[list[list[Vector3]], dict[str, float | int | str]]:
    wire_diameter = 2.0 * design.wire_radius
    groove_root_radius = (
        coil.radius - design.wire_radius - settings.radial_clearance
    )
    if groove_root_radius <= 0:
        raise ValueError(
            "coil radius is too small for wire radius and jig radial clearance"
        )

    groove_depth = min(
        settings.groove_depth_wire_diameters * wire_diameter,
        0.25 * groove_root_radius,
    )
    mandrel_radius = groove_root_radius + groove_depth
    groove_width = settings.groove_width_wire_diameters * wire_diameter
    groove_half_angle = min(math.pi / 3.0, groove_width / (2.0 * mandrel_radius))
    length = coil.turns * coil.pitch + 2.0 * settings.end_margin
    axial_segments = (
        int(coil.turns) * settings.axial_segments_per_turn
        + 2 * settings.end_margin_segments
    )
    theta_segments = settings.circumferential_segments
    handedness = coil.handedness.upper()
    direction = 1.0 if handedness == "RH" else -1.0

    rings: list[list[Vector3]] = []
    for axial_index in range(axial_segments + 1):
        z = length * axial_index / axial_segments
        winding_z = z - settings.end_margin
        ring: list[Vector3] = []
        for theta_index in range(theta_segments):
            theta = 2.0 * math.pi * theta_index / theta_segments
            phase = theta - direction * 2.0 * math.pi * winding_z / coil.pitch
            delta = math.atan2(math.sin(phase), math.cos(phase))
            if abs(delta) < groove_half_angle:
                groove_profile = math.cos(
                    math.pi * delta / (2.0 * groove_half_angle)
                ) ** 2
            else:
                groove_profile = 0.0
            radius = mandrel_radius - groove_depth * groove_profile
            ring.append(
                (
                    radius * math.cos(theta) * 1e3,
                    radius * math.sin(theta) * 1e3,
                    z * 1e3,
                )
            )
        rings.append(ring)

    metadata: dict[str, float | int | str] = {
        "handedness": handedness,
        "turns": int(coil.turns),
        "coil_centerline_radius_mm": coil.radius * 1e3,
        "coil_pitch_mm": coil.pitch * 1e3,
        "wire_diameter_mm": wire_diameter * 1e3,
        "radial_clearance_mm": settings.radial_clearance * 1e3,
        "mandrel_outer_radius_mm": mandrel_radius * 1e3,
        "mandrel_outer_diameter_mm": 2.0 * mandrel_radius * 1e3,
        "groove_root_radius_mm": groove_root_radius * 1e3,
        "formed_wire_centerline_radius_mm": (
            groove_root_radius + design.wire_radius
        ) * 1e3,
        "model_length_mm": length * 1e3,
        "groove_depth_mm": groove_depth * 1e3,
        "groove_width_mm": groove_width * 1e3,
        "circumferential_segments": theta_segments,
        "axial_segments": axial_segments,
    }
    return rings, metadata


def _triangles(rings: list[list[Vector3]]) -> Iterator[Triangle]:
    theta_segments = len(rings[0])
    for lower, upper in pairwise(rings):
        for index in range(theta_segments):
            following = (index + 1) % theta_segments
            yield (lower[index], lower[following], upper[following])
            yield (lower[index], upper[following], upper[index])

    bottom_center = (0.0, 0.0, rings[0][0][2])
    top_center = (0.0, 0.0, rings[-1][0][2])
    for index in range(theta_segments):
        following = (index + 1) % theta_segments
        yield (bottom_center, rings[0][following], rings[0][index])
        yield (top_center, rings[-1][index], rings[-1][following])


def export_jig_models(
    design: AntennaDesign,
    output: str | Path,
    *,
    settings: JigModelSettings | None = None,
) -> tuple[Path, ...]:
    """Export one grooved binary-STL winding mandrel per loading coil.

    The groove-root radius is the coil centerline radius minus wire radius and
    radial print clearance. A full-length helical groove follows the coil's
    pitch and handedness. ``jig_models.json`` records all derived dimensions.
    """
    design.validate()
    settings = settings or JigModelSettings()
    settings.validate()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)

    models: list[Path] = []
    manifest_models: list[dict[str, float | int | str]] = []
    for index, coil in enumerate(design.coils, 1):
        rings, metadata = _jig_geometry(design, coil, settings)
        theta_segments = settings.circumferential_segments
        count = 2 * theta_segments * (len(rings) - 1) + 2 * theta_segments
        path = destination / f"coil_{index:02d}_winding_jig.stl"
        _write_binary_stl(path, _triangles(rings), count)
        models.append(path)
        manifest_models.append({"coil_index": index, "file": path.name, **metadata})

    manifest = {
        "format": "binary STL",
        "units": "millimetres",
        "purpose": "coil winding mandrels with helical guide grooves",
        "settings": {
            "radial_clearance_mm": settings.radial_clearance * 1e3,
            "end_margin_mm": settings.end_margin * 1e3,
            "groove_depth_wire_diameters": (
                settings.groove_depth_wire_diameters
            ),
            "groove_width_wire_diameters": (
                settings.groove_width_wire_diameters
            ),
            "circumferential_segments": settings.circumferential_segments,
            "axial_segments_per_turn": settings.axial_segments_per_turn,
            "end_margin_segments": settings.end_margin_segments,
        },
        "models": manifest_models,
    }
    (destination / "jig_models.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return tuple(models)
