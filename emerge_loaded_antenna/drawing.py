"""Dimensioned fabrication drawings for optimized antenna designs.

This module intentionally reproduces the centerline equations used by
:mod:`emerge_loaded_antenna.geometry` without constructing EMerge/Gmsh CAD.
It can therefore turn an :class:`AntennaDesign` into a lightweight drawing
showing the X-Z, Y-Z, and X-Y orthographic views plus derived dimensions.

The ground-hub dimensions below match the current model in ``simulation.py``.
They are kept explicit here because they are part of the simulated physical
geometry even though they are not yet fields on ``AntennaDesign``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import AntennaDesign, CoilDesign

MM = 1e-3
MODEL_HUB_HEIGHT = 6.0 * MM
MODEL_HUB_CORE_RADIUS = 1.95 * MM / 2.0


@dataclass(frozen=True)
class CoilDrawingDimensions:
    """Derived dimensions for one smooth loading coil, in metres/degrees."""

    index: int
    z_start: float
    z_end: float
    radius: float
    diameter: float
    center_x: float
    turns: int
    pitch: float
    transition: float
    transition_offset: float
    handedness: str
    alpha_deg: float
    middle_rotation_deg: float
    middle_rise: float
    axial_height: float
    join_x: float
    join_y: float


@dataclass(frozen=True)
class AntennaDrawingDimensions:
    """Fabrication-oriented dimensions derived from an ``AntennaDesign``."""

    wire_diameter: float
    radiator_start_z: float
    radiator_tip_z: float
    radiator_height: float
    straight_ranges: tuple[tuple[float, float], ...]
    coils: tuple[CoilDrawingDimensions, ...]
    hub_radius: float
    hub_height: float
    radial_virtual_apex_z: float
    radial_model_start_radius: float
    radial_model_length: float
    radial_horizontal_radius: float
    radial_tip_z: float


def _coil_parameters(coil: CoilDesign) -> tuple[float, float, float]:
    """Return |alpha|, middle rotation, and middle axial rise."""
    alpha = 2.0 * math.asin(coil.transition_offset / (2.0 * coil.radius))
    middle_rotation = 2.0 * math.pi * int(coil.turns) - 2.0 * alpha
    if middle_rotation <= 0:
        raise ValueError("transition_offset is too large for the turn count")
    middle_rise = coil.pitch * middle_rotation / (2.0 * math.pi)
    return alpha, middle_rotation, middle_rise


def derive_drawing_dimensions(design: AntennaDesign) -> AntennaDrawingDimensions:
    """Derive all fabrication dimensions used by :func:`export_drawing`.

    The coil axial height follows the exact path construction used in
    ``geometry.py``: two transition rises plus the axial rise of the remaining
    helical rotation.
    """
    design.validate()

    z = design.port_height
    straight_ranges: list[tuple[float, float]] = []
    coil_dimensions: list[CoilDrawingDimensions] = []

    for index, straight_length in enumerate(design.straight_lengths):
        straight_start = z
        z += straight_length
        straight_ranges.append((straight_start, z))

        if index >= design.coil_count:
            continue

        coil = design.coils[index]
        alpha, middle_rotation, middle_rise = _coil_parameters(coil)
        coil_start = z
        coil_height = 2.0 * coil.transition + middle_rise
        coil_end = coil_start + coil_height
        signed_alpha = alpha if coil.handedness.upper() == "RH" else -alpha
        join_x = coil.radius * (math.cos(signed_alpha) - 1.0)
        join_y = coil.radius * math.sin(signed_alpha)

        coil_dimensions.append(
            CoilDrawingDimensions(
                index=index + 1,
                z_start=coil_start,
                z_end=coil_end,
                radius=coil.radius,
                diameter=2.0 * coil.radius,
                center_x=-coil.radius,
                turns=int(coil.turns),
                pitch=coil.pitch,
                transition=coil.transition,
                transition_offset=coil.transition_offset,
                handedness=coil.handedness.upper(),
                alpha_deg=math.degrees(alpha),
                middle_rotation_deg=math.degrees(middle_rotation),
                middle_rise=middle_rise,
                axial_height=coil_height,
                join_x=join_x,
                join_y=join_y,
            )
        )
        z = coil_end

    hub_radius = MODEL_HUB_CORE_RADIUS + 2.0 * design.wire_radius
    radial_model_start_radius = hub_radius - design.wire_radius
    radial_model_length = design.radial_length - radial_model_start_radius
    if radial_model_length <= 0:
        raise ValueError("radial_length is too short for the modeled ground hub")

    radial_angle = math.radians(design.radial_angle_deg)
    radial_virtual_apex_z = -MODEL_HUB_HEIGHT + 2.0 * design.wire_radius
    radial_horizontal_radius = design.radial_length * math.cos(radial_angle)
    radial_tip_z = radial_virtual_apex_z - design.radial_length * math.sin(radial_angle)

    return AntennaDrawingDimensions(
        wire_diameter=2.0 * design.wire_radius,
        radiator_start_z=design.port_height,
        radiator_tip_z=z,
        radiator_height=z - design.port_height,
        straight_ranges=tuple(straight_ranges),
        coils=tuple(coil_dimensions),
        hub_radius=hub_radius,
        hub_height=MODEL_HUB_HEIGHT,
        radial_virtual_apex_z=radial_virtual_apex_z,
        radial_model_start_radius=radial_model_start_radius,
        radial_model_length=radial_model_length,
        radial_horizontal_radius=radial_horizontal_radius,
        radial_tip_z=radial_tip_z,
    )


def _cubic_hermite(p0, p1, v0, v1, u) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    u = np.asarray(u, dtype=float)[:, None]
    return (
        (2 * u**3 - 3 * u**2 + 1) * p0
        + (-2 * u**3 + 3 * u**2) * p1
        + (u**3 - 2 * u**2 + u) * v0
        + (u**3 - u**2) * v1
    )


def _append_points(points: list[np.ndarray], coordinates: np.ndarray) -> None:
    coordinates = np.asarray(coordinates, dtype=float)
    if points and np.linalg.norm(coordinates[0] - points[-1]) < 1e-12:
        coordinates = coordinates[1:]
    points.extend(np.asarray(row, dtype=float) for row in coordinates)


def _sample_coil(
    start: np.ndarray,
    coil: CoilDesign,
    points_per_turn: int,
) -> np.ndarray:
    """Sample one coil with the same Hermite/helix equations as ``geometry.py``."""
    coil.validate()
    radius = coil.radius
    turns = int(coil.turns)
    pitch = coil.pitch
    transition = coil.transition
    transition_offset = coil.transition_offset
    sign = 1.0 if coil.handedness.upper() == "RH" else -1.0
    omega = sign * 2.0 * np.pi / pitch
    alpha = sign * 2.0 * np.arcsin(transition_offset / (2.0 * radius))
    middle_rotation = sign * 2.0 * np.pi * turns - 2.0 * alpha
    if sign * middle_rotation <= 0:
        raise ValueError("transition_offset is too large for the turn count")
    middle_length = abs(middle_rotation / omega)

    x0, y0, z0 = map(float, start)
    center_x = x0 - radius
    center_y = y0
    p_in_0 = np.array((x0, y0, z0))
    p_in_1 = np.array(
        (
            center_x + radius * np.cos(alpha),
            center_y + radius * np.sin(alpha),
            z0 + transition,
        )
    )
    d_in_0 = np.array((0.0, 0.0, 1.0))
    d_in_1 = np.array(
        (
            -radius * omega * np.sin(alpha),
            radius * omega * np.cos(alpha),
            1.0,
        )
    )
    d_in_1 /= np.linalg.norm(d_in_1)

    vertical_handle = 1.60 * transition
    helix_handle = 1.45 * transition
    transition_samples = max(8, int(np.ceil(points_per_turn / 4)) + 1)
    u = np.linspace(0.0, 1.0, transition_samples)
    incoming = _cubic_hermite(
        p_in_0,
        p_in_1,
        vertical_handle * d_in_0,
        helix_handle * d_in_1,
        u,
    )

    middle_samples = max(
        8,
        int(np.ceil(points_per_turn * middle_length / pitch)) + 1,
    )
    s = np.linspace(0.0, middle_length, middle_samples)
    theta_mid = alpha + omega * s
    z_mid = p_in_1[2] + s
    middle = np.column_stack(
        (
            center_x + radius * np.cos(theta_mid),
            center_y + radius * np.sin(theta_mid),
            z_mid,
        )
    )

    beta = alpha + middle_rotation
    z_out_start = float(z_mid[-1])
    p_out_0 = np.array(
        (
            center_x + radius * np.cos(beta),
            center_y + radius * np.sin(beta),
            z_out_start,
        )
    )
    p_out_1 = np.array((x0, y0, z_out_start + transition))
    d_out_0 = np.array(
        (
            -radius * omega * np.sin(beta),
            radius * omega * np.cos(beta),
            1.0,
        )
    )
    d_out_0 /= np.linalg.norm(d_out_0)
    d_out_1 = np.array((0.0, 0.0, 1.0))
    outgoing = _cubic_hermite(
        p_out_0,
        p_out_1,
        helix_handle * d_out_0,
        vertical_handle * d_out_1,
        u,
    )

    points: list[np.ndarray] = []
    _append_points(points, incoming)
    _append_points(points, middle)
    _append_points(points, outgoing)
    result = np.vstack(points)
    result[-1, 0] = x0
    result[-1, 1] = y0
    return result


def sample_centerline(
    design: AntennaDesign,
    points_per_turn: int = 160,
) -> np.ndarray:
    """Return an ``(N, 3)`` sampled radiator centerline in metres."""
    design.validate()
    if points_per_turn < 8:
        raise ValueError("points_per_turn must be at least eight")

    points: list[np.ndarray] = [np.array((0.0, 0.0, design.port_height))]
    current = points[-1].copy()

    for index, straight_length in enumerate(design.straight_lengths):
        straight_end = current + np.array((0.0, 0.0, straight_length))
        _append_points(points, np.vstack((current, straight_end)))
        current = straight_end

        if index < design.coil_count:
            sampled = _sample_coil(current, design.coils[index], points_per_turn)
            _append_points(points, sampled)
            current = sampled[-1].copy()

    return np.vstack(points)


def radial_centerlines(design: AntennaDesign) -> tuple[np.ndarray, ...]:
    """Return the modeled solid radial centerlines as ``(2, 3)`` arrays."""
    dims = derive_drawing_dimensions(design)
    angle = math.radians(design.radial_angle_deg)
    result: list[np.ndarray] = []

    for index in range(design.radial_count):
        phi = math.radians(index * 360.0 / design.radial_count)
        radii = np.array((dims.radial_model_start_radius, design.radial_length))
        horizontal = radii * math.cos(angle)
        z = dims.radial_virtual_apex_z - radii * math.sin(angle)
        result.append(
            np.column_stack(
                (
                    horizontal * math.cos(phi),
                    horizontal * math.sin(phi),
                    z,
                )
            )
        )
    return tuple(result)


def _mm(value: float) -> float:
    return value / MM


def _dimension_vertical(
    ax, x: float, y0: float, y1: float, label: str, ref_x: float = 0.0, *, rotate_text: bool = True
):
    ax.plot((ref_x, x), (y0, y0), linewidth=0.55)
    ax.plot((ref_x, x), (y1, y1), linewidth=0.55)
    ax.annotate(
        "",
        xy=(x, y0),
        xytext=(x, y1),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.7, "shrinkA": 0, "shrinkB": 0},
    )
    if rotate_text:
        ax.text(x + 1.6, (y0 + y1) / 2.0, label, rotation=90, va="center", fontsize=7)
    else:
        ax.text(x + 1.8, (y0 + y1) / 2.0, label, va="center", fontsize=7)


def _dimension_horizontal(ax, x0: float, x1: float, y: float, label: str):
    ax.annotate(
        "",
        xy=(x0, y),
        xytext=(x1, y),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.7, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text((x0 + x1) / 2.0, y + 2.0, label, ha="center", va="bottom", fontsize=7)


def _setup_orthographic_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


def _draw_side_view(ax, design: AntennaDesign, path_mm: np.ndarray, plane: str) -> None:
    dims = derive_drawing_dimensions(design)
    if plane == "xz":
        horizontal = path_mm[:, 0]
        radial_component = 0
        label = "X"
    else:
        horizontal = path_mm[:, 1]
        radial_component = 1
        label = "Y"

    ax.plot(horizontal, path_mm[:, 2], linewidth=1.25)

    # Simulation feed region and current hard-coded ground hub.
    hub_radius = _mm(dims.hub_radius)
    hub_height = _mm(dims.hub_height)
    ax.plot((-hub_radius, hub_radius, hub_radius, -hub_radius, -hub_radius),
            (-hub_height, -hub_height, 0.0, 0.0, -hub_height), linewidth=1.0)
    wire_radius = _mm(design.wire_radius)
    ax.plot((-wire_radius, -wire_radius, wire_radius, wire_radius),
            (0.0, _mm(design.port_height), _mm(design.port_height), 0.0),
            linestyle="--", linewidth=0.8)

    for radial in radial_centerlines(design):
        radial_mm = radial / MM
        ax.plot(radial_mm[:, radial_component], radial_mm[:, 2], linewidth=1.0)

    _setup_orthographic_axis(ax, f"{label} (mm)", "Z (mm)", f"{label}-Z view")


def _draw_top_view(ax, design: AntennaDesign, path_mm: np.ndarray) -> None:
    dims = derive_drawing_dimensions(design)
    ax.plot(path_mm[:, 0], path_mm[:, 1], linewidth=1.0)

    # Nominal coil circles make the construction radii obvious even though the
    # smooth transitions prevent the complete projected centerline from being a circle.
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    for coil in dims.coils:
        cx = _mm(coil.center_x)
        r = _mm(coil.radius)
        ax.plot(cx + r * np.cos(theta), r * np.sin(theta), linestyle="--", linewidth=0.65)
        ax.plot((cx,), (0.0,), marker="+", markersize=5)
        label_y = r + 2.0 if coil.index % 2 else -r - 6.0
        ax.text(cx, label_y, f"C{coil.index} R{r:.2f}", ha="center", fontsize=7)

    hub_radius = _mm(dims.hub_radius)
    ax.plot(hub_radius * np.cos(theta), hub_radius * np.sin(theta), linewidth=1.0)

    angle = math.radians(design.radial_angle_deg)
    start_horizontal = dims.radial_model_start_radius * math.cos(angle)
    end_horizontal = dims.radial_horizontal_radius
    for index in range(design.radial_count):
        phi = math.radians(index * 360.0 / design.radial_count)
        c, s = math.cos(phi), math.sin(phi)
        # Dashed virtual length inside the hub, then the modeled solid radial.
        ax.plot((0.0, _mm(start_horizontal) * c), (0.0, _mm(start_horizontal) * s),
                linestyle="--", linewidth=0.55)
        ax.plot((_mm(start_horizontal) * c, _mm(end_horizontal) * c),
                (_mm(start_horizontal) * s, _mm(end_horizontal) * s), linewidth=1.0)

    _setup_orthographic_axis(ax, "X (mm)", "Y (mm)", "X-Y top view")


def _draw_dimensions(ax, design: AntennaDesign, dims: AntennaDrawingDimensions) -> None:
    radial_extent = _mm(dims.radial_horizontal_radius)
    straight_x = radial_extent + 9.0
    coil_x = radial_extent + 22.0
    overall_x = radial_extent + 43.0

    # Dimension the three straight sections in one readable chain. Coil heights
    # use a separate lane because their short axial spans make rotated labels clash.
    for index, straight_range in enumerate(dims.straight_ranges):
        _dimension_vertical(
            ax,
            straight_x,
            _mm(straight_range[0]),
            _mm(straight_range[1]),
            f"S{index + 1} {_mm(straight_range[1]-straight_range[0]):.2f}",
        )

    for coil in dims.coils:
        _dimension_vertical(
            ax,
            coil_x,
            _mm(coil.z_start),
            _mm(coil.z_end),
            f"C{coil.index} H {_mm(coil.axial_height):.2f}",
            rotate_text=False,
        )

    _dimension_vertical(
        ax,
        overall_x,
        0.0,
        _mm(dims.radiator_tip_z),
        f"OVERALL {_mm(dims.radiator_tip_z):.2f}",
    )

    for coil in dims.coils:
        _dimension_horizontal(
            ax,
            -_mm(2.0 * coil.radius),
            0.0,
            _mm((coil.z_start + coil.z_end) / 2.0),
            f"C{coil.index} DIA {_mm(coil.diameter):.2f}",
        )

    apex_z = _mm(dims.radial_virtual_apex_z)
    tip_z = _mm(dims.radial_tip_z)
    tip_x = _mm(dims.radial_horizontal_radius)
    ax.plot((0.0, tip_x), (apex_z, apex_z), linestyle="--", linewidth=0.55)
    ax.text(
        tip_x * 0.48,
        (apex_z + tip_z) / 2.0 - 4.0,
        f"RADIAL L {_mm(design.radial_length):.2f} nominal\n"
        f"{design.radial_angle_deg:.2f} deg below horizontal",
        fontsize=7,
        ha="center",
    )


def _draw_table(ax, design: AntennaDesign, dims: AntennaDrawingDimensions) -> None:
    ax.axis("off")

    global_text = (
        f"Wire dia: {_mm(dims.wire_diameter):.3f} mm\n"
        f"Radiator height: {_mm(dims.radiator_height):.3f} mm\n"
        f"Radiator start Z: {_mm(dims.radiator_start_z):.3f} mm\n"
        f"Tip Z / overall: {_mm(dims.radiator_tip_z):.3f} mm\n"
        f"Ground hub: dia {_mm(2*dims.hub_radius):.3f} x {_mm(dims.hub_height):.3f} mm\n"
        f"Radials: {design.radial_count} x {_mm(design.radial_length):.3f} mm nominal, "
        f"{design.radial_angle_deg:.3f} deg below horizontal\n"
        f"Modeled radial cylinder: {_mm(dims.radial_model_length):.3f} mm; "
        f"virtual apex Z {_mm(dims.radial_virtual_apex_z):.3f} mm"
    )
    ax.text(0.0, 0.98, "GLOBAL DIMENSIONS", fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.0, 0.86, global_text, fontsize=7.2, va="top", family="monospace", transform=ax.transAxes)

    coil_lines = [
        "Coil   Hand Turns   Radius    Pitch    Trans.   Offset   alpha   helix   rise    height",
        "                   mm        mm       mm       mm       deg     deg     mm      mm",
    ]
    for coil in dims.coils:
        coil_lines.append(
            f"C{coil.index:<4}  {coil.handedness:<3}  {coil.turns:>3}   "
            f"{_mm(coil.radius):>7.3f}  {_mm(coil.pitch):>7.3f}  "
            f"{_mm(coil.transition):>7.3f}  {_mm(coil.transition_offset):>7.3f}  "
            f"{coil.alpha_deg:>6.2f}  {coil.middle_rotation_deg:>6.2f}  "
            f"{_mm(coil.middle_rise):>6.3f}  {_mm(coil.axial_height):>7.3f}"
        )

    ax.text(0.45, 0.98, "COIL DETAILS", fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.45, 0.86, "\n".join(coil_lines), fontsize=6.9, va="top", family="monospace", transform=ax.transAxes)

    notes = (
        "NOTES: Dimensions are centerline geometry unless stated otherwise. "
        "Coil radius is measured from the helix centerline. The smooth inlet/outlet "
        "transitions reproduce the simulation's cubic-Hermite geometry. Radial length "
        "is the nominal virtual-apex-to-tip dimension; the current CAD starts the radial "
        "solid inside the ground hub for overlap. The dashed feed region is the modeled "
        "lumped-port geometry and should be translated into the intended connector/insulator "
        "construction before fabrication."
    )
    ax.text(0.0, 0.12, notes, fontsize=6.8, va="bottom", wrap=True, transform=ax.transAxes)


def export_drawing(
    design: AntennaDesign,
    output: str | Path,
    *,
    title: str | None = None,
    points_per_turn: int = 160,
    dpi: int = 300,
) -> Path:
    """Export an A4-landscape fabrication sheet as PDF, SVG, or PNG."""
    design.validate()
    destination = Path(output)
    suffix = destination.suffix.lower()
    if suffix not in {".pdf", ".svg", ".png"}:
        raise ValueError("drawing output must end in .pdf, .svg, or .png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "drawing export requires matplotlib; install the package with the drawing extra"
        ) from error

    dims = derive_drawing_dimensions(design)
    path_mm = sample_centerline(design, points_per_turn=points_per_turn) / MM

    figure = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(2.25, 1.0))
    ax_xz = figure.add_subplot(grid[0, 0])
    ax_yz = figure.add_subplot(grid[0, 1])
    ax_xy = figure.add_subplot(grid[0, 2])
    ax_table = figure.add_subplot(grid[1, :])

    _draw_side_view(ax_xz, design, path_mm, "xz")
    _draw_side_view(ax_yz, design, path_mm, "yz")
    _draw_top_view(ax_xy, design, path_mm)
    _draw_dimensions(ax_xz, design, dims)
    _draw_table(ax_table, design, dims)

    figure.suptitle(title or "Loaded Antenna Fabrication Drawing", fontsize=14, fontweight="bold")
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point: ``python -m emerge_loaded_antenna.drawing``."""
    import argparse

    from .serialization import load_design

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", help="design JSON or optimizer-result JSON")
    parser.add_argument("output", help="output .pdf, .svg, or .png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--points-per-turn", type=int, default=160)
    args = parser.parse_args(list(argv) if argv is not None else None)

    export_drawing(
        load_design(args.design),
        args.output,
        title=args.title,
        dpi=args.dpi,
        points_per_turn=args.points_per_turn,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
