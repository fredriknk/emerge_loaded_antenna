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

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import AntennaDesign, CoilDesign

if TYPE_CHECKING:
    from .simulation import SimulationResult

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


def _draw_unavailable_plot(ax, title: str) -> None:
    """Draw a consistent placeholder when no solved RF result was supplied."""
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.text(
        0.5,
        0.5,
        "Solved simulation result not supplied",
        ha="center",
        va="center",
        fontsize=7,
        color="0.4",
        transform=ax.transAxes,
    )
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.tick_params(labelsize=7)


def _result_frequency_hz(result: SimulationResult) -> float | None:
    metrics = getattr(result, "farfield_metrics", None)
    value = getattr(metrics, "frequency_hz", None)
    if value is None:
        options = getattr(result, "options", None)
        value = getattr(options, "farfield_frequency", None)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _draw_s11(ax, result: SimulationResult | None) -> None:
    """Draw the solved reflection-coefficient sweep."""
    if result is None:
        _draw_unavailable_plot(ax, "S11 sweep")
        ax.set_xlabel("Frequency (MHz)", fontsize=8)
        ax.set_ylabel("S11 (dB)", fontsize=8)
        return

    frequencies = np.asarray(result.frequencies, dtype=float).reshape(-1)
    s11_db = np.asarray(result.s11_db, dtype=float).reshape(-1)
    if not frequencies.size or frequencies.size != s11_db.size:
        _draw_unavailable_plot(ax, "S11 sweep")
        ax.set_xlabel("Frequency (MHz)", fontsize=8)
        ax.set_ylabel("S11 (dB)", fontsize=8)
        return

    frequency_mhz = frequencies / 1e6
    marker = "o" if frequencies.size <= 20 else None
    ax.plot(frequency_mhz, s11_db, marker=marker, markersize=3, linewidth=1.1)
    ax.axhline(-10.0, color="tab:red", linestyle="--", linewidth=0.8, label="-10 dB")

    farfield_frequency = _result_frequency_hz(result)
    if farfield_frequency is not None:
        ax.axvline(
            farfield_frequency / 1e6,
            color="0.45",
            linestyle=":",
            linewidth=0.8,
            label="lobe frequency",
        )

    ax.set_title("S11 sweep", fontsize=9, fontweight="bold")
    ax.set_xlabel("Frequency (MHz)", fontsize=8)
    ax.set_ylabel("S11 (dB)", fontsize=8)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="best")


def _horizon_gain(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted horizon azimuth and realized gain from a sampled 3-D field."""
    farfield = getattr(result, "farfield_3d", None)
    if farfield is None:
        raise ValueError("far-field data is unavailable")

    theta = np.asarray(farfield.theta, dtype=float)
    phi = np.asarray(farfield.phi, dtype=float)
    norm_e = np.asarray(farfield.normE)
    try:
        theta, phi, norm_e = np.broadcast_arrays(theta, phi, norm_e)
    except ValueError as error:
        raise ValueError("far-field coordinate and value shapes do not match") from error
    if not theta.size:
        raise ValueError("far-field data is empty")

    theta_distance = np.abs(theta - np.pi / 2.0)
    nearest_distance = float(np.nanmin(theta_distance))
    horizon = np.isclose(theta_distance, nearest_distance, rtol=0.0, atol=1e-10)
    phi_values = np.mod(phi[horizon], 2.0 * np.pi)

    # EMerge's normE/EISO ratio is the realized-gain amplitude used throughout
    # the simulation and verification code.
    import emerge as em

    gain_values = 20.0 * np.log10(
        np.maximum(np.abs(norm_e[horizon]) / em.lib.EISO, 1e-12)
    )
    finite = np.isfinite(phi_values) & np.isfinite(gain_values)
    phi_values = phi_values[finite]
    gain_values = gain_values[finite]
    if not phi_values.size:
        raise ValueError("far-field horizon data is empty")

    order = np.argsort(phi_values)
    phi_values = phi_values[order]
    gain_values = gain_values[order]

    # A 0/360-degree endpoint pair is common in spherical grids. Average any
    # duplicate azimuth samples so the polar line closes without a false seam.
    rounded_phi = np.round(phi_values, decimals=12)
    unique_phi, inverse = np.unique(rounded_phi, return_inverse=True)
    sums = np.bincount(inverse, weights=gain_values)
    counts = np.bincount(inverse)
    gain_values = sums / counts
    phi_values = unique_phi

    if phi_values.size > 1:
        phi_values = np.append(phi_values, phi_values[0] + 2.0 * np.pi)
        gain_values = np.append(gain_values, gain_values[0])
    return phi_values, gain_values


def _draw_horizon_lobe(ax, result: SimulationResult | None) -> None:
    """Draw the XY/horizon realized-gain lobe on a polar axis."""
    if result is None:
        _draw_unavailable_plot(ax, "XY/horizon gain lobe")
        return
    try:
        phi, gain_db = _horizon_gain(result)
    except (AttributeError, TypeError, ValueError):
        _draw_unavailable_plot(ax, "XY/horizon gain lobe")
        return

    peak_gain = float(np.nanmax(gain_db))
    radial_floor = max(-40.0, float(math.floor(peak_gain - 30.0)))
    if radial_floor >= peak_gain:
        radial_floor = float(math.floor(peak_gain - 10.0))
    radial_ceiling = float(math.ceil(peak_gain + 1.0))
    ax.plot(phi, np.maximum(gain_db, radial_floor), linewidth=1.1)
    ax.set_rlim(radial_floor, radial_ceiling)
    frequency_hz = _result_frequency_hz(result)
    frequency_text = (
        f" at {frequency_hz / 1e6:g} MHz" if frequency_hz is not None else ""
    )
    ax.set_title(
        f"XY/horizon realized gain{frequency_text} (dBi)",
        fontsize=9,
        fontweight="bold",
        pad=12,
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.tick_params(labelsize=6.5)


def export_drawing(
    design: AntennaDesign,
    output: str | Path,
    *,
    result: SimulationResult | None = None,
    title: str | None = None,
    points_per_turn: int = 160,
    dpi: int = 300,
) -> Path:
    """Export an A4-landscape fabrication sheet as PDF, SVG, or PNG.

    Pass a solved ``SimulationResult`` to include its S11 sweep and sampled
    XY/horizon realized-gain lobe. Geometry-only callers remain supported and
    receive clearly marked placeholders for those two plots.
    """
    design.validate()
    destination = Path(output)
    suffix = destination.suffix.lower()
    if suffix not in {".pdf", ".svg", ".png"}:
        raise ValueError("drawing output must end in .pdf, .svg, or .png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        from matplotlib.figure import Figure
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "drawing export requires matplotlib; install the package with the drawing extra"
        ) from error

    dims = derive_drawing_dimensions(design)
    path_mm = sample_centerline(design, points_per_turn=points_per_turn) / MM

    # Build the figure directly instead of going through pyplot.  Exporting a
    # drawing is non-interactive and must not require a working Tk/Qt desktop
    # backend (for example in CI or on a headless solver machine).
    figure = Figure(figsize=(11.69, 8.27), constrained_layout=True)
    grid = figure.add_gridspec(4, 6, height_ratios=(0.18, 2.2, 1.2, 1.45))
    ax_title = figure.add_subplot(grid[0, :])
    ax_xz = figure.add_subplot(grid[1, 0:2])
    ax_yz = figure.add_subplot(grid[1, 2:4])
    ax_xy = figure.add_subplot(grid[1, 4:6])
    ax_s11 = figure.add_subplot(grid[2, 0:3])
    ax_lobe = figure.add_subplot(grid[2, 3:6], projection="polar")
    ax_table = figure.add_subplot(grid[3, :])

    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.5,
        title or "Loaded Antenna Fabrication Drawing",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=ax_title.transAxes,
    )

    _draw_side_view(ax_xz, design, path_mm, "xz")
    _draw_side_view(ax_yz, design, path_mm, "yz")
    _draw_top_view(ax_xy, design, path_mm)
    _draw_dimensions(ax_xz, design, dims)
    _draw_s11(ax_s11, result)
    _draw_horizon_lobe(ax_lobe, result)
    _draw_table(ax_table, design, dims)

    figure.savefig(destination, dpi=dpi)
    figure.clear()
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
