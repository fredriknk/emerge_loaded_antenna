"""Export 3-D-printable grooved coil-winding formers.

The solids in this module are fabrication aids only.  They are built in an
independent Gmsh model and are never added to the electromagnetic simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable
import uuid

import gmsh

from .config import AntennaDesign
from .geometry import AntennaPath, Segment


@dataclass(frozen=True)
class CoilFormerDimensions:
    """Derived dimensions for one grooved former, in metres."""

    index: int
    radius: float
    diameter: float
    inside_radius: float
    inside_diameter: float
    coil_height: float
    former_height: float
    groove_radius: float
    center_x: float


def derive_coil_former_dimensions(
    design: AntennaDesign,
    *,
    extra_length: float = 5e-3,
    groove_clearance: float = 0.1e-3,
    spacing: float = 5e-3,
) -> tuple[CoilFormerDimensions, ...]:
    """Return the blank and groove dimensions used by the CAD exporter."""
    design.validate()
    if extra_length <= 0:
        raise ValueError("extra_length must be positive")
    if groove_clearance < 0:
        raise ValueError("groove_clearance cannot be negative")
    if spacing < 0:
        raise ValueError("spacing cannot be negative")

    groove_radius = design.wire_radius + groove_clearance
    result: list[CoilFormerDimensions] = []
    next_left_edge = 0.0
    for index, coil in enumerate(design.coils, start=1):
        path = AntennaPath(x=coil.radius, z=0.0)
        path.coil(coil)
        coil_height = path.current_z
        center_x = next_left_edge + coil.radius
        result.append(
            CoilFormerDimensions(
                index=index,
                radius=coil.radius,
                diameter=2.0 * coil.radius,
                inside_radius=coil.radius - design.wire_radius,
                inside_diameter=2.0 * (coil.radius - design.wire_radius),
                coil_height=coil_height,
                former_height=coil_height + extra_length,
                groove_radius=groove_radius,
                center_x=center_x,
            )
        )
        next_left_edge += 2.0 * coil.radius + spacing
    return tuple(result)


def _add_occ_wire(segments: Iterable[Segment]) -> int:
    edge_tags: list[int] = []
    last_point_tag: int | None = None

    for kind, coordinates in segments:
        local_tags: list[int] = []
        if last_point_tag is not None:
            local_tags.append(last_point_tag)
            coordinates = coordinates[1:]
        for point in coordinates:
            local_tags.append(gmsh.model.occ.addPoint(*map(float, point)))

        if kind == "line":
            edge_tag = gmsh.model.occ.addLine(*local_tags)
        elif kind == "bezier":
            edge_tag = gmsh.model.occ.addBezier(local_tags)
        else:  # pragma: no cover - guarded by AntennaPath
            raise ValueError(f"unknown composite segment type: {kind}")
        edge_tags.append(edge_tag)
        last_point_tag = local_tags[-1]

    return gmsh.model.occ.addWire(edge_tags, checkClosed=False)


def _sweep_groove_cutter(
    coil,
    *,
    center_x: float,
    path_radius: float,
    groove_radius: float,
    z_margin: float,
    former_height: float,
) -> list[tuple[int, int]]:
    """Return the coil pipe plus overlapping cutters through both end faces."""
    overrun = max(2.0 * groove_radius, 1e-3)
    overlap = groove_radius
    path = AntennaPath(x=center_x + path_radius, z=z_margin)
    path.coil(coil)
    wire = _add_occ_wire(path.segments)
    cutter_profile = gmsh.model.occ.addDisk(
        center_x + path_radius,
        0.0,
        z_margin,
        groove_radius,
        groove_radius,
    )
    swept = gmsh.model.occ.addPipe(
        [(2, cutter_profile)],
        wire,
        trihedron="GuidePlan",
    )
    cutter_volumes = [entity for entity in swept if entity[0] == 3]
    if len(cutter_volumes) != 1:
        raise RuntimeError("failed to create one continuous groove cutter")
    # addPipe retains its source profile and guide as independent CAD entities.
    # Remove those construction entities so STEP/STL contains solids only.
    gmsh.model.occ.remove([(2, cutter_profile)], recursive=True)
    gmsh.model.occ.remove([(1, wire)], recursive=True)
    inlet = gmsh.model.occ.addCylinder(
        center_x + path_radius,
        0.0,
        -overrun,
        0.0,
        0.0,
        z_margin + overrun + overlap,
        groove_radius,
    )
    outlet_start = path.current_z - overlap
    outlet = gmsh.model.occ.addCylinder(
        center_x + path_radius,
        0.0,
        outlet_start,
        0.0,
        0.0,
        former_height + overrun - outlet_start,
        groove_radius,
    )
    fused, _ = gmsh.model.occ.fuse(
        [cutter_volumes[0]],
        [(3, inlet), (3, outlet)],
        removeObject=True,
        removeTool=True,
    )
    fused_volumes = [entity for entity in fused if entity[0] == 3]
    if len(fused_volumes) != 1:
        raise RuntimeError("failed to join the groove to both end-face cutters")
    return fused_volumes


def _transition_marker_cutters(
    coil,
    dimensions: CoilFormerDimensions,
    *,
    center_x: float,
    blank_radius: float,
    z_margin: float,
    marker_depth: float,
    marker_width: float,
    marker_length: float,
) -> list[tuple[int, int]]:
    """Create short transverse notches at all four transition boundaries."""
    if marker_depth == 0.0 or marker_width == 0.0 or marker_length == 0.0:
        return []
    if marker_depth >= blank_radius:
        raise ValueError("marker_depth must be smaller than the mandrel radius")
    sign = 1.0 if coil.handedness.upper() == "RH" else -1.0
    alpha = sign * 2.0 * math.asin(
        coil.transition_offset / (2.0 * coil.radius)
    )
    beta = sign * 2.0 * math.pi * int(coil.turns) - alpha
    marker_locations = (
        (z_margin, 0.0),
        (z_margin + coil.transition, alpha),
        (z_margin + dimensions.coil_height - coil.transition, beta),
        (z_margin + dimensions.coil_height, 0.0),
    )
    # Put each witness beside the wire groove instead of intersecting it.  The
    # shared Z coordinate still marks the boundary precisely, while separated
    # Boolean edges remain robust in both STEP and STL.
    marker_arc_offset = marker_length / 2.0 + marker_width + 0.25e-3
    angular_offset = marker_arc_offset / blank_radius
    cutters: list[tuple[int, int]] = []
    for z, angle in marker_locations:
        angle += angular_offset
        cutter = gmsh.model.occ.addBox(
            center_x + blank_radius - marker_depth,
            -marker_length / 2.0,
            z - marker_width / 2.0,
            2.0 * marker_depth,
            marker_length,
            marker_width,
        )
        if angle:
            gmsh.model.occ.rotate(
                [(3, cutter)],
                center_x,
                0.0,
                z,
                0.0,
                0.0,
                1.0,
                angle,
            )
        cutters.append((3, cutter))
    return cutters


def _build_former(
    design: AntennaDesign,
    dimensions: CoilFormerDimensions,
    *,
    center_x: float,
    blank_radius: float,
    groove_radius: float | None,
    extra_length: float,
    marker_depth: float,
    marker_width: float,
    marker_length: float,
) -> int:
    coil = design.coils[dimensions.index - 1]
    z_margin = extra_length / 2.0

    cylinder = gmsh.model.occ.addCylinder(
        center_x,
        0.0,
        0.0,
        0.0,
        0.0,
        dimensions.former_height,
        blank_radius,
    )

    cutters: list[tuple[int, int]] = []
    if groove_radius is not None:
        cutters.extend(
            _sweep_groove_cutter(
                coil,
                center_x=center_x,
                path_radius=dimensions.radius,
                groove_radius=groove_radius,
                z_margin=z_margin,
                former_height=dimensions.former_height,
            )
        )
    cutters.extend(
        _transition_marker_cutters(
            coil,
            dimensions,
            center_x=center_x,
            blank_radius=blank_radius,
            z_margin=z_margin,
            marker_depth=marker_depth,
            marker_width=marker_width,
            marker_length=marker_length,
        )
    )

    if cutters:
        cut, _ = gmsh.model.occ.cut(
            [(3, cylinder)],
            cutters,
            removeObject=True,
            removeTool=True,
        )
        volumes = [tag for dim, tag in cut if dim == 3]
    else:
        volumes = [cylinder]
    if len(volumes) != 1:
        raise RuntimeError(
            f"C{dimensions.index} former Boolean produced {len(volumes)} solids"
        )
    return volumes[0]


def export_coil_formers(
    design: AntennaDesign,
    output: str | Path,
    *,
    extra_length: float = 5e-3,
    groove_clearance: float = 0.1e-3,
    spacing: float = 5e-3,
    include_sizing_mandrels: bool = True,
    sizing_groove_depth: float = 0.2e-3,
    marker_depth: float = 0.15e-3,
    marker_width: float = 0.35e-3,
    marker_length: float = 2.0e-3,
    stl_mesh_size: float | None = None,
) -> Path:
    """Export grooved coil formers to one STEP or STL file.

    Each former has the coil centerline diameter before the wire-sized groove
    is subtracted.  ``extra_length`` is split equally above and below the coil.
    Multiple tools are placed side-by-side with ``spacing`` between blanks.
    By default, each coil also gets an inside-diameter sizing mandrel with only
    a shallow guide groove for correcting spring-back by hand.
    """
    destination = Path(output)
    if destination.suffix.lower() not in {".step", ".stp", ".stl"}:
        raise ValueError("coil-former output must end in .step, .stp, or .stl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dimensions = derive_coil_former_dimensions(
        design,
        extra_length=extra_length,
        groove_clearance=groove_clearance,
        spacing=spacing,
    )
    if not dimensions:
        raise ValueError("at least one coil is required to export a former")
    if stl_mesh_size is not None and stl_mesh_size <= 0:
        raise ValueError("stl_mesh_size must be positive")
    if sizing_groove_depth < 0:
        raise ValueError("sizing_groove_depth cannot be negative")
    if marker_depth < 0 or marker_width < 0 or marker_length < 0:
        raise ValueError("marker dimensions cannot be negative")

    owns_gmsh = not bool(gmsh.isInitialized())
    previous_model = ""
    if owns_gmsh:
        gmsh.initialize()
    else:
        previous_model = gmsh.model.getCurrent()
    previous_terminal = gmsh.option.getNumber("General.Terminal")
    previous_mesh_min = gmsh.option.getNumber("Mesh.MeshSizeMin")
    previous_mesh_max = gmsh.option.getNumber("Mesh.MeshSizeMax")

    model_name = f"coil_formers_{uuid.uuid4().hex}"
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(model_name)
        next_left_edge = 0.0
        for item in dimensions:
            winding_center_x = next_left_edge + item.radius
            volume = _build_former(
                design,
                item,
                center_x=winding_center_x,
                blank_radius=item.radius,
                groove_radius=item.groove_radius,
                extra_length=extra_length,
                marker_depth=marker_depth,
                marker_width=marker_width,
                marker_length=marker_length,
            )
            gmsh.model.setEntityName(3, volume, f"WindingFormerC{item.index}")
            next_left_edge += item.diameter + spacing

            if include_sizing_mandrels:
                sizing_center_x = next_left_edge + item.inside_radius
                sizing_cutter_radius = (
                    None
                    if sizing_groove_depth == 0.0
                    else design.wire_radius + sizing_groove_depth
                )
                volume = _build_former(
                    design,
                    item,
                    center_x=sizing_center_x,
                    blank_radius=item.inside_radius,
                    groove_radius=sizing_cutter_radius,
                    extra_length=extra_length,
                    marker_depth=marker_depth,
                    marker_width=marker_width,
                    marker_length=marker_length,
                )
                gmsh.model.setEntityName(
                    3,
                    volume,
                    f"SizingMandrelC{item.index}",
                )
                next_left_edge += item.inside_diameter + spacing
        gmsh.model.occ.synchronize()

        if destination.suffix.lower() == ".stl":
            mesh_size = stl_mesh_size or min(
                design.wire_radius / 3.0,
                min(coil.pitch for coil in design.coils) / 12.0,
            )
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
            gmsh.model.mesh.generate(2)
        gmsh.write(str(destination.resolve()))
    finally:
        if gmsh.isInitialized():
            if gmsh.model.getCurrent() == model_name:
                gmsh.model.remove()
            if previous_model:
                gmsh.model.setCurrent(previous_model)
            gmsh.option.setNumber("General.Terminal", previous_terminal)
            gmsh.option.setNumber("Mesh.MeshSizeMin", previous_mesh_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", previous_mesh_max)
            if owns_gmsh:
                gmsh.finalize()

    return destination


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point: ``python -m emerge_loaded_antenna.formers``."""
    import argparse

    from .serialization import load_design

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", help="design JSON or optimizer-result JSON")
    parser.add_argument("output", help="output .step, .stp, or .stl")
    parser.add_argument("--extra-length-mm", type=float, default=5.0)
    parser.add_argument("--groove-clearance-mm", type=float, default=0.1)
    parser.add_argument("--spacing-mm", type=float, default=5.0)
    parser.add_argument("--no-sizing-mandrels", action="store_true")
    parser.add_argument("--sizing-groove-depth-mm", type=float, default=0.2)
    parser.add_argument("--marker-depth-mm", type=float, default=0.15)
    parser.add_argument("--marker-width-mm", type=float, default=0.35)
    parser.add_argument("--marker-length-mm", type=float, default=2.0)
    parser.add_argument("--stl-mesh-size-mm", type=float, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    millimetres = 1e-3
    export_coil_formers(
        load_design(args.design),
        args.output,
        extra_length=args.extra_length_mm * millimetres,
        groove_clearance=args.groove_clearance_mm * millimetres,
        spacing=args.spacing_mm * millimetres,
        include_sizing_mandrels=not args.no_sizing_mandrels,
        sizing_groove_depth=args.sizing_groove_depth_mm * millimetres,
        marker_depth=args.marker_depth_mm * millimetres,
        marker_width=args.marker_width_mm * millimetres,
        marker_length=args.marker_length_mm * millimetres,
        stl_mesh_size=(
            None
            if args.stl_mesh_size_mm is None
            else args.stl_mesh_size_mm * millimetres
        ),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
