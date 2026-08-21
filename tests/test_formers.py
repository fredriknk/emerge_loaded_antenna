from __future__ import annotations

import math

import gmsh

from emerge_loaded_antenna.formers import (
    derive_coil_former_dimensions,
    export_coil_formers,
)
from test_drawing import example_design


def test_former_dimensions_use_centerline_diameter_and_five_mm_extension():
    design = example_design()
    dimensions = derive_coil_former_dimensions(design)

    assert len(dimensions) == 2
    for coil, former in zip(design.coils, dimensions):
        assert math.isclose(former.diameter, 2.0 * coil.radius)
        assert math.isclose(
            former.inside_diameter,
            2.0 * (coil.radius - design.wire_radius),
        )
        assert math.isclose(former.former_height, former.coil_height + 5e-3)
        assert math.isclose(former.groove_radius, design.wire_radius + 0.1e-3)
    assert math.isclose(
        dimensions[1].center_x - dimensions[1].radius,
        math.fsum((dimensions[0].center_x, dimensions[0].radius, 5e-3)),
    )


def test_step_export_contains_coil_tools_and_radial_gauge(tmp_path):
    destination = export_coil_formers(example_design(), tmp_path / "formers.step")

    assert destination.exists()
    assert destination.stat().st_size > 100_000

    owns_gmsh = not bool(gmsh.isInitialized())
    previous_model = ""
    if owns_gmsh:
        gmsh.initialize()
    else:
        previous_model = gmsh.model.getCurrent()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("former_import_check")
        gmsh.model.occ.importShapes(str(destination.resolve()))
        gmsh.model.occ.synchronize()
        assert len(gmsh.model.getEntities(3)) == 2 * example_design().coil_count + 1
    finally:
        if gmsh.isInitialized():
            if gmsh.model.getCurrent() == "former_import_check":
                gmsh.model.remove()
            if previous_model:
                gmsh.model.setCurrent(previous_model)
            if owns_gmsh:
                gmsh.finalize()
