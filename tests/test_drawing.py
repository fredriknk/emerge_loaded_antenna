from __future__ import annotations

import math

import numpy as np

from emerge_loaded_antenna import AntennaDesign, CoilDesign
from emerge_loaded_antenna.drawing import (
    derive_drawing_dimensions,
    export_drawing,
    radial_centerlines,
    sample_centerline,
)


def example_design() -> AntennaDesign:
    return AntennaDesign(
        wire_radius=0.0008,
        radial_length=0.09,
        radial_angle_deg=30.116935985992047,
        radial_count=4,
        straight_lengths=(
            0.09612517992329095,
            0.0751706730240138,
            0.11207200818321847,
        ),
        coils=(
            CoilDesign(
                radius=0.014969084631026586,
                turns=1,
                pitch=0.007444323042496868,
                transition=0.0059896492236917774,
                transition_offset=0.004741805635422657,
                handedness="RH",
            ),
            CoilDesign(
                radius=0.009744240482572868,
                turns=1,
                pitch=0.006079221404596115,
                transition=0.0059896492236917774,
                transition_offset=0.004741805635422657,
                handedness="RH",
            ),
        ),
        port_height=0.0019965497412305923,
        port_impedance=50.0,
    )


def test_example_dimensions():
    dims = derive_drawing_dimensions(example_design())

    assert math.isclose(dims.wire_diameter, 0.0016)
    assert math.isclose(dims.coils[0].diameter, 0.029938169262053172)
    assert math.isclose(dims.coils[1].diameter, 0.019488480965145736)
    assert math.isclose(dims.coils[0].alpha_deg, 18.227, abs_tol=0.002)
    assert math.isclose(dims.coils[1].alpha_deg, 28.164, abs_tol=0.002)
    assert math.isclose(dims.coils[0].middle_rotation_deg, 323.547, abs_tol=0.003)
    assert math.isclose(dims.coils[1].middle_rotation_deg, 303.671, abs_tol=0.003)
    assert math.isclose(dims.radiator_height, 0.319145, abs_tol=2e-6)
    assert math.isclose(dims.radiator_tip_z, 0.321142, abs_tol=2e-6)


def test_centerline_returns_to_axis_and_matches_derived_tip():
    design = example_design()
    dims = derive_drawing_dimensions(design)
    path = sample_centerline(design, points_per_turn=40)

    np.testing.assert_allclose(path[0], (0.0, 0.0, design.port_height))
    assert math.isclose(path[-1, 0], 0.0, abs_tol=1e-12)
    assert math.isclose(path[-1, 1], 0.0, abs_tol=1e-12)
    assert math.isclose(path[-1, 2], dims.radiator_tip_z, abs_tol=1e-12)


def test_transition_offset_is_chord_distance():
    dims = derive_drawing_dimensions(example_design())
    for coil in dims.coils:
        chord = math.hypot(coil.join_x, coil.join_y)
        assert math.isclose(chord, coil.transition_offset, rel_tol=1e-12)


def test_radial_centerlines_have_expected_modeled_length():
    design = example_design()
    dims = derive_drawing_dimensions(design)
    for radial in radial_centerlines(design):
        assert math.isclose(np.linalg.norm(radial[1] - radial[0]), dims.radial_model_length, rel_tol=1e-12)


def test_svg_export(tmp_path):
    destination = export_drawing(example_design(), tmp_path / "antenna.svg")
    assert destination.exists()
    assert destination.stat().st_size > 10_000
