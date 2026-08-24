from __future__ import annotations

import math
from types import SimpleNamespace

import emerge as em
import numpy as np

from emerge_loaded_antenna import AntennaDesign, CoilDesign
from emerge_loaded_antenna.drawing import (
    A3_LANDSCAPE_SIZE_INCHES,
    _azimuth_ring_gain,
    _draw_gain_lobes,
    _horizon_gain,
    _xz_gain,
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
    assert math.isclose(dims.coils[0].inside_diameter, 0.028338169262053172)
    assert math.isclose(dims.coils[1].inside_diameter, 0.017888480965145736)
    assert math.isclose(dims.coils[0].transition_radius, 0.007863117754520154)
    assert math.isclose(dims.coils[1].transition_radius, 0.007395249073612345)
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

    header = destination.read_text(encoding="utf-8")[:1000]
    width_points = A3_LANDSCAPE_SIZE_INCHES[0] * 72.0
    height_points = A3_LANDSCAPE_SIZE_INCHES[1] * 72.0
    assert f'width="{width_points:.6f}pt"' in header
    assert f'height="{height_points:.6f}pt"' in header


def solved_result():
    theta_axis = np.array((0.0, np.pi / 2.0, np.pi))
    phi_axis = np.linspace(0.0, 2.0 * np.pi, 9)
    phi, theta = np.meshgrid(phi_axis, theta_axis, indexing="ij")
    horizon_gain_db = 1.0 + 2.0 * np.cos(phi)
    norm_e = em.lib.EISO * 10.0 ** (horizon_gain_db / 20.0)
    return SimpleNamespace(
        frequencies=np.array((850e6, 868e6, 886e6)),
        s11_db=np.array((-8.0, -15.0, -9.0)),
        farfield_3d=SimpleNamespace(theta=theta, phi=phi, normE=norm_e),
        farfield_metrics=SimpleNamespace(frequency_hz=868e6),
    )


def multi_ring_result():
    theta_axis = np.deg2rad((0.0, 90.0, 130.0, 180.0))
    phi_axis = np.linspace(0.0, 2.0 * np.pi, 9)
    phi, theta = np.meshgrid(phi_axis, theta_axis, indexing="ij")
    gain_db = np.rad2deg(theta) / 100.0 + np.cos(phi)
    norm_e = em.lib.EISO * 10.0 ** (gain_db / 20.0)
    return SimpleNamespace(
        frequencies=np.array((850e6, 868e6, 886e6)),
        s11_db=np.array((-8.0, -15.0, -9.0)),
        farfield_3d=SimpleNamespace(theta=theta, phi=phi, normE=norm_e),
        farfield_metrics=SimpleNamespace(frequency_hz=868e6),
    )


def test_horizon_gain_extracts_and_closes_polar_cut():
    phi, gain_db = _horizon_gain(solved_result())

    assert phi.size == 9
    assert math.isclose(phi[0], 0.0, abs_tol=1e-12)
    assert math.isclose(phi[-1], 2.0 * np.pi, abs_tol=1e-12)
    assert math.isclose(gain_db[0], 3.0, abs_tol=1e-12)
    assert math.isclose(gain_db[-1], gain_db[0], abs_tol=1e-12)


def test_azimuth_ring_gain_extracts_requested_verified_theta():
    phi, gain_db = _azimuth_ring_gain(multi_ring_result(), 130.0)

    assert phi.size == 9
    assert math.isclose(gain_db[0], 2.3, abs_tol=1e-12)
    assert math.isclose(gain_db[-1], gain_db[0], abs_tol=1e-12)


def test_xz_gain_extracts_both_meridians_and_closes_cut():
    angle, gain_db = _xz_gain(solved_result())

    np.testing.assert_allclose(
        angle,
        (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi),
        atol=1e-12,
    )
    np.testing.assert_allclose(gain_db, (3.0, 3.0, 3.0, -1.0, -1.0))


def test_gain_lobes_use_clockwise_compass_orientation():
    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.add_subplot(111, projection="polar")
    _draw_gain_lobes(axis, solved_result())

    assert math.isclose(axis.get_theta_offset(), np.pi / 2.0)
    assert axis.get_theta_direction() == -1.0
    figure.clear()


def test_gain_lobes_label_and_overlay_verified_target_rings():
    from matplotlib.figure import Figure

    figure = Figure()
    axis = figure.add_subplot(111, projection="polar")
    _draw_gain_lobes(
        axis,
        multi_ring_result(),
        target_ring_thetas_deg=(90.0, 130.0),
    )

    labels = [line.get_label() for line in axis.lines]
    assert (
        "XY/horizon - target 90 deg\n"
        "min -0.10 | max 1.90 | avg 0.96 dBi"
    ) in labels
    assert (
        "Target ring 130 deg\n"
        "min 0.30 | max 2.30 | avg 1.36 dBi"
    ) in labels
    assert "XZ/elevation" in labels
    figure.clear()


def test_svg_export_includes_solved_rf_plots(tmp_path):
    destination = export_drawing(
        example_design(),
        tmp_path / "antenna-with-rf.svg",
        result=solved_result(),
    )

    assert destination.exists()
    assert destination.stat().st_size > 20_000
