from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import main as example


def test_main_uses_visible_static_design_dimensions():
    design = example.DESIGN

    assert design.wire_radius == pytest.approx(0.8e-3)
    assert design.radial_length == pytest.approx(100e-3)
    assert design.radial_angle_deg == pytest.approx(30.1)
    assert design.straight_lengths == pytest.approx((96e-3, 75e-3, 112e-3))
    assert [coil.radius for coil in design.coils] == pytest.approx((15e-3, 9.75e-3))
    assert [coil.pitch for coil in design.coils] == pytest.approx((7.45e-3, 6.08e-3))
    assert all(coil.transition == pytest.approx(1e-3) for coil in design.coils)


def test_main_exports_design_sheet_and_jig_examples():
    with TemporaryDirectory() as temporary:
        output = Path(temporary)
        sheet = output / "design_sheet.pdf"
        jig = output / "jig_models" / "coil_01_winding_jig.stl"
        result = SimpleNamespace(solved=True)
        with (
            patch.object(example, "EXAMPLE_OUTPUT", output),
            patch.object(example, "export_drawing", return_value=sheet) as drawing,
            patch.object(example, "export_jig_models", return_value=(jig,)) as jigs,
        ):
            exported_sheet, exported_jigs = example.export_example_artifacts(result)

    assert exported_sheet == sheet
    assert exported_jigs == (jig,)
    drawing.assert_called_once_with(
        example.DESIGN,
        sheet,
        result=result,
        title="869.5 MHz Static Example",
    )
    jigs.assert_called_once_with(example.DESIGN, output / "jig_models")
