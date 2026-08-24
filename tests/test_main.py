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
    assert all(coil.transition == pytest.approx(6e-3) for coil in design.coils)


def test_main_exports_design_sheet_and_former_examples():
    with TemporaryDirectory() as temporary:
        output = Path(temporary)
        sheet = output / "design_sheet.pdf"
        former = output / "coil_formers.step"
        result = SimpleNamespace(solved=True)
        with (
            patch.object(example, "EXAMPLE_OUTPUT", output),
            patch.object(example, "export_drawing", return_value=sheet) as drawing,
            patch.object(
                example,
                "export_coil_formers",
                return_value=former,
            ) as formers,
        ):
            exported_sheet, exported_formers = example.export_example_artifacts(result)

    assert exported_sheet == sheet
    assert exported_formers == former
    drawing.assert_called_once_with(
        example.DESIGN,
        sheet,
        result=result,
        title="869.5 MHz Static Example",
    )
    formers.assert_called_once_with(
        example.DESIGN,
        former,
        extra_length=5e-3,
        groove_clearance=0.1e-3,
        spacing=5e-3,
    )
