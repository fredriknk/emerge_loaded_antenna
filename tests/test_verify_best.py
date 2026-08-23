from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from emerge_loaded_antenna import AntennaDesign, MeshSettings
from examples.verify_best import export_fabrication_artifacts, options, parse_args


class VerifyBestTests(unittest.TestCase):
    def test_fabrication_flags_are_parsed(self):
        with patch(
            "sys.argv",
            ["verify_best.py", "--design-sheet", "--jig-models"],
        ):
            args = parse_args()

        self.assertTrue(args.design_sheet)
        self.assertTrue(args.jig_models)

    def test_fabrication_flags_generate_named_verification_artifacts(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            jig_path = output / "jig_models" / "coil_01_winding_jig.stl"
            with (
                patch(
                    "examples.verify_best.export_drawing",
                    return_value=output / "design_sheet.pdf",
                ) as drawing,
                patch(
                    "examples.verify_best.export_jig_models",
                    return_value=(jig_path,),
                ) as jigs,
            ):
                artifacts = export_fabrication_artifacts(
                    AntennaDesign(),
                    object(),
                    output,
                    868e6,
                    design_sheet=True,
                    jig_models=True,
                )

        drawing.assert_called_once()
        jigs.assert_called_once()
        self.assertEqual(artifacts["design_sheet"], "design_sheet.pdf")
        self.assertEqual(len(artifacts["jig_models"]), 1)
        self.assertEqual(
            Path(artifacts["jig_models"][0]),
            Path("jig_models/coil_01_winding_jig.stl"),
        )
        self.assertEqual(
            Path(artifacts["jig_manifest"]),
            Path("jig_models/jig_models.json"),
        )

    def test_model_and_mesh_viewer_flags_are_parsed(self):
        with patch(
            "sys.argv",
            ["verify_best.py", "--show-model", "--show-mesh"],
        ):
            args = parse_args()

        self.assertTrue(args.show_model)
        self.assertTrue(args.show_mesh)

    def test_viewer_flags_map_to_simulation_options(self):
        verification = options(
            MeshSettings(),
            angular_step=0.5,
            points=13,
            solver="auto",
            frequency_hz=868e6,
            sweep_bandwidth_hz=30e6,
            show_model=True,
            show_mesh=True,
        )

        self.assertTrue(verification.show_geometry)
        self.assertTrue(verification.show_mesh)

    def test_viewers_are_disabled_by_default(self):
        verification = options(
            MeshSettings(),
            angular_step=0.5,
            points=13,
            solver="auto",
            frequency_hz=868e6,
            sweep_bandwidth_hz=30e6,
        )

        self.assertFalse(verification.show_geometry)
        self.assertFalse(verification.show_mesh)


if __name__ == "__main__":
    unittest.main()
