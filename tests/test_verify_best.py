from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from emerge_loaded_antenna import AntennaDesign, MeshSettings
from examples.verify_best import (
    export_fabrication_artifacts,
    options,
    parse_args,
    pattern_target_from_payload,
    verification_quality,
)


class VerifyBestTests(unittest.TestCase):
    def test_directional_target_is_recovered_from_campaign_metadata(self):
        target = pattern_target_from_payload(
            {
                "simulation": {
                    "objective": {
                        "pattern_mode": "directional",
                        "target_theta_deg": 100.0,
                        "target_phi_deg": 20.0,
                        "target_beamwidth_deg": 55.0,
                    }
                }
            }
        )

        self.assertEqual(
            target,
            {
                "mode": "directional",
                "theta_deg": 100.0,
                "phi_deg": 20.0,
                "beamwidth_deg": 55.0,
            },
        )
        self.assertIsNone(
            pattern_target_from_payload(
                {"simulation": {"objective": {"pattern_mode": "horizon"}}}
            )
        )

    def test_ring_target_is_recovered_without_phi(self):
        target = pattern_target_from_payload(
            {
                "simulation": {
                    "objective": {
                        "pattern_mode": "ring",
                        "target_theta_deg": 100.0,
                        "target_phi_deg": 45.0,
                        "target_beamwidth_deg": 50.0,
                    }
                }
            }
        )

        self.assertEqual(
            target,
            {
                "mode": "ring",
                "theta_deg": 100.0,
                "phi_deg": None,
                "beamwidth_deg": 50.0,
            },
        )

    def test_verification_quality_checks_matching_and_mesh_agreement(self):
        quality = verification_quality(
            {
                "fine": {
                    "worst_s11_db": -9.5,
                    "target_beamwidth_deg": 50.0,
                    "ring_beamwidth_deg": 53.0,
                },
                "convergence": {
                    "peak_gain_delta_db": 0.2,
                    "s11_target_delta_db": 0.7,
                },
            },
            {
                "simulation": {
                    "objective": {"maximum_s11_db": -10.0}
                }
            },
        )

        self.assertEqual(quality["status"], "warning")
        self.assertFalse(quality["checks"]["fine_worst_s11"]["passed"])
        self.assertFalse(quality["checks"]["coarse_fine_agreement"]["passed"])
        self.assertEqual(quality["observations"]["beamwidth"]["error_deg"], 3.0)

    def test_verification_quality_warns_for_uncertified_open_region(self):
        quality = verification_quality(
            {"fine": {"worst_s11_db": -12.0}},
            {
                "simulation": {
                    "convergence_status": "warning",
                    "convergence_warning": "certificate did not pass",
                    "objective": {"maximum_s11_db": -10.0},
                }
            },
        )

        self.assertEqual(quality["status"], "warning")
        self.assertFalse(quality["checks"]["open_region_preflight"]["passed"])

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
            former_path = output / "coil_formers.step"
            with (
                patch(
                    "examples.verify_best.export_drawing",
                    return_value=output / "design_sheet.pdf",
                ) as drawing,
                patch(
                    "examples.verify_best.export_coil_formers",
                    return_value=former_path,
                ) as formers,
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
        formers.assert_called_once_with(
            AntennaDesign(),
            output / "coil_formers.step",
        )
        self.assertEqual(artifacts["design_sheet"], "design_sheet.pdf")
        self.assertEqual(len(artifacts["jig_models"]), 1)
        self.assertEqual(
            Path(artifacts["jig_models"][0]),
            Path("coil_formers.step"),
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
