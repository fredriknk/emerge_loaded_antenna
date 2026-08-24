from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from emerge_loaded_antenna import (
    AntennaDesign,
    MeshSettings,
    OpenRegionSettings,
)
from examples.verify_best import (
    export_fabrication_artifacts,
    latest_optimizer_result,
    options,
    parse_args,
    pattern_target_from_payload,
    result_summary,
    verification_quality,
)


class VerifyBestTests(unittest.TestCase):
    def test_latest_optimizer_result_uses_newest_campaign_winner(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "optimization_results"
            old = root / "old_run" / "campaign_best.json"
            new = root / "new_run" / "campaign_best.json"
            ignored = root / "newer_run" / "turns_1x1_best.json"
            for path in (old, new, ignored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            os.utime(old, (1_000_000_000, 1_000_000_000))
            os.utime(new, (1_100_000_000, 1_100_000_000))
            os.utime(ignored, (1_200_000_000, 1_200_000_000))

            selected = latest_optimizer_result(root)

        self.assertEqual(selected, new)

    def test_latest_cli_resolves_result_and_default_output(self):
        selected = Path("optimization_results/new_run/campaign_best.json")
        with (
            patch("sys.argv", ["verify_best.py", "--latest"]),
            patch(
                "examples.verify_best.latest_optimizer_result",
                return_value=selected,
            ) as latest,
        ):
            args = parse_args()

        latest.assert_called_once_with()
        self.assertEqual(args.result, selected)
        self.assertEqual(
            args.output,
            Path("optimization_results/new_run/campaign_best_verification"),
        )

    def test_latest_cli_rejects_explicit_result_and_missing_results(self):
        with (
            patch(
                "sys.argv",
                ["verify_best.py", "design.json", "--latest"],
            ),
            self.assertRaises(SystemExit),
        ):
            parse_args()

        with (
            patch("sys.argv", ["verify_best.py", "--latest"]),
            patch(
                "examples.verify_best.latest_optimizer_result",
                side_effect=FileNotFoundError("no campaign winners"),
            ),
            self.assertRaises(SystemExit),
        ):
            parse_args()

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
                "theta_degrees": (100.0,),
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
                "theta_degrees": (100.0,),
                "phi_deg": None,
                "beamwidth_deg": 50.0,
            },
        )

    def test_multiple_ring_targets_are_recovered_from_campaign_metadata(self):
        target = pattern_target_from_payload(
            {
                "simulation": {
                    "objective": {
                        "pattern_mode": "ring",
                        "target_theta_deg": 90.0,
                        "target_theta_degrees": [90.0, 130.0],
                    }
                }
            }
        )

        self.assertEqual(target["theta_deg"], 90.0)
        self.assertEqual(target["theta_degrees"], (90.0, 130.0))
        self.assertIsNone(target["phi_deg"])

    def test_verification_summary_reports_each_ring_and_the_weakest(self):
        rings = {
            90.0: SimpleNamespace(
                sampled_theta_deg=90.0,
                min_gain_dbi=3.0,
                p10_gain_dbi=4.0,
                mean_gain_dbi=4.2,
                p90_gain_dbi=5.0,
                peak_gain_dbi=5.2,
                ripple_p90_p10_db=1.0,
                peak_to_null_db=2.2,
            ),
            130.0: SimpleNamespace(
                sampled_theta_deg=130.0,
                min_gain_dbi=1.0,
                p10_gain_dbi=2.0,
                mean_gain_dbi=2.2,
                p90_gain_dbi=3.5,
                peak_gain_dbi=3.8,
                ripple_p90_p10_db=1.5,
                peak_to_null_db=2.8,
            ),
        }
        pattern = SimpleNamespace(
            peak_gain_dbi=5.2,
            peak_theta_deg=90.0,
            peak_phi_deg=0.0,
            peak_elevation_deg=0.0,
            horizon_min_gain_dbi=3.0,
            horizon_p10_gain_dbi=4.0,
            horizon_mean_gain_dbi=4.2,
            horizon_peak_gain_dbi=5.2,
            horizon_ripple_p90_p10_db=1.0,
            horizon_peak_to_null_db=2.2,
        )
        result = SimpleNamespace(
            farfield_metrics=pattern,
            s11_db_at=lambda frequency: -12.0,
            s11_db=np.asarray((-12.0, -11.0, -10.5)),
            frequencies=np.asarray((863e6, 868e6, 873e6)),
            antenna_height=0.5,
            artifacts=SimpleNamespace(
                mesh_nodes=10,
                mesh_elements=20,
                volume_elements=15,
                farfield_selection=SimpleNamespace(tags=(1, 2)),
                termination_selection=None,
                outer_boundary_tags=(3, 4),
            ),
            options=SimpleNamespace(
                open_region=OpenRegionSettings(),
                mesh=MeshSettings(),
            ),
            azimuth_ring_metrics=lambda theta: rings[theta],
            azimuth_ring_beamwidth_deg=lambda theta: {
                90.0: 50.0,
                130.0: 70.0,
            }[theta],
        )

        summary = result_summary(
            result,
            868e6,
            {
                "mode": "ring",
                "theta_deg": 90.0,
                "theta_degrees": (90.0, 130.0),
                "phi_deg": None,
                "beamwidth_deg": 60.0,
            },
        )

        self.assertEqual(summary["ring_p10_gain_dbi"], 2.0)
        self.assertEqual(summary["ring_worst_target_theta_deg"], 130.0)
        self.assertEqual(
            [ring["p10_gain_dbi"] for ring in summary["rings"]],
            [4.0, 2.0],
        )
        self.assertEqual(summary["ring_beamwidths_deg"], [50.0, 70.0])

    def test_verification_quality_checks_matching_and_mesh_agreement(self):
        quality = verification_quality(
            {
                "fine": {
                    "worst_s11_db": -9.5,
                    "target_beamwidth_deg": 50.0,
                    "ring_beamwidths_deg": [53.0, 46.0],
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
        self.assertEqual(
            quality["observations"]["beamwidth"]["errors_deg"],
            [3.0, -4.0],
        )
        self.assertEqual(
            quality["observations"]["beamwidth"]["rms_error_deg"],
            2.5*np.sqrt(2),
        )

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
