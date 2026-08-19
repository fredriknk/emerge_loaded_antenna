from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from emerge_loaded_antenna import (
    AntennaDesign,
    CoilDesign,
    DesignSpace,
    DesignVariable,
    EvaluationRecord,
    FrequencySweep,
    GainMatchObjective,
    MeshSettings,
    OpenRegionSettings,
    RobustGainObjective,
    SOLVER_CHOICES,
    SimulationOptions,
    build_centerline,
    build_model,
    design_fingerprint,
    design_from_dict,
    selected_open_region_configuration,
    validate_convergence_certificate,
)
from emerge_loaded_antenna.simulation import _configure_solver


class DesignTests(unittest.TestCase):
    def test_default_centerline_is_compact_and_returns_to_axis(self):
        design = AntennaDesign()
        path = build_centerline(design)
        x, y, z = path.arrays()

        self.assertEqual(len(path.segments), 13)
        self.assertEqual(sum(kind == "line" for kind, _ in path.segments), 3)
        self.assertTrue(np.isclose(x[0], 0.0))
        self.assertTrue(np.isclose(y[0], 0.0))
        self.assertTrue(np.isclose(x[-1], 0.0))
        self.assertTrue(np.isclose(y[-1], 0.0))
        self.assertGreater(z[-1], z[0])

    def test_zero_coil_centerline_is_one_straight_section(self):
        design = AntennaDesign(straight_lengths=(86e-3,), coils=())
        path = build_centerline(design)
        x, y, z = path.arrays()

        self.assertEqual(design.coil_count, 0)
        self.assertEqual(len(path.segments), 1)
        self.assertEqual(path.segments[0][0], "line")
        self.assertTrue(np.allclose((x[-1], y[-1]), (0.0, 0.0)))
        self.assertAlmostEqual(z[-1] - z[0], 86e-3)

    def test_three_coil_centerline_returns_to_axis(self):
        design = AntennaDesign(
            straight_lengths=(60e-3, 70e-3, 80e-3, 90e-3),
            coils=(CoilDesign(), CoilDesign(), CoilDesign()),
        )
        path = build_centerline(design)
        x, y, _ = path.arrays()

        self.assertEqual(design.coil_count, 3)
        self.assertEqual(sum(kind == "line" for kind, _ in path.segments), 4)
        self.assertTrue(np.isclose(x[-1], 0.0))
        self.assertTrue(np.isclose(y[-1], 0.0))

    def test_nested_design_space_mapping(self):
        base = AntennaDesign()
        space = DesignSpace(
            base,
            (
                DesignVariable("straight_lengths.0", 100e-3, 180e-3),
                DesignVariable("coils.0.pitch", 4e-3, 10e-3),
                DesignVariable("coils.0.turns", 1, 4, kind="int"),
            ),
        )
        design = space.decode((150e-3, 8e-3, 2.4))

        self.assertAlmostEqual(design.straight_lengths[0], 150e-3)
        self.assertAlmostEqual(design.coils[0].pitch, 8e-3)
        self.assertEqual(design.coils[0].turns, 2)
        self.assertEqual(space.names[1], "coils.0.pitch")
        self.assertEqual(space.bounds[2], (1, 4))

    def test_normalized_vector_round_trip(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("straight_lengths.1", 180e-3, 260e-3),),
        )
        vector = np.array((220e-3,))
        np.testing.assert_allclose(
            space.denormalize(space.normalize(vector)),
            vector,
        )

    def test_invalid_design_fails_before_meshing(self):
        design = replace(AntennaDesign(), wire_radius=-1e-3)
        with self.assertRaisesRegex(ValueError, "wire_radius"):
            design.validate()

    def test_single_frequency_options(self):
        sweep = FrequencySweep.single(868e6)
        options = SimulationOptions(sweep=sweep, compute_farfield=False)
        options.validate()
        self.assertEqual(sweep.start, sweep.stop)
        self.assertEqual(sweep.points, 1)

    def test_farfield_angular_step_is_validated(self):
        options = SimulationOptions(farfield_angular_step_deg=0.0)
        with self.assertRaisesRegex(ValueError, "angular_step"):
            options.validate()

    def test_solver_selection_is_validated(self):
        self.assertIn("cudss", SOLVER_CHOICES)
        SimulationOptions(solver="cudss").validate()
        with self.assertRaisesRegex(ValueError, "solver must be one of"):
            SimulationOptions(solver="cuda").validate()

    def test_open_region_settings_are_validated(self):
        OpenRegionSettings(mode="abc").validate()
        OpenRegionSettings(mode="pml").validate()
        with self.assertRaisesRegex(ValueError, "open-region mode"):
            OpenRegionSettings(mode="reflector").validate()
        with self.assertRaisesRegex(ValueError, "pml_mesh_layers"):
            OpenRegionSettings(pml_mesh_layers=1).validate()

    def test_abc_has_closed_separate_huygens_and_termination_surfaces(self):
        design = AntennaDesign(
            radial_length=55e-3,
            straight_lengths=(50e-3,),
            coils=(),
        )
        options = SimulationOptions(
            mesh=MeshSettings(
                wavelength_resolution=0.5,
                air_margin_wavelengths=0.10,
            ),
            open_region=OpenRegionSettings(
                mode="abc",
                abc_buffer_wavelengths=0.10,
            ),
            solve=False,
            verbose=False,
        )

        artifacts = build_model(design, options)

        self.assertEqual(len(artifacts.farfield_selection.tags), 6)
        self.assertEqual(len(artifacts.open_region_volumes), 7)
        self.assertEqual(
            set(artifacts.termination_selection.tags),
            set(artifacts.outer_boundary_tags),
        )
        self.assertFalse(
            set(artifacts.farfield_selection.tags)
            & set(artifacts.termination_selection.tags)
        )

    def test_solver_selection_maps_to_emerge_and_reports_missing_backend(self):
        model = SimpleNamespace(set_solver=Mock())
        _configure_solver(model, "cudss")
        self.assertEqual(model.set_solver.call_args.args[0].name, "CUDSS")

        model.set_solver.side_effect = KeyError("cudss")
        with self.assertRaisesRegex(RuntimeError, "install-solver cudss"):
            _configure_solver(model, "cudss")

    def test_design_json_mapping_round_trip(self):
        original = replace(
            AntennaDesign(),
            coils=(
                replace(AntennaDesign().coils[0], turns=2, radius=12e-3),
                AntennaDesign().coils[1],
            ),
        )
        restored = design_from_dict(asdict(original))

        self.assertEqual(restored, original)

    def test_unknown_design_fields_are_rejected(self):
        values = asdict(AntennaDesign())
        values["unknown_parameter"] = 1.0

        with self.assertRaisesRegex(ValueError, "unsupported AntennaDesign fields"):
            design_from_dict(values)

    def test_unknown_coil_fields_are_rejected(self):
        values = asdict(AntennaDesign())
        values["coils"][0]["unknown_parameter"] = 1.0

        with self.assertRaisesRegex(ValueError, "unsupported CoilDesign fields"):
            design_from_dict(values)

    def test_straight_section_count_must_match_coils(self):
        design = AntennaDesign(
            straight_lengths=(100e-3,),
            coils=(CoilDesign(),),
        )
        with self.assertRaisesRegex(ValueError, "exactly one more"):
            design.validate()

    def test_matching_convergence_certificate_is_accepted(self):
        design = AntennaDesign()
        mesh = replace(
            MeshSettings(),
            wavelength_resolution=0.33,
            air_margin_wavelengths=0.25,
        )
        open_region = OpenRegionSettings()
        payload = {
            "schema_version": 1,
            "passed": True,
            "frequency_hz": 868e6,
            "design_fingerprint": design_fingerprint(design),
            "selected_configuration": selected_open_region_configuration(
                mesh, open_region
            ),
            "samples": {
                "selected": {"peak_gain_dbi": 4.0},
                "air": {"peak_gain_dbi": 4.0},
                "boundary": {"peak_gain_dbi": 4.0},
                "mesh": {"peak_gain_dbi": 4.0},
            },
            "comparisons": [
                {"passed": True, "selected": "selected", "reference": "air"},
                {
                    "passed": True,
                    "selected": "selected",
                    "reference": "boundary",
                },
                {"passed": True, "selected": "selected", "reference": "mesh"},
            ],
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"certificate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored = validate_convergence_certificate(
                path,
                design,
                mesh,
                open_region,
                868e6,
            )

        self.assertTrue(restored["passed"])

    def test_mismatched_convergence_certificate_is_rejected(self):
        design = AntennaDesign()
        mesh = MeshSettings(wavelength_resolution=0.33)
        open_region = OpenRegionSettings()
        payload = {
            "schema_version": 1,
            "passed": True,
            "frequency_hz": 868e6,
            "design_fingerprint": design_fingerprint(design),
            "selected_configuration": selected_open_region_configuration(
                mesh, open_region
            ),
            "samples": {
                "selected": {},
                "air": {},
                "boundary": {},
                "mesh": {},
            },
            "comparisons": [
                {"passed": True, "selected": "selected", "reference": "air"},
                {
                    "passed": True,
                    "selected": "selected",
                    "reference": "boundary",
                },
                {"passed": True, "selected": "selected", "reference": "mesh"},
            ],
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"certificate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                validate_convergence_certificate(
                    path,
                    design,
                    replace(mesh, wavelength_resolution=0.25),
                    open_region,
                    868e6,
                )

    def test_objective_reports_best_successful_record(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("straight_lengths.1", 180e-3, 260e-3),),
        )
        objective = GainMatchObjective(space)
        objective.history.extend(
            (
                EvaluationRecord((0.20,), 1000.0, None, None, "mesh failed"),
                EvaluationRecord((0.21,), -3.0, -11.0, 3.0),
                EvaluationRecord((0.22,), -4.5, -10.5, 4.5),
            )
        )

        self.assertEqual(objective.best_record, objective.history[-1])

    def test_objective_calls_progress_callback(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("straight_lengths.1", 180e-3, 260e-3),),
        )
        reported = []
        objective = GainMatchObjective(space, on_evaluation=reported.append)
        fake_result = SimpleNamespace(
            peak_gain_dbi=4.0,
            s11_db_at=lambda frequency: -12.0,
        )

        with patch("emerge_loaded_antenna.optimize.simulate", return_value=fake_result):
            score = objective((220e-3,))

        self.assertEqual(score, -4.0)
        self.assertEqual(reported, objective.history)

    def test_robust_objective_uses_horizon_and_band_metrics(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("straight_lengths.1", 180e-3, 260e-3),),
        )
        pattern = SimpleNamespace(
            horizon_p10_gain_dbi=4.0,
            horizon_min_gain_dbi=3.0,
            horizon_mean_gain_dbi=4.2,
            horizon_ripple_p90_p10_db=1.0,
            peak_theta_deg=90.0,
            peak_phi_deg=0.0,
        )
        fake_result = SimpleNamespace(
            peak_gain_dbi=5.0,
            farfield_metrics=pattern,
            frequencies=np.array((863e6, 868e6, 873e6)),
            s11_db=np.array((-12.0, -11.0, -10.5)),
            s11_db_at=lambda frequency: -11.0,
            antenna_height=0.55,
            gain_db_at=lambda theta, phi: 4.5,
        )
        objective = RobustGainObjective(space)

        with patch("emerge_loaded_antenna.optimize.simulate", return_value=fake_result):
            score = objective((220e-3,))

        self.assertEqual(score, -4.0)
        self.assertEqual(objective.best_record.metrics["worst_s11_db"], -10.5)


if __name__ == "__main__":
    unittest.main()
