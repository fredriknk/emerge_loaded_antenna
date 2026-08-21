from __future__ import annotations

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from emerge_loaded_antenna import (
    CONVERGENCE_SCHEMA_VERSION,
    SOLVER_CHOICES,
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
    SimulationOptions,
    build_centerline,
    build_model,
    design_fingerprint,
    design_from_dict,
    selected_open_region_configuration,
    validate_convergence_certificate,
)
from emerge_loaded_antenna.simulation import _configure_solver


def _robust_result(
    useful_gain_dbi: float = 4.0,
    s11_db: tuple[float, float, float] = (-12.0, -11.0, -10.5),
):
    pattern = SimpleNamespace(
        horizon_p10_gain_dbi=useful_gain_dbi,
        horizon_min_gain_dbi=3.0,
        horizon_mean_gain_dbi=useful_gain_dbi + 0.2,
        horizon_ripple_p90_p10_db=1.0,
        peak_theta_deg=90.0,
        peak_phi_deg=0.0,
    )
    values = np.asarray(s11_db, dtype=float)
    return SimpleNamespace(
        peak_gain_dbi=useful_gain_dbi,
        farfield_metrics=pattern,
        frequencies=np.array((863e6, 868e6, 873e6)),
        s11_db=values,
        s11_db_at=lambda frequency: float(values[1]),
        antenna_height=0.55,
        gain_db_at=lambda theta, phi: useful_gain_dbi,
    )


class DesignTests(unittest.TestCase):
    def test_transition_offset_is_derived_and_tracks_replacement(self):
        coil = CoilDesign(transition=12e-3, transition_offset=1e-3)

        self.assertAlmostEqual(coil.transition_offset, 12e-3*19/24)

        resized = replace(coil, transition=6e-3)
        self.assertAlmostEqual(resized.transition_offset, 4.75e-3)

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            CoilDesign(transition=float("nan")).validate()

    def test_transition_must_be_at_least_five_quarters_wire_radius(self):
        at_limit = AntennaDesign(
            wire_radius=0.8e-3,
            straight_lengths=(0.1, 0.1),
            coils=(CoilDesign(transition=1e-3),),
        )
        at_limit.validate()

        below_limit = replace(
            at_limit,
            coils=(replace(at_limit.coils[0], transition=0.99e-3),),
        )
        with self.assertRaisesRegex(ValueError, "at least 1.25 times wire_radius"):
            below_limit.validate()

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

    def test_linked_design_variable_updates_multiple_fields(self):
        base = AntennaDesign(
            straight_lengths=(0.1, 0.1, 0.1, 0.1),
            coils=(CoilDesign(), CoilDesign(), CoilDesign()),
        )
        space = DesignSpace(
            base,
            (
                DesignVariable(
                    "coils.0.pitch",
                    4e-3,
                    12e-3,
                    linked_paths=("coils.1.pitch", "coils.2.pitch"),
                    label="shared_coil_pitch",
                ),
            ),
        )

        design = space.decode((9e-3,))

        self.assertEqual(space.names, ("shared_coil_pitch",))
        self.assertTrue(
            all(np.isclose(coil.pitch, 9e-3) for coil in design.coils)
        )

    def test_linked_design_variable_paths_cannot_overlap(self):
        with self.assertRaisesRegex(ValueError, "paths must be unique"):
            DesignSpace(
                AntennaDesign(),
                (
                    DesignVariable(
                        "coils.0.pitch",
                        4e-3,
                        12e-3,
                        linked_paths=("coils.1.pitch",),
                    ),
                    DesignVariable("coils.1.pitch", 4e-3, 12e-3),
                ),
            )

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

    def test_legacy_transition_offset_json_is_normalized(self):
        values = asdict(AntennaDesign())
        values["coils"][0]["transition"] = 12e-3
        values["coils"][0]["transition_offset"] = 1e-6

        restored = design_from_dict(values)

        self.assertAlmostEqual(restored.coils[0].transition_offset, 9.5e-3)
        self.assertAlmostEqual(
            asdict(restored)["coils"][0]["transition_offset"],
            9.5e-3,
        )

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
            "schema_version": CONVERGENCE_SCHEMA_VERSION,
            "passed": True,
            "frequency_hz": 868e6,
            "farfield_angular_step_deg": 2.0,
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
                farfield_angular_step_deg=2.0,
            )
            with self.assertRaisesRegex(RuntimeError, "angular step"):
                validate_convergence_certificate(
                    path,
                    design,
                    mesh,
                    open_region,
                    868e6,
                    farfield_angular_step_deg=4.0,
                )

        self.assertTrue(restored["passed"])

    def test_mismatched_convergence_certificate_is_rejected(self):
        design = AntennaDesign()
        mesh = MeshSettings(wavelength_resolution=0.33)
        open_region = OpenRegionSettings()
        payload = {
            "schema_version": CONVERGENCE_SCHEMA_VERSION,
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

        with patch(
            "emerge_loaded_antenna.optimize.simulate",
            return_value=fake_result,
        ) as simulation:
            score = objective((220e-3,))

        self.assertEqual(score, -4.0)
        simulation.assert_called_once()
        self.assertEqual(
            objective.best_record.confirmation_status,
            "not_requested",
        )
        self.assertEqual(objective.best_record.metrics["worst_s11_db"], -10.5)
        self.assertEqual(objective.best_record.metrics["s11_low_db"], -12.0)
        self.assertEqual(objective.best_record.metrics["s11_high_db"], -10.5)

    def test_robust_objective_rewards_worst_band_s11_margin_to_target(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("straight_lengths.1", 180e-3, 260e-3),),
        )
        cases = (
            ((-12.0, -11.0, -10.0), 0.0),
            ((-13.0, -12.0, -11.0), 1.0),
            ((-15.0, -14.0, -13.0), 2.0),
        )

        for pattern_mode in ("horizon", "directional", "peak"):
            for s11_values, expected_margin in cases:
                with self.subTest(
                    pattern_mode=pattern_mode,
                    s11_values=s11_values,
                ):
                    objective = RobustGainObjective(
                        space,
                        pattern_mode=pattern_mode,
                        s11_margin_target_db=-12.0,
                        s11_margin_weight=0.5,
                    )
                    with patch(
                        "emerge_loaded_antenna.optimize.simulate",
                        return_value=_robust_result(s11_db=s11_values),
                    ):
                        score = objective((220e-3,))

                    expected_reward = 0.5*expected_margin
                    self.assertAlmostEqual(score, -4.0 - expected_reward)
                    self.assertAlmostEqual(
                        objective.best_record.metrics["s11_margin_db"],
                        expected_margin,
                    )
                    self.assertAlmostEqual(
                        objective.best_record.metrics["s11_margin_reward"],
                        expected_reward,
                    )

    def test_robust_objective_confirms_and_quarantines_new_incumbents(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("straight_lengths.1", 180e-3, 260e-3),),
        )
        reported = []
        confirmations = []
        objective = RobustGainObjective(
            space,
            confirmation_runs=3,
            confirmation_score_tolerance=0.5,
            on_evaluation=reported.append,
            on_confirmation=confirmations.append,
        )

        with patch(
            "emerge_loaded_antenna.optimize.simulate",
            side_effect=(
                _robust_result(4.0),
                _robust_result(4.2),
                _robust_result(4.1),
            ),
        ) as simulation:
            confirmed_score = objective((220e-3,))

        self.assertAlmostEqual(confirmed_score, -4.1)
        self.assertEqual(simulation.call_count, 3)
        self.assertEqual(len(objective.history), 1)
        self.assertEqual(objective.history[0].simulation_runs, 3)
        self.assertEqual(objective.simulation_evaluations, 3)
        self.assertEqual(reported, objective.history)
        self.assertEqual(len(confirmations), 1)
        self.assertEqual(confirmations[0].status, "confirmed")
        self.assertEqual(len(confirmations[0].records), 3)
        self.assertEqual(objective.best_record.confirmation_status, "confirmed")
        self.assertAlmostEqual(
            objective.best_record.metrics["confirmation_confirmed_score"],
            -4.1,
        )

        with patch(
            "emerge_loaded_antenna.optimize.simulate",
            side_effect=(
                _robust_result(30.0),
                _robust_result(3.0),
                _robust_result(3.0),
            ),
        ) as simulation:
            quarantined_score = objective((221e-3,))

        self.assertAlmostEqual(quarantined_score, -3.0)
        self.assertEqual(simulation.call_count, 3)
        self.assertEqual(len(objective.history), 2)
        self.assertEqual(len(reported), 2)
        self.assertEqual(len(confirmations), 2)
        self.assertEqual(confirmations[-1].status, "confirmed_with_outliers")
        self.assertAlmostEqual(confirmations[-1].confirmed_score, -3.0)
        self.assertEqual(confirmations[-1].outlier_runs, 1)
        self.assertIsNone(confirmations[-1].reason)
        self.assertEqual(
            objective.history[-1].confirmation_status,
            "confirmed_with_outliers",
        )
        self.assertIsNone(objective.history[-1].error)
        self.assertIs(objective.best_record, objective.history[0])
        self.assertEqual(objective.simulation_evaluations, 6)

        with patch(
            "emerge_loaded_antenna.optimize.simulate",
            side_effect=(
                _robust_result(30.0),
                _robust_result(3.0),
                _robust_result(2.0),
            ),
        ) as simulation:
            quarantined_score = objective((222e-3,))

        self.assertEqual(quarantined_score, objective.failure_penalty)
        self.assertEqual(simulation.call_count, 3)
        self.assertEqual(len(objective.history), 3)
        self.assertEqual(len(reported), 3)
        self.assertEqual(len(confirmations), 3)
        self.assertEqual(confirmations[-1].status, "quarantined")
        self.assertIn("agreed with the median", confirmations[-1].reason)
        self.assertEqual(
            objective.history[-1].confirmation_status,
            "quarantined",
        )
        self.assertIsNotNone(objective.history[-1].error)
        self.assertIs(objective.best_record, objective.history[0])
        self.assertEqual(objective.simulation_evaluations, 9)

        with patch(
            "emerge_loaded_antenna.optimize.simulate",
            return_value=_robust_result(3.0),
        ) as simulation:
            objective((223e-3,))

        simulation.assert_called_once()
        self.assertEqual(
            objective.history[-1].confirmation_status,
            "not_needed",
        )
        self.assertEqual(len(reported), 4)
        self.assertEqual(len(confirmations), 3)
        self.assertEqual(objective.simulation_evaluations, 10)

    def test_robust_objective_validates_confirmation_and_margin_settings(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("straight_lengths.1", 180e-3, 260e-3),),
        )

        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            RobustGainObjective(space, confirmation_runs=2)
        with self.assertRaisesRegex(ValueError, "below maximum_s11_db"):
            RobustGainObjective(space, s11_margin_target_db=-9.0)
        with self.assertRaisesRegex(ValueError, "maximum_s11_db must be finite"):
            RobustGainObjective(space, maximum_s11_db=float("nan"))

    def test_robust_objective_confirms_new_best_feasible_candidate(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("straight_lengths.1", 180e-3, 260e-3),),
        )
        infeasible = _robust_result(30.0, (-9.0, -9.0, -9.0))
        feasible = _robust_result(4.0, (-12.0, -11.0, -10.5))
        objective = RobustGainObjective(
            space,
            confirmation_runs=3,
            confirmation_score_tolerance=0.5,
        )

        with patch(
            "emerge_loaded_antenna.optimize.simulate",
            side_effect=(infeasible, infeasible, infeasible),
        ):
            infeasible_score = objective((220e-3,))
        with patch(
            "emerge_loaded_antenna.optimize.simulate",
            side_effect=(feasible, feasible, feasible),
        ) as simulation:
            feasible_score = objective((221e-3,))

        self.assertLess(infeasible_score, feasible_score)
        self.assertIs(objective.best_record, objective.history[0])
        self.assertIs(objective.best_feasible_record, objective.history[1])
        self.assertEqual(objective.history[1].confirmation_status, "confirmed")
        self.assertEqual(objective.history[1].simulation_runs, 3)
        self.assertEqual(objective.simulation_evaluations, 6)
        self.assertEqual(simulation.call_count, 3)


if __name__ == "__main__":
    unittest.main()
