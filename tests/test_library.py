from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from emerge_loaded_antenna import (
    AntennaDesign,
    CoilDesign,
    DesignSpace,
    DesignVariable,
    EvaluationRecord,
    FrequencySweep,
    GainMatchObjective,
    RobustGainObjective,
    SimulationOptions,
    build_centerline,
    design_from_dict,
)


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

    def test_straight_section_count_must_match_coils(self):
        design = AntennaDesign(
            straight_lengths=(100e-3,),
            coils=(CoilDesign(),),
        )
        with self.assertRaisesRegex(ValueError, "exactly one more"):
            design.validate()

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
