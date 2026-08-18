from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from emerge_loaded_antenna import (
    AntennaDesign,
    DesignSpace,
    DesignVariable,
    EvaluationRecord,
    FrequencySweep,
    GainMatchObjective,
    SimulationOptions,
    build_centerline,
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

    def test_nested_design_space_mapping(self):
        base = AntennaDesign()
        space = DesignSpace(
            base,
            (
                DesignVariable("bottom_length", 100e-3, 180e-3),
                DesignVariable("coil1.pitch", 4e-3, 10e-3),
                DesignVariable("coil1.turns", 1, 4, kind="int"),
            ),
        )
        design = space.decode((150e-3, 8e-3, 2.4))

        self.assertAlmostEqual(design.bottom_length, 150e-3)
        self.assertAlmostEqual(design.coil1.pitch, 8e-3)
        self.assertEqual(design.coil1.turns, 2)
        self.assertEqual(space.names[1], "coil1.pitch")
        self.assertEqual(space.bounds[2], (1, 4))

    def test_normalized_vector_round_trip(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("middle_length", 180e-3, 260e-3),),
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

    def test_objective_reports_best_successful_record(self):
        space = DesignSpace(
            AntennaDesign(),
            (DesignVariable("middle_length", 180e-3, 260e-3),),
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
            (DesignVariable("middle_length", 180e-3, 260e-3),),
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


if __name__ == "__main__":
    unittest.main()
