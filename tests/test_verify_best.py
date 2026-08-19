from __future__ import annotations

import unittest
from unittest.mock import patch

from emerge_loaded_antenna import MeshSettings
from examples.verify_best import options, parse_args


class VerifyBestTests(unittest.TestCase):
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
