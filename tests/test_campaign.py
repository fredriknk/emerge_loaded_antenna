from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from emerge_loaded_antenna import (
    AntennaDesign,
    EvaluationRecord,
    MeshSettings,
    OpenRegionSettings,
    REFERENCE_868MHZ_DESIGN_PATH,
    load_reference_868mhz_design,
)
from examples.optimize_gain import (
    CampaignProgress,
    design_for_coil_count,
    ensure_convergence_certificate,
    iterations_per_run,
    load_baseline,
    make_space,
    parse_args,
    parse_turn_cases,
)


class CampaignTests(unittest.TestCase):
    def test_fresh_clone_uses_tracked_reference_design(self):
        with patch("sys.argv", ["optimize_gain.py"]):
            args = parse_args()

        self.assertEqual(args.warm_start, REFERENCE_868MHZ_DESIGN_PATH)
        self.assertTrue(args.warm_start.is_file())
        self.assertEqual(
            load_baseline(args.warm_start),
            load_reference_868mhz_design(),
        )

    def test_missing_warm_start_does_not_silently_change_design(self):
        with TemporaryDirectory() as temporary:
            missing = Path(temporary)/"missing.json"
            with self.assertRaisesRegex(SystemExit, "WARM START FAILED"):
                load_baseline(missing)

    def test_twelve_hour_budget_is_split_across_seeds(self):
        args = Namespace(
            maxiter=None,
            hours=12.0,
            seconds_per_eval=8.0,
            popsize=8,
        )

        self.assertEqual(iterations_per_run(args, variables=9, run_count=4), 17)

    def test_turn_cases_are_parsed_as_discrete_searches(self):
        self.assertEqual(
            parse_turn_cases("1x1,1x2,2x1"),
            ((1, 1), (1, 2), (2, 1)),
        )
        self.assertEqual(parse_turn_cases("none"), ((),))
        self.assertEqual(parse_turn_cases("1x2x3"), ((1, 2, 3),))

    def test_design_can_be_resized_to_any_coil_count(self):
        base = AntennaDesign()
        zero = design_for_coil_count(base, 0)
        three = design_for_coil_count(base, 3)

        self.assertEqual(zero.coil_count, 0)
        self.assertEqual(len(zero.straight_lengths), 1)
        self.assertAlmostEqual(sum(zero.straight_lengths), sum(base.straight_lengths))
        self.assertEqual(three.coil_count, 3)
        self.assertEqual(len(three.straight_lengths), 4)
        self.assertEqual(len(make_space(zero).variables), 3)
        self.assertEqual(len(make_space(three).variables), 12)

    def test_cli_accepts_zero_and_arbitrary_coil_counts(self):
        with patch("sys.argv", ["optimize_gain.py", "--coil-count", "0"]):
            zero = parse_args()
        with patch(
            "sys.argv",
            [
                "optimize_gain.py",
                "--coil-count",
                "3",
                "--turn-cases",
                "1x2x1,2x2x1",
                "--solver",
                "cudss",
            ],
        ):
            three = parse_args()

        self.assertEqual(zero.turn_cases, ((),))
        self.assertEqual(three.turn_cases, ((1, 2, 1), (2, 2, 1)))
        self.assertEqual(three.solver, "cudss")

    def test_missing_certificate_is_generated_automatically(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                convergence_report=root/"convergence.json",
                warm_start=root/"campaign_best.json",
                output=root/"campaign",
                solver="auto",
                no_auto_convergence=False,
            )
            certificate = {"passed": True}
            with (
                patch(
                    "examples.optimize_gain.validate_convergence_certificate",
                    side_effect=(RuntimeError("not found"), certificate),
                ) as validate,
                patch(
                    "examples.optimize_gain.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ) as run,
            ):
                result = ensure_convergence_certificate(
                    args,
                    AntennaDesign(),
                    MeshSettings(),
                    OpenRegionSettings(),
                )

            self.assertEqual(result, certificate)
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertIn("check_open_region.py", command[2])
            self.assertIn("--selected-resolution", command)
            self.assertTrue((args.output/"baseline_design.json").exists())

    def test_matching_certificate_is_reused_without_subprocess(self):
        args = Namespace(
            convergence_report=Path("convergence.json"),
            warm_start=Path("campaign_best.json"),
            output=Path("campaign"),
            solver="auto",
            no_auto_convergence=False,
        )
        certificate = {"passed": True}
        with (
            patch(
                "examples.optimize_gain.validate_convergence_certificate",
                return_value=certificate,
            ),
            patch("examples.optimize_gain.subprocess.run") as run,
        ):
            result = ensure_convergence_certificate(
                args,
                AntennaDesign(),
                MeshSettings(),
                OpenRegionSettings(),
            )

        self.assertEqual(result, certificate)
        run.assert_not_called()

    def test_no_auto_convergence_preserves_fail_fast_behavior(self):
        args = Namespace(
            convergence_report=Path("missing.json"),
            warm_start=Path("campaign_best.json"),
            output=Path("campaign"),
            solver="auto",
            no_auto_convergence=True,
        )
        with (
            patch(
                "examples.optimize_gain.validate_convergence_certificate",
                side_effect=RuntimeError("not found"),
            ),
            patch("examples.optimize_gain.subprocess.run") as run,
            self.assertRaisesRegex(SystemExit, "Automatic convergence is disabled"),
        ):
            ensure_convergence_certificate(
                args,
                AntennaDesign(),
                MeshSettings(),
                OpenRegionSettings(),
            )

        run.assert_not_called()

    def test_progress_checkpoints_each_new_best(self):
        space = make_space(AntennaDesign())
        record = EvaluationRecord(
            tuple(space.initial_vector),
            -3.0,
            -11.0,
            4.0,
            metrics={
                "worst_s11_db": -10.5,
                "horizon_p10_gain_dbi": 3.0,
            },
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            progress = CampaignProgress(
                output,
                total=1,
                report_every=1,
                variable_names=space.names,
            )
            progress.set_context(space, (1, 1), seed=2)
            progress(record)
            progress.close()

            payload = json.loads(
                (output/"campaign_best.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["objective"], -3.0)
            self.assertEqual(payload["turn_case"], [1, 1])
            self.assertEqual(len((output/"evaluations.csv").read_text().splitlines()), 2)

    def test_zero_coil_progress_uses_none_case(self):
        space = make_space(design_for_coil_count(AntennaDesign(), 0))
        record = EvaluationRecord(
            tuple(space.initial_vector),
            -1.0,
            -12.0,
            2.0,
            metrics={"worst_s11_db": -11.0, "horizon_p10_gain_dbi": 1.0},
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            progress = CampaignProgress(
                output,
                total=1,
                report_every=1,
                variable_names=space.names,
            )
            progress.set_context(space, (), seed=2)
            progress(record)
            progress.close()

            payload = json.loads(
                (output/"campaign_best.json").read_text(encoding="utf-8")
            )
            rows = (output/"evaluations.csv").read_text().splitlines()
            self.assertEqual(payload["coil_count"], 0)
            self.assertEqual(payload["turn_case"], [])
            self.assertIn("none", rows[1])


if __name__ == "__main__":
    unittest.main()
