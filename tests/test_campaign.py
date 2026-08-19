from __future__ import annotations

from argparse import ArgumentTypeError, Namespace
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from emerge_loaded_antenna import (
    AntennaDesign,
    CoilDesign,
    EvaluationRecord,
    MeshSettings,
    OpenRegionSettings,
    REFERENCE_DESIGN_FREQUENCY_HZ,
    load_reference_design,
)
from examples.optimize_gain import (
    CampaignProgress,
    build_case_schedules,
    design_for_coil_count,
    ensure_convergence_certificate,
    iterations_per_run,
    load_baseline,
    make_space,
    parse_args,
    parse_coil_counts,
    parse_turn_cases,
    resolve_topology,
    run_campaign,
)


class CampaignTests(unittest.TestCase):
    def test_fresh_clone_synthesizes_a_frequency_scaled_design(self):
        with patch("sys.argv", ["optimize_gain.py"]):
            args = parse_args()

        self.assertIsNone(args.warm_start)
        self.assertIsNone(args.coil_count)
        self.assertIsNone(args.coil_counts)
        self.assertIsNone(args.turn_cases)
        self.assertEqual(
            load_baseline(None, args.frequency_hz),
            load_reference_design(args.frequency_hz),
        )

        half_frequency = REFERENCE_DESIGN_FREQUENCY_HZ/2
        scaled = load_baseline(None, half_frequency)
        reference = load_reference_design()
        self.assertAlmostEqual(scaled.wire_radius, 2*reference.wire_radius)
        self.assertAlmostEqual(scaled.radial_length, 2*reference.radial_length)
        self.assertAlmostEqual(
            sum(scaled.straight_lengths),
            2*sum(reference.straight_lengths),
        )

    def test_missing_warm_start_does_not_silently_change_design(self):
        with TemporaryDirectory() as temporary:
            missing = Path(temporary)/"missing.json"
            with self.assertRaisesRegex(SystemExit, "WARM START FAILED"):
                load_baseline(missing, REFERENCE_DESIGN_FREQUENCY_HZ)

    def test_frequency_scales_defaults_and_search_bounds(self):
        with patch(
            "sys.argv",
            ["optimize_gain.py", "--frequency-mhz", "434"],
        ):
            args = parse_args()

        self.assertEqual(args.frequency_hz, 434e6)
        self.assertAlmostEqual(args.match_bandwidth_mhz, 5.0)
        self.assertAlmostEqual(args.maximum_height_mm, 1200.0)
        self.assertIn("434000000hz", str(args.convergence_report))
        bounds_868 = make_space(load_reference_design(868e6), 868e6).bounds
        bounds_434 = make_space(load_reference_design(434e6), 434e6).bounds
        for first, second in zip(bounds_868[:-1], bounds_434[:-1]):
            self.assertAlmostEqual(second[0], 2*first[0])
            self.assertAlmostEqual(second[1], 2*first[1])
        self.assertEqual(bounds_868[-1], bounds_434[-1])

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
        self.assertEqual(
            parse_turn_cases("none,1,1x1,1x1x1"),
            ((), (1,), (1, 1), (1, 1, 1)),
        )

    def test_coil_counts_are_non_negative_unique_and_ordered(self):
        self.assertEqual(parse_coil_counts("3,0,1,3"), (3, 0, 1))
        with self.assertRaisesRegex(ArgumentTypeError, "non-negative"):
            parse_coil_counts("0,-1")

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
        self.assertEqual(len(make_space(three).variables), 8)
        self.assertEqual(len(make_space(three, finetune=True).variables), 12)

    def test_broad_space_shares_coil_pitch_and_radius(self):
        custom = AntennaDesign(
            straight_lengths=(0.1, 0.2, 0.1),
            coils=(
                CoilDesign(pitch=6e-3, radius=9e-3),
                CoilDesign(pitch=10e-3, radius=13e-3),
            ),
        )

        broad = make_space(custom)
        vector = broad.initial_vector.copy()
        vector[broad.names.index("shared_coil_pitch")] = 8.5e-3
        vector[broad.names.index("shared_coil_radius")] = 12e-3
        decoded = broad.decode(vector)
        fine = make_space(custom, finetune=True)

        self.assertEqual(len(broad.variables), 7)
        self.assertEqual(len(fine.variables), 9)
        self.assertTrue(
            all(coil.pitch == 8.5e-3 for coil in decoded.coils)
        )
        self.assertTrue(
            all(coil.radius == 12e-3 for coil in decoded.coils)
        )
        self.assertIn("coils.1.pitch", fine.names)
        self.assertIn("coils.1.radius", fine.names)

    def test_custom_start_is_inside_the_generated_search_space(self):
        custom = AntennaDesign(
            radial_length=0.2,
            radial_angle_deg=80.0,
            straight_lengths=(0.25, 0.35, 0.25),
        )
        space = make_space(custom, REFERENCE_DESIGN_FREQUENCY_HZ)

        for value, (lower, upper) in zip(space.initial_vector, space.bounds):
            self.assertLessEqual(lower, value)
            self.assertGreaterEqual(upper, value)

    def test_unspecified_topology_is_inferred_from_custom_start(self):
        custom = AntennaDesign(
            straight_lengths=(0.1, 0.1, 0.1, 0.1),
            coils=(
                CoilDesign(turns=1),
                CoilDesign(turns=2),
                CoilDesign(turns=3),
            ),
        )
        args = Namespace(coil_count=None, coil_counts=None, turn_cases=None)

        resolve_topology(args, custom)

        self.assertEqual(args.coil_count, 3)
        self.assertEqual(args.coil_counts, (3,))
        self.assertEqual(args.turn_cases, ((1, 2, 3),))

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
        self.assertFalse(three.finetune)

        with patch("sys.argv", ["optimize_gain.py", "--finetune"]):
            fine = parse_args()
        self.assertTrue(fine.finetune)

    def test_cli_builds_multi_coil_topology_campaign(self):
        with patch(
            "sys.argv",
            ["optimize_gain.py", "--coil-counts", "0,1,2,3"],
        ):
            args = parse_args()

        self.assertEqual(args.coil_counts, (0, 1, 2, 3))
        self.assertEqual(
            args.turn_cases,
            ((), (1,), (1, 1), (1, 1, 1)),
        )
        resolve_topology(args, AntennaDesign())
        self.assertIsNone(args.coil_count)

    def test_cli_accepts_explicit_mixed_topologies(self):
        with patch(
            "sys.argv",
            [
                "optimize_gain.py",
                "--turn-cases",
                "none,1,1x1,1x2,1x1x1",
            ],
        ):
            args = parse_args()

        resolve_topology(args, AntennaDesign())
        self.assertEqual(args.coil_counts, (0, 1, 2, 3))
        self.assertEqual(
            args.turn_cases,
            ((), (1,), (1, 1), (1, 2), (1, 1, 1)),
        )

    def test_coil_counts_and_explicit_turn_cases_conflict(self):
        with (
            patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--coil-counts",
                    "0,1,3",
                    "--turn-cases",
                    "none,1,1x1x1",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_mixed_topologies_receive_equal_estimated_time(self):
        args = Namespace(
            seeds=(2, 3),
            turn_cases=((), (1,), (1, 1, 1)),
            maxiter=None,
            hours=1.0,
            seconds_per_eval=3.0,
            popsize=8,
        )

        schedules = build_case_schedules(
            args,
            AntennaDesign(),
            REFERENCE_DESIGN_FREQUENCY_HZ,
        )

        self.assertEqual(
            [len(schedule.space.variables) for schedule in schedules],
            [3, 6, 8],
        )
        self.assertEqual(
            [schedule.population for schedule in schedules],
            [24, 48, 64],
        )
        self.assertEqual(
            [schedule.evaluations_per_run for schedule in schedules],
            [192, 192, 192],
        )

        args.finetune = True
        fine_schedules = build_case_schedules(
            args,
            AntennaDesign(),
            REFERENCE_DESIGN_FREQUENCY_HZ,
        )
        self.assertEqual(
            [len(schedule.space.variables) for schedule in fine_schedules],
            [3, 6, 12],
        )

    def test_campaign_executes_and_ranks_every_requested_coil_count(self):
        class FakeObjective:
            def __init__(self, space, *, on_evaluation, **_kwargs):
                coil_count = space.base.coil_count
                record = EvaluationRecord(
                    tuple(space.initial_vector),
                    -float(coil_count),
                    -12.0,
                    2.0 + coil_count,
                    metrics={
                        "worst_s11_db": -11.0,
                        "horizon_p10_gain_dbi": 1.0 + coil_count,
                        "horizon_min_gain_dbi": 0.5 + coil_count,
                    },
                )
                self.best_record = record
                self.history = [record]
                on_evaluation(record)

        with TemporaryDirectory() as temporary:
            output = Path(temporary)/"campaign"
            with patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--coil-counts",
                    "0,1,3",
                    "--seeds",
                    "2",
                    "--maxiter",
                    "0",
                    "--skip-convergence-check",
                    "--output",
                    str(output),
                ],
            ):
                args = parse_args()
            with (
                patch(
                    "examples.optimize_gain.RobustGainObjective",
                    FakeObjective,
                ),
                patch(
                    "examples.optimize_gain.differential_evolution",
                    return_value=SimpleNamespace(success=True, message="fake"),
                ) as optimize,
            ):
                run_campaign(args)

            leaderboard = json.loads(
                (output/"topology_leaderboard.json").read_text(encoding="utf-8")
            )["topologies"]
            best = json.loads(
                (output/"campaign_best.json").read_text(encoding="utf-8")
            )

            self.assertEqual(optimize.call_count, 3)
            self.assertEqual(
                [item["coil_count"] for item in leaderboard],
                [3, 1, 0],
            )
            self.assertEqual(best["coil_count"], 3)
            self.assertEqual(
                best["simulation"]["coil_parameterization"],
                "shared",
            )

    def test_strict_and_skipped_convergence_flags_conflict(self):
        with (
            patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--skip-convergence-check",
                    "--require-convergence",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_missing_certificate_is_generated_automatically(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                convergence_report=root/"convergence.json",
                warm_start=root/"campaign_best.json",
                output=root/"campaign",
                solver="auto",
                no_auto_convergence=False,
                require_convergence=False,
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
                    REFERENCE_DESIGN_FREQUENCY_HZ,
                )

            self.assertEqual(result, certificate)
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertIn("check_open_region.py", command[2])
            self.assertIn("--frequency-mhz", command)
            self.assertIn("--selected-resolution", command)
            self.assertTrue(
                (args.output/"convergence_reference_design.json").exists()
            )

    def test_matching_certificate_is_reused_without_subprocess(self):
        args = Namespace(
            convergence_report=Path("convergence.json"),
            warm_start=Path("campaign_best.json"),
            output=Path("campaign"),
            solver="auto",
            no_auto_convergence=False,
            require_convergence=False,
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
                REFERENCE_DESIGN_FREQUENCY_HZ,
            )

        self.assertEqual(result, certificate)
        run.assert_not_called()

    def test_no_auto_convergence_warns_when_not_strict(self):
        args = Namespace(
            convergence_report=Path("missing.json"),
            warm_start=Path("campaign_best.json"),
            output=Path("campaign"),
            solver="auto",
            no_auto_convergence=True,
            require_convergence=False,
        )
        with (
            patch(
                "examples.optimize_gain.validate_convergence_certificate",
                side_effect=RuntimeError("not found"),
            ),
            patch("examples.optimize_gain.subprocess.run") as run,
        ):
            result = ensure_convergence_certificate(
                args,
                AntennaDesign(),
                MeshSettings(),
                OpenRegionSettings(),
                REFERENCE_DESIGN_FREQUENCY_HZ,
            )

        self.assertIsNone(result)
        self.assertIn("Automatic convergence is disabled", args.convergence_warning)
        run.assert_not_called()

    def test_require_convergence_makes_missing_certificate_fatal(self):
        args = Namespace(
            convergence_report=Path("missing.json"),
            output=Path("campaign"),
            solver="auto",
            no_auto_convergence=True,
            require_convergence=True,
        )
        with (
            patch(
                "examples.optimize_gain.validate_convergence_certificate",
                side_effect=RuntimeError("not found"),
            ),
            patch("examples.optimize_gain.subprocess.run") as run,
            self.assertRaisesRegex(SystemExit, "REQUIRED BUT NOT CERTIFIED"),
        ):
            ensure_convergence_certificate(
                args,
                AntennaDesign(),
                MeshSettings(),
                OpenRegionSettings(),
                REFERENCE_DESIGN_FREQUENCY_HZ,
            )

        run.assert_not_called()

    def test_failed_automatic_convergence_warns_or_fails_in_strict_mode(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = dict(
                convergence_report=root/"failed.json",
                output=root/"campaign",
                solver="auto",
                no_auto_convergence=False,
            )
            with (
                patch(
                    "examples.optimize_gain.validate_convergence_certificate",
                    side_effect=RuntimeError("not found"),
                ),
                patch(
                    "examples.optimize_gain.subprocess.run",
                    return_value=SimpleNamespace(returncode=2),
                ),
            ):
                warning_args = Namespace(
                    **common,
                    require_convergence=False,
                )
                result = ensure_convergence_certificate(
                    warning_args,
                    AntennaDesign(),
                    MeshSettings(),
                    OpenRegionSettings(),
                    REFERENCE_DESIGN_FREQUENCY_HZ,
                )
                self.assertIsNone(result)
                self.assertIn("convergence failed", warning_args.convergence_warning)

                strict_args = Namespace(
                    **common,
                    require_convergence=True,
                )
                with self.assertRaisesRegex(
                    SystemExit,
                    "REQUIRED BUT NOT CERTIFIED",
                ):
                    ensure_convergence_certificate(
                        strict_args,
                        AntennaDesign(),
                        MeshSettings(),
                        OpenRegionSettings(),
                        REFERENCE_DESIGN_FREQUENCY_HZ,
                    )

    def test_matching_failed_report_warns_without_rerunning_solves(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root/"failed.json"
            report.write_text("{}", encoding="utf-8")
            args = Namespace(
                convergence_report=report,
                output=root/"campaign",
                solver="auto",
                no_auto_convergence=False,
                require_convergence=False,
            )
            with (
                patch(
                    "examples.optimize_gain.validate_convergence_certificate",
                    side_effect=(
                        RuntimeError("did not pass"),
                        {"passed": False},
                    ),
                ) as validate,
                patch("examples.optimize_gain.subprocess.run") as run,
            ):
                result = ensure_convergence_certificate(
                    args,
                    AntennaDesign(),
                    MeshSettings(),
                    OpenRegionSettings(),
                    REFERENCE_DESIGN_FREQUENCY_HZ,
                )

            self.assertIsNone(result)
            self.assertEqual(validate.call_count, 2)
            self.assertIn("existing matching", args.convergence_warning)
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
                frequency_hz=REFERENCE_DESIGN_FREQUENCY_HZ,
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
                frequency_hz=REFERENCE_DESIGN_FREQUENCY_HZ,
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

    def test_progress_csv_supports_different_topology_dimensions(self):
        zero = make_space(design_for_coil_count(AntennaDesign(), 0))
        three = make_space(design_for_coil_count(AntennaDesign(), 3))
        variable_names = tuple(dict.fromkeys((*zero.names, *three.names)))
        records = (
            EvaluationRecord(
                tuple(zero.initial_vector),
                -1.0,
                -11.0,
                2.0,
                metrics={"worst_s11_db": -10.0},
            ),
            EvaluationRecord(
                tuple(three.initial_vector),
                -2.0,
                -12.0,
                3.0,
                metrics={"worst_s11_db": -11.0},
            ),
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            progress = CampaignProgress(
                output,
                total=2,
                report_every=10,
                variable_names=variable_names,
                frequency_hz=REFERENCE_DESIGN_FREQUENCY_HZ,
            )
            progress.set_context(zero, (), seed=2)
            progress(records[0])
            progress.set_context(three, (1, 1, 1), seed=2)
            progress(records[1])
            progress.close()

            with (output/"evaluations.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            payload = json.loads(
                (output/"campaign_best.json").read_text(encoding="utf-8")
            )

            self.assertEqual(rows[0]["shared_coil_pitch"], "")
            self.assertNotEqual(rows[1]["shared_coil_pitch"], "")
            self.assertEqual(rows[0]["coil_count"], "0")
            self.assertEqual(rows[1]["coil_count"], "3")
            self.assertEqual(payload["coil_count"], 3)
            leaderboard = json.loads(
                (output/"topology_leaderboard.json").read_text(encoding="utf-8")
            )["topologies"]
            self.assertEqual([item["coil_count"] for item in leaderboard], [3, 0])
            self.assertTrue((output/"turns_none_best.json").is_file())
            self.assertTrue((output/"turns_1x1x1_best.json").is_file())


if __name__ == "__main__":
    unittest.main()
