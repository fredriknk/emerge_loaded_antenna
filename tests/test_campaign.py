from __future__ import annotations

import csv
import json
import unittest
from argparse import ArgumentTypeError, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from emerge_loaded_antenna import (
    REFERENCE_DESIGN_FREQUENCY_HZ,
    AntennaDesign,
    CoilDesign,
    DesignSpace,
    DesignVariable,
    EvaluationRecord,
    MeshSettings,
    OpenRegionSettings,
    load_reference_design,
)
from examples.optimize_gain import (
    BASE_SECTION_START_LAMBDA,
    C0,
    COLLINEAR_SECTION_RANGE_LAMBDA,
    COLLINEAR_SECTION_START_LAMBDA,
    MONOPOLE_LENGTH_RANGE_LAMBDA,
    CampaignOutcome,
    CampaignProgress,
    CaseSchedule,
    CoordinatePolishStats,
    GenerationMonitor,
    _automatic_budget_model,
    apply_design_overrides,
    build_case_schedules,
    build_finetune_population,
    coordinate_polish,
    default_maximum_height,
    design_for_coil_count,
    ensure_convergence_certificate,
    iterations_per_run,
    load_baseline,
    make_space,
    normalized_pattern_search,
    parse_args,
    parse_coil_counts,
    parse_turn_cases,
    progress_goal_text,
    random_seeds,
    resolve_topology,
    restart_seed,
    run_automatic_pipeline,
    run_automatic_verification,
    run_campaign,
    run_finetune_optimizer,
    select_automatic_winner,
    split_finetune_budget,
)


class CampaignTests(unittest.TestCase):
    def test_progress_goal_text_follows_active_ring_and_beamwidth(self):
        record = EvaluationRecord(
            (1.0,),
            2.0,
            -11.0,
            5.0,
            metrics={
                "ring_p10_gain_dbi": -7.65,
                "ring_sampled_theta_deg": 134.0,
                "ring_beamwidth_deg": 48.0,
                "horizon_p10_gain_dbi": 5.15,
            },
        )

        text = progress_goal_text(
            record,
            {
                "objective": {
                    "pattern_mode": "ring",
                    "target_theta_deg": 135.0,
                    "target_beamwidth_deg": 50.0,
                }
            },
        )

        self.assertIn("Ring P10 -7.65 dBi", text)
        self.assertIn("@ 134 deg (goal 135 deg)", text)
        self.assertIn("BW 48 deg (goal 50 deg)", text)
        self.assertNotIn("Horizon", text)

    def test_progress_goal_text_reports_every_requested_ring(self):
        record = EvaluationRecord(
            (1.0,),
            -2.0,
            -11.0,
            5.0,
            metrics={
                "ring_p10_gain_dbi": 1.25,
                "ring_0_sampled_theta_deg": 90.0,
                "ring_0_p10_gain_dbi": 3.5,
                "ring_1_sampled_theta_deg": 130.0,
                "ring_1_p10_gain_dbi": 1.25,
            },
        )

        text = progress_goal_text(
            record,
            {
                "objective": {
                    "pattern_mode": "ring",
                    "target_theta_degrees": [90.0, 130.0],
                }
            },
        )

        self.assertIn("90 deg: 3.50", text)
        self.assertIn("130 deg: 1.25", text)
        self.assertIn("worst 1.25 dBi", text)

    def test_progress_goal_text_follows_directional_and_peak_modes(self):
        directional = EvaluationRecord(
            (1.0,),
            -2.0,
            -12.0,
            6.0,
            metrics={
                "target_gain_dbi": 4.25,
                "elevation_beamwidth_deg": 55.0,
                "azimuth_beamwidth_deg": 62.0,
            },
        )
        directional_text = progress_goal_text(
            directional,
            {
                "objective": {
                    "pattern_mode": "directional",
                    "target_theta_deg": 70.0,
                    "target_phi_deg": 25.0,
                    "target_beamwidth_deg": 60.0,
                }
            },
        )
        peak_text = progress_goal_text(
            directional,
            {"objective": {"pattern_mode": "peak"}},
        )

        self.assertIn("Target gain  4.25 dBi @ theta 70, phi 25 deg", directional_text)
        self.assertIn("BW el/az 55/62 deg (goal 60 deg)", directional_text)
        self.assertEqual(peak_text, "Peak gain  6.00 dBi")

    def test_automatic_polish_reserve_covers_smallest_topology_population(self):
        with patch(
            "sys.argv",
            [
                "optimize_gain.py",
                "--automatic",
                "--coil-counts",
                "0,3",
                "--polish-evaluations",
                "100",
            ],
        ):
            args = parse_args()

        model = _automatic_budget_model(args)
        zero_coil_space = make_space(
            design_for_coil_count(AntennaDesign(), 0),
            args.frequency_hz,
        )
        zero_coil_population = args.popsize*len(zero_coil_space.variables)

        self.assertEqual(
            model.fine_required_batches,
            1 + int(np.ceil(100/zero_coil_population)),
        )

    def test_automatic_winner_does_not_regress_confirmed_rough_design(self):
        rough = {"feasible": True, "objective": -5.0, "design": "rough"}
        noisy_fine = {"feasible": True, "objective": -4.0, "design": "fine"}

        stage, payload = select_automatic_winner(rough, noisy_fine)

        self.assertEqual(stage, "rough")
        self.assertIs(payload, rough)

    def test_coordinate_polish_converges_one_parameter_at_a_time(self):
        space = DesignSpace(
            AntennaDesign(radial_length=0.5),
            (DesignVariable("radial_length", 0.0, 1.0),),
        )

        class QuadraticObjective:
            def __init__(self):
                self.history = []
                self.simulation_evaluations = 0
                self.seen = []
                self((0.5,))

            @property
            def best_record(self):
                return min(self.history, key=lambda record: record.score)

            def __call__(self, vector):
                values = tuple(float(value) for value in vector)
                self.seen.append(values)
                score = (values[0] - 0.73) ** 2
                record = EvaluationRecord(
                    values,
                    score,
                    -12.0,
                    2.0,
                    metrics={"worst_s11_db": -12.0},
                )
                self.history.append(record)
                self.simulation_evaluations += 1
                return score

        objective = QuadraticObjective()
        stats = coordinate_polish(
            objective,
            space,
            -10.0,
            100,
            initial_step=0.2,
            minimum_step=0.0125,
            minimum_improvement=0.0,
        )

        self.assertTrue(stats.converged)
        self.assertLess(stats.evaluations, 100)
        self.assertGreater(stats.improvements, 0)
        self.assertLessEqual(abs(objective.best_record.vector[0] - 0.73), 0.0125)

    def test_coordinate_polish_prefers_feasibility_over_raw_score(self):
        space = DesignSpace(
            AntennaDesign(radial_length=0.5),
            (DesignVariable("radial_length", 0.0, 1.0),),
        )

        class FeasibilityObjective:
            def __init__(self):
                self.history = []
                self.seen = []
                self((0.5,))

            @property
            def best_record(self):
                return self.history[0]

            def __call__(self, vector):
                value = float(vector[0])
                self.seen.append(value)
                feasible = value > 0.5
                record = EvaluationRecord(
                    (value,),
                    100.0 if feasible else -100.0,
                    -12.0 if feasible else -5.0,
                    2.0,
                    metrics={"worst_s11_db": -12.0 if feasible else -5.0},
                )
                self.history.append(record)
                return record.score

        objective = FeasibilityObjective()
        stats = coordinate_polish(
            objective,
            space,
            -10.0,
            3,
            initial_step=0.2,
            minimum_step=0.1,
        )

        self.assertEqual(stats.evaluations, 3)
        self.assertEqual(stats.improvements, 1)
        self.assertAlmostEqual(objective.seen[-1], 0.9)

    def test_coordinate_polish_rejects_quarantined_candidate(self):
        space = DesignSpace(
            AntennaDesign(radial_length=0.5),
            (DesignVariable("radial_length", 0.0, 1.0),),
        )

        class ConfirmationObjective:
            def __init__(self):
                self.history = []
                self.seen = []
                self((0.5,))

            @property
            def best_record(self):
                return self.history[0]

            def __call__(self, vector):
                value = float(vector[0])
                self.seen.append(value)
                score = -100.0 if value > 0.5 else -1.0 if value < 0.5 else 0.0
                record = EvaluationRecord(
                    (value,),
                    score,
                    -12.0,
                    2.0,
                    metrics={"worst_s11_db": -12.0},
                    confirmation_status=(
                        "quarantined" if value > 0.5 else "not_requested"
                    ),
                )
                self.history.append(record)
                return record.score

        objective = ConfirmationObjective()
        coordinate_polish(
            objective,
            space,
            -10.0,
            3,
            initial_step=0.2,
            minimum_step=0.1,
        )

        self.assertAlmostEqual(objective.seen[-1], 0.5)

    def test_target_theta_alone_selects_omnidirectional_ring_mode(self):
        with patch(
            "sys.argv",
            ["optimize_gain.py", "--target-theta", "100"],
        ):
            args = parse_args()

        self.assertEqual(args.pattern, "ring")
        self.assertEqual(args.target_theta, 100.0)
        self.assertEqual(args.target_thetas, (100.0,))
        self.assertEqual(args.target_phi, 0.0)

    def test_theta_alias_accepts_multiple_omnidirectional_rings(self):
        with patch(
            "sys.argv",
            ["optimize_gain.py", "--theta", "90", "130"],
        ):
            args = parse_args()

        self.assertEqual(args.pattern, "ring")
        self.assertEqual(args.target_theta, 90.0)
        self.assertEqual(args.target_thetas, (90.0, 130.0))

    def test_multiple_theta_values_reject_directional_phi(self):
        with (
            patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--theta",
                    "90",
                    "130",
                    "--target-phi",
                    "20",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_target_phi_selects_single_direction_mode(self):
        with patch(
            "sys.argv",
            ["optimize_gain.py", "--target-theta", "100", "--target-phi", "20"],
        ):
            args = parse_args()

        self.assertEqual(args.pattern, "directional")
        self.assertEqual(args.target_theta, 100.0)
        self.assertEqual(args.target_phi, 20.0)

    def test_cli_accepts_wire_diameter_and_directional_lobe_target(self):
        with patch(
            "sys.argv",
            [
                "optimize_gain.py",
                "--wire-diameter-mm",
                "1.6",
                "--target-theta",
                "72.5",
                "--target-phi",
                "35",
                "--target-beamwidth-deg",
                "70",
                "--beamwidth-weight",
                "2.5",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.wire_diameter_mm, 1.6)
        self.assertEqual(args.pattern, "directional")
        self.assertEqual(args.target_theta, 72.5)
        self.assertEqual(args.target_phi, 35.0)
        self.assertEqual(args.target_beamwidth_deg, 70.0)
        self.assertEqual(args.beamwidth_weight, 2.5)

    def test_cli_rejects_invalid_wire_diameter_and_lobe_angles(self):
        invalid_cases = (
            ("--wire-diameter-mm", "0"),
            ("--wire-diameter-mm", "-1"),
            ("--wire-diameter-mm", "nan"),
            ("--wire-diameter-mm", "inf"),
            ("--target-theta", "-1"),
            ("--target-theta", "181"),
            ("--target-theta", "nan"),
            ("--target-phi", "inf"),
        )

        for flag, value in invalid_cases:
            with (
                self.subTest(flag=flag, value=value),
                patch("sys.argv", ["optimize_gain.py", flag, value]),
                self.assertRaises(SystemExit),
            ):
                parse_args()

    def test_beamwidth_goal_requires_directional_mode_and_valid_settings(self):
        invalid_commands = (
            [
                "--pattern",
                "directional",
                "--target-beamwidth-deg",
                "0",
            ],
            ["--pattern", "peak", "--target-beamwidth-deg", "60"],
            ["--target-theta", "100", "--target-beamwidth-deg", "181"],
            ["--beamwidth-weight", "-1"],
        )

        for command in invalid_commands:
            with (
                self.subTest(command=command),
                patch("sys.argv", ["optimize_gain.py", *command]),
                self.assertRaises(SystemExit),
            ):
                parse_args()

    def test_explicit_non_directional_pattern_rejects_lobe_target(self):
        for pattern in ("horizon", "peak"):
            with (
                self.subTest(pattern=pattern),
                patch(
                    "sys.argv",
                    [
                        "optimize_gain.py",
                        "--pattern",
                        pattern,
                        "--target-theta",
                        "100",
                    ],
                ),
                self.assertRaises(SystemExit),
            ):
                parse_args()

        with (
            patch(
                "sys.argv",
                ["optimize_gain.py", "--pattern", "ring", "--target-phi", "20"],
            ),
            self.assertRaises(SystemExit),
        ):
            parse_args()

        for conflicting in (
            ["--polish", "--scipy-polish"],
            ["--finetune", "--scipy-polish"],
        ):
            with (
                patch("sys.argv", ["optimize_gain.py", *conflicting]),
                self.assertRaises(SystemExit),
            ):
                parse_args()

    def test_design_override_converts_diameter_and_leaves_input_unchanged(self):
        original = AntennaDesign(wire_radius=0.7e-3, radial_angle_deg=60.0)

        overridden = apply_design_overrides(
            original,
            wire_diameter_mm=2.4,
        )

        self.assertIsNot(overridden, original)
        self.assertAlmostEqual(overridden.wire_radius, 1.2e-3)
        self.assertEqual(overridden.radial_angle_deg, 60.0)
        self.assertAlmostEqual(original.wire_radius, 0.7e-3)
        self.assertEqual(original.radial_angle_deg, 60.0)

    def test_search_modes_choose_safe_optimizer_defaults(self):
        with patch("sys.argv", ["optimize_gain.py"]):
            broad = parse_args()
        with patch("sys.argv", ["optimize_gain.py", "--finetune"]):
            fine = parse_args()

        self.assertEqual(len(broad.seeds), 4)
        self.assertEqual(len(set(broad.seeds)), 4)
        self.assertEqual(broad.seed_source, "system_random")
        self.assertEqual(broad.pattern, "horizon")
        self.assertEqual(len(fine.seeds), 2)
        self.assertEqual(len(set(fine.seeds)), 2)
        self.assertEqual(fine.seed_source, "system_random")
        self.assertEqual(fine.confirmation_runs, 3)
        self.assertEqual(fine.s11_margin_target_db, -12.0)
        self.assertEqual(fine.s11_margin_weight, 0.10)
        self.assertEqual(fine.restart_min_improvement, 0.05)
        self.assertFalse(broad.automatic)
        self.assertFalse(broad.polish)
        self.assertFalse(broad.scipy_polish)

    def test_random_seeds_are_distinct_and_retry_collisions(self):
        with patch(
            "examples.optimize_gain.secrets.randbelow",
            side_effect=(7, 7, 8, 9),
        ):
            generated = random_seeds(3)

        self.assertEqual(generated, (7, 8, 9))

    def test_automatic_cli_records_separate_random_stage_seeds(self):
        with patch("sys.argv", ["optimize_gain.py", "--automatic"]):
            args = parse_args()

        self.assertTrue(args.automatic)
        self.assertFalse(args.seeds_explicit)
        self.assertEqual(args.seed_source, "system_random")
        self.assertEqual(len(args.seeds), 4)
        self.assertEqual(len(args.automatic_finetune_seeds), 2)
        self.assertEqual(
            len({*args.seeds, *args.automatic_finetune_seeds}),
            6,
        )

        with patch(
            "sys.argv",
            ["optimize_gain.py", "--automatic", "--seeds", "10,11"],
        ):
            explicit = parse_args()
        self.assertTrue(explicit.seeds_explicit)
        self.assertEqual(explicit.seed_source, "command_line")
        self.assertEqual(explicit.seeds, (10, 11))
        self.assertEqual(explicit.automatic_finetune_seeds, (10, 11))

        with (
            patch(
                "sys.argv",
                ["optimize_gain.py", "--automatic", "--scipy-polish"],
            ),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_automatic_pipeline_runs_rough_then_fine_polish_then_verify(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "automatic"
            with patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--automatic",
                    "--maxiter",
                    "20",
                    "--coil-counts",
                    "1,2",
                    "--skip-convergence-check",
                    "--output",
                    str(output),
                ],
            ):
                args = parse_args()
            expected_rough_seeds = args.seeds
            expected_fine_seeds = args.automatic_finetune_seeds

            stage_args = []

            def fake_campaign(stage):
                stage_args.append(stage)
                if not stage.finetune and stage.maximum_height_mm is None:
                    stage.maximum_height_mm = 432.1
                stage.output.mkdir(parents=True, exist_ok=True)
                best = stage.output / "campaign_best.json"
                best.write_text(
                    json.dumps(
                        {
                            "frequency_hz": stage.frequency_hz,
                            "turn_case": [1, 1],
                            "coil_count": 2,
                            "design": {"coils": [{}, {}]},
                            "simulation": {
                                "objective": {"pattern_mode": "horizon"}
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                if stage.finetune:
                    (stage.output / "run_summaries.json").write_text(
                        json.dumps(
                            {
                                "runs": [
                                    {
                                        "optimizer": {
                                            "transition_reason": "fine_stagnation",
                                            "coordinate_polish": {
                                                "evaluations": 12,
                                                "converged": True,
                                            },
                                        }
                                    }
                                ]
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    (stage.output / "evaluations.csv").write_text(
                        "candidate\n1\n",
                        encoding="utf-8",
                    )
                return CampaignOutcome(best, False)

            artifacts = (output / "verification.json", output / "design_sheet.pdf")

            def fake_verification(_args, _best):
                (output / "verification.json").write_text(
                    json.dumps({"quality": {"status": "passed", "checks": {}}}),
                    encoding="utf-8",
                )
                return artifacts

            with (
                patch(
                    "examples.optimize_gain.run_campaign",
                    side_effect=fake_campaign,
                ),
                patch(
                    "examples.optimize_gain.run_automatic_verification",
                    side_effect=fake_verification,
                ) as verify,
            ):
                final_best = run_automatic_pipeline(args)

            self.assertEqual(len(stage_args), 2)
            rough, fine = stage_args
            self.assertFalse(rough.finetune)
            self.assertEqual(rough.pipeline_stage, "rough")
            self.assertEqual(rough.seeds, expected_rough_seeds)
            self.assertTrue(fine.finetune)
            self.assertTrue(fine.polish)
            self.assertEqual(fine.pipeline_stage, "fine_polish")
            self.assertEqual(fine.turn_cases, ((1, 1),))
            self.assertEqual(fine.seeds, expected_fine_seeds)
            self.assertEqual(fine.maximum_height_mm, 432.1)
            self.assertEqual(
                fine.maximum_height_source_override,
                "automatic_campaign_shared",
            )
            self.assertGreater(fine.maxiter, 10)
            self.assertEqual(final_best, output / "campaign_best.json")
            verify.assert_called_once_with(args, final_best)
            manifest = json.loads(
                (output / "automatic_pipeline.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["seeds"]["source"], "system_random")
            self.assertEqual(
                manifest["seeds"]["rough"],
                list(expected_rough_seeds),
            )
            self.assertEqual(
                manifest["seeds"]["fine_polish"],
                list(expected_fine_seeds),
            )
            self.assertAlmostEqual(
                manifest["budget"]["estimated_rough_candidate_fraction"],
                0.65,
                delta=0.03,
            )
            self.assertGreater(
                manifest["budget"]["fine_polish"]["rolled_forward_candidates"],
                0,
            )

        with TemporaryDirectory() as temporary:
            with patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--automatic",
                    "--maxiter",
                    "0",
                    "--output",
                    str(Path(temporary) / "too_small"),
                ],
            ):
                too_small = parse_args()
            with self.assertRaisesRegex(ValueError, "coordinate polish"):
                run_automatic_pipeline(too_small)

    def test_automatic_pipeline_records_stage_interrupt(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "automatic"
            with patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--automatic",
                    "--maxiter",
                    "3",
                    "--output",
                    str(output),
                ],
            ):
                args = parse_args()

            with (
                patch(
                    "examples.optimize_gain.run_campaign",
                    side_effect=KeyboardInterrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                run_automatic_pipeline(args)

            manifest = json.loads(
                (output / "automatic_pipeline.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "interrupted")
            self.assertEqual(manifest["stages"]["rough"]["status"], "interrupted")

    def test_automatic_verification_uses_same_output_and_fabrication_flags(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            best = output / "campaign_best.json"
            best.write_text(
                json.dumps(
                    {
                        "coil_count": 0,
                        "design": {"coils": []},
                        "simulation": {
                            "objective": {"pattern_mode": "horizon"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            for name in (
                "verification.json",
                "s11_verified.png",
                "horizon_gain.png",
                "principal_plane_gain.png",
                "design_sheet.pdf",
            ):
                (output / name).write_bytes(b"artifact")
            args = Namespace(
                output=output,
                match_bandwidth_mhz=20.0,
                solver="cudss",
            )
            with patch(
                "examples.optimize_gain.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ) as run:
                artifacts = run_automatic_verification(args, best)

        command = run.call_args.args[0]
        self.assertIn(str(best.resolve()), command)
        self.assertIn("--design-sheet", command)
        self.assertIn("--jig-models", command)
        self.assertEqual(command[command.index("--output") + 1], str(output.resolve()))
        self.assertNotIn(output / "coil_formers.step", artifacts)

    def test_automatic_verification_requires_ring_and_coil_artifacts(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            best = output / "campaign_best.json"
            best.write_text(
                json.dumps(
                    {
                        "coil_count": 1,
                        "design": {"coils": [{}]},
                        "simulation": {
                            "objective": {"pattern_mode": "ring"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            for name in (
                "verification.json",
                "s11_verified.png",
                "horizon_gain.png",
                "principal_plane_gain.png",
                "design_sheet.pdf",
            ):
                (output / name).write_bytes(b"artifact")
            args = Namespace(
                output=output,
                match_bandwidth_mhz=20.0,
                solver="cudss",
            )
            with (
                patch(
                    "examples.optimize_gain.subprocess.run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "target_ring_gain.*coil_formers",
                ),
            ):
                run_automatic_verification(args, best)

    def test_finetune_population_is_multiscale_bounded_and_reproducible(self):
        space = make_space(AntennaDesign())
        population = build_finetune_population(space, 20, seed=17)
        repeated = build_finetune_population(space, 20, seed=17)
        normalized = np.asarray([space.normalize(row) for row in population])
        center = space.normalize(space.initial_vector)

        np.testing.assert_allclose(population, repeated)
        np.testing.assert_allclose(population[0], space.initial_vector)
        self.assertEqual(population.shape, (20, len(space.variables)))
        self.assertTrue(np.all(np.abs(normalized[1:10] - center) <= 0.03))
        self.assertTrue(np.all(np.abs(normalized[10:16] - center) <= 0.10))
        self.assertTrue(np.all((0 <= normalized) & (normalized <= 1)))

    def test_finetune_population_samples_inside_boundary_intersections(self):
        space = make_space(AntennaDesign())
        lower = np.asarray([bound[0] for bound in space.bounds])
        population = build_finetune_population(
            space,
            20,
            seed=9,
            center=lower,
        )
        normalized = np.asarray([space.normalize(row) for row in population])

        self.assertTrue(np.all(normalized[1:10] >= 0))
        self.assertTrue(np.all(normalized[1:10] <= 0.03))
        self.assertFalse(np.any(np.all(normalized[1:10] == 0, axis=1)))

    def test_finetune_budget_reserves_whole_population_batches(self):
        budget = split_finetune_budget(72 * 21, 72, requested_local=24)

        self.assertEqual(budget.differential_evolution, 72 * 20)
        self.assertEqual(budget.local_search, 72)
        self.assertEqual(
            budget.differential_evolution + budget.local_search,
            72 * 21,
        )
        self.assertEqual(
            split_finetune_budget(72, 72, requested_local=24).local_search,
            0,
        )

    def test_generation_monitor_reports_stagnation_and_diversity(self):
        objective = SimpleNamespace(history=[])
        space = make_space(AntennaDesign())
        population = build_finetune_population(space, 10, seed=3)
        monitor = GenerationMonitor(
            objective,
            space,
            "test_run",
            restart_after=2,
            incumbent_score=1.0,
        )
        intermediate = SimpleNamespace(fun=1.0, population=population)

        self.assertFalse(monitor(intermediate))
        self.assertTrue(monitor(intermediate))
        self.assertEqual(monitor.stage_generations, 2)
        self.assertEqual(monitor.stagnation_generations, 2)
        self.assertGreater(monitor.last_diversity, 0)

    def test_generation_monitor_resets_when_feasibility_first_appears(self):
        space = make_space(AntennaDesign())
        population = build_finetune_population(space, 10, seed=3)
        objective = SimpleNamespace(
            history=[
                EvaluationRecord(
                    tuple(space.initial_vector),
                    1.0,
                    -5.0,
                    2.0,
                    metrics={"worst_s11_db": -5.0},
                )
            ]
        )
        monitor = GenerationMonitor(
            objective,
            space,
            "test_run",
            restart_after=2,
            incumbent_score=1.0,
            improvement_tolerance=0.05,
        )
        self.assertFalse(monitor(SimpleNamespace(fun=1.0, population=population)))

        objective.history.append(
            EvaluationRecord(
                tuple(space.initial_vector),
                0.99,
                -12.0,
                2.0,
                metrics={"worst_s11_db": -12.0},
            )
        )

        self.assertFalse(monitor(SimpleNamespace(fun=0.99, population=population)))
        self.assertEqual(monitor.stagnation_generations, 0)

    def test_restart_seeds_are_stable_and_do_not_overlap_adjacent_runs(self):
        self.assertEqual(restart_seed(2, 1), restart_seed(2, 1))
        self.assertNotEqual(restart_seed(2, 1), restart_seed(3, 0))

    def test_pattern_search_is_bounded_budgeted_and_improves_feasible_elite(self):
        space = make_space(design_for_coil_count(AntennaDesign(), 0))
        start = space.normalize(space.initial_vector)
        target = np.clip(start + np.asarray((0.08, -0.05, 0.04)), 0, 1)

        class QuadraticObjective:
            def __init__(self):
                self.history = []
                self.simulation_evaluations = 0
                self(space.initial_vector)

            def __call__(self, vector):
                normalized = space.normalize(vector)
                score = float(np.sum((normalized - target) ** 2))
                record = EvaluationRecord(
                    tuple(vector),
                    score,
                    -12.0,
                    2.0,
                    metrics={
                        "worst_s11_db": -12.0,
                        "mismatch_penalty": 0.0,
                        "pattern_penalty": 0.0,
                        "height_penalty": 0.0,
                    },
                )
                self.history.append(record)
                self.simulation_evaluations += 1
                return score

        objective = QuadraticObjective()
        initial_score = objective.history[0].score
        stats = normalized_pattern_search(
            objective,
            space,
            -10.0,
            12,
            initial_step=0.05,
            minimum_step=0.005,
            elite_count=1,
        )

        self.assertEqual(stats.evaluations, 12)
        self.assertEqual(stats.simulation_evaluations, 12)
        self.assertGreater(stats.improvements, 0)
        self.assertLess(
            min(record.score for record in objective.history), initial_score
        )
        for record in objective.history:
            normalized = space.normalize(record.vector)
            self.assertTrue(np.all((0 <= normalized) & (normalized <= 1)))

    def test_pattern_search_spends_budget_without_a_feasible_elite(self):
        space = make_space(design_for_coil_count(AntennaDesign(), 0))

        class InfeasibleObjective:
            def __init__(self):
                self.history = []
                self.simulation_evaluations = 0
                self(space.initial_vector)

            def __call__(self, vector):
                record = EvaluationRecord(
                    tuple(vector),
                    1.0,
                    -9.0,
                    2.0,
                    metrics={
                        "worst_s11_db": -9.0,
                        "mismatch_penalty": 1.0,
                        "pattern_penalty": 0.0,
                        "height_penalty": 0.0,
                    },
                )
                self.history.append(record)
                self.simulation_evaluations += 1
                return record.score

        objective = InfeasibleObjective()
        stats = normalized_pattern_search(
            objective,
            space,
            -10.0,
            17,
            initial_step=0.01,
            minimum_step=0.01,
            elite_count=3,
        )

        self.assertEqual(stats.evaluations, 17)
        self.assertEqual(stats.simulation_evaluations, 17)
        self.assertEqual(stats.elite_count, 0)

        tiny_step_stats = normalized_pattern_search(
            objective,
            space,
            -10.0,
            5,
            initial_step=1e-20,
            minimum_step=1e-20,
            elite_count=1,
        )
        self.assertEqual(tiny_step_stats.evaluations, 5)

    def test_finetune_runner_restarts_without_exceeding_candidate_budget(self):
        space = make_space(design_for_coil_count(AntennaDesign(), 0))
        schedule = CaseSchedule(
            (), space, maxiter=3, population=5, evaluations_per_run=20
        )

        class FlatObjective:
            def __init__(self):
                self.history = []
                self.simulation_evaluations = 0

            @property
            def best_record(self):
                return min(self.history, key=lambda record: record.score, default=None)

            def __call__(self, vector):
                record = EvaluationRecord(
                    tuple(vector),
                    1.0,
                    -12.0,
                    2.0,
                    metrics={"worst_s11_db": -12.0},
                )
                self.history.append(record)
                self.simulation_evaluations += 1
                return 1.0

        objective = FlatObjective()

        def fake_de(function, *, init, maxiter, callback, **_kwargs):
            for row in init:
                function(row)
            generations = 0
            for _ in range(maxiter):
                for row in init:
                    function(row)
                generations += 1
                if callback(SimpleNamespace(fun=1.0, population=init)):
                    break
            return SimpleNamespace(
                success=False,
                message="fake",
                nit=generations,
            )

        args = Namespace(
            popsize=1,
            local_search_evaluations=0,
            finetune_near_radius=0.03,
            finetune_wide_radius=0.10,
            finetune_mutation=(0.2, 0.6),
            finetune_recombination=0.45,
            restart_stagnation_generations=1,
            s11_limit_db=-10.0,
        )
        with patch(
            "examples.optimize_gain.differential_evolution",
            side_effect=fake_de,
        ) as optimize:
            result = run_finetune_optimizer(
                objective,
                space,
                schedule,
                args,
                seed=2,
                run_name="flat",
            )

        self.assertEqual(len(objective.history), 20)
        self.assertEqual(result.differential_evolution_evaluations, 20)
        self.assertEqual(result.restarts, 1)
        self.assertEqual(result.generations, 2)
        self.assertEqual(result.effective_stagnation_generations, 1)
        self.assertEqual(optimize.call_count, 2)

    def test_finetune_stagnation_transitions_to_coordinate_polish(self):
        space = make_space(design_for_coil_count(AntennaDesign(), 0))
        schedule = CaseSchedule(
            (), space, maxiter=3, population=5, evaluations_per_run=20
        )

        class FlatObjective:
            def __init__(self):
                self.history = []
                self.simulation_evaluations = 0

            @property
            def best_record(self):
                return min(self.history, key=lambda record: record.score, default=None)

            def __call__(self, vector):
                record = EvaluationRecord(
                    tuple(vector),
                    1.0,
                    -12.0,
                    2.0,
                    metrics={"worst_s11_db": -12.0},
                )
                self.history.append(record)
                self.simulation_evaluations += 1
                return 1.0

        objective = FlatObjective()

        def fake_de(function, *, init, maxiter, callback, **_kwargs):
            for row in init:
                function(row)
            generations = 0
            for _ in range(maxiter):
                for row in init:
                    function(row)
                generations += 1
                if callback(SimpleNamespace(fun=1.0, population=init)):
                    break
            return SimpleNamespace(success=False, message="fake", nit=generations)

        args = Namespace(
            popsize=1,
            polish=True,
            polish_evaluations=5,
            polish_min_improvement=0.001,
            local_search_step=0.03,
            local_search_min_step=0.001,
            finetune_near_radius=0.03,
            finetune_wide_radius=0.10,
            finetune_mutation=(0.2, 0.6),
            finetune_recombination=0.45,
            restart_stagnation_generations=1,
            restart_min_improvement=0.05,
            s11_limit_db=-10.0,
        )
        polish_stats = CoordinatePolishStats(4, 4, 0, 1, 0.001, True)
        with (
            patch(
                "examples.optimize_gain.differential_evolution",
                side_effect=fake_de,
            ) as optimize,
            patch(
                "examples.optimize_gain.coordinate_polish",
                return_value=polish_stats,
            ) as polish,
        ):
            result = run_finetune_optimizer(
                objective,
                space,
                schedule,
                args,
                seed=2,
                run_name="flat",
            )

        self.assertEqual(optimize.call_count, 1)
        polish.assert_called_once()
        self.assertEqual(result.transition_reason, "fine_stagnation")
        self.assertEqual(result.polish, polish_stats)

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

        half_frequency = REFERENCE_DESIGN_FREQUENCY_HZ / 2
        scaled = load_baseline(None, half_frequency)
        reference = load_reference_design()
        self.assertAlmostEqual(scaled.wire_radius, 2 * reference.wire_radius)
        self.assertAlmostEqual(scaled.radial_length, 2 * reference.radial_length)
        self.assertAlmostEqual(
            sum(scaled.straight_lengths),
            2 * sum(reference.straight_lengths),
        )

    def test_missing_warm_start_does_not_silently_change_design(self):
        with TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
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
        self.assertIsNone(args.maximum_height_mm)
        self.assertAlmostEqual(
            1e3 * default_maximum_height(434e6, 2),
            1.7 * C0 / 434e6 * 1e3,
        )
        self.assertIn("434000000hz", str(args.convergence_report))
        bounds_868 = make_space(load_reference_design(868e6), 868e6).bounds
        bounds_434 = make_space(load_reference_design(434e6), 434e6).bounds
        for first, second in zip(bounds_868[:-1], bounds_434[:-1]):
            self.assertAlmostEqual(second[0], 2 * first[0])
            self.assertAlmostEqual(second[1], 2 * first[1])
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
        wavelength = C0 / REFERENCE_DESIGN_FREQUENCY_HZ

        self.assertEqual(zero.coil_count, 0)
        self.assertEqual(len(zero.straight_lengths), 1)
        self.assertAlmostEqual(
            zero.straight_lengths[0],
            BASE_SECTION_START_LAMBDA * wavelength,
        )
        self.assertEqual(three.coil_count, 3)
        self.assertEqual(len(three.straight_lengths), 4)
        self.assertAlmostEqual(
            three.straight_lengths[0],
            BASE_SECTION_START_LAMBDA * wavelength,
        )
        for length in three.straight_lengths[1:]:
            self.assertAlmostEqual(
                length,
                COLLINEAR_SECTION_START_LAMBDA * wavelength,
            )
        self.assertEqual(len(make_space(zero).variables), 3)
        self.assertEqual(len(make_space(three).variables), 12)
        self.assertEqual(len(make_space(three, lock_coils=True).variables), 8)

    def test_electrical_length_priors_are_broad_but_finite(self):
        frequency_hz = REFERENCE_DESIGN_FREQUENCY_HZ
        wavelength = C0 / frequency_hz
        zero = make_space(
            design_for_coil_count(AntennaDesign(), 0, frequency_hz),
            frequency_hz,
        )
        three = make_space(
            design_for_coil_count(AntennaDesign(), 3, frequency_hz),
            frequency_hz,
        )
        zero_bounds = dict(zip(zero.names, zero.bounds))
        three_bounds = dict(zip(three.names, three.bounds))

        self.assertEqual(
            zero_bounds["straight_lengths.0"],
            tuple(value * wavelength for value in MONOPOLE_LENGTH_RANGE_LAMBDA),
        )
        expected_loaded = tuple(
            value * wavelength for value in COLLINEAR_SECTION_RANGE_LAMBDA
        )
        for index in range(4):
            self.assertEqual(
                three_bounds[f"straight_lengths.{index}"],
                expected_loaded,
            )

    def test_thick_wire_expands_coil_geometry_floors(self):
        coils = (
            CoilDesign(pitch=8e-3, radius=12e-3),
            CoilDesign(pitch=8e-3, radius=12e-3),
        )
        thin = make_space(AntennaDesign(coils=coils))
        thick = make_space(AntennaDesign(wire_radius=2e-3, coils=coils))
        thin_bounds = dict(zip(thin.names, thin.bounds))
        thick_bounds = dict(zip(thick.names, thick.bounds))

        self.assertGreater(
            thick_bounds["coils.0.pitch"][0],
            thin_bounds["coils.0.pitch"][0],
        )
        self.assertGreater(
            thick_bounds["coils.0.radius"][0],
            thin_bounds["coils.0.radius"][0],
        )

    def test_valid_warm_start_can_expand_below_preferred_wire_floor(self):
        coils = (
            CoilDesign(pitch=4e-3, radius=6e-3),
            CoilDesign(pitch=4e-3, radius=6e-3),
        )
        space = make_space(AntennaDesign(wire_radius=2e-3, coils=coils))

        for value, (lower, upper) in zip(space.initial_vector, space.bounds):
            self.assertLessEqual(lower, value)
            self.assertGreaterEqual(upper, value)

    def test_coil_geometry_is_independent_unless_explicitly_locked(self):
        custom = AntennaDesign(
            straight_lengths=(0.1, 0.2, 0.1),
            coils=(
                CoilDesign(pitch=6e-3, radius=9e-3),
                CoilDesign(pitch=10e-3, radius=13e-3),
            ),
        )

        independent = make_space(custom)
        locked = make_space(custom, lock_coils=True)
        vector = locked.initial_vector.copy()
        vector[locked.names.index("shared_coil_pitch")] = 8.5e-3
        vector[locked.names.index("shared_coil_radius")] = 12e-3
        decoded = locked.decode(vector)

        self.assertEqual(len(independent.variables), 9)
        self.assertEqual(len(locked.variables), 7)
        self.assertTrue(all(coil.pitch == 8.5e-3 for coil in decoded.coils))
        self.assertTrue(all(coil.radius == 12e-3 for coil in decoded.coils))
        self.assertIn("coils.1.pitch", independent.names)
        self.assertIn("coils.1.radius", independent.names)

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
        self.assertFalse(three.lock_coils)

        with patch("sys.argv", ["optimize_gain.py", "--finetune"]):
            fine = parse_args()
        self.assertTrue(fine.finetune)
        self.assertFalse(fine.lock_coils)
        self.assertEqual(fine.finetune_near_radius, 0.03)
        self.assertEqual(fine.finetune_wide_radius, 0.10)
        self.assertEqual(fine.finetune_mutation, (0.2, 0.6))
        self.assertEqual(fine.finetune_recombination, 0.30)
        self.assertEqual(fine.restart_stagnation_generations, 10)
        self.assertEqual(fine.local_search_evaluations, 24)
        self.assertEqual(fine.s11_margin_target_db, -12.0)
        self.assertEqual(fine.s11_margin_weight, 0.10)
        self.assertEqual(fine.confirmation_runs, 3)

        with patch("sys.argv", ["optimize_gain.py", "--lock-coils"]):
            locked = parse_args()
        self.assertTrue(locked.lock_coils)
        self.assertFalse(locked.finetune)

        with patch(
            "sys.argv",
            ["optimize_gain.py", "--lock-coils", "--finetune"],
        ):
            locked_fine = parse_args()
        self.assertTrue(locked_fine.lock_coils)
        self.assertTrue(locked_fine.finetune)

    def test_cli_rejects_nonfinite_objective_and_grid_inputs(self):
        with (
            patch("sys.argv", ["optimize_gain.py", "--s11-limit-db", "nan"]),
            self.assertRaises(SystemExit),
        ):
            parse_args()

        with (
            patch("sys.argv", ["optimize_gain.py", "--angular-step", "nan"]),
            self.assertRaises(SystemExit),
        ):
            parse_args()

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
            [3, 6, 12],
        )
        self.assertEqual(
            [schedule.population for schedule in schedules],
            [24, 48, 96],
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

        args.lock_coils = True
        locked_schedules = build_case_schedules(
            args,
            AntennaDesign(),
            REFERENCE_DESIGN_FREQUENCY_HZ,
        )
        self.assertEqual(
            [len(schedule.space.variables) for schedule in locked_schedules],
            [3, 6, 8],
        )

    def test_campaign_executes_and_ranks_every_requested_coil_count(self):
        objective_options = []

        class FakeObjective:
            def __init__(self, space, *, on_evaluation, **kwargs):
                objective_options.append(kwargs)
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
            output = Path(temporary) / "campaign"
            with patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--coil-counts",
                    "0,1,3",
                    "--wire-diameter-mm",
                    "1.6",
                    "--pattern",
                    "directional",
                    "--target-theta",
                    "70",
                    "--target-phi",
                    "25",
                    "--target-beamwidth-deg",
                    "80",
                    "--beamwidth-weight",
                    "1.5",
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
                (output / "topology_leaderboard.json").read_text(encoding="utf-8")
            )["topologies"]
            best = json.loads(
                (output / "campaign_best.json").read_text(encoding="utf-8")
            )
            recorded_seeds = json.loads(
                (output / "campaign_seeds.json").read_text(encoding="utf-8")
            )

            self.assertEqual(optimize.call_count, 3)
            self.assertEqual(recorded_seeds["source"], "command_line")
            self.assertEqual(recorded_seeds["seeds"], [2])
            self.assertTrue(
                all(item["confirmation_runs"] == 3 for item in objective_options)
            )
            self.assertTrue(
                all(
                    item["s11_margin_target_db"] == -12.0
                    for item in objective_options
                )
            )
            self.assertTrue(
                all(item["s11_margin_weight"] == 0.10 for item in objective_options)
            )
            self.assertTrue(
                all(item["target_beamwidth_deg"] == 80.0 for item in objective_options)
            )
            self.assertTrue(
                all(item["beamwidth_weight"] == 1.5 for item in objective_options)
            )
            self.assertTrue(
                all(item["target_theta_degrees"] == (70.0,) for item in objective_options)
            )
            self.assertEqual(
                [item["coil_count"] for item in leaderboard],
                [3, 1, 0],
            )
            self.assertEqual(best["coil_count"], 3)
            self.assertEqual(
                best["simulation"]["coil_parameterization"],
                "independent",
            )
            self.assertEqual(
                best["simulation"]["seeds"],
                {"source": "command_line", "values": [2]},
            )
            self.assertEqual(
                best["simulation"]["search_bounds"]["policy"],
                "wavelength_wire_v1",
            )
            self.assertAlmostEqual(best["design"]["wire_radius"], 0.8e-3)
            self.assertIn("radial_angle_deg", best["search_space"]["bounds"])
            self.assertAlmostEqual(
                best["simulation"]["search_bounds"]["wire_diameter_m"],
                1.6e-3,
            )
            self.assertEqual(
                best["simulation"]["search_bounds"]["wire_diameter_source"],
                "command_line",
            )
            self.assertEqual(
                best["simulation"]["objective"]["target_beamwidth_deg"],
                80.0,
            )
            self.assertEqual(
                best["simulation"]["objective"]["pattern_mode"],
                "directional",
            )
            self.assertEqual(
                best["simulation"]["objective"]["target_theta_deg"],
                70.0,
            )
            self.assertEqual(
                best["simulation"]["objective"]["target_theta_degrees"],
                [70.0],
            )
            self.assertEqual(
                best["simulation"]["objective"]["target_phi_deg"],
                25.0,
            )
            self.assertEqual(
                best["simulation"]["objective"]["beamwidth_weight"],
                1.5,
            )
            self.assertAlmostEqual(
                best["simulation"]["search_bounds"]["maximum_height_wavelengths"],
                2.2,
            )
            self.assertEqual(
                best["simulation"]["search_bounds"]["maximum_height_source"],
                "automatic",
            )

    def test_campaign_records_multiple_ring_targets_and_csv_metrics(self):
        objective_options = []

        class FakeObjective:
            def __init__(self, space, *, on_evaluation, **kwargs):
                objective_options.append(kwargs)
                record = EvaluationRecord(
                    tuple(space.initial_vector),
                    -1.0,
                    -12.0,
                    3.0,
                    metrics={
                        "worst_s11_db": -11.0,
                        "ring_p10_gain_dbi": 1.0,
                        "ring_0_p10_gain_dbi": 2.0,
                        "ring_1_p10_gain_dbi": 1.0,
                    },
                )
                self.best_record = record
                self.history = [record]
                on_evaluation(record)

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"
            with patch(
                "sys.argv",
                [
                    "optimize_gain.py",
                    "--theta",
                    "90",
                    "130",
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
                ),
            ):
                run_campaign(args)

            best = json.loads(
                (output / "campaign_best.json").read_text(encoding="utf-8")
            )
            with (output / "evaluations.csv").open(encoding="utf-8") as csv_file:
                fieldnames = csv.DictReader(csv_file).fieldnames

        self.assertEqual(
            objective_options[0]["target_theta_degrees"],
            (90.0, 130.0),
        )
        self.assertEqual(
            best["simulation"]["objective"]["target_theta_degrees"],
            [90.0, 130.0],
        )
        self.assertIn("ring_0_p10_gain_dbi", fieldnames)
        self.assertIn("ring_1_p10_gain_dbi", fieldnames)

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
                convergence_report=root / "convergence.json",
                warm_start=root / "campaign_best.json",
                output=root / "campaign",
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
            self.assertEqual(command[command.index("--angular-step") + 1], "2")
            self.assertTrue(
                (args.output / "convergence_reference_design.json").exists()
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
            common = {
                "convergence_report": root / "failed.json",
                "output": root / "campaign",
                "solver": "auto",
                "no_auto_convergence": False,
            }
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
            report = root / "failed.json"
            report.write_text("{}", encoding="utf-8")
            args = Namespace(
                convergence_report=report,
                output=root / "campaign",
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
            confirmation_status="confirmed",
            simulation_runs=3,
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            progress = CampaignProgress(
                output,
                total=1,
                report_every=1,
                variable_names=space.names,
                frequency_hz=REFERENCE_DESIGN_FREQUENCY_HZ,
                stage="fine_polish",
            )
            progress.set_context(space, (1, 1), seed=2)
            progress(record)
            progress.close()

            payload = json.loads(
                (output / "campaign_best.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["objective"], -3.0)
            self.assertEqual(payload["evaluations_at_save"], 1)
            self.assertEqual(payload["simulation_evaluations_at_save"], 3)
            self.assertEqual(payload["confirmation_status"], "confirmed")
            self.assertEqual(payload["turn_case"], [1, 1])
            self.assertEqual(
                set(payload["search_space"]["bounds"]),
                set(space.names),
            )
            self.assertEqual(
                len((output / "evaluations.csv").read_text().splitlines()), 2
            )
            with (output / "evaluations.csv").open(encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(rows[0]["stage"], "fine_polish")

    def test_progress_prefers_confirmed_feasible_checkpoint(self):
        space = make_space(AntennaDesign())
        records = (
            EvaluationRecord(
                tuple(space.initial_vector),
                -10.0,
                -9.0,
                12.0,
                metrics={
                    "worst_s11_db": -9.0,
                    "mismatch_penalty": 2.0,
                    "pattern_penalty": 0.0,
                    "height_penalty": 0.0,
                },
                confirmation_status="confirmed",
            ),
            EvaluationRecord(
                tuple(space.initial_vector),
                -3.0,
                -12.0,
                4.0,
                metrics={
                    "worst_s11_db": -11.0,
                    "mismatch_penalty": 0.0,
                    "pattern_penalty": 0.0,
                    "height_penalty": 0.0,
                },
                confirmation_status="confirmed",
            ),
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            progress = CampaignProgress(
                output,
                total=2,
                report_every=10,
                variable_names=space.names,
                frequency_hz=REFERENCE_DESIGN_FREQUENCY_HZ,
            )
            progress.set_context(space, (1, 1), seed=2)
            for record in records:
                progress(record)
            progress.close()

            payload = json.loads(
                (output / "campaign_best.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["objective"], -3.0)
            self.assertEqual(payload["metrics"]["worst_s11_db"], -11.0)
            self.assertTrue(payload["feasible"])

    def test_progress_refuses_to_truncate_an_existing_campaign_log(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            log = output / "evaluations.csv"
            log.write_text("existing campaign\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "resume is not supported"):
                CampaignProgress(
                    output,
                    total=1,
                    report_every=1,
                    variable_names=("straight_lengths.0",),
                    frequency_hz=REFERENCE_DESIGN_FREQUENCY_HZ,
                )

            self.assertEqual(
                log.read_text(encoding="utf-8"),
                "existing campaign\n",
            )

    def test_progress_refuses_any_nonempty_output_directory(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            best = output / "campaign_best.json"
            best.write_text('{"existing": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                CampaignProgress(
                    output,
                    total=1,
                    report_every=1,
                    variable_names=("straight_lengths.0",),
                    frequency_hz=REFERENCE_DESIGN_FREQUENCY_HZ,
                )

            self.assertEqual(
                best.read_text(encoding="utf-8"),
                '{"existing": true}\n',
            )

    def test_progress_allows_artifact_created_by_current_preflight(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            reference = output / "convergence_reference_design.json"
            reference.write_text('{"design": true}\n', encoding="utf-8")

            progress = CampaignProgress(
                output,
                total=1,
                report_every=1,
                variable_names=("straight_lengths.0",),
                frequency_hz=REFERENCE_DESIGN_FREQUENCY_HZ,
                allowed_existing=(reference,),
            )
            progress.close()

            self.assertTrue(reference.is_file())
            self.assertTrue((output / "evaluations.csv").is_file())

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
                (output / "campaign_best.json").read_text(encoding="utf-8")
            )
            rows = (output / "evaluations.csv").read_text().splitlines()
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

            with (output / "evaluations.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            payload = json.loads(
                (output / "campaign_best.json").read_text(encoding="utf-8")
            )

            self.assertEqual(rows[0]["coils.0.pitch"], "")
            self.assertNotEqual(rows[1]["coils.0.pitch"], "")
            self.assertNotEqual(rows[1]["coils.2.pitch"], "")
            self.assertEqual(rows[0]["coil_count"], "0")
            self.assertEqual(rows[1]["coil_count"], "3")
            self.assertEqual(payload["coil_count"], 3)
            leaderboard = json.loads(
                (output / "topology_leaderboard.json").read_text(encoding="utf-8")
            )["topologies"]
            self.assertEqual([item["coil_count"] for item in leaderboard], [3, 0])
            self.assertTrue((output / "turns_none_best.json").is_file())
            self.assertTrue((output / "turns_1x1x1_best.json").is_file())


if __name__ == "__main__":
    unittest.main()
