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
    CampaignProgress,
    CaseSchedule,
    GenerationMonitor,
    apply_design_overrides,
    build_case_schedules,
    build_finetune_population,
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
    resolve_topology,
    restart_seed,
    run_campaign,
    run_finetune_optimizer,
    split_finetune_budget,
)


class CampaignTests(unittest.TestCase):
    def test_cli_accepts_wire_diameter_and_directional_lobe_target(self):
        with patch(
            "sys.argv",
            [
                "optimize_gain.py",
                "--wire-diameter-mm",
                "1.6",
                "--pattern",
                "directional",
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
            ["--target-beamwidth-deg", "60"],
            [
                "--pattern",
                "directional",
                "--target-beamwidth-deg",
                "0",
            ],
            ["--beamwidth-weight", "-1"],
        )

        for command in invalid_commands:
            with (
                self.subTest(command=command),
                patch("sys.argv", ["optimize_gain.py", *command]),
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

        self.assertEqual(broad.seeds, (2, 3, 4, 5))
        self.assertEqual(fine.seeds, (2, 3))
        self.assertEqual(fine.confirmation_runs, 3)
        self.assertEqual(fine.s11_margin_target_db, -12.0)
        self.assertEqual(fine.s11_margin_weight, 0.10)
        self.assertEqual(fine.restart_min_improvement, 0.05)

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

            self.assertEqual(optimize.call_count, 3)
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
