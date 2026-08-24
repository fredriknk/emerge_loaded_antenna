"""Frequency-scalable, multi-seed robust loaded-antenna optimization."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("EMERGE_STD_LOGLEVEL", "ERROR")

import numpy as np
from scipy.optimize import differential_evolution

from emerge_loaded_antenna import (
    REFERENCE_DESIGN_FREQUENCY_HZ,
    SOLVER_CHOICES,
    AntennaDesign,
    DesignSpace,
    DesignVariable,
    EvaluationRecord,
    FrequencySweep,
    MeshSettings,
    OpenRegionSettings,
    RobustGainObjective,
    SimulationOptions,
    load_design,
    load_reference_design,
    save_design,
    validate_convergence_certificate,
)

METRIC_FIELDS = (
    "s11_low_db",
    "center_s11_db",
    "s11_high_db",
    "worst_s11_db",
    "useful_gain_dbi",
    "target_theta_deg",
    "target_phi_deg",
    "target_gain_dbi",
    "ring_sampled_theta_deg",
    "ring_min_gain_dbi",
    "ring_p10_gain_dbi",
    "ring_mean_gain_dbi",
    "ring_p90_gain_dbi",
    "ring_peak_gain_dbi",
    "ring_ripple_db",
    "ring_peak_to_null_db",
    "ring_beamwidth_deg",
    "ring_target_count",
    "ring_worst_target_theta_deg",
    "ring_minimum_across_targets_dbi",
    "ring_maximum_ripple_across_targets_db",
    "ring_beamwidth_rms_error_deg",
    "peak_gain_dbi",
    "peak_theta_deg",
    "peak_phi_deg",
    "horizon_min_gain_dbi",
    "horizon_p10_gain_dbi",
    "horizon_mean_gain_dbi",
    "horizon_ripple_db",
    "antenna_height_m",
    "mismatch_penalty",
    "s11_margin_db",
    "s11_margin_reward",
    "s11_margin_target_db",
    "pattern_penalty",
    "target_beamwidth_deg",
    "elevation_beamwidth_deg",
    "azimuth_beamwidth_deg",
    "beamwidth_error_deg",
    "beamwidth_penalty",
    "height_penalty",
    "confirmation_requested_runs",
    "confirmation_successful_runs",
    "confirmation_preliminary_score",
    "confirmation_confirmed_score",
    "confirmation_score_min",
    "confirmation_score_max",
    "confirmation_score_spread",
    "confirmation_score_tolerance",
    "confirmation_consensus_runs",
    "confirmation_outlier_runs",
    "confirmation_consistent",
    "confirmation_quarantined",
    "confirmation_incumbent_score",
    "confirmation_returned_score",
)

C0 = 299_792_458.0
BASE_SECTION_START_LAMBDA = 0.25
COLLINEAR_SECTION_START_LAMBDA = 0.50
MONOPOLE_LENGTH_RANGE_LAMBDA = (0.18, 0.70)
COLLINEAR_SECTION_RANGE_LAMBDA = (0.15, 0.72)
RADIAL_LENGTH_RANGE_LAMBDA = (0.15, 0.40)
COIL_PITCH_RANGE_LAMBDA = (0.010, 0.040)
COIL_RADIUS_RANGE_LAMBDA = (0.015, 0.050)
MINIMUM_STRAIGHT_WIRE_DIAMETERS = 12.0
MINIMUM_PITCH_WIRE_DIAMETERS = 1.5
MINIMUM_RADIUS_WIRE_DIAMETERS = 2.5

FINETUNE_NEAR_FRACTION = 0.50
FINETUNE_WIDE_FRACTION = 0.30
FINETUNE_GLOBAL_FRACTION = 0.20
DEFAULT_FINETUNE_NEAR_RADIUS = 0.03
DEFAULT_FINETUNE_WIDE_RADIUS = 0.10
DEFAULT_FINETUNE_MUTATION = (0.20, 0.60)
DEFAULT_FINETUNE_RECOMBINATION = 0.30
DEFAULT_RESTART_STAGNATION_GENERATIONS = 10
DEFAULT_RESTART_MIN_IMPROVEMENT = 0.05
DEFAULT_LOCAL_SEARCH_EVALUATIONS = 24
DEFAULT_LOCAL_SEARCH_STEP = 0.03
DEFAULT_LOCAL_SEARCH_MIN_STEP = 0.001
DEFAULT_POLISH_MIN_IMPROVEMENT = 0.001
DEFAULT_AUTOMATIC_ROUGH_FRACTION = 0.65
DEFAULT_BROAD_SEED_COUNT = 4
DEFAULT_FINE_SEED_COUNT = 2
SEED_UPPER_BOUND = 2**32


def elapsed_text(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return result


def random_seeds(count: int) -> tuple[int, ...]:
    """Generate distinct system-random uint32 seeds for one campaign."""
    if count < 1:
        raise ValueError("seed count must be positive")
    generated: list[int] = []
    seen: set[int] = set()
    while len(generated) < count:
        seed = secrets.randbelow(SEED_UPPER_BOUND)
        if seed in seen:
            continue
        seen.add(seed)
        generated.append(seed)
    return tuple(generated)


def parse_coil_counts(value: str) -> tuple[int, ...]:
    """Parse unique non-negative coil counts while preserving CLI order."""
    try:
        counts = parse_int_list(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "coil counts must be comma-separated non-negative integers"
        ) from error
    if any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError("coil counts must be non-negative")
    return tuple(dict.fromkeys(counts))


def parse_turn_cases(value: str) -> tuple[tuple[int, ...], ...]:
    cases = []
    try:
        for item in value.split(","):
            item = item.lower().strip()
            if item in {"0", "none"}:
                case = ()
            else:
                parts = item.split("x")
                if not parts or any(not part for part in parts):
                    raise ValueError
                case = tuple(int(part) for part in parts)
            if any(turns < 1 for turns in case):
                raise ValueError
            cases.append(case)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "turn cases must look like none, 1, 1x2 or 1x2x1"
        ) from error
    if not cases:
        raise argparse.ArgumentTypeError("provide at least one turn case")
    return tuple(cases)


def format_turn_case(turn_case: tuple[int, ...]) -> str:
    return "x".join(str(turns) for turns in turn_case) or "none"


def format_parameter(name: str, value: float) -> str:
    if name.endswith(("turns", "radial_count")):
        return f"{name}={value:.0f}"
    if name.endswith("_deg"):
        return f"{name}={value:.1f} deg"
    return f"{name}={value * 1e3:.2f} mm"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def free_space_wavelength(frequency_hz: float) -> float:
    """Return free-space wavelength after validating a frequency."""
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    return C0 / frequency_hz


def default_maximum_height(
    frequency_hz: float,
    maximum_coil_count: int,
) -> float:
    """Return the topology-aware soft height allowance in metres."""
    if (
        isinstance(maximum_coil_count, bool)
        or int(maximum_coil_count) != maximum_coil_count
        or maximum_coil_count < 0
    ):
        raise ValueError("maximum_coil_count must be a non-negative integer")
    allowance_wavelengths = MONOPOLE_LENGTH_RANGE_LAMBDA[
        1
    ] + COLLINEAR_SECTION_START_LAMBDA * int(maximum_coil_count)
    return allowance_wavelengths * free_space_wavelength(frequency_hz)


def design_for_coil_count(
    base: AntennaDesign,
    coil_count: int,
    frequency_hz: float = REFERENCE_DESIGN_FREQUENCY_HZ,
) -> AntennaDesign:
    """Resize a design using monopole and collinear wavelength priors."""
    if isinstance(coil_count, bool) or int(coil_count) != coil_count or coil_count < 0:
        raise ValueError("coil_count must be a non-negative integer")
    coil_count = int(coil_count)
    if base.coil_count == coil_count:
        return base

    wavelength = free_space_wavelength(frequency_hz)
    wire_diameter = 2 * base.wire_radius
    minimum_straight = MINIMUM_STRAIGHT_WIRE_DIAMETERS * wire_diameter
    base_section = max(BASE_SECTION_START_LAMBDA * wavelength, minimum_straight)
    collinear_section = max(
        COLLINEAR_SECTION_START_LAMBDA * wavelength,
        minimum_straight,
    )
    straight_lengths = (
        (base_section,)
        if coil_count == 0
        else (base_section,) + (collinear_section,) * coil_count
    )
    coils = base.coils[:coil_count]
    template = (
        base.coils[-1] if base.coils else load_reference_design(frequency_hz).coils[0]
    )
    coils += tuple(template for _ in range(coil_count - len(coils)))
    return replace(
        base,
        straight_lengths=straight_lengths,
        coils=coils,
    )


def design_for_turn_case(
    base: AntennaDesign,
    turn_case: tuple[int, ...],
    frequency_hz: float = REFERENCE_DESIGN_FREQUENCY_HZ,
) -> AntennaDesign:
    design = design_for_coil_count(base, len(turn_case), frequency_hz)
    return replace(
        design,
        coils=tuple(
            replace(coil, turns=turns) for coil, turns in zip(design.coils, turn_case)
        ),
    )


def apply_design_overrides(
    design: AntennaDesign,
    wire_diameter_mm: float | None = None,
) -> AntennaDesign:
    """Apply a fixed physical wire diameter to a starting design."""
    if wire_diameter_mm is None:
        return design
    if not np.isfinite(wire_diameter_mm) or wire_diameter_mm <= 0:
        raise ValueError("wire_diameter_mm must be finite and positive")
    overridden = replace(
        design,
        wire_radius=float(wire_diameter_mm) * 1e-3 / 2,
    )
    overridden.validate()
    return overridden


def make_space(
    base: AntennaDesign,
    frequency_hz: float = REFERENCE_DESIGN_FREQUENCY_HZ,
    lock_coils: bool = False,
) -> DesignSpace:
    """Create wavelength-scaled bounds with optional shared coil geometry."""
    wavelength = free_space_wavelength(frequency_hz)
    wire_diameter = 2 * base.wire_radius

    def enclose(
        bounds: tuple[float, float],
        value: float,
    ) -> tuple[float, float]:
        lower, upper = bounds
        return min(lower, 0.8 * value), max(upper, 1.2 * value)

    original_pitches = tuple(coil.pitch for coil in base.coils)
    original_radii = tuple(coil.radius for coil in base.coils)
    if base.coils and lock_coils:
        shared_pitch = float(np.mean(original_pitches))
        minimum_radius = max(0.5001 * coil.transition_offset for coil in base.coils)
        shared_radius = max(float(np.mean(original_radii)), minimum_radius)
        base = replace(
            base,
            coils=tuple(
                replace(
                    coil,
                    pitch=shared_pitch,
                    radius=shared_radius,
                )
                for coil in base.coils
            ),
        )

    straight_range = (
        MONOPOLE_LENGTH_RANGE_LAMBDA
        if base.coil_count == 0
        else COLLINEAR_SECTION_RANGE_LAMBDA
    )
    minimum_straight = max(
        straight_range[0] * wavelength,
        MINIMUM_STRAIGHT_WIRE_DIAMETERS * wire_diameter,
    )
    maximum_straight = max(
        straight_range[1] * wavelength,
        1.5 * minimum_straight,
    )
    default_straight_bounds = ((minimum_straight, maximum_straight),) * len(
        base.straight_lengths
    )
    straight_bounds = tuple(
        enclose(bounds, value)
        for bounds, value in zip(
            default_straight_bounds,
            base.straight_lengths,
        )
    )

    variables = [
        DesignVariable(f"straight_lengths.{index}", lower, upper)
        for index, (lower, upper) in enumerate(straight_bounds)
    ]
    if base.coils and lock_coils:
        minimum_pitch = max(
            COIL_PITCH_RANGE_LAMBDA[0] * wavelength,
            MINIMUM_PITCH_WIRE_DIAMETERS * wire_diameter,
        )
        pitch_bounds = (
            minimum_pitch,
            max(
                COIL_PITCH_RANGE_LAMBDA[1] * wavelength,
                6.0 * wire_diameter,
                1.5 * minimum_pitch,
            ),
        )
        for pitch in original_pitches:
            pitch_bounds = enclose(pitch_bounds, pitch)
        variables.append(
            DesignVariable(
                "coils.0.pitch",
                *pitch_bounds,
                linked_paths=tuple(
                    f"coils.{index}.pitch" for index in range(1, base.coil_count)
                ),
                label="shared_coil_pitch",
            )
        )
        hard_minimum_radius = max(
            0.5001 * coil.transition_offset for coil in base.coils
        )
        preferred_minimum_radius = max(
            COIL_RADIUS_RANGE_LAMBDA[0] * wavelength,
            MINIMUM_RADIUS_WIRE_DIAMETERS * wire_diameter,
            max(
                0.5001 * coil.transition_offset + base.wire_radius
                for coil in base.coils
            ),
        )
        radius_bounds = (
            preferred_minimum_radius,
            max(
                COIL_RADIUS_RANGE_LAMBDA[1] * wavelength,
                8.0 * wire_diameter,
                1.5 * preferred_minimum_radius,
            ),
        )
        for radius in original_radii:
            radius_bounds = enclose(radius_bounds, radius)
        radius_bounds = (
            max(hard_minimum_radius, radius_bounds[0]),
            radius_bounds[1],
        )
        variables.append(
            DesignVariable(
                "coils.0.radius",
                *radius_bounds,
                linked_paths=tuple(
                    f"coils.{index}.radius" for index in range(1, base.coil_count)
                ),
                label="shared_coil_radius",
            )
        )
    else:
        for index, coil in enumerate(base.coils):
            minimum_pitch = max(
                COIL_PITCH_RANGE_LAMBDA[0] * wavelength,
                MINIMUM_PITCH_WIRE_DIAMETERS * wire_diameter,
            )
            pitch_bounds = enclose(
                (
                    minimum_pitch,
                    max(
                        COIL_PITCH_RANGE_LAMBDA[1] * wavelength,
                        6.0 * wire_diameter,
                        1.5 * minimum_pitch,
                    ),
                ),
                coil.pitch,
            )
            variables.append(DesignVariable(f"coils.{index}.pitch", *pitch_bounds))
        for index, coil in enumerate(base.coils):
            hard_minimum_radius = 0.5001 * coil.transition_offset
            preferred_minimum_radius = max(
                COIL_RADIUS_RANGE_LAMBDA[0] * wavelength,
                MINIMUM_RADIUS_WIRE_DIAMETERS * wire_diameter,
                0.5001 * coil.transition_offset + base.wire_radius,
            )
            radius_bounds = enclose(
                (
                    preferred_minimum_radius,
                    max(
                        COIL_RADIUS_RANGE_LAMBDA[1] * wavelength,
                        8.0 * wire_diameter,
                        1.5 * preferred_minimum_radius,
                    ),
                ),
                coil.radius,
            )
            radius_bounds = (
                max(hard_minimum_radius, radius_bounds[0]),
                radius_bounds[1],
            )
            variables.append(DesignVariable(f"coils.{index}.radius", *radius_bounds))
    minimum_radial = max(
        RADIAL_LENGTH_RANGE_LAMBDA[0] * wavelength,
        MINIMUM_STRAIGHT_WIRE_DIAMETERS * wire_diameter,
    )
    radial_bounds = enclose(
        (
            minimum_radial,
            max(
                RADIAL_LENGTH_RANGE_LAMBDA[1] * wavelength,
                1.5 * minimum_radial,
            ),
        ),
        base.radial_length,
    )
    angle_bounds = (
        max(0.1, min(5.0, base.radial_angle_deg - 10.0)),
        min(89.9, max(75.0, base.radial_angle_deg + 10.0)),
    )
    variables.extend(
        (
            DesignVariable("radial_length", *radial_bounds),
            DesignVariable("radial_angle_deg", *angle_bounds),
        )
    )
    return DesignSpace(
        base,
        variables,
    )


def finetune_population_counts(population: int) -> tuple[int, int, int]:
    """Split a fine-tuning population into near, wide, and global samples."""
    if population < 5:
        raise ValueError("population must contain at least five candidates")
    near = max(1, round(FINETUNE_NEAR_FRACTION * population))
    wide = round(FINETUNE_WIDE_FRACTION * population)
    global_count = population - near - wide
    if global_count < 1:
        global_count = 1
        wide = population - near - global_count
    return near, wide, global_count


def build_finetune_population(
    space: DesignSpace,
    population: int,
    seed: int,
    center: tuple[float, ...] | np.ndarray | None = None,
    near_radius: float = DEFAULT_FINETUNE_NEAR_RADIUS,
    wide_radius: float = DEFAULT_FINETUNE_WIDE_RADIUS,
) -> np.ndarray:
    """Build a deterministic 50/30/20 local, wider-local, and global mix.

    Radii are fractions of each variable's complete bound span.  The first
    candidate is always the clipped incumbent, so a restart cannot discard it.
    """
    if not 0 < near_radius <= wide_radius <= 1:
        raise ValueError("fine-tune radii must satisfy 0 < near <= wide <= 1")
    near_count, wide_count, global_count = finetune_population_counts(population)
    center_vector = np.asarray(
        space.initial_vector if center is None else center,
        dtype=float,
    )
    if center_vector.shape != (len(space.variables),):
        raise ValueError("fine-tune population center has the wrong dimension")
    center_normalized = np.clip(space.normalize(center_vector), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    normalized = np.empty((population, len(space.variables)), dtype=float)
    normalized[0] = center_normalized
    if near_count > 1:
        normalized[1:near_count] = rng.uniform(
            np.maximum(0.0, center_normalized - near_radius),
            np.minimum(1.0, center_normalized + near_radius),
            size=(near_count - 1, len(space.variables)),
        )
    wide_start = near_count
    wide_stop = wide_start + wide_count
    if wide_count:
        normalized[wide_start:wide_stop] = rng.uniform(
            np.maximum(0.0, center_normalized - wide_radius),
            np.minimum(1.0, center_normalized + wide_radius),
            size=(wide_count, len(space.variables)),
        )
    normalized[wide_stop:] = rng.uniform(
        0.0,
        1.0,
        size=(global_count, len(space.variables)),
    )
    return space.denormalize(normalized)


def normalized_population_diversity(
    population: np.ndarray | None,
    space: DesignSpace,
) -> float:
    """Return mean per-variable standard deviation in normalized bounds."""
    if population is None:
        return float("nan")
    values = np.asarray(population, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(space.variables):
        return float("nan")
    normalized = np.clip(
        np.asarray([space.normalize(row) for row in values]),
        0.0,
        1.0,
    )
    return float(np.mean(np.std(normalized, axis=0)))


def record_is_feasible(
    record: EvaluationRecord,
    maximum_s11_db: float,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether a successful record clears every campaign constraint."""
    if record.error is not None or not np.isfinite(record.score):
        return False
    worst_s11 = record.metrics.get("worst_s11_db", record.s11_db)
    if worst_s11 is None or worst_s11 > maximum_s11_db + tolerance:
        return False
    return all(
        record.metrics.get(name, 0.0) <= tolerance
        for name in ("mismatch_penalty", "pattern_penalty", "height_penalty")
    )


def record_is_confirmation_eligible(record: EvaluationRecord) -> bool:
    """Return whether a record is safe to use as a search incumbent."""
    return record.confirmation_status in {
        "not_requested",
        "confirmed",
        "confirmed_with_outliers",
    }


def record_preference_key(
    record: EvaluationRecord,
    maximum_s11_db: float,
) -> tuple[int, float]:
    """Rank feasible records ahead of infeasible ones, then use objective score."""
    return (0 if record_is_feasible(record, maximum_s11_db) else 1, record.score)


def confirmed_elites(
    objective: object,
    limit: int,
) -> list[EvaluationRecord]:
    """Return distinct, score-ordered successful confirmed records."""
    records = getattr(objective, "history", ())
    ordered = sorted(
        (
            record
            for record in records
            if record.error is None
            and np.isfinite(record.score)
            and record_is_confirmation_eligible(record)
        ),
        key=lambda record: record.score,
    )
    result = []
    seen: set[tuple[float, ...]] = set()
    for record in ordered:
        vector = tuple(float(value) for value in record.vector)
        if vector in seen:
            continue
        seen.add(vector)
        result.append(record)
        if len(result) == limit:
            break
    return result


def feasible_elites(
    objective: object,
    maximum_s11_db: float,
    limit: int,
) -> list[EvaluationRecord]:
    """Return distinct, score-ordered feasible records from one run."""
    records = getattr(objective, "history", ())
    ordered = sorted(
        (
            record
            for record in records
            if record_is_confirmation_eligible(record)
            and record_is_feasible(record, maximum_s11_db)
        ),
        key=lambda record: record.score,
    )
    result = []
    seen: set[tuple[float, ...]] = set()
    for record in ordered:
        vector = tuple(float(value) for value in record.vector)
        if vector in seen:
            continue
        seen.add(vector)
        result.append(record)
        if len(result) == limit:
            break
    return result


def preferred_incumbent(
    objective: object,
    maximum_s11_db: float,
) -> EvaluationRecord | None:
    """Prefer a confirmation-aware incumbent, then a feasible run record.

    Confirmation is implemented by the objective layer.  Looking for its
    conventional attributes here keeps the campaign runner compatible with
    both confirming and lightweight/test objectives.
    """
    for name in (
        "verified_best_record",
        "best_verified_record",
        "confirmed_best_record",
        "best_confirmed_record",
    ):
        candidate = getattr(objective, name, None)
        candidate = candidate() if callable(candidate) else candidate
        if isinstance(candidate, EvaluationRecord) and record_is_confirmation_eligible(
            candidate
        ):
            return candidate
    elites = feasible_elites(objective, maximum_s11_db, 1)
    if elites:
        return elites[0]
    candidate = getattr(objective, "best_record", None)
    candidate = candidate() if callable(candidate) else candidate
    return (
        candidate
        if isinstance(candidate, EvaluationRecord)
        and record_is_confirmation_eligible(candidate)
        else None
    )


@dataclass(frozen=True)
class FinetuneBudget:
    differential_evolution: int
    local_search: int


def split_finetune_budget(
    total: int,
    population: int,
    requested_local: int,
) -> FinetuneBudget:
    """Reserve whole DE population batches while retaining the total budget."""
    if total < population or population < 5:
        raise ValueError("the evaluation budget must contain one population")
    if requested_local <= 0 or total < 2 * population:
        return FinetuneBudget(total, 0)
    local_batches = max(1, int(np.ceil(requested_local / population)))
    local = min(total - population, local_batches * population)
    return FinetuneBudget(total - local, local)


@dataclass
class GenerationMonitor:
    """Track and print true DE generations, stagnation, and diversity."""

    objective: object
    space: DesignSpace
    run_name: str
    generation_offset: int = 0
    restart_index: int = 0
    restart_after: int = 0
    incumbent_score: float | None = None
    improvement_tolerance: float = DEFAULT_RESTART_MIN_IMPROVEMENT
    maximum_s11_db: float = -10.0
    incumbent_feasible: bool = False
    stage_generations: int = 0
    stagnation_generations: int = 0
    last_diversity: float = float("nan")
    stopped_for_stagnation: bool = False

    def __call__(self, intermediate_result: object) -> bool:
        self.stage_generations += 1
        current_score = float(getattr(intermediate_result, "fun", float("inf")))
        feasible_now = any(
            record_is_confirmation_eligible(record)
            and record_is_feasible(record, self.maximum_s11_db)
            for record in getattr(self.objective, "history", ())
        )
        if (
            self.incumbent_score is None
            or current_score < self.incumbent_score - self.improvement_tolerance
            or (feasible_now and not self.incumbent_feasible)
        ):
            self.incumbent_score = current_score
            self.stagnation_generations = 0
        else:
            self.stagnation_generations += 1
        self.incumbent_feasible = self.incumbent_feasible or feasible_now
        self.last_diversity = normalized_population_diversity(
            getattr(intermediate_result, "population", None),
            self.space,
        )
        generation = self.generation_offset + self.stage_generations
        diversity_text = (
            f"{self.last_diversity:.4f}" if np.isfinite(self.last_diversity) else "n/a"
        )
        print(
            f"    generation {generation:4d} | run best {current_score:8.4f} | "
            f"stagnant {self.stagnation_generations:2d} | "
            f"diversity {diversity_text} | restart {self.restart_index}",
            flush=True,
        )
        self.stopped_for_stagnation = (
            self.restart_after > 0 and self.stagnation_generations >= self.restart_after
        )
        return self.stopped_for_stagnation


@dataclass(frozen=True)
class LocalSearchStats:
    evaluations: int
    simulation_evaluations: int
    elite_count: int
    improvements: int
    final_step: float


@dataclass(frozen=True)
class CoordinatePolishStats:
    evaluations: int
    simulation_evaluations: int
    improvements: int
    sweeps: int
    final_step: float
    converged: bool


def coordinate_polish(
    objective: object,
    space: DesignSpace,
    maximum_s11_db: float,
    evaluation_budget: int,
    *,
    initial_step: float = DEFAULT_LOCAL_SEARCH_STEP,
    minimum_step: float = DEFAULT_LOCAL_SEARCH_MIN_STEP,
    minimum_improvement: float = DEFAULT_POLISH_MIN_IMPROVEMENT,
) -> CoordinatePolishStats:
    """Polish one incumbent by changing one normalized coordinate at a time."""
    if evaluation_budget <= 0:
        return CoordinatePolishStats(0, 0, 0, 0, initial_step, False)
    if not 0 < minimum_step <= initial_step <= 1:
        raise ValueError("polish steps must satisfy 0 < minimum <= initial <= 1")
    if not np.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("polish minimum improvement must be non-negative")

    incumbent = preferred_incumbent(objective, maximum_s11_db)
    evaluations = 0
    simulation_evaluations = 0
    if incumbent is None:
        history = getattr(objective, "history", None)
        history_before = len(history) if history is not None else 0
        simulations_before = objective_evaluation_count(objective)
        objective(space.initial_vector)
        simulation_evaluations += max(
            1,
            objective_evaluation_count(objective) - simulations_before,
        )
        evaluations += 1
        incumbent = (
            history[-1]
            if history is not None and len(history) > history_before
            else None
        )
    if not isinstance(incumbent, EvaluationRecord):
        return CoordinatePolishStats(
            evaluations,
            simulation_evaluations,
            0,
            0,
            initial_step,
            False,
        )

    point = np.clip(space.normalize(incumbent.vector), 0.0, 1.0)
    current = incumbent
    step = initial_step
    improvements = 0
    sweeps = 0
    converged = False

    def candidate_is_better(
        candidate: EvaluationRecord,
        reference: EvaluationRecord,
    ) -> bool:
        candidate_feasible = record_is_feasible(candidate, maximum_s11_db)
        reference_feasible = record_is_feasible(reference, maximum_s11_db)
        if candidate_feasible != reference_feasible:
            return candidate_feasible
        return candidate.score < reference.score - minimum_improvement

    while evaluations < evaluation_budget:
        sweep_improved = False
        completed_sweep = True
        for coordinate in range(len(point)):
            if evaluations >= evaluation_budget:
                completed_sweep = False
                break
            origin = point.copy()
            coordinate_candidates: list[tuple[EvaluationRecord, np.ndarray]] = []
            decoded_seen: set[tuple[float, ...]] = {tuple(current.vector)}
            for direction in (1.0, -1.0):
                if evaluations >= evaluation_budget:
                    completed_sweep = False
                    break
                candidate = origin.copy()
                candidate[coordinate] = float(
                    np.clip(candidate[coordinate] + direction*step, 0.0, 1.0)
                )
                decoded = tuple(float(value) for value in space.denormalize(candidate))
                if decoded in decoded_seen:
                    continue
                decoded_seen.add(decoded)
                history = getattr(objective, "history", None)
                history_before = len(history) if history is not None else 0
                simulations_before = objective_evaluation_count(objective)
                objective(decoded)
                simulation_evaluations += max(
                    1,
                    objective_evaluation_count(objective) - simulations_before,
                )
                evaluations += 1
                record = (
                    history[-1]
                    if history is not None and len(history) > history_before
                    else None
                )
                if (
                    isinstance(record, EvaluationRecord)
                    and record_is_confirmation_eligible(record)
                ):
                    coordinate_candidates.append((record, candidate))
            if coordinate_candidates:
                best_record, best_point = min(
                    coordinate_candidates,
                    key=lambda item: record_preference_key(
                        item[0],
                        maximum_s11_db,
                    ),
                )
                if candidate_is_better(best_record, current):
                    current = best_record
                    point = best_point
                    improvements += 1
                    sweep_improved = True
        if not completed_sweep:
            break
        sweeps += 1
        if sweep_improved:
            continue
        if step <= minimum_step:
            converged = True
            break
        step = max(minimum_step, 0.5*step)

    return CoordinatePolishStats(
        evaluations,
        simulation_evaluations,
        improvements,
        sweeps,
        step,
        converged,
    )


def normalized_pattern_search(
    objective: object,
    space: DesignSpace,
    maximum_s11_db: float,
    evaluation_budget: int,
    *,
    initial_step: float = DEFAULT_LOCAL_SEARCH_STEP,
    minimum_step: float = DEFAULT_LOCAL_SEARCH_MIN_STEP,
    elite_count: int = 3,
) -> LocalSearchStats:
    """Run deterministic bounded coordinate searches around feasible elites."""
    if evaluation_budget <= 0:
        return LocalSearchStats(0, 0, 0, 0, initial_step)
    if not 0 < minimum_step <= initial_step <= 1:
        raise ValueError("local-search steps must satisfy 0 < minimum <= initial <= 1")
    if elite_count < 1:
        raise ValueError("elite_count must be positive")
    feasible = feasible_elites(objective, maximum_s11_db, elite_count)
    feasible_count = len(feasible)
    elites = feasible or confirmed_elites(objective, elite_count)
    if not elites:
        elites = [
            EvaluationRecord(
                tuple(float(value) for value in space.initial_vector),
                float("inf"),
                None,
                None,
            )
        ]

    base_share, remainder = divmod(evaluation_budget, len(elites))
    evaluations = 0
    simulation_evaluations = 0
    improvements = 0
    final_step = initial_step
    for elite_index, elite in enumerate(elites):
        budget = base_share + (1 if elite_index < remainder else 0)
        point = np.clip(space.normalize(elite.vector), 0.0, 1.0)
        score = elite.score
        point_is_feasible = record_is_feasible(elite, maximum_s11_db)
        step = initial_step
        spent = 0
        while spent < budget:
            cycle_improved = False
            for coordinate in range(len(point)):
                if spent >= budget:
                    break
                for direction in (1.0, -1.0):
                    candidate = point.copy()
                    candidate_value = float(np.clip(
                        candidate[coordinate] + direction * step,
                        0.0,
                        1.0,
                    ))
                    if candidate_value == point[coordinate]:
                        boundary = 1.0 if direction > 0 else 0.0
                        candidate_value = float(
                            np.nextafter(point[coordinate], boundary)
                        )
                    if candidate_value == point[coordinate]:
                        continue
                    candidate[coordinate] = candidate_value
                    history = getattr(objective, "history", None)
                    history_before = len(history) if history is not None else 0
                    simulations_before = objective_evaluation_count(objective)
                    candidate_score = float(objective(space.denormalize(candidate)))
                    simulations_after = objective_evaluation_count(objective)
                    simulation_evaluations += max(
                        1,
                        simulations_after - simulations_before,
                    )
                    spent += 1
                    evaluations += 1
                    record = (
                        history[-1]
                        if history is not None and len(history) > history_before
                        else None
                    )
                    candidate_is_eligible = (
                        isinstance(record, EvaluationRecord)
                        and record_is_confirmation_eligible(record)
                    )
                    candidate_is_feasible = candidate_is_eligible and record_is_feasible(
                        record,
                        maximum_s11_db,
                    )
                    if candidate_is_eligible and (
                        (candidate_is_feasible and not point_is_feasible)
                        or (
                            candidate_is_feasible == point_is_feasible
                            and candidate_score < score
                        )
                    ):
                        point = candidate
                        score = candidate_score
                        point_is_feasible = candidate_is_feasible
                        cycle_improved = True
                        improvements += 1
                        break
                    if spent >= budget:
                        break
            step = (
                min(initial_step, 1.25 * step)
                if cycle_improved
                else max(minimum_step, 0.5 * step)
            )
        final_step = min(final_step, step)
    return LocalSearchStats(
        evaluations,
        simulation_evaluations,
        feasible_count,
        improvements,
        final_step,
    )


def result_payload(
    record: EvaluationRecord,
    space: DesignSpace,
    turn_case: tuple[int, ...],
    seed: int,
    evaluations: int,
    frequency_hz: float,
    simulation_metadata: dict | None = None,
    candidate_evaluations: int | None = None,
    simulation_evaluations: int | None = None,
    maximum_s11_db: float | None = None,
) -> dict:
    design = space.decode(record.vector)
    payload = {
        "frequency_hz": frequency_hz,
        "objective": record.score,
        "s11_db": record.s11_db,
        "peak_gain_dbi": record.peak_gain_dbi,
        "metrics": dict(record.metrics),
        "coil_count": len(turn_case),
        "turn_case": list(turn_case),
        "seed": seed,
        "evaluations_at_save": evaluations,
        "confirmation_status": getattr(
            record,
            "confirmation_status",
            "not_requested",
        ),
        "simulation_runs": int(getattr(record, "simulation_runs", 1)),
        "variables": dict(zip(space.names, record.vector)),
        "search_space": {
            "initial_variables": {
                name: float(value)
                for name, value in zip(space.names, space.initial_vector)
            },
            "bounds": {
                name: [lower, upper]
                for name, (lower, upper) in zip(space.names, space.bounds)
            },
        },
        "design": asdict(design),
    }
    if simulation_metadata is not None:
        payload["simulation"] = simulation_metadata
    if candidate_evaluations is not None:
        payload["candidate_evaluations_at_save"] = candidate_evaluations
    if simulation_evaluations is not None:
        payload["simulation_evaluations_at_save"] = simulation_evaluations
    if maximum_s11_db is not None:
        payload["feasible"] = record_is_feasible(record, maximum_s11_db)
    return payload


def progress_goal_text(
    record: EvaluationRecord,
    simulation_metadata: dict | None,
) -> str:
    """Format the active pattern goal for live campaign reporting."""
    objective = (
        simulation_metadata.get("objective", {})
        if isinstance(simulation_metadata, dict)
        else {}
    )
    mode = objective.get("pattern_mode", "horizon")
    metrics = record.metrics
    if mode == "ring":
        raw_targets = objective.get("target_theta_degrees")
        target_thetas = (
            tuple(float(value) for value in raw_targets)
            if isinstance(raw_targets, (list, tuple)) and raw_targets
            else (float(objective.get("target_theta_deg", 90.0)),)
        )
        if len(target_thetas) > 1:
            labels = []
            for index, target_theta in enumerate(target_thetas):
                sampled_theta = float(
                    metrics.get(
                        f"ring_{index}_sampled_theta_deg",
                        target_theta,
                    )
                )
                sampled_label = (
                    f"{target_theta:g}->{sampled_theta:g} deg"
                    if sampled_theta != target_theta
                    else f"{target_theta:g} deg"
                )
                labels.append(
                    f"{sampled_label}: "
                    f"{metrics.get(f'ring_{index}_p10_gain_dbi', float('nan')):.2f}"
                )
            text = (
                "Ring P10s ["
                + ", ".join(labels)
                + "] dBi | worst "
                + f"{metrics.get('ring_p10_gain_dbi', float('nan')):.2f} dBi"
            )
            beamwidth_goal = objective.get("target_beamwidth_deg")
            if beamwidth_goal is not None:
                widths = [
                    metrics.get(f"ring_{index}_beamwidth_deg")
                    for index in range(len(target_thetas))
                ]
                if all(width is not None for width in widths):
                    text += (
                        " | BWs "
                        + "/".join(f"{float(width):g}" for width in widths)
                        + f" deg (goal {float(beamwidth_goal):g} each)"
                    )
            return text
        target_theta = target_thetas[0]
        sampled_theta = float(
            metrics.get("ring_sampled_theta_deg", target_theta)
        )
        text = (
            f"Ring P10 {metrics.get('ring_p10_gain_dbi', float('nan')):5.2f} "
            f"dBi @ {sampled_theta:g} deg (goal {target_theta:g} deg)"
        )
        beamwidth_goal = objective.get("target_beamwidth_deg")
        if beamwidth_goal is not None and "ring_beamwidth_deg" in metrics:
            text += (
                f" | BW {metrics['ring_beamwidth_deg']:g} deg "
                f"(goal {float(beamwidth_goal):g} deg)"
            )
        return text
    if mode == "directional":
        target_theta = float(objective.get("target_theta_deg", 90.0))
        target_phi = float(objective.get("target_phi_deg", 0.0))
        text = (
            f"Target gain {metrics.get('target_gain_dbi', float('nan')):5.2f} "
            f"dBi @ theta {target_theta:g}, phi {target_phi:g} deg"
        )
        beamwidth_goal = objective.get("target_beamwidth_deg")
        if (
            beamwidth_goal is not None
            and "elevation_beamwidth_deg" in metrics
            and "azimuth_beamwidth_deg" in metrics
        ):
            text += (
                f" | BW el/az {metrics['elevation_beamwidth_deg']:g}/"
                f"{metrics['azimuth_beamwidth_deg']:g} deg "
                f"(goal {float(beamwidth_goal):g} deg)"
            )
        return text
    if mode == "peak":
        peak_gain = (
            float(record.peak_gain_dbi)
            if record.peak_gain_dbi is not None
            else float("nan")
        )
        return f"Peak gain {peak_gain:5.2f} dBi"
    return (
        "Horizon P10 "
        f"{metrics.get('horizon_p10_gain_dbi', float('nan')):5.2f} dBi "
        "@ 90 deg"
    )


class CampaignProgress:
    """Persistent evaluation log and concise whole-campaign reporting."""

    def __init__(
        self,
        output: Path,
        total: int,
        report_every: int,
        variable_names: tuple[str, ...],
        frequency_hz: float,
        simulation_metadata: dict | None = None,
        allowed_existing: tuple[Path, ...] = (),
        maximum_s11_db: float = -10.0,
        stage: str = "optimization",
        additional_metric_fields: tuple[str, ...] = (),
    ):
        self.output = output
        self.total = total
        self.report_every = max(1, report_every)
        self.started = time.perf_counter()
        self.count = 0
        self.candidate_count = 0
        self.simulation_count = 0
        self.failures = 0
        self.best: EvaluationRecord | None = None
        self.best_space: DesignSpace | None = None
        self.best_turn_case: tuple[int, ...] = ()
        self.best_seed = 0
        self.topology_records: dict[tuple[int, ...], EvaluationRecord] = {}
        self.topology_payloads: dict[tuple[int, ...], dict] = {}
        self.last_reported_best: EvaluationRecord | None = None
        self.frequency_hz = frequency_hz
        self.maximum_s11_db = maximum_s11_db
        self.simulation_metadata = simulation_metadata
        self.stage = stage
        self.space: DesignSpace | None = None
        self.turn_case: tuple[int, ...] = ()
        self.seed = 0
        output.mkdir(parents=True, exist_ok=True)
        allowed = {path.resolve() for path in allowed_existing}
        existing = next(
            (path for path in output.iterdir() if path.resolve() not in allowed),
            None,
        )
        if existing is not None:
            raise RuntimeError(
                f"Campaign output directory is not empty: {output}. Choose a new "
                "--output directory; campaign resume is not supported."
            )
        try:
            self.file = (output / "evaluations.csv").open(
                "x",
                newline="",
                encoding="utf-8",
            )
        except FileExistsError as error:
            raise RuntimeError(
                f"Campaign output already contains evaluations.csv: {output}. "
                "Choose a new --output directory; campaign resume is not supported."
            ) from error
        extra_metrics = tuple(
            name for name in additional_metric_fields if name not in METRIC_FIELDS
        )
        fields = [
            "stage",
            "evaluation",
            "candidate",
            "simulation_evaluation",
            "simulation_runs",
            "confirmation_status",
            "elapsed_seconds",
            "coil_count",
            "turn_case",
            "seed",
            "score",
            "s11_db",
            "error",
            *METRIC_FIELDS,
            *extra_metrics,
            *variable_names,
        ]
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=fields,
            extrasaction="ignore",
        )
        self.writer.writeheader()

    def set_context(
        self,
        space: DesignSpace,
        turn_case: tuple[int, ...],
        seed: int,
    ) -> None:
        self.space = space
        self.turn_case = turn_case
        self.seed = seed

    def __call__(self, record: EvaluationRecord) -> None:
        if self.space is None:
            raise RuntimeError("progress context was not initialized")
        simulation_runs = int(getattr(record, "simulation_runs", 1))
        self.count += 1
        self.candidate_count += 1
        self.simulation_count += simulation_runs
        elapsed = time.perf_counter() - self.started
        if record.error is not None:
            self.failures += 1
        elif record_is_confirmation_eligible(record):
            topology_best = self.topology_records.get(self.turn_case)
            if topology_best is None or record_preference_key(
                record,
                self.maximum_s11_db,
            ) < record_preference_key(topology_best, self.maximum_s11_db):
                self.topology_records[self.turn_case] = record
                topology_payload = result_payload(
                    record,
                    self.space,
                    self.turn_case,
                    self.seed,
                    self.count,
                    self.frequency_hz,
                    self.simulation_metadata,
                    candidate_evaluations=self.candidate_count,
                    simulation_evaluations=self.simulation_count,
                    maximum_s11_db=self.maximum_s11_db,
                )
                self.topology_payloads[self.turn_case] = topology_payload
                topology_name = format_turn_case(self.turn_case)
                write_json(
                    self.output / f"turns_{topology_name}_best.json",
                    topology_payload,
                )
            if self.best is None or record_preference_key(
                record,
                self.maximum_s11_db,
            ) < record_preference_key(self.best, self.maximum_s11_db):
                self.best = record
                self.best_space = self.space
                self.best_turn_case = self.turn_case
                self.best_seed = self.seed
                payload = result_payload(
                    record,
                    self.space,
                    self.turn_case,
                    self.seed,
                    self.count,
                    self.frequency_hz,
                    self.simulation_metadata,
                    candidate_evaluations=self.candidate_count,
                    simulation_evaluations=self.simulation_count,
                    maximum_s11_db=self.maximum_s11_db,
                )
                write_json(self.output / "campaign_best.json", payload)

        row = {
            "stage": self.stage,
            "evaluation": self.count,
            "candidate": self.candidate_count,
            "simulation_evaluation": self.simulation_count,
            "simulation_runs": simulation_runs,
            "confirmation_status": getattr(
                record,
                "confirmation_status",
                "not_requested",
            ),
            "elapsed_seconds": f"{elapsed:.3f}",
            "coil_count": len(self.turn_case),
            "turn_case": format_turn_case(self.turn_case),
            "seed": self.seed,
            "score": record.score,
            "s11_db": record.s11_db,
            "peak_gain_dbi": record.peak_gain_dbi,
            "error": record.error or "",
        }
        row.update(record.metrics)
        row.update(dict(zip(self.space.names, record.vector)))
        self.writer.writerow(row)
        self.file.flush()

        confirmation_status = row["confirmation_status"]
        if confirmation_status == "confirmed_with_outliers":
            outliers = int(record.metrics.get("confirmation_outlier_runs", 0))
            print(
                f"    confirmation accepted median consensus; "
                f"discarded {outliers} outlier run(s)",
                flush=True,
            )
        elif confirmation_status == "quarantined":
            print(f"    {record.error}", flush=True)

        should_report = (
            self.candidate_count == 1
            or self.candidate_count % self.report_every == 0
            or self.candidate_count == self.total
        )
        if not should_report:
            return
        progress = min(self.candidate_count / self.total, 1.0)
        eta = elapsed / progress - elapsed if progress else 0.0
        if self.best is None:
            result_text = "no successful candidate yet"
        else:
            metrics = self.best.metrics
            result_text = (
                f"best {self.best.score:7.3f} | "
                f"S11 worst {metrics.get('worst_s11_db', float('nan')):6.2f} "
                f"dB (goal <= {self.maximum_s11_db:g} dB) | "
                f"{progress_goal_text(self.best, self.simulation_metadata)}"
            )
        print(
            f"[{self.candidate_count:5d}/{self.total} {100 * progress:5.1f}% | "
            f"{self.simulation_count:5d} simulations] "
            f"elapsed {elapsed_text(elapsed):>8} | ETA {elapsed_text(eta):>8} | "
            f"{result_text} | failed {self.failures}",
            flush=True,
        )
        if self.best is not None and self.best is not self.last_reported_best:
            assert self.best_space is not None
            parameters = ", ".join(
                format_parameter(name, value)
                for name, value in zip(self.best_space.names, self.best.vector)
            )
            print(
                f"    best case {format_turn_case(self.best_turn_case)} "
                f"seed {self.best_seed}: {parameters}",
                flush=True,
            )
            self.last_reported_best = self.best

    def topology_leaderboard(self) -> list[dict]:
        ordered_cases = sorted(
            self.topology_payloads,
            key=lambda turn_case: record_preference_key(
                self.topology_records[turn_case],
                self.maximum_s11_db,
            ),
        )
        return [self.topology_payloads[turn_case] for turn_case in ordered_cases]

    def close(self) -> None:
        self.file.close()
        write_json(
            self.output / "topology_leaderboard.json",
            {"topologies": self.topology_leaderboard()},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed optimization of useful gain, broadband match, "
            "pattern quality and antenna height at any target frequency."
        ),
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        default=REFERENCE_DESIGN_FREQUENCY_HZ / 1e6,
        help="target frequency in MHz (default: %(default)g)",
    )
    parser.add_argument(
        "--match-bandwidth-mhz",
        type=float,
        help=(
            "total three-point S11 span in MHz; defaults to the same "
            "fractional bandwidth as 10 MHz at 868 MHz"
        ),
    )
    parser.add_argument(
        "--wire-diameter-mm",
        "--wire-diameter",
        type=float,
        help=(
            "fixed conductor diameter in mm; overrides the reference or "
            "warm-start design"
        ),
    )
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--maxiter",
        type=int,
        help=(
            "DE generations per run; automatic mode treats maxiter + 1 as a "
            "population-batch budget shared across its optimizer stages"
        ),
    )
    budget.add_argument(
        "--hours",
        type=float,
        help=(
            "divide an estimated wall-time budget across every case and seed; "
            "new-incumbent confirmation solves add variable overhead that can "
            "be material during polish"
        ),
    )
    parser.add_argument("--seconds-per-eval", type=float, default=8.0)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        help=(
            "comma-separated reproducible RNG seeds; when omitted, generate "
            "and record two system-random seeds for fine-tuning or four for "
            "broad search"
        ),
    )
    topology = parser.add_mutually_exclusive_group()
    topology.add_argument(
        "--coil-count",
        type=int,
        help="number of loading coils; otherwise inferred from the start design",
    )
    topology.add_argument(
        "--coil-counts",
        type=parse_coil_counts,
        help=(
            "compare several coil counts in one campaign, for example 0,1,2,3; "
            "each generated topology starts with one turn per coil"
        ),
    )
    parser.add_argument(
        "--turn-cases",
        type=parse_turn_cases,
        help=(
            "comma-separated discrete topologies, which may mix counts; use "
            "none for zero coils, 1 for one coil, or forms such as 1x2x1"
        ),
    )
    parser.add_argument(
        "--lock-coils",
        action="store_true",
        help="use one shared pitch and radius across all loading coils",
    )
    parser.add_argument(
        "--finetune",
        "--fine-tune",
        action="store_true",
        help=(
            "use multiscale populations, restarts, and a bounded local search "
            "around the starting design"
        ),
    )
    parser.add_argument(
        "--finetune-near-radius",
        type=float,
        default=DEFAULT_FINETUNE_NEAR_RADIUS,
        help="near-population radius as a fraction of each bound span",
    )
    parser.add_argument(
        "--finetune-wide-radius",
        type=float,
        default=DEFAULT_FINETUNE_WIDE_RADIUS,
        help="wider local-population radius as a fraction of each bound span",
    )
    parser.add_argument(
        "--finetune-mutation",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_FINETUNE_MUTATION,
        help="fine-tune DE mutation dithering range",
    )
    parser.add_argument(
        "--finetune-recombination",
        type=float,
        default=DEFAULT_FINETUNE_RECOMBINATION,
        help="fine-tune DE crossover probability",
    )
    parser.add_argument(
        "--restart-stagnation-generations",
        type=int,
        default=DEFAULT_RESTART_STAGNATION_GENERATIONS,
        help=(
            "manual fine mode restarts after this many unimproved generations; "
            "automatic mode advances rough to fine and fine to polish; short "
            "budgets reduce this automatically; 0 disables stagnation triggers"
        ),
    )
    parser.add_argument(
        "--restart-min-improvement",
        type=float,
        default=DEFAULT_RESTART_MIN_IMPROVEMENT,
        help="minimum confirmed score decrease that resets stagnation",
    )
    parser.add_argument(
        "--local-search-evaluations",
        type=int,
        default=DEFAULT_LOCAL_SEARCH_EVALUATIONS,
        help=(
            "minimum evaluations reserved for fine-tune coordinate search; "
            "the reserve is rounded to a whole DE population batch"
        ),
    )
    parser.add_argument(
        "--local-search-step",
        type=float,
        default=DEFAULT_LOCAL_SEARCH_STEP,
        help="initial normalized coordinate-search step",
    )
    parser.add_argument(
        "--local-search-min-step",
        type=float,
        default=DEFAULT_LOCAL_SEARCH_MIN_STEP,
        help="smallest normalized coordinate-search step",
    )
    parser.add_argument(
        "--local-search-elites",
        type=int,
        default=3,
        help="number of feasible elites refined by local search",
    )
    parser.add_argument(
        "--polish-evaluations",
        type=int,
        help=(
            "coordinate-polish reserve; defaults to 12 times the number of "
            "search variables and is capped by the candidate budget"
        ),
    )
    parser.add_argument(
        "--polish-min-improvement",
        type=float,
        default=DEFAULT_POLISH_MIN_IMPROVEMENT,
        help="minimum score decrease accepted by coordinate polish",
    )
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument(
        "--warm-start",
        type=Path,
        help=(
            "optional design or optimizer-result JSON; when omitted, a "
            "wavelength-scaled starting geometry is synthesized"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="run directory; defaults to a new timestamped directory",
    )
    parser.add_argument(
        "--pattern",
        choices=("horizon", "ring", "directional", "peak"),
        help=(
            "gain objective; defaults to the theta-90 horizon ring, while "
            "target theta alone selects ring and target phi selects directional"
        ),
    )
    parser.add_argument(
        "--automatic",
        action="store_true",
        help=(
            "run rough search until stagnation, fine-tune the winner, apply "
            "coordinate polish, then verify and export fabrication artifacts"
        ),
    )
    parser.add_argument(
        "--automatic-rough-fraction",
        type=float,
        default=DEFAULT_AUTOMATIC_ROUGH_FRACTION,
        help=(
            "fraction of an automatic optimizer budget reserved for rough "
            "search; the remainder goes to fine-tuning and polish"
        ),
    )
    parser.add_argument(
        "--target-theta",
        "--theta",
        dest="target_theta",
        type=float,
        nargs="+",
        help=(
            "one or more spherical theta angles in degrees (0=+Z, "
            "90=horizon, 180=-Z); multiple angles optimize the weakest "
            "omnidirectional ring and cannot be combined with phi"
        ),
    )
    parser.add_argument(
        "--target-phi",
        type=float,
        help=(
            "lobe target: azimuth phi in degrees (0=+X, 90=+Y); selects a "
            "single directional target"
        ),
    )
    parser.add_argument(
        "--target-beamwidth-deg",
        type=float,
        help=(
            "HPBW goal: elevation width for a ring target, or both orthogonal "
            "cuts when target phi is supplied"
        ),
    )
    parser.add_argument(
        "--beamwidth-weight",
        type=float,
        default=1.0,
        help=(
            "beamwidth penalty weight; a 10-degree RMS error contributes "
            "this value to the objective (default: %(default)g)"
        ),
    )
    parser.add_argument(
        "--maximum-height-mm",
        type=float,
        help=(
            "height before penalty; by default uses (0.70 + 0.50 per coil) "
            "wavelengths for the largest requested topology"
        ),
    )
    parser.add_argument("--s11-limit-db", type=float, default=-10.0)
    parser.add_argument("--mismatch-weight", type=float, default=2.0)
    parser.add_argument(
        "--s11-margin-target-db",
        type=float,
        default=-12.0,
        help="reward useful match margin down to this worst-band S11",
    )
    parser.add_argument(
        "--s11-margin-weight",
        type=float,
        default=0.10,
        help="linear reward per dB of useful S11 margin",
    )
    parser.add_argument(
        "--confirmation-runs",
        type=int,
        default=3,
        help="odd number of simulations used to confirm a new incumbent",
    )
    parser.add_argument(
        "--confirmation-score-tolerance",
        type=float,
        default=1.0,
        help="maximum score distance from the median counted as consensus",
    )
    parser.add_argument("--minimum-horizon-gain-dbi", type=float, default=2.0)
    parser.add_argument("--null-weight", type=float, default=0.25)
    parser.add_argument("--maximum-ripple-db", type=float, default=1.5)
    parser.add_argument("--ripple-weight", type=float, default=0.15)
    parser.add_argument("--height-weight", type=float, default=0.10)
    parser.add_argument("--angular-step", type=float, default=2.0)
    parser.add_argument("--air-margin-wavelengths", type=float, default=0.25)
    parser.add_argument("--abc-buffer-wavelengths", type=float, default=1.00)
    parser.add_argument("--wavelength-resolution", type=float, default=0.33)
    parser.add_argument(
        "--convergence-report",
        type=Path,
        help="numerical benchmark certificate to reuse or generate",
    )
    parser.add_argument(
        "--skip-convergence-check",
        action="store_true",
        help="run without a matching open-region certificate (not recommended)",
    )
    parser.add_argument(
        "--no-auto-convergence",
        action="store_true",
        help="do not generate a missing or mismatched certificate",
    )
    parser.add_argument(
        "--require-convergence",
        action="store_true",
        help="abort unless a matching convergence certificate passes",
    )
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help="linear solver backend (default: %(default)s)",
    )
    parser.add_argument(
        "--polish",
        action="store_true",
        help=(
            "finish with deterministic one-parameter-at-a-time coordinate "
            "polish; automatic mode enables this itself"
        ),
    )
    parser.add_argument(
        "--scipy-polish",
        action="store_true",
        help="enable SciPy's legacy unbudgeted differential-evolution polish",
    )
    args = parser.parse_args()
    args.seeds_explicit = args.seeds is not None
    theta_supplied = args.target_theta is not None
    phi_supplied = args.target_phi is not None
    beamwidth_supplied = args.target_beamwidth_deg is not None
    pattern_target_supplied = theta_supplied or phi_supplied or beamwidth_supplied
    if args.pattern is None:
        if phi_supplied:
            args.pattern = "directional"
        elif theta_supplied or beamwidth_supplied:
            args.pattern = "ring"
        else:
            args.pattern = "horizon"
    elif args.pattern == "ring" and phi_supplied:
        parser.error("--target-phi conflicts with --pattern ring")
    elif args.pattern in {"horizon", "peak"} and pattern_target_supplied:
        parser.error(
            "--target-theta, --target-phi, and --target-beamwidth-deg conflict "
            f"with --pattern {args.pattern}"
        )
    args.target_thetas = tuple(
        dict.fromkeys(
            [90.0]
            if args.target_theta is None
            else (float(value) for value in args.target_theta)
        )
    )
    args.target_theta = args.target_thetas[0]
    if len(args.target_thetas) > 1 and args.pattern != "ring":
        parser.error(
            "multiple --theta values require ring mode and cannot be combined "
            "with --target-phi"
        )
    if args.target_phi is None:
        args.target_phi = 0.0
    if args.seeds is None:
        args.seed_source = "system_random"
        if args.automatic:
            generated = random_seeds(
                DEFAULT_BROAD_SEED_COUNT + DEFAULT_FINE_SEED_COUNT
            )
            args.seeds = generated[:DEFAULT_BROAD_SEED_COUNT]
            args.automatic_finetune_seeds = generated[DEFAULT_BROAD_SEED_COUNT:]
        elif args.finetune:
            args.seeds = random_seeds(DEFAULT_FINE_SEED_COUNT)
            args.automatic_finetune_seeds = args.seeds
        else:
            args.seeds = random_seeds(DEFAULT_BROAD_SEED_COUNT)
            args.automatic_finetune_seeds = args.seeds
    else:
        args.seed_source = "command_line"
        args.automatic_finetune_seeds = args.seeds
    if args.polish and args.scipy_polish:
        parser.error("--polish and --scipy-polish are mutually exclusive")
    if args.automatic and args.scipy_polish:
        parser.error("--scipy-polish is not used by --automatic")
    if args.finetune and args.scipy_polish:
        parser.error("--scipy-polish is only available for broad search")
    if args.skip_convergence_check and args.require_convergence:
        parser.error("--skip-convergence-check and --require-convergence conflict")
    if not np.isfinite(args.frequency_mhz) or args.frequency_mhz <= 0:
        parser.error("--frequency-mhz must be finite and positive")
    if args.wire_diameter_mm is not None and (
        not np.isfinite(args.wire_diameter_mm) or args.wire_diameter_mm <= 0
    ):
        parser.error("--wire-diameter-mm must be finite and positive")
    if any(
        not np.isfinite(value) or not 0 <= value <= 180
        for value in args.target_thetas
    ):
        parser.error("--theta values must be finite and between 0 and 180")
    if not np.isfinite(args.target_phi):
        parser.error("--target-phi must be finite")
    beamwidth_limit = 180.0 if args.pattern == "ring" else 360.0
    if args.target_beamwidth_deg is not None and (
        not np.isfinite(args.target_beamwidth_deg)
        or not 0 < args.target_beamwidth_deg <= beamwidth_limit
    ):
        parser.error(
            "--target-beamwidth-deg must be finite and between 0 and "
            f"{beamwidth_limit:g} for {args.pattern} mode"
        )
    args.frequency_hz = args.frequency_mhz * 1e6
    frequency_scale = REFERENCE_DESIGN_FREQUENCY_HZ / args.frequency_hz
    if args.match_bandwidth_mhz is None:
        args.match_bandwidth_mhz = 10.0 / frequency_scale
    if (
        not np.isfinite(args.match_bandwidth_mhz)
        or args.match_bandwidth_mhz <= 0
        or args.match_bandwidth_mhz >= 2 * args.frequency_mhz
    ):
        parser.error(
            "--match-bandwidth-mhz must be positive and below twice the target"
        )
    if args.maximum_height_mm is not None and (
        not np.isfinite(args.maximum_height_mm) or args.maximum_height_mm <= 0
    ):
        parser.error("--maximum-height-mm must be finite and positive")
    if args.maxiter is None and args.hours is None:
        args.maxiter = 20
    if args.maxiter is not None and args.maxiter < 0:
        parser.error("--maxiter must be non-negative")
    if args.hours is not None and args.hours <= 0:
        parser.error("--hours must be positive")
    if args.popsize < 1 or args.seconds_per_eval <= 0:
        parser.error("--popsize and --seconds-per-eval must be positive")
    if not (0 < args.finetune_near_radius <= args.finetune_wide_radius <= 1):
        parser.error(
            "fine-tune radii must satisfy 0 < --finetune-near-radius "
            "<= --finetune-wide-radius <= 1"
        )
    mutation_low, mutation_high = args.finetune_mutation
    if not (0 <= mutation_low <= mutation_high < 2):
        parser.error("--finetune-mutation must satisfy 0 <= MIN <= MAX < 2")
    args.finetune_mutation = (mutation_low, mutation_high)
    if not 0 <= args.finetune_recombination <= 1:
        parser.error("--finetune-recombination must be between zero and one")
    if args.restart_stagnation_generations < 0:
        parser.error("--restart-stagnation-generations must be non-negative")
    if not 0 < args.automatic_rough_fraction < 1:
        parser.error("--automatic-rough-fraction must be between zero and one")
    if args.polish_evaluations is not None and args.polish_evaluations <= 0:
        parser.error("--polish-evaluations must be positive")
    if (
        not np.isfinite(args.polish_min_improvement)
        or args.polish_min_improvement < 0
    ):
        parser.error("--polish-min-improvement must be non-negative")
    if (
        not np.isfinite(args.restart_min_improvement)
        or args.restart_min_improvement < 0
    ):
        parser.error("--restart-min-improvement must be finite and non-negative")
    if args.local_search_evaluations < 0:
        parser.error("--local-search-evaluations must be non-negative")
    if not (0 < args.local_search_min_step <= args.local_search_step <= 1):
        parser.error("local-search steps must satisfy 0 < MIN_STEP <= STEP <= 1")
    if args.local_search_elites < 1:
        parser.error("--local-search-elites must be positive")
    numerical_values = (
        args.air_margin_wavelengths,
        args.abc_buffer_wavelengths,
        args.wavelength_resolution,
    )
    if any(not np.isfinite(value) or value <= 0 for value in numerical_values):
        parser.error("open-region and mesh values must be finite and positive")
    if not np.isfinite(args.angular_step) or not 0 < args.angular_step <= 10:
        parser.error("--angular-step must be finite and between zero and 10 degrees")
    if args.coil_count is not None and args.coil_count < 0:
        parser.error("--coil-count must be non-negative")
    if args.coil_counts is not None and args.turn_cases is not None:
        parser.error(
            "--coil-counts and --turn-cases conflict; list mixed topologies "
            "directly with --turn-cases"
        )
    if args.coil_counts is not None:
        args.turn_cases = tuple((1,) * count for count in args.coil_counts)
    elif args.turn_cases is None:
        if args.coil_count is not None:
            args.turn_cases = ((1,) * args.coil_count,)
    elif args.coil_count is not None:
        if any(len(turn_case) != args.coil_count for turn_case in args.turn_cases):
            parser.error("every --turn-cases entry must match --coil-count")
    else:
        case_counts = {len(turn_case) for turn_case in args.turn_cases}
        if len(case_counts) == 1:
            args.coil_count = next(iter(case_counts))
    weights = (
        args.mismatch_weight,
        args.s11_margin_weight,
        args.beamwidth_weight,
        args.null_weight,
        args.ripple_weight,
        args.height_weight,
    )
    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        parser.error("objective weights must be finite and non-negative")
    if not np.isfinite(args.s11_limit_db):
        parser.error("--s11-limit-db must be finite")
    if (
        not np.isfinite(args.s11_margin_target_db)
        or args.s11_margin_target_db >= args.s11_limit_db
    ):
        parser.error("--s11-margin-target-db must be below --s11-limit-db")
    if args.confirmation_runs < 1 or args.confirmation_runs % 2 == 0:
        parser.error("--confirmation-runs must be a positive odd integer")
    if (
        not np.isfinite(args.confirmation_score_tolerance)
        or args.confirmation_score_tolerance < 0
    ):
        parser.error("--confirmation-score-tolerance must be non-negative")
    if args.output is None:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        frequency_label = f"{args.frequency_mhz:g}mhz".replace(".", "p")
        args.output = Path("optimization_results") / f"{frequency_label}_{stamp}"
    if args.convergence_report is None:
        frequency_hz = round(args.frequency_hz)
        args.convergence_report = (
            Path("optimization_results")
            / f"open_region_convergence_{frequency_hz}hz.json"
        )
    return args


def load_baseline(path: Path | None, frequency_hz: float) -> AntennaDesign:
    if path is None:
        design = load_reference_design(frequency_hz)
        scale = REFERENCE_DESIGN_FREQUENCY_HZ / frequency_hz
        print(
            "Initial design  : synthesized wavelength-scaled geometry "
            f"(scale {scale:.6g})"
        )
        return design
    if not path.is_file():
        raise SystemExit(
            "WARM START FAILED\n"
            f"Design file not found: {path}\n"
            "Pass an existing design or optimizer-result JSON with --warm-start."
        )
    design = load_design(path)
    print(f"Initial design  : {path.resolve()}")
    return design


def resolve_topology(
    args: argparse.Namespace,
    initial: AntennaDesign,
) -> None:
    """Resolve all requested discrete topologies, including mixed counts."""
    coil_count = getattr(args, "coil_count", None)
    coil_counts = getattr(args, "coil_counts", None)
    turn_cases = getattr(args, "turn_cases", None)
    if turn_cases is None:
        if coil_counts is not None:
            turn_cases = tuple((1,) * count for count in coil_counts)
        elif coil_count is not None:
            turn_cases = ((1,) * coil_count,)
        else:
            turn_cases = (tuple(coil.turns for coil in initial.coils),)

    args.turn_cases = tuple(dict.fromkeys(turn_cases))
    args.coil_counts = tuple(
        dict.fromkeys(len(turn_case) for turn_case in args.turn_cases)
    )
    args.coil_count = args.coil_counts[0] if len(args.coil_counts) == 1 else None


def iterations_per_run(
    args: argparse.Namespace,
    variables: int,
    run_count: int,
) -> int:
    if args.maxiter is not None:
        return args.maxiter
    evaluations = args.hours * 3600 / args.seconds_per_eval / run_count
    population = max(5, args.popsize * variables)
    generations = max(1, int(evaluations / population))
    return max(0, generations - 1)


@dataclass(frozen=True)
class CaseSchedule:
    """One fixed-dimensional differential-evolution topology search."""

    turn_case: tuple[int, ...]
    space: DesignSpace
    maxiter: int
    population: int
    evaluations_per_run: int


def build_case_schedules(
    args: argparse.Namespace,
    initial: AntennaDesign,
    frequency_hz: float,
) -> tuple[CaseSchedule, ...]:
    """Build fixed-dimensional searches and allocate any wall-time budget."""
    run_count = len(args.seeds) * len(args.turn_cases)
    schedules = []
    for turn_case in args.turn_cases:
        design = design_for_turn_case(initial, turn_case, frequency_hz)
        space = make_space(
            design,
            frequency_hz,
            lock_coils=getattr(args, "lock_coils", False),
        )
        variable_count = len(space.variables)
        maxiter = iterations_per_run(args, variable_count, run_count)
        population = max(5, args.popsize * variable_count)
        schedules.append(
            CaseSchedule(
                turn_case=turn_case,
                space=space,
                maxiter=maxiter,
                population=population,
                evaluations_per_run=population * (maxiter + 1),
            )
        )
    return tuple(schedules)


def _number_list(*values: float) -> str:
    return ",".join(f"{value:.12g}" for value in values)


def handle_uncertified_convergence(
    args: argparse.Namespace,
    detail: str,
) -> None:
    """Warn by default, or abort when strict convergence was requested."""
    if getattr(args, "require_convergence", False):
        raise SystemExit(
            f"OPEN-REGION CONVERGENCE REQUIRED BUT NOT CERTIFIED\n{detail}"
        )
    args.convergence_warning = detail
    print("\nWARNING: OPEN-REGION CONVERGENCE NOT CERTIFIED")
    print(detail)
    print(
        "Optimization will continue with uncertified numerical settings. "
        "Use --require-convergence to make this fatal.\n",
        flush=True,
    )


def ensure_convergence_certificate(
    args: argparse.Namespace,
    benchmark: AntennaDesign,
    mesh: MeshSettings,
    open_region: OpenRegionSettings,
    frequency_hz: float,
) -> dict | None:
    """Certify numerical settings on a frequency-scaled reference problem."""
    angular_step = float(getattr(args, "angular_step", 2.0))
    try:
        return validate_convergence_certificate(
            args.convergence_report,
            benchmark,
            mesh,
            open_region,
            frequency_hz,
            farfield_angular_step_deg=angular_step,
        )
    except RuntimeError as error:
        if args.convergence_report.is_file():
            try:
                existing_report = validate_convergence_certificate(
                    args.convergence_report,
                    benchmark,
                    mesh,
                    open_region,
                    frequency_hz,
                    require_passed=False,
                    farfield_angular_step_deg=angular_step,
                )
            except RuntimeError:
                pass
            else:
                if existing_report.get("passed") is False:
                    handle_uncertified_convergence(
                        args,
                        "The existing matching convergence report failed.\n"
                        f"See {args.convergence_report.resolve()} for details.",
                    )
                    return None
        if args.no_auto_convergence:
            handle_uncertified_convergence(
                args,
                f"Certificate unavailable: {error}\n"
                "Automatic convergence is disabled by --no-auto-convergence.",
            )
            return None

        print("OPEN-REGION PREFLIGHT")
        print(f"Certificate     : unavailable ({error})")
        print("Action          : running automatic convergence (7 isolated solves)")
        print("The optimization timer starts after this one-time check.\n", flush=True)

        air_margin = mesh.air_margin_wavelengths
        abc_buffer = open_region.abc_buffer_wavelengths
        resolution = mesh.wavelength_resolution
        source = args.output / "convergence_reference_design.json"
        save_design(benchmark, source)
        print(f"Benchmark       : {source.resolve()}")
        command = (
            sys.executable,
            "-u",
            str(Path(__file__).with_name("check_open_region.py").resolve()),
            str(source.resolve()),
            "--frequency-mhz",
            f"{frequency_hz / 1e6:.12g}",
            "--output",
            str(args.convergence_report.resolve()),
            "--air-margins",
            _number_list(0.8 * air_margin, air_margin, 1.4 * air_margin),
            "--abc-buffers",
            _number_list(0.75 * abc_buffer, abc_buffer, 1.25 * abc_buffer),
            "--mesh-resolutions",
            _number_list(1.5 * resolution, resolution, 0.75 * resolution),
            "--selected-air-margin",
            f"{air_margin:.12g}",
            "--selected-abc-buffer",
            f"{abc_buffer:.12g}",
            "--selected-resolution",
            f"{resolution:.12g}",
            "--angular-step",
            f"{angular_step:.12g}",
            "--solver",
            args.solver,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            handle_uncertified_convergence(
                args,
                "Automatic open-region convergence failed.\n"
                f"See {args.convergence_report.resolve()} for the failed checks.",
            )
            return None

        try:
            return validate_convergence_certificate(
                args.convergence_report,
                benchmark,
                mesh,
                open_region,
                frequency_hz,
                farfield_angular_step_deg=angular_step,
            )
        except RuntimeError as validation_error:
            handle_uncertified_convergence(
                args,
                "Automatic convergence produced an invalid report:\n"
                f"{validation_error}",
            )
            return None


def objective_evaluation_count(objective: object) -> int:
    """Return a confirmation-aware evaluation count when one is available."""
    for name in (
        "evaluation_count",
        "simulation_evaluations",
        "total_evaluations",
    ):
        value = getattr(objective, name, None)
        value = value() if callable(value) else value
        if isinstance(value, (int, np.integer)):
            return int(value)
    return len(getattr(objective, "history", ()))


def restart_seed(seed: int, restart_index: int) -> int:
    """Derive independent deterministic RNG seeds for restart stages."""
    sequence = np.random.SeedSequence((int(seed), int(restart_index)))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class FinetuneRunResult:
    result: object
    generations: int
    restarts: int
    differential_evolution_evaluations: int
    differential_evolution_simulations: int
    local_search: LocalSearchStats
    polish: CoordinatePolishStats | None
    final_diversity: float
    planned_budget: FinetuneBudget
    effective_stagnation_generations: int
    transition_reason: str


@dataclass(frozen=True)
class CampaignOutcome:
    best_path: Path | None
    interrupted: bool


def run_finetune_optimizer(
    objective: object,
    space: DesignSpace,
    schedule: CaseSchedule,
    args: argparse.Namespace,
    seed: int,
    run_name: str,
) -> FinetuneRunResult:
    """Run restartable local DE and bounded pattern search within one budget."""
    polish_enabled = bool(getattr(args, "polish", False))
    requested_polish = getattr(args, "polish_evaluations", None)
    requested_local = (
        int(requested_polish)
        if polish_enabled and requested_polish is not None
        else max(DEFAULT_LOCAL_SEARCH_EVALUATIONS, 12*len(space.variables))
        if polish_enabled
        else int(
            getattr(
                args,
                "local_search_evaluations",
                DEFAULT_LOCAL_SEARCH_EVALUATIONS,
            )
        )
    )
    budget = split_finetune_budget(
        schedule.evaluations_per_run,
        schedule.population,
        requested_local,
    )
    near_radius = float(
        getattr(args, "finetune_near_radius", DEFAULT_FINETUNE_NEAR_RADIUS)
    )
    wide_radius = float(
        getattr(args, "finetune_wide_radius", DEFAULT_FINETUNE_WIDE_RADIUS)
    )
    mutation = tuple(getattr(args, "finetune_mutation", DEFAULT_FINETUNE_MUTATION))
    recombination = float(
        getattr(
            args,
            "finetune_recombination",
            DEFAULT_FINETUNE_RECOMBINATION,
        )
    )
    restart_after = int(
        getattr(
            args,
            "restart_stagnation_generations",
            DEFAULT_RESTART_STAGNATION_GENERATIONS,
        )
    )
    minimum_improvement = float(
        getattr(
            args,
            "restart_min_improvement",
            DEFAULT_RESTART_MIN_IMPROVEMENT,
        )
    )
    maximum_s11_db = float(getattr(args, "s11_limit_db", -10.0))
    initial_candidates = len(getattr(objective, "history", ()))
    de_target = max(0, budget.differential_evolution - initial_candidates)
    available_generations = max(0, de_target // schedule.population - 1)
    effective_restart_after = (
        min(restart_after, max(1, available_generations // 2))
        if restart_after > 0 and available_generations > 0
        else 0
    )
    if 0 < effective_restart_after < restart_after:
        print(
            f"    stagnation patience reduced from {restart_after} to "
            f"{effective_restart_after} generations for this candidate budget",
            flush=True,
        )
    de_evaluations = 0
    de_simulations = 0
    generations = 0
    restarts = 0
    final_diversity = float("nan")
    result: object | None = None
    transition_reason = "candidate_budget_exhausted"

    while de_target - de_evaluations >= schedule.population:
        remaining = de_target - de_evaluations
        population_batches = remaining // schedule.population
        incumbent = preferred_incumbent(objective, maximum_s11_db)
        center = (
            np.asarray(incumbent.vector, dtype=float)
            if incumbent is not None
            else np.asarray(space.initial_vector, dtype=float)
        )
        stage_seed = restart_seed(seed, restarts)
        population = build_finetune_population(
            space,
            schedule.population,
            stage_seed,
            center=center,
            near_radius=near_radius,
            wide_radius=wide_radius,
        )
        monitor = GenerationMonitor(
            objective,
            space,
            run_name,
            generation_offset=generations,
            restart_index=restarts,
            restart_after=effective_restart_after,
            incumbent_score=(incumbent.score if incumbent is not None else None),
            improvement_tolerance=minimum_improvement,
            maximum_s11_db=maximum_s11_db,
            incumbent_feasible=(
                record_is_feasible(incumbent, maximum_s11_db)
                if incumbent is not None
                else False
            ),
        )
        history_before = len(getattr(objective, "history", ()))
        simulations_before = objective_evaluation_count(objective)
        result = differential_evolution(
            objective,
            bounds=space.bounds,
            maxiter=max(0, population_batches - 1),
            popsize=getattr(args, "popsize", 8),
            polish=False,
            seed=stage_seed,
            workers=1,
            updating="immediate",
            init=population,
            mutation=mutation,
            recombination=recombination,
            callback=monitor,
            tol=1e-3,
        )
        consumed = len(getattr(objective, "history", ())) - history_before
        simulation_runs = objective_evaluation_count(objective) - simulations_before
        if consumed <= 0:
            break
        de_evaluations += consumed
        de_simulations += simulation_runs
        generations += int(getattr(result, "nit", monitor.stage_generations))
        final_diversity = monitor.last_diversity
        if (
            polish_enabled
            and monitor.stopped_for_stagnation
            and preferred_incumbent(objective, maximum_s11_db) is not None
        ):
            transition_reason = "fine_stagnation"
            print(
                f"    polishing {run_name}: fine search stagnated after "
                f"{monitor.stagnation_generations} generations",
                flush=True,
            )
            break
        if (
            polish_enabled
            and de_target - de_evaluations >= schedule.population
            and preferred_incumbent(objective, maximum_s11_db) is not None
        ):
            transition_reason = "fine_de_converged"
            print(
                f"    polishing {run_name}: fine differential evolution "
                "converged before its candidate cap",
                flush=True,
            )
            break
        if de_target - de_evaluations >= schedule.population:
            restarts += 1
            reason = (
                f"{monitor.stagnation_generations} stagnant generations"
                if monitor.stopped_for_stagnation
                else "early DE convergence"
            )
            print(
                f"    restarting {run_name}: {reason}; "
                f"{de_target - de_evaluations} DE evaluations remain",
                flush=True,
            )

    if result is None:
        raise RuntimeError("fine-tune budget did not contain one DE population")

    total_used = len(getattr(objective, "history", ()))
    local_budget = max(0, schedule.evaluations_per_run - total_used)
    polish_stats = None
    if polish_enabled:
        local_stats = LocalSearchStats(0, 0, 0, 0, DEFAULT_LOCAL_SEARCH_STEP)
        polish_stats = coordinate_polish(
            objective,
            space,
            maximum_s11_db,
            local_budget,
            initial_step=float(
                getattr(args, "local_search_step", DEFAULT_LOCAL_SEARCH_STEP)
            ),
            minimum_step=float(
                getattr(
                    args,
                    "local_search_min_step",
                    DEFAULT_LOCAL_SEARCH_MIN_STEP,
                )
            ),
            minimum_improvement=float(
                getattr(
                    args,
                    "polish_min_improvement",
                    DEFAULT_POLISH_MIN_IMPROVEMENT,
                )
            ),
        )
        if local_budget:
            print(
                f"    coordinate polish | {polish_stats.evaluations}/"
                f"{local_budget} evaluations | {polish_stats.improvements} "
                f"improvements | step {polish_stats.final_step:.4g} | "
                f"{'converged' if polish_stats.converged else 'budget-limited'}",
                flush=True,
            )
    else:
        local_stats = normalized_pattern_search(
            objective,
            space,
            maximum_s11_db,
            local_budget,
            initial_step=float(
                getattr(args, "local_search_step", DEFAULT_LOCAL_SEARCH_STEP)
            ),
            minimum_step=float(
                getattr(
                    args,
                    "local_search_min_step",
                    DEFAULT_LOCAL_SEARCH_MIN_STEP,
                )
            ),
            elite_count=int(getattr(args, "local_search_elites", 3)),
        )
    if local_budget and not polish_enabled:
        print(
            f"    local search | {local_stats.evaluations}/{local_budget} evaluations "
            f"| {local_stats.elite_count} feasible elites | "
            f"{local_stats.improvements} improvements",
            flush=True,
        )
    return FinetuneRunResult(
        result=result,
        generations=generations,
        restarts=restarts,
        differential_evolution_evaluations=de_evaluations,
        differential_evolution_simulations=de_simulations,
        local_search=local_stats,
        polish=polish_stats,
        final_diversity=final_diversity,
        planned_budget=budget,
        effective_stagnation_generations=effective_restart_after,
        transition_reason=transition_reason,
    )


def run_campaign(args: argparse.Namespace) -> CampaignOutcome:
    frequency_hz = args.frequency_hz
    args.finetune = bool(getattr(args, "finetune", False))
    args.lock_coils = bool(getattr(args, "lock_coils", False))
    args.target_thetas = tuple(
        float(value)
        for value in getattr(args, "target_thetas", (args.target_theta,))
    )
    args.target_theta = args.target_thetas[0]
    if getattr(args, "seeds", None) is None:
        args.seeds = random_seeds(
            DEFAULT_FINE_SEED_COUNT
            if args.finetune
            else DEFAULT_BROAD_SEED_COUNT
        )
        args.seed_source = "system_random"
    elif not hasattr(args, "seed_source"):
        args.seed_source = "command_line"
    args.s11_margin_target_db = float(
        getattr(args, "s11_margin_target_db", -12.0)
    )
    args.s11_margin_weight = float(getattr(args, "s11_margin_weight", 0.10))
    args.target_beamwidth_deg = getattr(args, "target_beamwidth_deg", None)
    args.beamwidth_weight = float(getattr(args, "beamwidth_weight", 1.0))
    args.confirmation_runs = int(getattr(args, "confirmation_runs", 3))
    args.confirmation_score_tolerance = float(
        getattr(args, "confirmation_score_tolerance", 1.0)
    )
    if args.output.exists() and next(args.output.iterdir(), None) is not None:
        raise RuntimeError(
            f"Campaign output directory is not empty: {args.output}. Choose a new "
            "--output directory; campaign resume is not supported."
        )
    initial = load_baseline(args.warm_start, frequency_hz)
    initial = apply_design_overrides(
        initial,
        wire_diameter_mm=getattr(args, "wire_diameter_mm", None),
    )
    resolve_topology(args, initial)
    automatic_height = args.maximum_height_mm is None
    if automatic_height:
        args.maximum_height_mm = 1e3 * default_maximum_height(
            frequency_hz,
            max(args.coil_counts),
        )
    schedules = build_case_schedules(args, initial, frequency_hz)
    convergence_benchmark = load_reference_design(frequency_hz)
    mesh = replace(
        MeshSettings(),
        air_margin_wavelengths=args.air_margin_wavelengths,
        wavelength_resolution=args.wavelength_resolution,
    )
    open_region = replace(
        OpenRegionSettings(),
        mode="abc",
        abc_buffer_wavelengths=args.abc_buffer_wavelengths,
    )
    certificate = None
    args.convergence_warning = None
    if not args.skip_convergence_check:
        certificate = ensure_convergence_certificate(
            args,
            convergence_benchmark,
            mesh,
            open_region,
            frequency_hz,
        )
    simulation_metadata = {
        "pipeline_stage": getattr(args, "pipeline_stage", "optimization"),
        "mesh": asdict(mesh),
        "open_region": asdict(open_region),
        "solver": args.solver,
        "seeds": {
            "source": args.seed_source,
            "values": list(args.seeds),
        },
        "target_frequency_hz": frequency_hz,
        "match_bandwidth_hz": args.match_bandwidth_mhz * 1e6,
        "farfield_angular_step_deg": args.angular_step,
        "optimizer_budget_unit": "candidate_evaluations",
        "confirmation_simulations_counted_separately": True,
        "objective": {
            "pattern_mode": args.pattern,
            "target_theta_deg": args.target_theta,
            "target_theta_degrees": list(args.target_thetas),
            "target_phi_deg": args.target_phi,
            "maximum_s11_db": args.s11_limit_db,
            "mismatch_weight": args.mismatch_weight,
            "s11_margin_target_db": args.s11_margin_target_db,
            "s11_margin_weight": args.s11_margin_weight,
            "target_beamwidth_deg": args.target_beamwidth_deg,
            "beamwidth_weight": args.beamwidth_weight,
            "confirmation_runs": args.confirmation_runs,
            "confirmation_score_tolerance": (
                args.confirmation_score_tolerance
            ),
        },
        "coil_counts": list(args.coil_counts),
        "turn_cases": [list(turn_case) for turn_case in args.turn_cases],
        "coil_parameterization": (
            "shared" if args.lock_coils else "independent"
        ),
        "optimizer": (
            {
                "mode": "finetune_multiscale",
                "population_fractions": {
                    "near": FINETUNE_NEAR_FRACTION,
                    "wide": FINETUNE_WIDE_FRACTION,
                    "global": FINETUNE_GLOBAL_FRACTION,
                },
                "near_radius_normalized": float(
                    getattr(
                        args,
                        "finetune_near_radius",
                        DEFAULT_FINETUNE_NEAR_RADIUS,
                    )
                ),
                "wide_radius_normalized": float(
                    getattr(
                        args,
                        "finetune_wide_radius",
                        DEFAULT_FINETUNE_WIDE_RADIUS,
                    )
                ),
                "mutation": list(
                    getattr(
                        args,
                        "finetune_mutation",
                        DEFAULT_FINETUNE_MUTATION,
                    )
                ),
                "recombination": float(
                    getattr(
                        args,
                        "finetune_recombination",
                        DEFAULT_FINETUNE_RECOMBINATION,
                    )
                ),
                "restart_stagnation_generations": int(
                    getattr(
                        args,
                        "restart_stagnation_generations",
                        DEFAULT_RESTART_STAGNATION_GENERATIONS,
                    )
                ),
                "restart_min_improvement": float(
                    getattr(
                        args,
                        "restart_min_improvement",
                        DEFAULT_RESTART_MIN_IMPROVEMENT,
                    )
                ),
                "local_search": {
                    "method": (
                        "single_incumbent_coordinate_polish"
                        if getattr(args, "polish", False)
                        else "normalized_coordinate_pattern"
                    ),
                    "requested_evaluations": (
                        getattr(args, "polish_evaluations", None)
                        if getattr(args, "polish", False)
                        else int(
                            getattr(
                                args,
                                "local_search_evaluations",
                                DEFAULT_LOCAL_SEARCH_EVALUATIONS,
                            )
                        )
                    ),
                    "initial_step": float(
                        getattr(
                            args,
                            "local_search_step",
                            DEFAULT_LOCAL_SEARCH_STEP,
                        )
                    ),
                    "minimum_step": float(
                        getattr(
                            args,
                            "local_search_min_step",
                            DEFAULT_LOCAL_SEARCH_MIN_STEP,
                        )
                    ),
                    "elite_count": int(getattr(args, "local_search_elites", 3)),
                },
            }
            if args.finetune
            else {
                "mode": "global_differential_evolution",
                "initialization": "scipy_default_with_warm_x0",
                "coordinate_polish": bool(getattr(args, "polish", False)),
                "scipy_polish": bool(getattr(args, "scipy_polish", False)),
                "stagnation_transition_generations": int(
                    getattr(args, "rough_stagnation_generations", 0)
                ),
            }
        ),
        "search_bounds": {
            "policy": "wavelength_wire_v1",
            "wavelength_m": free_space_wavelength(frequency_hz),
            "wire_diameter_m": 2 * initial.wire_radius,
            "wire_diameter_source": (
                "command_line"
                if getattr(args, "wire_diameter_mm", None) is not None
                else "start_design"
            ),
            "maximum_height_m": args.maximum_height_mm * 1e-3,
            "maximum_height_wavelengths": (
                args.maximum_height_mm * 1e-3 / free_space_wavelength(frequency_hz)
            ),
            "maximum_height_source": (
                getattr(args, "maximum_height_source_override", None)
                or ("automatic" if automatic_height else "command_line")
            ),
        },
        "convergence_status": (
            "skipped"
            if args.skip_convergence_check
            else "passed"
            if certificate
            else "warning"
        ),
        "convergence_warning": args.convergence_warning,
        "convergence_report": (
            None
            if args.skip_convergence_check
            else str(args.convergence_report.resolve())
        ),
    }
    variable_names = tuple(
        dict.fromkeys(name for schedule in schedules for name in schedule.space.names)
    )
    ring_metric_fields = (
        tuple(
            field
            for index in range(len(args.target_thetas))
            for field in (
                f"ring_{index}_target_theta_deg",
                f"ring_{index}_sampled_theta_deg",
                f"ring_{index}_min_gain_dbi",
                f"ring_{index}_p10_gain_dbi",
                f"ring_{index}_mean_gain_dbi",
                f"ring_{index}_ripple_db",
                f"ring_{index}_beamwidth_deg",
                f"ring_{index}_beamwidth_error_deg",
            )
        )
        if args.pattern == "ring"
        else ()
    )
    total = len(args.seeds) * sum(
        schedule.evaluations_per_run for schedule in schedules
    )
    progress = CampaignProgress(
        args.output,
        total,
        args.report_every,
        variable_names,
        frequency_hz,
        simulation_metadata,
        allowed_existing=tuple(
            path
            for path in (
                args.output / "convergence_reference_design.json",
                args.convergence_report,
            )
            if path.exists()
        ),
        maximum_s11_db=args.s11_limit_db,
        stage=getattr(args, "pipeline_stage", "optimization"),
        additional_metric_fields=ring_metric_fields,
    )
    write_json(
        args.output / "campaign_seeds.json",
        {
            "source": args.seed_source,
            "seeds": list(args.seeds),
            "pipeline_stage": getattr(args, "pipeline_stage", "optimization"),
            "frequency_hz": frequency_hz,
        },
    )
    run_summaries = []

    low_mhz = args.frequency_mhz - args.match_bandwidth_mhz / 2
    high_mhz = args.frequency_mhz + args.match_bandwidth_mhz / 2
    print(f"ROBUST {args.frequency_mhz:g} MHz ANTENNA CAMPAIGN")
    print(f"Pattern target  : {args.pattern}")
    if args.pattern == "directional":
        print(
            f"Gain direction  : theta {args.target_theta:g} deg, "
            f"phi {args.target_phi:g} deg"
        )
    elif args.pattern == "ring":
        if len(args.target_thetas) == 1:
            print(
                f"Gain ring       : theta {args.target_theta:g} deg, all phi angles"
            )
        else:
            theta_text = ", ".join(f"{value:g}" for value in args.target_thetas)
            print(
                f"Gain rings      : theta {theta_text} deg, all phi angles; "
                "maximize the weakest ring P10"
            )
    print(f"Match samples   : {low_mhz:g}, {args.frequency_mhz:g} and {high_mhz:g} MHz")
    print(
        f"Match objective : limit {args.s11_limit_db:g} dB; margin reward to "
        f"{args.s11_margin_target_db:g} dB"
    )
    print(
        f"Confirmation    : {args.confirmation_runs} runs, score tolerance "
        f"{args.confirmation_score_tolerance:g}"
    )
    if args.pattern in {"horizon", "ring"}:
        print(
            f"Pattern limits  : azimuth-ring min "
            f"{args.minimum_horizon_gain_dbi:.1f} dBi, "
            f"P90-P10 ripple {args.maximum_ripple_db:.1f} dB"
        )
    if args.target_beamwidth_deg is not None:
        beamwidth_description = (
            "azimuthal-P10 elevation profile"
            if args.pattern == "ring"
            else "both orthogonal cuts"
        )
        print(
            f"Beamwidth goal  : {args.target_beamwidth_deg:g} deg HPBW on "
            f"{beamwidth_description}, weight {args.beamwidth_weight:g}"
        )
    height_wavelengths = (
        args.maximum_height_mm * 1e-3 / free_space_wavelength(frequency_hz)
    )
    print(
        f"Physical limit  : {args.maximum_height_mm:.1f} mm "
        f"({height_wavelengths:.2f} lambda) maximum height before penalty"
        + (" [automatic]" if automatic_height else "")
    )
    print("Length priors   : bare 0.18-0.70 lambda; loaded sections 0.15-0.72 lambda")
    print(
        "Coil/radial     : pitch 0.010-0.040 lambda, radius "
        "0.015-0.050 lambda, radials 0.15-0.40 lambda; wire floors apply"
    )
    print(f"Wire diameter   : {2 * initial.wire_radius * 1e3:.3f} mm")
    print("Coil counts     : " + ", ".join(str(count) for count in args.coil_counts))
    print(
        "Coil geometry   : "
        + (
            "shared pitch/radius (--lock-coils)"
            if args.lock_coils
            else "independent pitch/radius"
        )
    )
    if args.finetune:
        near_radius = getattr(
            args,
            "finetune_near_radius",
            DEFAULT_FINETUNE_NEAR_RADIUS,
        )
        wide_radius = getattr(
            args,
            "finetune_wide_radius",
            DEFAULT_FINETUNE_WIDE_RADIUS,
        )
        print(
            "Fine population : 50% +/-"
            f"{100 * near_radius:g}%, 30% +/-{100 * wide_radius:g}%, 20% global"
        )
        print(
            "Fine DE         : mutation "
            f"{tuple(getattr(args, 'finetune_mutation', DEFAULT_FINETUNE_MUTATION))}, "
            "recombination "
            f"{getattr(args, 'finetune_recombination', DEFAULT_FINETUNE_RECOMBINATION):g}"
        )
        if getattr(args, "polish", False):
            print(
                "Fine polish     : one-parameter coordinate descent after "
                "stagnation or DE convergence"
            )
    print(f"Linear solver   : {args.solver}")
    print(
        f"Open region     : inner Huygens box at "
        f"{mesh.air_margin_wavelengths:.2f} lambda + "
        f"{open_region.abc_buffer_wavelengths:.2f} lambda all-face ABC buffer"
    )
    print(f"Air resolution  : {mesh.wavelength_resolution:.2f} wavelengths")
    if args.skip_convergence_check:
        print("Preflight       : SKIPPED BY USER")
    elif certificate is None:
        print(
            "Preflight       : WARNING - NOT CERTIFIED "
            f"({args.convergence_report.resolve()})"
        )
    else:
        print(f"Preflight       : PASS ({args.convergence_report.resolve()})")
    print(
        "Turn cases      : "
        + ", ".join(format_turn_case(case) for case in args.turn_cases)
    )
    print(f"Seeds           : {args.seeds} [{args.seed_source}]")
    print("Search schedule :")
    for schedule in schedules:
        allocation = ""
        if args.finetune:
            requested_local = (
                getattr(args, "polish_evaluations", None)
                if getattr(args, "polish", False)
                else int(
                    getattr(
                        args,
                        "local_search_evaluations",
                        DEFAULT_LOCAL_SEARCH_EVALUATIONS,
                    )
                )
            )
            if requested_local is None:
                requested_local = max(
                    DEFAULT_LOCAL_SEARCH_EVALUATIONS,
                    12*len(schedule.space.variables),
                )
            budget = split_finetune_budget(
                schedule.evaluations_per_run,
                schedule.population,
                int(requested_local),
            )
            allocation = (
                f" | DE/{'polish' if getattr(args, 'polish', False) else 'local'} "
                f"{budget.differential_evolution}/{budget.local_search}"
            )
        print(
            f"  {format_turn_case(schedule.turn_case):>8} : "
            f"{len(schedule.space.variables):2d} variables | "
            f"population {schedule.population:3d} | "
            f"iterations {schedule.maxiter:4d} | "
            f"{schedule.evaluations_per_run:5d} candidates/seed"
            f"{allocation}"
        )
    print(f"Candidate budget: {total} (confirmation simulations reported separately)")
    print(f"Campaign log    : {(args.output / 'evaluations.csv').resolve()}")
    print("Global and per-topology bests are checkpointed after every improvement.")
    print("Press Ctrl+C to stop safely; completed evaluations remain on disk.\n")

    interrupted = False
    try:
        for seed in args.seeds:
            for schedule in schedules:
                turn_case = schedule.turn_case
                space = schedule.space
                run_name = f"turns_{format_turn_case(turn_case)}_seed_{seed}"
                print(f"\nStarting {run_name}", flush=True)
                progress.set_context(space, turn_case, seed)
                options = SimulationOptions(
                    sweep=FrequencySweep(
                        center=frequency_hz,
                        span=args.match_bandwidth_mhz * 1e6,
                        points=3,
                    ),
                    mesh=mesh,
                    open_region=open_region,
                    solve=True,
                    solver=args.solver,
                    compute_farfield=True,
                    farfield_frequency=frequency_hz,
                    farfield_angular_step_deg=args.angular_step,
                    verbose=False,
                )
                objective = RobustGainObjective(
                    space,
                    target_frequency=frequency_hz,
                    pattern_mode=args.pattern,
                    target_theta_deg=args.target_theta,
                    target_theta_degrees=args.target_thetas,
                    target_phi_deg=args.target_phi,
                    target_beamwidth_deg=args.target_beamwidth_deg,
                    beamwidth_weight=args.beamwidth_weight,
                    maximum_s11_db=args.s11_limit_db,
                    mismatch_weight=args.mismatch_weight,
                    s11_margin_target_db=args.s11_margin_target_db,
                    s11_margin_weight=args.s11_margin_weight,
                    minimum_horizon_gain_dbi=args.minimum_horizon_gain_dbi,
                    null_weight=args.null_weight,
                    maximum_horizon_ripple_db=args.maximum_ripple_db,
                    ripple_weight=args.ripple_weight,
                    maximum_height=args.maximum_height_mm * 1e-3,
                    height_weight=args.height_weight,
                    options=options,
                    on_evaluation=progress,
                    confirmation_runs=args.confirmation_runs,
                    confirmation_score_tolerance=(
                        args.confirmation_score_tolerance
                    ),
                )
                lower = np.asarray([bound[0] for bound in space.bounds])
                upper = np.asarray([bound[1] for bound in space.bounds])
                warm_vector = np.clip(space.initial_vector, lower, upper)
                if args.finetune:
                    fine_result = run_finetune_optimizer(
                        objective,
                        space,
                        schedule,
                        args,
                        seed,
                        run_name,
                    )
                    result = fine_result.result
                    optimizer_details = {
                        "generations": fine_result.generations,
                        "restarts": fine_result.restarts,
                        "effective_stagnation_generations": (
                            fine_result.effective_stagnation_generations
                        ),
                        "final_normalized_diversity": (
                            fine_result.final_diversity
                            if np.isfinite(fine_result.final_diversity)
                            else None
                        ),
                        "differential_evolution_candidates": (
                            fine_result.differential_evolution_evaluations
                        ),
                        "differential_evolution_simulations": (
                            fine_result.differential_evolution_simulations
                        ),
                        "local_search_candidates": (
                            fine_result.local_search.evaluations
                        ),
                        "local_search_simulations": (
                            fine_result.local_search.simulation_evaluations
                        ),
                        "local_search_elites": (fine_result.local_search.elite_count),
                        "local_search_improvements": (
                            fine_result.local_search.improvements
                        ),
                        "coordinate_polish": (
                            asdict(fine_result.polish)
                            if fine_result.polish is not None
                            else None
                        ),
                        "transition_reason": fine_result.transition_reason,
                        "planned_candidate_budget": {
                            "differential_evolution": (
                                fine_result.planned_budget.differential_evolution
                            ),
                            "local_search": (fine_result.planned_budget.local_search),
                        },
                    }
                else:
                    polish_enabled = bool(getattr(args, "polish", False))
                    requested_polish = getattr(args, "polish_evaluations", None)
                    polish_reserve = (
                        int(requested_polish)
                        if requested_polish is not None
                        else max(
                            DEFAULT_LOCAL_SEARCH_EVALUATIONS,
                            12*len(space.variables),
                        )
                    )
                    global_budget = (
                        split_finetune_budget(
                            schedule.evaluations_per_run,
                            schedule.population,
                            polish_reserve,
                        )
                        if polish_enabled
                        else FinetuneBudget(schedule.evaluations_per_run, 0)
                    )
                    monitor = GenerationMonitor(
                        objective,
                        space,
                        run_name,
                        restart_after=int(
                            getattr(args, "rough_stagnation_generations", 0)
                        ),
                        improvement_tolerance=float(
                            getattr(
                                args,
                                "restart_min_improvement",
                                DEFAULT_RESTART_MIN_IMPROVEMENT,
                            )
                        ),
                        maximum_s11_db=args.s11_limit_db,
                    )
                    result = differential_evolution(
                        objective,
                        bounds=space.bounds,
                        maxiter=max(
                            0,
                            global_budget.differential_evolution
                            // schedule.population
                            - 1,
                        ),
                        popsize=args.popsize,
                        polish=bool(getattr(args, "scipy_polish", False)),
                        seed=seed,
                        workers=1,
                        updating="immediate",
                        x0=warm_vector,
                        callback=monitor,
                        tol=1e-3,
                    )
                    polish_stats = None
                    if polish_enabled:
                        polish_stats = coordinate_polish(
                            objective,
                            space,
                            args.s11_limit_db,
                            max(
                                0,
                                schedule.evaluations_per_run
                                - len(objective.history),
                            ),
                            initial_step=float(
                                getattr(
                                    args,
                                    "local_search_step",
                                    DEFAULT_LOCAL_SEARCH_STEP,
                                )
                            ),
                            minimum_step=float(
                                getattr(
                                    args,
                                    "local_search_min_step",
                                    DEFAULT_LOCAL_SEARCH_MIN_STEP,
                                )
                            ),
                            minimum_improvement=float(
                                getattr(
                                    args,
                                    "polish_min_improvement",
                                    DEFAULT_POLISH_MIN_IMPROVEMENT,
                                )
                            ),
                        )
                    optimizer_details = {
                        "generations": int(
                            getattr(result, "nit", monitor.stage_generations)
                        ),
                        "restarts": 0,
                        "stopped_for_stagnation": monitor.stopped_for_stagnation,
                        "final_normalized_diversity": (
                            monitor.last_diversity
                            if np.isfinite(monitor.last_diversity)
                            else None
                        ),
                        "coordinate_polish": (
                            asdict(polish_stats)
                            if polish_stats is not None
                            else None
                        ),
                    }
                confirmations = list(
                    getattr(objective, "confirmation_history", ())
                )
                optimizer_details.update(
                    confirmation_checks=len(confirmations),
                    confirmation_checks_with_outliers=sum(
                        item.status == "confirmed_with_outliers"
                        for item in confirmations
                    ),
                    quarantined_confirmations=sum(
                        item.status == "quarantined" for item in confirmations
                    ),
                )
                best = preferred_incumbent(objective, args.s11_limit_db)
                if best is None:
                    print(f"{run_name}: no successful candidate")
                    continue
                summary = result_payload(
                    best,
                    space,
                    turn_case,
                    seed,
                    progress.count,
                    frequency_hz,
                    simulation_metadata,
                    candidate_evaluations=progress.candidate_count,
                    simulation_evaluations=progress.simulation_count,
                    maximum_s11_db=args.s11_limit_db,
                )
                summary.update(
                    optimizer_success=bool(getattr(result, "success", False)),
                    optimizer_message=str(getattr(result, "message", "")),
                    run_evaluations=len(objective.history),
                    run_simulation_evaluations=(
                        objective_evaluation_count(objective)
                    ),
                    run_candidate_evaluations=len(objective.history),
                    optimizer=optimizer_details,
                )
                write_json(args.output / f"{run_name}.json", summary)
                run_summaries.append(summary)
    except KeyboardInterrupt:
        interrupted = True
        print("\nCampaign interrupted; CSV and campaign_best.json are current.")
    finally:
        progress.close()

    if run_summaries:
        run_summaries.sort(
            key=lambda item: (not item.get("feasible", False), item["objective"])
        )
        write_json(args.output / "run_summaries.json", {"runs": run_summaries})
    leaderboard = progress.topology_leaderboard()
    if leaderboard:
        useful_label, useful_key = {
            "horizon": ("H10", "horizon_p10_gain_dbi"),
            "ring": (
                "Worst ring P10"
                if len(args.target_thetas) > 1
                else "Ring P10",
                "ring_p10_gain_dbi",
            ),
            "directional": ("Target", "target_gain_dbi"),
            "peak": ("Peak", "peak_gain_dbi"),
        }[args.pattern]
        print("\nTOPOLOGY LEADERBOARD")
        for rank, summary in enumerate(leaderboard, start=1):
            turn_case = tuple(summary["turn_case"])
            metrics = summary.get("metrics", {})
            print(
                f"{rank:2d}. {format_turn_case(turn_case):>8} | "
                f"objective {summary['objective']:7.3f} | "
                f"S11 {metrics.get('worst_s11_db', float('nan')):6.2f} dB | "
                f"{useful_label} "
                f"{metrics.get(useful_key, float('nan')):5.2f} dBi | "
                f"peak {summary['peak_gain_dbi']:5.2f} dBi | "
                f"seed {summary['seed']}"
            )
        print(
            f"Leaderboard     : {(args.output / 'topology_leaderboard.json').resolve()}"
        )
    best_path = args.output / "campaign_best.json"
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        metrics = best.get("metrics", {})
        turn_case = tuple(best.get("turn_case", ()))
        print("\nCAMPAIGN BEST")
        print(f"Objective       : {best['objective']:.4f}")
        print(
            f"Topology        : {best.get('coil_count', len(turn_case))} coils, "
            f"turns {format_turn_case(turn_case)}"
        )
        print(f"Worst-band S11  : {metrics.get('worst_s11_db', float('nan')):.3f} dB")
        if args.pattern == "ring":
            if len(args.target_thetas) == 1:
                print(
                    f"Ring P10        : "
                    f"{metrics.get('ring_p10_gain_dbi', float('nan')):.3f} dBi"
                )
                print(
                    f"Ring minimum    : "
                    f"{metrics.get('ring_min_gain_dbi', float('nan')):.3f} dBi"
                )
            else:
                for index, theta_deg in enumerate(args.target_thetas):
                    print(
                        f"Ring {theta_deg:g} P10   : "
                        f"{metrics.get(f'ring_{index}_p10_gain_dbi', float('nan')):.3f} "
                        "dBi"
                    )
                print(
                    "Worst ring P10  : "
                    f"{metrics.get('ring_p10_gain_dbi', float('nan')):.3f} dBi"
                )
        elif args.pattern == "directional":
            print(
                f"Target gain     : "
                f"{metrics.get('target_gain_dbi', float('nan')):.3f} dBi"
            )
        elif args.pattern == "horizon":
            print(
                f"Horizon P10     : "
                f"{metrics.get('horizon_p10_gain_dbi', float('nan')):.3f} dBi"
            )
            print(
                f"Horizon minimum : "
                f"{metrics.get('horizon_min_gain_dbi', float('nan')):.3f} dBi"
            )
        print(f"Peak gain       : {best['peak_gain_dbi']:.3f} dBi")
        print(f"Result file     : {best_path.resolve()}")
    return CampaignOutcome(
        best_path=best_path if best_path.exists() else None,
        interrupted=interrupted,
    )


@dataclass(frozen=True)
class AutomaticBudgetModel:
    rough_batch_candidates: int
    fine_batch_candidates: int
    fine_required_batches: int


def _automatic_budget_model(args: argparse.Namespace) -> AutomaticBudgetModel:
    """Estimate the candidate cost of one generation batch in each stage."""
    initial = (
        load_reference_design(args.frequency_hz)
        if args.warm_start is None
        else load_design(args.warm_start)
    )
    initial = apply_design_overrides(
        initial,
        wire_diameter_mm=getattr(args, "wire_diameter_mm", None),
    )
    topology_args = copy.deepcopy(args)
    resolve_topology(topology_args, initial)
    populations = []
    variable_counts = []
    for turn_case in topology_args.turn_cases:
        design = design_for_turn_case(initial, turn_case, args.frequency_hz)
        space = make_space(
            design,
            args.frequency_hz,
            lock_coils=getattr(args, "lock_coils", False),
        )
        variable_counts.append(len(space.variables))
        populations.append(max(5, args.popsize*len(space.variables)))

    rough_seed_count = len(args.seeds)
    fine_seed_count = len(
        getattr(args, "automatic_finetune_seeds", args.seeds)
    )
    largest_population = max(populations)
    requested_polish = getattr(args, "polish_evaluations", None)
    polish_batches = max(
        int(
            np.ceil(
                (
                    int(requested_polish)
                    if requested_polish is not None
                    else max(DEFAULT_LOCAL_SEARCH_EVALUATIONS, 12*variables)
                )
                / population
            )
        )
        for variables, population in zip(variable_counts, populations)
    )
    return AutomaticBudgetModel(
        rough_batch_candidates=rough_seed_count*sum(populations),
        fine_batch_candidates=fine_seed_count*largest_population,
        fine_required_batches=1 + polish_batches,
    )


def _automatic_stage_budgets(
    args: argparse.Namespace,
    model: AutomaticBudgetModel | None = None,
) -> tuple[tuple[float | None, int | None], tuple[float | None, int | None]]:
    """Split one requested optimizer budget between rough and fine stages."""
    model = model or _automatic_budget_model(args)
    fraction = float(args.automatic_rough_fraction)
    if args.hours is not None:
        rough_hours = args.hours*fraction
        fine_hours = args.hours*(1.0 - fraction)
        seconds_per_candidate = float(args.seconds_per_eval)
        rough_minimum = (
            model.rough_batch_candidates*seconds_per_candidate/3600.0
        )
        fine_minimum = (
            model.fine_required_batches
            * model.fine_batch_candidates
            * seconds_per_candidate
            / 3600.0
        )
        if rough_hours + 1e-12 < rough_minimum:
            raise ValueError(
                "--automatic rough allocation is smaller than one population "
                f"batch ({rough_minimum:.3g} h required); increase --hours, "
                "reduce --popsize/seeds/topologies, or increase "
                "--automatic-rough-fraction"
            )
        if fine_hours + 1e-12 < fine_minimum:
            raise ValueError(
                "--automatic fine allocation cannot reserve differential "
                "evolution plus coordinate polish "
                f"({fine_minimum:.3g} h required); increase --hours, reduce "
                "--popsize/seeds, lower --polish-evaluations, or decrease "
                "--automatic-rough-fraction"
            )
        return ((rough_hours, None), (fine_hours, None))

    total_batches = int(args.maxiter) + 1
    minimum_batches = 1 + model.fine_required_batches
    if total_batches < minimum_batches:
        raise ValueError(
            "--automatic needs at least "
            f"--maxiter {minimum_batches - 1} so rough search, fine search, "
            "and coordinate polish all receive candidate capacity"
        )
    candidates = []
    for rough_batches in range(1, total_batches - model.fine_required_batches + 1):
        fine_batches = total_batches - rough_batches
        rough_candidates = rough_batches*model.rough_batch_candidates
        fine_candidates = fine_batches*model.fine_batch_candidates
        actual_fraction = rough_candidates/(rough_candidates + fine_candidates)
        candidates.append(
            (abs(actual_fraction - fraction), rough_batches, fine_batches)
        )
    _, rough_batches, fine_batches = min(candidates)
    return ((None, rough_batches - 1), (None, fine_batches - 1))


def _logged_candidate_count(path: Path) -> int | None:
    """Count completed optimizer candidates in a campaign CSV."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _roll_forward_automatic_budget(
    rough_csv: Path,
    rough_candidate_estimate: float,
    fine_budget: tuple[float | None, int | None],
    budget_model: AutomaticBudgetModel,
    seconds_per_candidate: float,
) -> tuple[tuple[float | None, int | None], dict[str, int]]:
    """Move unused rough candidate capacity into fine tuning."""
    rough_candidates_used = _logged_candidate_count(rough_csv)
    if rough_candidates_used is None:
        return fine_budget, {}
    unused = max(
        0,
        int(np.floor(rough_candidate_estimate)) - rough_candidates_used,
    )
    if fine_budget[0] is not None:
        adjusted = (
            fine_budget[0] + unused*seconds_per_candidate/3600.0,
            None,
        )
        rolled_forward = unused
    else:
        extra_batches = unused // budget_model.fine_batch_candidates
        adjusted = (None, int(fine_budget[1]) + extra_batches)
        rolled_forward = extra_batches*budget_model.fine_batch_candidates
    return adjusted, {
        "actual_rough_candidates": rough_candidates_used,
        "rolled_forward_candidates": rolled_forward,
    }


def select_automatic_winner(
    rough_payload: dict,
    fine_payload: dict,
) -> tuple[str, dict]:
    """Keep a feasible rough incumbent when noisy fine tuning regresses it."""

    def preference(payload: dict) -> tuple[int, float]:
        return (
            0 if payload.get("feasible", False) else 1,
            float(payload.get("objective", float("inf"))),
        )

    if preference(fine_payload) <= preference(rough_payload):
        return "fine_polish", fine_payload
    return "rough", rough_payload


def _finalize_automatic_winner(
    output: Path,
    rough_best_path: Path,
    rough_payload: dict,
    fine_output: Path,
    fine_best_path: Path,
    winning_turn_case: tuple[int, ...],
) -> tuple[Path, dict | None, str]:
    """Select and write the canonical winner after fine-stage completion."""
    fine_optimizer = None
    fine_summaries_path = fine_output / "run_summaries.json"
    if fine_summaries_path.is_file():
        fine_summaries = json.loads(
            fine_summaries_path.read_text(encoding="utf-8")
        ).get("runs", ())
        if fine_summaries:
            fine_optimizer = fine_summaries[0].get("optimizer")
            polish_summary = (
                fine_optimizer.get("coordinate_polish")
                if isinstance(fine_optimizer, dict)
                else None
            )
            if not polish_summary or polish_summary.get("evaluations", 0) <= 0:
                raise RuntimeError(
                    "automatic fine-tuning ended without running coordinate "
                    "polish; increase the optimizer budget"
                )

    fine_payload = json.loads(fine_best_path.read_text(encoding="utf-8"))
    selected_stage, selected_payload = select_automatic_winner(
        rough_payload,
        fine_payload,
    )
    final_payload = copy.deepcopy(selected_payload)
    final_payload["automatic_pipeline"] = {
        "rough_best": str(rough_best_path.resolve()),
        "fine_best": str(fine_best_path.resolve()),
        "winning_turn_case": list(winning_turn_case),
        "selected_stage": selected_stage,
        "fine_optimizer": fine_optimizer,
    }
    final_best = output / "campaign_best.json"
    write_json(final_best, final_payload)
    return final_best, fine_optimizer, selected_stage


def run_automatic_verification(
    args: argparse.Namespace,
    best_path: Path,
) -> tuple[Path, ...]:
    """Run isolated fine verification and fabrication export in the run root."""
    verify_script = Path(__file__).with_name("verify_best.py").resolve()
    command = [
        sys.executable,
        "-u",
        str(verify_script),
        str(best_path.resolve()),
        "--output",
        str(args.output.resolve()),
        "--sweep-bandwidth-mhz",
        f"{args.match_bandwidth_mhz:.12g}",
        "--solver",
        args.solver,
        "--design-sheet",
        "--jig-models",
    ]
    print("\nAUTOMATIC STAGE 4/4: FINE VERIFICATION + FABRICATION")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "automatic verification failed with exit code "
            f"{completed.returncode}; optimizer results remain in {args.output}"
        )

    payload = json.loads(best_path.read_text(encoding="utf-8"))
    required = [
        args.output / "verification.json",
        args.output / "s11_verified.png",
        args.output / "horizon_gain.png",
        args.output / "principal_plane_gain.png",
        args.output / "design_sheet.pdf",
    ]
    objective = payload.get("simulation", {}).get("objective", {})
    if objective.get("pattern_mode") == "ring":
        required.append(args.output / "target_ring_gain.png")
    if payload.get("coil_count", len(payload.get("design", {}).get("coils", ()))):
        required.append(args.output / "coil_formers.step")
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            "automatic verification did not create required artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return tuple(required)


def run_automatic_pipeline(args: argparse.Namespace) -> Path:
    """Run rough, fine, coordinate-polish, and verification stages."""
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "--automatic requires verification dependencies; install with "
            "pip install -e \".[optimize,verify]\""
        )
    if args.output.exists() and next(args.output.iterdir(), None) is not None:
        raise RuntimeError(
            f"Automatic output directory is not empty: {args.output}. Choose a "
            "new --output directory."
        )
    budget_model = _automatic_budget_model(args)
    rough_budget, fine_budget = _automatic_stage_budgets(args, budget_model)
    args.output.mkdir(parents=True, exist_ok=True)
    rough_candidate_estimate = (
        rough_budget[0]*3600/args.seconds_per_eval
        if rough_budget[0] is not None
        else (int(rough_budget[1]) + 1)*budget_model.rough_batch_candidates
    )
    fine_candidate_estimate = (
        fine_budget[0]*3600/args.seconds_per_eval
        if fine_budget[0] is not None
        else (int(fine_budget[1]) + 1)*budget_model.fine_batch_candidates
    )
    manifest_path = args.output / "automatic_pipeline.json"
    manifest: dict[str, object] = {
        "status": "running",
        "frequency_hz": args.frequency_hz,
        "output": str(args.output.resolve()),
        "stages": {},
        "budget": {
            "requested_hours": args.hours,
            "requested_maxiter": args.maxiter,
            "rough_fraction": args.automatic_rough_fraction,
            "rough": {
                "hours": rough_budget[0],
                "maxiter": rough_budget[1],
                "estimated_candidates": rough_candidate_estimate,
            },
            "fine_polish": {
                "hours": fine_budget[0],
                "maxiter": fine_budget[1],
                "estimated_candidates": fine_candidate_estimate,
                "minimum_batches": budget_model.fine_required_batches,
            },
            "estimated_rough_candidate_fraction": (
                rough_candidate_estimate
                / (rough_candidate_estimate + fine_candidate_estimate)
            ),
            "confirmation_simulations_counted_separately": True,
            "verification_outside_optimizer_budget": True,
        },
        "seeds": {
            "source": getattr(args, "seed_source", "command_line"),
            "rough": list(args.seeds),
            "fine_polish": list(
                getattr(args, "automatic_finetune_seeds", args.seeds)
            ),
        },
    }
    write_json(manifest_path, manifest)

    rough = copy.deepcopy(args)
    rough.automatic = False
    rough.finetune = False
    rough.polish = False
    rough.scipy_polish = False
    rough.output = args.output / "rough_search"
    rough.hours, rough.maxiter = rough_budget
    rough.seeds = tuple(args.seeds)
    rough.rough_stagnation_generations = args.restart_stagnation_generations
    rough.pipeline_stage = "rough"
    print("\nAUTOMATIC STAGE 1/4: ROUGH TOPOLOGY SEARCH")
    try:
        rough_outcome = run_campaign(rough)
    except BaseException as error:
        stage_status = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["status"] = stage_status
        manifest["stages"]["rough"] = {
            "status": stage_status,
            "error": f"{type(error).__name__}: {error}",
        }
        rough_checkpoint = rough.output / "campaign_best.json"
        if rough_checkpoint.is_file():
            manifest["stages"]["rough"]["best"] = str(
                rough_checkpoint.resolve()
            )
        write_json(manifest_path, manifest)
        raise
    if rough_outcome.interrupted:
        manifest["status"] = "interrupted"
        manifest["stages"]["rough"] = {"status": "interrupted"}
        write_json(manifest_path, manifest)
        raise KeyboardInterrupt
    if rough_outcome.best_path is None:
        manifest["status"] = "failed"
        manifest["stages"]["rough"] = {"status": "no_successful_candidate"}
        write_json(manifest_path, manifest)
        raise RuntimeError("automatic rough search produced no usable candidate")
    manifest["stages"]["rough"] = {
        "status": "complete",
        "best": str(rough_outcome.best_path.resolve()),
    }
    try:
        fine_budget, rollover = _roll_forward_automatic_budget(
            rough.output / "evaluations.csv",
            rough_candidate_estimate,
            fine_budget,
            budget_model,
            args.seconds_per_eval,
        )
        if rollover:
            rolled_forward = rollover["rolled_forward_candidates"]
            manifest["budget"]["rough"]["actual_candidates"] = rollover[
                "actual_rough_candidates"
            ]
            manifest["budget"]["fine_polish"][
                "rolled_forward_candidates"
            ] = rolled_forward
            manifest["budget"]["fine_polish"]["estimated_candidates"] = (
                fine_candidate_estimate + rolled_forward
            )
            manifest["budget"]["fine_polish"]["hours"] = fine_budget[0]
            manifest["budget"]["fine_polish"]["maxiter"] = fine_budget[1]
        write_json(manifest_path, manifest)
    except BaseException as error:
        stage_status = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["status"] = stage_status
        manifest["stages"]["rough"] = {
            "status": stage_status,
            "error": f"{type(error).__name__}: {error}",
            "best": str(rough_outcome.best_path.resolve()),
        }
        write_json(manifest_path, manifest)
        raise

    fine_output = args.output / "fine_tune"
    try:
        rough_best = json.loads(
            rough_outcome.best_path.read_text(encoding="utf-8")
        )
        winning_turn_case = tuple(int(value) for value in rough_best["turn_case"])
        fine = copy.deepcopy(args)
        fine.automatic = False
        fine.finetune = True
        fine.polish = True
        fine.scipy_polish = False
        fine.output = fine_output
        fine.warm_start = rough_outcome.best_path
        fine.turn_cases = (winning_turn_case,)
        fine.coil_counts = None
        fine.coil_count = len(winning_turn_case)
        fine.maximum_height_mm = rough.maximum_height_mm
        fine.maximum_height_source_override = (
            "automatic_campaign_shared"
            if args.maximum_height_mm is None
            else "command_line"
        )
        fine.hours, fine.maxiter = fine_budget
        fine.seeds = tuple(
            getattr(args, "automatic_finetune_seeds", args.seeds)
        )
        fine.pipeline_stage = "fine_polish"
        print("\nAUTOMATIC STAGES 2-3/4: FINE SEARCH + COORDINATE POLISH")
        fine_outcome = run_campaign(fine)
    except BaseException as error:
        stage_status = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["status"] = stage_status
        manifest["stages"]["fine_polish"] = {
            "status": stage_status,
            "error": f"{type(error).__name__}: {error}",
        }
        fine_checkpoint = fine_output / "campaign_best.json"
        if fine_checkpoint.is_file():
            manifest["stages"]["fine_polish"]["best"] = str(
                fine_checkpoint.resolve()
            )
        write_json(manifest_path, manifest)
        raise
    if fine_outcome.interrupted:
        manifest["status"] = "interrupted"
        manifest["stages"]["fine_polish"] = {"status": "interrupted"}
        write_json(manifest_path, manifest)
        raise KeyboardInterrupt
    if fine_outcome.best_path is None:
        manifest["status"] = "failed"
        manifest["stages"]["fine_polish"] = {
            "status": "no_successful_candidate"
        }
        write_json(manifest_path, manifest)
        raise RuntimeError("automatic fine-tuning produced no usable candidate")

    try:
        final_best, fine_optimizer, selected_stage = _finalize_automatic_winner(
            args.output,
            rough_outcome.best_path,
            rough_best,
            fine_output,
            fine_outcome.best_path,
            winning_turn_case,
        )
    except BaseException as error:
        stage_status = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["status"] = stage_status
        manifest["stages"]["fine_polish"] = {
            "status": stage_status,
            "error": f"{type(error).__name__}: {error}",
            "best": str(fine_outcome.best_path.resolve()),
        }
        write_json(manifest_path, manifest)
        raise
    manifest["stages"]["fine_polish"] = {
        "status": "complete",
        "best": str(fine_outcome.best_path.resolve()),
        "canonical_best": str(final_best.resolve()),
        "selected_stage": selected_stage,
        "optimizer": fine_optimizer,
    }
    write_json(manifest_path, manifest)

    try:
        artifacts = run_automatic_verification(args, final_best)
        verification_report = json.loads(
            (args.output / "verification.json").read_text(encoding="utf-8")
        )
        quality = verification_report.get("quality", {})
        quality_status = quality.get("status", "not_assessed")
    except BaseException as error:
        stage_status = (
            "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "verification_failed"
        )
        manifest["status"] = stage_status
        manifest["stages"]["verification"] = {
            "status": (
                "interrupted" if stage_status == "interrupted" else "failed"
            ),
            "error": f"{type(error).__name__}: {error}",
        }
        write_json(manifest_path, manifest)
        raise
    completion_status = (
        "complete" if quality_status == "passed" else "complete_with_warnings"
    )
    manifest["status"] = completion_status
    manifest["stages"]["verification"] = {
        "status": completion_status,
        "quality": quality,
        "artifacts": [str(path.resolve()) for path in artifacts],
    }
    write_json(manifest_path, manifest)
    suffix = "" if completion_status == "complete" else " (verification warnings)"
    print(f"\nAutomatic pipeline complete{suffix}: {args.output.resolve()}")
    return final_best


def main() -> None:
    args = parse_args()
    if args.automatic:
        run_automatic_pipeline(args)
    else:
        run_campaign(args)


if __name__ == "__main__":
    main()
