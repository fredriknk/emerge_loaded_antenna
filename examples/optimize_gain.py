"""Frequency-scalable, multi-seed robust loaded-antenna optimization."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("EMERGE_STD_LOGLEVEL", "ERROR")

from scipy.optimize import differential_evolution
import numpy as np

from emerge_loaded_antenna import (
    AntennaDesign,
    DesignSpace,
    DesignVariable,
    EvaluationRecord,
    FrequencySweep,
    MeshSettings,
    OpenRegionSettings,
    REFERENCE_DESIGN_FREQUENCY_HZ,
    RobustGainObjective,
    SOLVER_CHOICES,
    SimulationOptions,
    load_design,
    load_reference_design,
    save_design,
    selected_open_region_configuration,
    validate_convergence_certificate,
)

METRIC_FIELDS = (
    "s11_low_db",
    "center_s11_db",
    "s11_high_db",
    "worst_s11_db",
    "useful_gain_dbi",
    "peak_gain_dbi",
    "peak_theta_deg",
    "peak_phi_deg",
    "horizon_min_gain_dbi",
    "horizon_p10_gain_dbi",
    "horizon_mean_gain_dbi",
    "horizon_ripple_db",
    "antenna_height_m",
    "mismatch_penalty",
    "pattern_penalty",
    "height_penalty",
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


def elapsed_text(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return result


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
    if name.endswith("turns") or name.endswith("radial_count"):
        return f"{name}={value:.0f}"
    if name.endswith("_deg"):
        return f"{name}={value:.1f} deg"
    return f"{name}={value*1e3:.2f} mm"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def free_space_wavelength(frequency_hz: float) -> float:
    """Return free-space wavelength after validating a frequency."""
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    return C0/frequency_hz


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
    allowance_wavelengths = (
        MONOPOLE_LENGTH_RANGE_LAMBDA[1]
        + COLLINEAR_SECTION_START_LAMBDA*int(maximum_coil_count)
    )
    return allowance_wavelengths*free_space_wavelength(frequency_hz)


def design_for_coil_count(
    base: AntennaDesign,
    coil_count: int,
    frequency_hz: float = REFERENCE_DESIGN_FREQUENCY_HZ,
) -> AntennaDesign:
    """Resize a design using monopole and collinear wavelength priors."""
    if (
        isinstance(coil_count, bool)
        or int(coil_count) != coil_count
        or coil_count < 0
    ):
        raise ValueError("coil_count must be a non-negative integer")
    coil_count = int(coil_count)
    if base.coil_count == coil_count:
        return base

    wavelength = free_space_wavelength(frequency_hz)
    wire_diameter = 2*base.wire_radius
    minimum_straight = MINIMUM_STRAIGHT_WIRE_DIAMETERS*wire_diameter
    base_section = max(BASE_SECTION_START_LAMBDA*wavelength, minimum_straight)
    collinear_section = max(
        COLLINEAR_SECTION_START_LAMBDA*wavelength,
        minimum_straight,
    )
    straight_lengths = (
        (base_section,)
        if coil_count == 0
        else (base_section,) + (collinear_section,)*coil_count
    )
    coils = base.coils[:coil_count]
    template = (
        base.coils[-1]
        if base.coils
        else load_reference_design(frequency_hz).coils[0]
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
            replace(coil, turns=turns)
            for coil, turns in zip(design.coils, turn_case)
        ),
    )


def make_space(
    base: AntennaDesign,
    frequency_hz: float = REFERENCE_DESIGN_FREQUENCY_HZ,
    finetune: bool = False,
) -> DesignSpace:
    """Create wavelength-scaled bounds for broad or fine topology searches."""
    wavelength = free_space_wavelength(frequency_hz)
    wire_diameter = 2*base.wire_radius

    def enclose(
        bounds: tuple[float, float],
        value: float,
    ) -> tuple[float, float]:
        lower, upper = bounds
        return min(lower, 0.8*value), max(upper, 1.2*value)

    original_pitches = tuple(coil.pitch for coil in base.coils)
    original_radii = tuple(coil.radius for coil in base.coils)
    if base.coils and not finetune:
        shared_pitch = float(np.mean(original_pitches))
        minimum_radius = max(
            0.5001*coil.transition_offset for coil in base.coils
        )
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
        straight_range[0]*wavelength,
        MINIMUM_STRAIGHT_WIRE_DIAMETERS*wire_diameter,
    )
    maximum_straight = max(
        straight_range[1]*wavelength,
        1.5*minimum_straight,
    )
    default_straight_bounds = (
        (minimum_straight, maximum_straight),
    )*len(base.straight_lengths)
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
    if base.coils and not finetune:
        minimum_pitch = max(
            COIL_PITCH_RANGE_LAMBDA[0]*wavelength,
            MINIMUM_PITCH_WIRE_DIAMETERS*wire_diameter,
        )
        pitch_bounds = (
            minimum_pitch,
            max(
                COIL_PITCH_RANGE_LAMBDA[1]*wavelength,
                6.0*wire_diameter,
                1.5*minimum_pitch,
            ),
        )
        for pitch in original_pitches:
            pitch_bounds = enclose(pitch_bounds, pitch)
        variables.append(
            DesignVariable(
                "coils.0.pitch",
                *pitch_bounds,
                linked_paths=tuple(
                    f"coils.{index}.pitch"
                    for index in range(1, base.coil_count)
                ),
                label="shared_coil_pitch",
            )
        )
        hard_minimum_radius = max(
            0.5001*coil.transition_offset for coil in base.coils
        )
        preferred_minimum_radius = max(
            COIL_RADIUS_RANGE_LAMBDA[0]*wavelength,
            MINIMUM_RADIUS_WIRE_DIAMETERS*wire_diameter,
            max(
                0.5001*coil.transition_offset + base.wire_radius
                for coil in base.coils
            ),
        )
        radius_bounds = (
            preferred_minimum_radius,
            max(
                COIL_RADIUS_RANGE_LAMBDA[1]*wavelength,
                8.0*wire_diameter,
                1.5*preferred_minimum_radius,
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
                    f"coils.{index}.radius"
                    for index in range(1, base.coil_count)
                ),
                label="shared_coil_radius",
            )
        )
    else:
        for index, coil in enumerate(base.coils):
            minimum_pitch = max(
                COIL_PITCH_RANGE_LAMBDA[0]*wavelength,
                MINIMUM_PITCH_WIRE_DIAMETERS*wire_diameter,
            )
            pitch_bounds = enclose(
                (
                    minimum_pitch,
                    max(
                        COIL_PITCH_RANGE_LAMBDA[1]*wavelength,
                        6.0*wire_diameter,
                        1.5*minimum_pitch,
                    ),
                ),
                coil.pitch,
            )
            variables.append(
                DesignVariable(f"coils.{index}.pitch", *pitch_bounds)
            )
        for index, coil in enumerate(base.coils):
            hard_minimum_radius = 0.5001*coil.transition_offset
            preferred_minimum_radius = max(
                COIL_RADIUS_RANGE_LAMBDA[0]*wavelength,
                MINIMUM_RADIUS_WIRE_DIAMETERS*wire_diameter,
                0.5001*coil.transition_offset + base.wire_radius,
            )
            radius_bounds = enclose(
                (
                    preferred_minimum_radius,
                    max(
                        COIL_RADIUS_RANGE_LAMBDA[1]*wavelength,
                        8.0*wire_diameter,
                        1.5*preferred_minimum_radius,
                    ),
                ),
                coil.radius,
            )
            radius_bounds = (
                max(hard_minimum_radius, radius_bounds[0]),
                radius_bounds[1],
            )
            variables.append(
                DesignVariable(f"coils.{index}.radius", *radius_bounds)
            )
    minimum_radial = max(
        RADIAL_LENGTH_RANGE_LAMBDA[0]*wavelength,
        MINIMUM_STRAIGHT_WIRE_DIAMETERS*wire_diameter,
    )
    radial_bounds = enclose(
        (
            minimum_radial,
            max(
                RADIAL_LENGTH_RANGE_LAMBDA[1]*wavelength,
                1.5*minimum_radial,
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


def result_payload(
    record: EvaluationRecord,
    space: DesignSpace,
    turn_case: tuple[int, ...],
    seed: int,
    evaluations: int,
    frequency_hz: float,
    simulation_metadata: dict | None = None,
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
    return payload


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
    ):
        self.output = output
        self.total = total
        self.report_every = max(1, report_every)
        self.started = time.perf_counter()
        self.count = 0
        self.failures = 0
        self.best: EvaluationRecord | None = None
        self.best_space: DesignSpace | None = None
        self.best_turn_case: tuple[int, ...] = ()
        self.best_seed = 0
        self.topology_records: dict[tuple[int, ...], EvaluationRecord] = {}
        self.topology_payloads: dict[tuple[int, ...], dict] = {}
        self.last_reported_best: EvaluationRecord | None = None
        self.frequency_hz = frequency_hz
        self.simulation_metadata = simulation_metadata
        self.space: DesignSpace | None = None
        self.turn_case: tuple[int, ...] = ()
        self.seed = 0
        output.mkdir(parents=True, exist_ok=True)
        self.file = (output/"evaluations.csv").open(
            "w", newline="", encoding="utf-8"
        )
        fields = [
            "evaluation",
            "elapsed_seconds",
            "coil_count",
            "turn_case",
            "seed",
            "score",
            "s11_db",
            "error",
            *METRIC_FIELDS,
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
        self.count += 1
        elapsed = time.perf_counter() - self.started
        if record.error is not None:
            self.failures += 1
        else:
            topology_best = self.topology_records.get(self.turn_case)
            if topology_best is None or record.score < topology_best.score:
                self.topology_records[self.turn_case] = record
                topology_payload = result_payload(
                    record,
                    self.space,
                    self.turn_case,
                    self.seed,
                    self.count,
                    self.frequency_hz,
                    self.simulation_metadata,
                )
                self.topology_payloads[self.turn_case] = topology_payload
                topology_name = format_turn_case(self.turn_case)
                write_json(
                    self.output/f"turns_{topology_name}_best.json",
                    topology_payload,
                )
            if self.best is None or record.score < self.best.score:
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
                )
                write_json(self.output/"campaign_best.json", payload)

        row = {
            "evaluation": self.count,
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

        should_report = (
            self.count == 1
            or self.count % self.report_every == 0
            or self.count == self.total
        )
        if not should_report:
            return
        progress = min(self.count/self.total, 1.0)
        eta = elapsed/progress - elapsed if progress else 0.0
        if self.best is None:
            result_text = "no successful candidate yet"
        else:
            metrics = self.best.metrics
            result_text = (
                f"best {self.best.score:7.3f} | "
                f"worst S11 {metrics.get('worst_s11_db', float('nan')):6.2f} | "
                f"H10 {metrics.get('horizon_p10_gain_dbi', float('nan')):5.2f} | "
                f"peak {self.best.peak_gain_dbi:5.2f} dBi"
            )
        print(
            f"[{self.count:5d}/{self.total} {100*progress:5.1f}%] "
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
        return sorted(
            self.topology_payloads.values(),
            key=lambda payload: payload["objective"],
        )

    def close(self) -> None:
        self.file.close()
        write_json(
            self.output/"topology_leaderboard.json",
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
        default=REFERENCE_DESIGN_FREQUENCY_HZ/1e6,
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
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--maxiter", type=int)
    budget.add_argument(
        "--hours",
        type=float,
        help="divide an estimated wall-time budget across every case and seed",
    )
    parser.add_argument("--seconds-per-eval", type=float, default=8.0)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--seeds", type=parse_int_list, default=(2, 3, 4, 5))
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
        "--finetune",
        "--fine-tune",
        action="store_true",
        help=(
            "optimize every coil pitch and radius independently; broad "
            "searches share one pitch and radius across all coils"
        ),
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
        choices=("horizon", "directional", "peak"),
        default="horizon",
    )
    parser.add_argument("--target-theta", type=float, default=90.0)
    parser.add_argument("--target-phi", type=float, default=0.0)
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
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args()
    if args.skip_convergence_check and args.require_convergence:
        parser.error(
            "--skip-convergence-check and --require-convergence conflict"
        )
    if not np.isfinite(args.frequency_mhz) or args.frequency_mhz <= 0:
        parser.error("--frequency-mhz must be finite and positive")
    args.frequency_hz = args.frequency_mhz*1e6
    frequency_scale = REFERENCE_DESIGN_FREQUENCY_HZ/args.frequency_hz
    if args.match_bandwidth_mhz is None:
        args.match_bandwidth_mhz = 10.0/frequency_scale
    if (
        not np.isfinite(args.match_bandwidth_mhz)
        or args.match_bandwidth_mhz <= 0
        or args.match_bandwidth_mhz >= 2*args.frequency_mhz
    ):
        parser.error(
            "--match-bandwidth-mhz must be positive and below twice the target"
        )
    if (
        args.maximum_height_mm is not None
        and (
            not np.isfinite(args.maximum_height_mm)
            or args.maximum_height_mm <= 0
        )
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
    numerical_values = (
        args.air_margin_wavelengths,
        args.abc_buffer_wavelengths,
        args.wavelength_resolution,
    )
    if any(not np.isfinite(value) or value <= 0 for value in numerical_values):
        parser.error("open-region and mesh values must be finite and positive")
    if args.coil_count is not None and args.coil_count < 0:
        parser.error("--coil-count must be non-negative")
    if args.coil_counts is not None and args.turn_cases is not None:
        parser.error(
            "--coil-counts and --turn-cases conflict; list mixed topologies "
            "directly with --turn-cases"
        )
    if args.coil_counts is not None:
        args.turn_cases = tuple((1,)*count for count in args.coil_counts)
    elif args.turn_cases is None:
        if args.coil_count is not None:
            args.turn_cases = ((1,)*args.coil_count,)
    elif args.coil_count is not None:
        if any(len(turn_case) != args.coil_count for turn_case in args.turn_cases):
            parser.error("every --turn-cases entry must match --coil-count")
    else:
        case_counts = {len(turn_case) for turn_case in args.turn_cases}
        if len(case_counts) == 1:
            args.coil_count = next(iter(case_counts))
    weights = (
        args.mismatch_weight,
        args.null_weight,
        args.ripple_weight,
        args.height_weight,
    )
    if any(weight < 0 for weight in weights):
        parser.error("objective weights must be non-negative")
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frequency_label = f"{args.frequency_mhz:g}mhz".replace(".", "p")
        args.output = Path("optimization_results")/f"{frequency_label}_{stamp}"
    if args.convergence_report is None:
        frequency_hz = round(args.frequency_hz)
        args.convergence_report = Path(
            "optimization_results"
        )/f"open_region_convergence_{frequency_hz}hz.json"
    return args


def load_baseline(path: Path | None, frequency_hz: float) -> AntennaDesign:
    if path is None:
        design = load_reference_design(frequency_hz)
        scale = REFERENCE_DESIGN_FREQUENCY_HZ/frequency_hz
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
            turn_cases = tuple((1,)*count for count in coil_counts)
        elif coil_count is not None:
            turn_cases = ((1,)*coil_count,)
        else:
            turn_cases = (tuple(coil.turns for coil in initial.coils),)

    args.turn_cases = tuple(dict.fromkeys(turn_cases))
    args.coil_counts = tuple(
        dict.fromkeys(len(turn_case) for turn_case in args.turn_cases)
    )
    args.coil_count = (
        args.coil_counts[0] if len(args.coil_counts) == 1 else None
    )


def iterations_per_run(
    args: argparse.Namespace,
    variables: int,
    run_count: int,
) -> int:
    if args.maxiter is not None:
        return args.maxiter
    evaluations = args.hours*3600/args.seconds_per_eval/run_count
    population = max(5, args.popsize*variables)
    generations = max(1, int(evaluations/population))
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
    run_count = len(args.seeds)*len(args.turn_cases)
    schedules = []
    for turn_case in args.turn_cases:
        design = design_for_turn_case(initial, turn_case, frequency_hz)
        space = make_space(
            design,
            frequency_hz,
            finetune=getattr(args, "finetune", False),
        )
        variable_count = len(space.variables)
        maxiter = iterations_per_run(args, variable_count, run_count)
        population = max(5, args.popsize*variable_count)
        schedules.append(
            CaseSchedule(
                turn_case=turn_case,
                space=space,
                maxiter=maxiter,
                population=population,
                evaluations_per_run=population*(maxiter + 1),
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
            "OPEN-REGION CONVERGENCE REQUIRED BUT NOT CERTIFIED\n"
            f"{detail}"
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
    try:
        return validate_convergence_certificate(
            args.convergence_report,
            benchmark,
            mesh,
            open_region,
            frequency_hz,
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
        source = args.output/"convergence_reference_design.json"
        save_design(benchmark, source)
        print(f"Benchmark       : {source.resolve()}")
        command = (
            sys.executable,
            "-u",
            str(Path(__file__).with_name("check_open_region.py").resolve()),
            str(source.resolve()),
            "--frequency-mhz",
            f"{frequency_hz/1e6:.12g}",
            "--output",
            str(args.convergence_report.resolve()),
            "--air-margins",
            _number_list(0.8*air_margin, air_margin, 1.4*air_margin),
            "--abc-buffers",
            _number_list(0.75*abc_buffer, abc_buffer, 1.25*abc_buffer),
            "--mesh-resolutions",
            _number_list(1.5*resolution, resolution, 0.75*resolution),
            "--selected-air-margin",
            f"{air_margin:.12g}",
            "--selected-abc-buffer",
            f"{abc_buffer:.12g}",
            "--selected-resolution",
            f"{resolution:.12g}",
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
            )
        except RuntimeError as validation_error:
            handle_uncertified_convergence(
                args,
                "Automatic convergence produced an invalid report:\n"
                f"{validation_error}",
            )
            return None


def run_campaign(args: argparse.Namespace) -> None:
    frequency_hz = args.frequency_hz
    args.finetune = bool(getattr(args, "finetune", False))
    initial = load_baseline(args.warm_start, frequency_hz)
    resolve_topology(args, initial)
    automatic_height = args.maximum_height_mm is None
    if automatic_height:
        args.maximum_height_mm = 1e3*default_maximum_height(
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
        "mesh": asdict(mesh),
        "open_region": asdict(open_region),
        "solver": args.solver,
        "target_frequency_hz": frequency_hz,
        "match_bandwidth_hz": args.match_bandwidth_mhz*1e6,
        "farfield_angular_step_deg": args.angular_step,
        "coil_counts": list(args.coil_counts),
        "turn_cases": [list(turn_case) for turn_case in args.turn_cases],
        "coil_parameterization": (
            "independent" if args.finetune else "shared"
        ),
        "search_bounds": {
            "policy": "wavelength_wire_v1",
            "wavelength_m": free_space_wavelength(frequency_hz),
            "wire_diameter_m": 2*initial.wire_radius,
            "maximum_height_m": args.maximum_height_mm*1e-3,
            "maximum_height_wavelengths": (
                args.maximum_height_mm*1e-3
                / free_space_wavelength(frequency_hz)
            ),
            "maximum_height_source": (
                "automatic" if automatic_height else "command_line"
            ),
        },
        "convergence_status": (
            "skipped"
            if args.skip_convergence_check
            else "passed" if certificate else "warning"
        ),
        "convergence_warning": args.convergence_warning,
        "convergence_report": (
            None
            if args.skip_convergence_check
            else str(args.convergence_report.resolve())
        ),
    }
    variable_names = tuple(
        dict.fromkeys(
            name
            for schedule in schedules
            for name in schedule.space.names
        )
    )
    total = len(args.seeds)*sum(
        schedule.evaluations_per_run for schedule in schedules
    )
    progress = CampaignProgress(
        args.output,
        total,
        args.report_every,
        variable_names,
        frequency_hz,
        simulation_metadata,
    )
    run_summaries = []

    low_mhz = args.frequency_mhz - args.match_bandwidth_mhz/2
    high_mhz = args.frequency_mhz + args.match_bandwidth_mhz/2
    print(f"ROBUST {args.frequency_mhz:g} MHz ANTENNA CAMPAIGN")
    print(f"Pattern target  : {args.pattern}")
    print(
        f"Match samples   : {low_mhz:g}, {args.frequency_mhz:g} and "
        f"{high_mhz:g} MHz"
    )
    print(
        f"Pattern limits  : horizon min {args.minimum_horizon_gain_dbi:.1f} dBi, "
        f"P90-P10 ripple {args.maximum_ripple_db:.1f} dB"
    )
    height_wavelengths = (
        args.maximum_height_mm*1e-3/free_space_wavelength(frequency_hz)
    )
    print(
        f"Physical limit  : {args.maximum_height_mm:.1f} mm "
        f"({height_wavelengths:.2f} lambda) maximum height before penalty"
        + (" [automatic]" if automatic_height else "")
    )
    print(
        "Length priors   : bare 0.18-0.70 lambda; loaded sections "
        "0.15-0.72 lambda"
    )
    print(
        "Coil/radial     : pitch 0.010-0.040 lambda, radius "
        "0.015-0.050 lambda, radials 0.15-0.40 lambda; wire floors apply"
    )
    print(
        "Coil counts     : "
        + ", ".join(str(count) for count in args.coil_counts)
    )
    print(
        "Coil geometry   : "
        + (
            "independent pitch/radius (--finetune)"
            if args.finetune
            else "shared pitch/radius (broad search)"
        )
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
    print(f"Seeds           : {args.seeds}")
    print("Search schedule :")
    for schedule in schedules:
        print(
            f"  {format_turn_case(schedule.turn_case):>8} : "
            f"{len(schedule.space.variables):2d} variables | "
            f"population {schedule.population:3d} | "
            f"iterations {schedule.maxiter:4d} | "
            f"{schedule.evaluations_per_run:5d} solves/seed"
        )
    print(f"Planned solves  : {total}")
    print(f"Campaign log    : {(args.output/'evaluations.csv').resolve()}")
    print("Global and per-topology bests are checkpointed after every improvement.")
    print("Press Ctrl+C to stop safely; completed evaluations remain on disk.\n")

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
                        span=args.match_bandwidth_mhz*1e6,
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
                    target_phi_deg=args.target_phi,
                    maximum_s11_db=args.s11_limit_db,
                    mismatch_weight=args.mismatch_weight,
                    minimum_horizon_gain_dbi=args.minimum_horizon_gain_dbi,
                    null_weight=args.null_weight,
                    maximum_horizon_ripple_db=args.maximum_ripple_db,
                    ripple_weight=args.ripple_weight,
                    maximum_height=args.maximum_height_mm*1e-3,
                    height_weight=args.height_weight,
                    options=options,
                    on_evaluation=progress,
                )
                lower = np.asarray([bound[0] for bound in space.bounds])
                upper = np.asarray([bound[1] for bound in space.bounds])
                warm_vector = np.clip(space.initial_vector, lower, upper)
                result = differential_evolution(
                    objective,
                    bounds=space.bounds,
                    maxiter=schedule.maxiter,
                    popsize=args.popsize,
                    polish=args.polish,
                    seed=seed,
                    workers=1,
                    updating="immediate",
                    x0=warm_vector,
                    tol=1e-3,
                )
                best = objective.best_record
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
                )
                summary.update(
                    optimizer_success=bool(result.success),
                    optimizer_message=str(result.message),
                    run_evaluations=len(objective.history),
                )
                write_json(args.output/f"{run_name}.json", summary)
                run_summaries.append(summary)
    except KeyboardInterrupt:
        print("\nCampaign interrupted; CSV and campaign_best.json are current.")
    finally:
        progress.close()

    if run_summaries:
        run_summaries.sort(key=lambda item: item["objective"])
        write_json(args.output/"run_summaries.json", {"runs": run_summaries})
    leaderboard = progress.topology_leaderboard()
    if leaderboard:
        print("\nTOPOLOGY LEADERBOARD")
        for rank, summary in enumerate(leaderboard, start=1):
            turn_case = tuple(summary["turn_case"])
            metrics = summary.get("metrics", {})
            print(
                f"{rank:2d}. {format_turn_case(turn_case):>8} | "
                f"objective {summary['objective']:7.3f} | "
                f"S11 {metrics.get('worst_s11_db', float('nan')):6.2f} dB | "
                f"H10 "
                f"{metrics.get('horizon_p10_gain_dbi', float('nan')):5.2f} dBi | "
                f"peak {summary['peak_gain_dbi']:5.2f} dBi | "
                f"seed {summary['seed']}"
            )
        print(
            "Leaderboard     : "
            f"{(args.output/'topology_leaderboard.json').resolve()}"
        )
    best_path = args.output/"campaign_best.json"
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
        print(f"Horizon P10     : {metrics.get('horizon_p10_gain_dbi', float('nan')):.3f} dBi")
        print(f"Horizon minimum : {metrics.get('horizon_min_gain_dbi', float('nan')):.3f} dBi")
        print(f"Peak gain       : {best['peak_gain_dbi']:.3f} dBi")
        print(f"Result file     : {best_path.resolve()}")


def main() -> None:
    run_campaign(parse_args())


if __name__ == "__main__":
    main()
