"""Warm-started, multi-seed robust gain optimization at 868 MHz."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time

os.environ.setdefault("EMERGE_STD_LOGLEVEL", "ERROR")

from scipy.optimize import differential_evolution
import numpy as np

from emerge_loaded_antenna import (
    AntennaDesign,
    CoilDesign,
    DesignSpace,
    DesignVariable,
    EvaluationRecord,
    FrequencySweep,
    MeshSettings,
    OpenRegionSettings,
    RobustGainObjective,
    SOLVER_CHOICES,
    SimulationOptions,
    load_design,
    selected_open_region_configuration,
    validate_convergence_certificate,
)

FREQUENCY = 868e6
METRIC_FIELDS = (
    "s11_863_mhz_db",
    "s11_868_mhz_db",
    "s11_873_mhz_db",
    "center_s11_db",
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


def elapsed_text(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


def parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return result


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


def design_for_coil_count(base: AntennaDesign, coil_count: int) -> AntennaDesign:
    """Resize a warm-start design while preserving total straight length."""
    if (
        isinstance(coil_count, bool)
        or int(coil_count) != coil_count
        or coil_count < 0
    ):
        raise ValueError("coil_count must be a non-negative integer")
    coil_count = int(coil_count)
    if base.coil_count == coil_count:
        return base

    section_length = sum(base.straight_lengths)/(coil_count + 1)
    coils = base.coils[:coil_count]
    coils += tuple(CoilDesign() for _ in range(coil_count - len(coils)))
    return replace(
        base,
        straight_lengths=(section_length,)*(coil_count + 1),
        coils=coils,
    )


def design_for_turn_case(
    base: AntennaDesign,
    turn_case: tuple[int, ...],
) -> AntennaDesign:
    design = design_for_coil_count(base, len(turn_case))
    return replace(
        design,
        coils=tuple(
            replace(coil, turns=turns)
            for coil, turns in zip(design.coils, turn_case)
        ),
    )


def make_space(base: AntennaDesign) -> DesignSpace:
    """Continuous variables; coil turns are fixed by the surrounding case."""
    if base.coil_count == 2:
        straight_bounds = (
            (80e-3, 200e-3),
            (120e-3, 280e-3),
            (80e-3, 200e-3),
        )
    elif base.coil_count == 0:
        straight_bounds = ((40e-3, 600e-3),)
    else:
        straight_bounds = ((40e-3, 300e-3),)*len(base.straight_lengths)

    variables = [
        DesignVariable(f"straight_lengths.{index}", lower, upper)
        for index, (lower, upper) in enumerate(straight_bounds)
    ]
    variables.extend(
        DesignVariable(f"coils.{index}.pitch", 4e-3, 12e-3)
        for index in range(base.coil_count)
    )
    variables.extend(
        DesignVariable(f"coils.{index}.radius", 7e-3, 16e-3)
        for index in range(base.coil_count)
    )
    variables.extend(
        (
            DesignVariable("radial_length", 55e-3, 120e-3),
            DesignVariable("radial_angle_deg", 20.0, 70.0),
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
    simulation_metadata: dict | None = None,
) -> dict:
    design = space.decode(record.vector)
    payload = {
        "frequency_hz": FREQUENCY,
        "objective": record.score,
        "s11_db": record.s11_db,
        "peak_gain_dbi": record.peak_gain_dbi,
        "metrics": dict(record.metrics),
        "coil_count": len(turn_case),
        "turn_case": list(turn_case),
        "seed": seed,
        "evaluations_at_save": evaluations,
        "variables": dict(zip(space.names, record.vector)),
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
        self.last_reported_best: EvaluationRecord | None = None
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
        elif self.best is None or record.score < self.best.score:
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
                self.simulation_metadata,
            )
            write_json(self.output/"campaign_best.json", payload)

        row = {
            "evaluation": self.count,
            "elapsed_seconds": f"{elapsed:.3f}",
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

    def close(self) -> None:
        self.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm-started multi-seed optimization of useful 868 MHz gain, "
            "broadband match, pattern quality and antenna height."
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
    parser.add_argument(
        "--coil-count",
        type=int,
        help="number of loading coils; defaults to two",
    )
    parser.add_argument(
        "--turn-cases",
        type=parse_turn_cases,
        help=(
            "separate discrete searches; use none for zero coils, 1 for one "
            "coil, or forms such as 1x2 and 1x2x1"
        ),
    )
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=Path("optimization_results/best_result.json"),
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
    parser.add_argument("--maximum-height-mm", type=float, default=600.0)
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
        default=Path("optimization_results/open_region_convergence.json"),
        help="passing report from examples/check_open_region.py",
    )
    parser.add_argument(
        "--skip-convergence-check",
        action="store_true",
        help="run without a matching open-region certificate (not recommended)",
    )
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help="linear solver backend (default: %(default)s)",
    )
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args()
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
    if args.turn_cases is None:
        args.coil_count = 2 if args.coil_count is None else args.coil_count
        args.turn_cases = ((1,)*args.coil_count,)
    else:
        case_counts = {len(turn_case) for turn_case in args.turn_cases}
        if len(case_counts) != 1:
            parser.error("all --turn-cases must use the same number of coils")
        parsed_count = next(iter(case_counts))
        if args.coil_count is None:
            args.coil_count = parsed_count
        elif args.coil_count != parsed_count:
            parser.error("--turn-cases do not match --coil-count")
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
        args.output = Path("optimization_results")/f"robust_{stamp}"
    return args


def load_baseline(path: Path) -> AntennaDesign:
    if path.exists():
        design = load_design(path)
        print(f"Warm start      : {path.resolve()}")
        return design
    print(f"Warm start      : {path} not found; using library defaults")
    return AntennaDesign()


def iterations_per_run(
    args: argparse.Namespace,
    variables: int,
    run_count: int,
) -> int:
    if args.maxiter is not None:
        return args.maxiter
    evaluations = args.hours*3600/args.seconds_per_eval/run_count
    population = args.popsize*variables
    generations = max(1, int(evaluations/population))
    return max(0, generations - 1)


def run_campaign(args: argparse.Namespace) -> None:
    baseline = design_for_coil_count(load_baseline(args.warm_start), args.coil_count)
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
    if not args.skip_convergence_check:
        try:
            certificate = validate_convergence_certificate(
                args.convergence_report,
                baseline,
                mesh,
                open_region,
                FREQUENCY,
            )
        except RuntimeError as error:
            raise SystemExit(
                f"OPEN-REGION PREFLIGHT FAILED\n{error}\n\n"
                "Run examples/check_open_region.py for this warm start and "
                "configuration, or use --skip-convergence-check explicitly."
            ) from error
    simulation_metadata = {
        "mesh": asdict(mesh),
        "open_region": asdict(open_region),
        "solver": args.solver,
        "farfield_angular_step_deg": args.angular_step,
        "convergence_report": (
            str(args.convergence_report.resolve()) if certificate else None
        ),
    }
    run_count = len(args.seeds)*len(args.turn_cases)
    baseline_space = make_space(baseline)
    variable_count = len(baseline_space.variables)
    maxiter = iterations_per_run(args, variable_count, run_count)
    population = args.popsize*variable_count
    evaluations_per_run = population*(maxiter + 1)
    total = evaluations_per_run*run_count
    progress = CampaignProgress(
        args.output,
        total,
        args.report_every,
        baseline_space.names,
        simulation_metadata,
    )
    run_summaries = []

    print("ROBUST 868 MHz ANTENNA CAMPAIGN")
    print(f"Pattern target  : {args.pattern}")
    print("Match samples   : 863, 868 and 873 MHz")
    print(
        f"Pattern limits  : horizon min {args.minimum_horizon_gain_dbi:.1f} dBi, "
        f"P90-P10 ripple {args.maximum_ripple_db:.1f} dB"
    )
    print(
        f"Physical limit  : {args.maximum_height_mm:.1f} mm maximum height "
        "before penalty"
    )
    print(f"Loading coils   : {args.coil_count}")
    print(f"Linear solver   : {args.solver}")
    print(
        f"Open region     : inner Huygens box at "
        f"{mesh.air_margin_wavelengths:.2f} lambda + "
        f"{open_region.abc_buffer_wavelengths:.2f} lambda all-face ABC buffer"
    )
    print(f"Air resolution  : {mesh.wavelength_resolution:.2f} wavelengths")
    if certificate is None:
        print("Preflight       : SKIPPED BY USER")
    else:
        print(f"Preflight       : PASS ({args.convergence_report.resolve()})")
    print(f"Variables       : {variable_count} continuous")
    print(
        "Turn cases      : "
        + ", ".join(format_turn_case(case) for case in args.turn_cases)
    )
    print(f"Seeds           : {args.seeds}")
    print(f"Population/run  : {population}")
    print(f"Iterations/run  : {maxiter}")
    print(f"Planned solves  : {total}")
    print(f"Campaign log    : {(args.output/'evaluations.csv').resolve()}")
    print("Current best is checkpointed after every improvement.")
    print("Press Ctrl+C to stop safely; completed evaluations remain on disk.\n")

    try:
        for turn_case in args.turn_cases:
            case_design = design_for_turn_case(baseline, turn_case)
            space = make_space(case_design)
            for seed in args.seeds:
                run_name = f"turns_{format_turn_case(turn_case)}_seed_{seed}"
                print(f"\nStarting {run_name}", flush=True)
                progress.set_context(space, turn_case, seed)
                options = SimulationOptions(
                    sweep=FrequencySweep(center=FREQUENCY, span=10e6, points=3),
                    mesh=mesh,
                    open_region=open_region,
                    solve=True,
                    solver=args.solver,
                    compute_farfield=True,
                    farfield_frequency=FREQUENCY,
                    farfield_angular_step_deg=args.angular_step,
                    verbose=False,
                )
                objective = RobustGainObjective(
                    space,
                    target_frequency=FREQUENCY,
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
                    maxiter=maxiter,
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
    best_path = args.output/"campaign_best.json"
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        metrics = best.get("metrics", {})
        print("\nCAMPAIGN BEST")
        print(f"Objective       : {best['objective']:.4f}")
        print(f"Worst-band S11  : {metrics.get('worst_s11_db', float('nan')):.3f} dB")
        print(f"Horizon P10     : {metrics.get('horizon_p10_gain_dbi', float('nan')):.3f} dBi")
        print(f"Horizon minimum : {metrics.get('horizon_min_gain_dbi', float('nan')):.3f} dBi")
        print(f"Peak gain       : {best['peak_gain_dbi']:.3f} dBi")
        print(f"Result file     : {best_path.resolve()}")


def main() -> None:
    run_campaign(parse_args())


if __name__ == "__main__":
    main()
