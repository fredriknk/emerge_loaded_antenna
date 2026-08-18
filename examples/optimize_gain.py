"""Comprehensive matched peak-gain optimization at 868 MHz."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import time

# Suppress EMerge's routine per-model INFO messages before importing it. Errors
# still appear, while failed candidates are also recorded in the CSV log.
os.environ.setdefault("EMERGE_STD_LOGLEVEL", "ERROR")

from scipy.optimize import differential_evolution

from emerge_loaded_antenna import (
    AntennaDesign,
    DesignSpace,
    DesignVariable,
    EvaluationRecord,
    GainMatchObjective,
)


def elapsed_text(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


def format_parameter(name: str, value: float) -> str:
    if name.endswith("turns") or name.endswith("radial_count"):
        return f"{name}={value:.0f}"
    if name.endswith("_deg"):
        return f"{name}={value:.1f} deg"
    return f"{name}={value*1e3:.2f} mm"


class ProgressLogger:
    """Write every candidate to CSV and print throttled progress summaries."""

    def __init__(
        self,
        space: DesignSpace,
        total: int,
        csv_path: Path,
        report_every: int,
    ):
        self.space = space
        self.total = total
        self.report_every = max(1, report_every)
        self.started = time.perf_counter()
        self.count = 0
        self.failures = 0
        self.best: EvaluationRecord | None = None
        self.last_reported_best: EvaluationRecord | None = None
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = csv_path.open("w", newline="", encoding="utf-8")
        fields = [
            "evaluation",
            "elapsed_seconds",
            "score",
            "s11_db",
            "peak_gain_dbi",
            "error",
            *space.names,
        ]
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    def __call__(self, record: EvaluationRecord) -> None:
        self.count += 1
        elapsed = time.perf_counter() - self.started
        if record.error is not None:
            self.failures += 1
        elif self.best is None or record.score < self.best.score:
            self.best = record

        row = {
            "evaluation": self.count,
            "elapsed_seconds": f"{elapsed:.3f}",
            "score": record.score,
            "s11_db": record.s11_db,
            "peak_gain_dbi": record.peak_gain_dbi,
            "error": record.error or "",
        }
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
            result_text = (
                f"best {self.best.score:8.3f} | "
                f"S11 {self.best.s11_db:7.2f} dB | "
                f"gain {self.best.peak_gain_dbi:6.2f} dBi"
            )
        print(
            f"[{self.count:4d}/{self.total} {100*progress:5.1f}%] "
            f"elapsed {elapsed_text(elapsed):>8} | ETA {elapsed_text(eta):>8} | "
            f"{result_text} | failed {self.failures}",
            flush=True,
        )
        if self.best is not None and self.best is not self.last_reported_best:
            parameters = ", ".join(
                format_parameter(name, value)
                for name, value in zip(self.space.names, self.best.vector)
            )
            print(f"    best geometry: {parameters}", flush=True)
            self.last_reported_best = self.best

    def close(self) -> None:
        self.file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize matched peak gain at 868 MHz.",
    )
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optimization_results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    space = DesignSpace(
        AntennaDesign(),
        (
            DesignVariable("bottom_length", 100e-3, 180e-3),
            DesignVariable("middle_length", 160e-3, 260e-3),
            DesignVariable("top_length", 100e-3, 180e-3),
            DesignVariable("coil1.pitch", 5e-3, 10e-3),
            DesignVariable("coil2.pitch", 5e-3, 10e-3),
            DesignVariable("radial_length", 60e-3, 100e-3),
        ),
    )
    population = args.popsize*len(space.variables)
    total = population*(args.maxiter + 1)
    args.output.mkdir(parents=True, exist_ok=True)
    history_path = args.output/"evaluations.csv"
    result_path = args.output/"best_result.json"
    progress = ProgressLogger(space, total, history_path, args.report_every)

    print("868 MHz MATCHED-GAIN OPTIMIZATION")
    print(f"Variables       : {len(space.variables)}")
    print(f"Population      : {population}")
    print(f"Generations     : {args.maxiter + 1} (including initial population)")
    print(f"Planned solves  : {total}")
    print(f"Candidate log   : {history_path.resolve()}")
    print("Objective       : maximize gain with S11 <= -10 dB")
    print("Press Ctrl+C to stop; completed evaluations remain in the CSV.\n")

    objective = GainMatchObjective(
        space,
        target_frequency=868e6,
        maximum_s11_db=-10.0,
        mismatch_weight=2.0,
        gain_weight=1.0,
        on_evaluation=progress,
    )
    try:
        result = differential_evolution(
            objective,
            bounds=space.bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            polish=False,
            seed=args.seed,
            workers=1,
            updating="immediate",
        )
    finally:
        progress.close()

    best = objective.best_record
    if best is None:
        raise RuntimeError("optimization produced no successful candidate")
    best_design = space.decode(best.vector)
    payload = {
        "frequency_hz": 868e6,
        "objective": best.score,
        "s11_db": best.s11_db,
        "peak_gain_dbi": best.peak_gain_dbi,
        "evaluations": len(objective.history),
        "failed_evaluations": sum(r.error is not None for r in objective.history),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "variables": dict(zip(space.names, best.vector)),
        "design": asdict(best_design),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nOPTIMIZATION COMPLETE")
    print(f"Best objective  : {best.score:.4f}")
    print(f"S11 at 868 MHz  : {best.s11_db:.3f} dB")
    print(f"Peak gain       : {best.peak_gain_dbi:.3f} dBi")
    print(f"Best parameters : {dict(zip(space.names, best.vector))}")
    print(f"Result file     : {result_path.resolve()}")


if __name__ == "__main__":
    main()
