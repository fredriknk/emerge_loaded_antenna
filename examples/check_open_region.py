"""Run and certify open-region and mesh convergence at any frequency."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("EMERGE_STD_LOGLEVEL", "ERROR")

import numpy as np

from emerge_loaded_antenna import (
    CONVERGENCE_SCHEMA_VERSION,
    FrequencySweep,
    MeshSettings,
    OpenRegionSettings,
    REFERENCE_DESIGN_FREQUENCY_HZ,
    SOLVER_CHOICES,
    SimulationOptions,
    design_fingerprint,
    load_design,
    load_reference_design,
    save_design,
    selected_open_region_configuration,
    simulate,
)

DEFAULT_TOLERANCES = {
    "reflection_magnitude": 0.05,
    "peak_gain_dbi": 0.50,
    "horizon_min_gain_dbi": 0.35,
    "horizon_p10_gain_dbi": 0.25,
    "horizon_mean_gain_dbi": 0.25,
    "horizon_ripple_db": 0.40,
}


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be finite and positive")
    return values


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def sample_key(air_margin: float, abc_buffer: float, resolution: float) -> str:
    return (
        f"air={air_margin:.6g},buffer={abc_buffer:.6g},"
        f"resolution={resolution:.6g}"
    )


def run_sample(
    design,
    frequency_hz: float,
    air_margin: float,
    abc_buffer: float,
    resolution: float,
    angular_step: float,
    solver: str,
) -> dict:
    mesh = replace(
        MeshSettings(),
        air_margin_wavelengths=air_margin,
        wavelength_resolution=resolution,
    )
    open_region = replace(
        OpenRegionSettings(),
        mode="abc",
        abc_buffer_wavelengths=abc_buffer,
    )
    options = SimulationOptions(
        sweep=FrequencySweep.single(frequency_hz),
        mesh=mesh,
        open_region=open_region,
        solve=True,
        solver=solver,
        compute_farfield=True,
        farfield_frequency=frequency_hz,
        farfield_angular_step_deg=angular_step,
        verbose=False,
    )
    started = time.perf_counter()
    result = simulate(design, options)
    elapsed = time.perf_counter() - started
    metrics = result.farfield_metrics
    if metrics is None or result.s11.size != 1:
        raise RuntimeError("convergence sample did not return S11 and far-field data")
    s11 = complex(result.s11[0])
    return {
        "configuration": selected_open_region_configuration(mesh, open_region),
        "elapsed_seconds": elapsed,
        "mesh_nodes": result.artifacts.mesh_nodes,
        "mesh_elements": result.artifacts.mesh_elements,
        "volume_elements": result.artifacts.volume_elements,
        "huygens_face_count": len(result.artifacts.farfield_selection.tags),
        "termination_face_count": len(result.artifacts.termination_selection.tags),
        "huygens_touches_termination": bool(
            set(result.artifacts.farfield_selection.tags)
            & set(result.artifacts.termination_selection.tags)
        ),
        "s11_real": s11.real,
        "s11_imag": s11.imag,
        "s11_db": result.s11_db_at(frequency_hz),
        "peak_gain_dbi": metrics.peak_gain_dbi,
        "peak_theta_deg": metrics.peak_theta_deg,
        "peak_phi_deg": metrics.peak_phi_deg,
        "peak_elevation_deg": metrics.peak_elevation_deg,
        "horizon_min_gain_dbi": metrics.horizon_min_gain_dbi,
        "horizon_p10_gain_dbi": metrics.horizon_p10_gain_dbi,
        "horizon_mean_gain_dbi": metrics.horizon_mean_gain_dbi,
        "horizon_ripple_db": metrics.horizon_ripple_p90_p10_db,
    }


def run_sample_isolated(
    design,
    frequency_hz: float,
    air_margin: float,
    abc_buffer: float,
    resolution: float,
    angular_step: float,
    solver: str,
    timeout: float,
) -> dict:
    """Run one EMerge model in a fresh process to isolate its global CAD state."""
    with tempfile.TemporaryDirectory(prefix="antenna-convergence-") as temporary:
        source = Path(temporary)/"design.json"
        output = Path(temporary)/"sample.json"
        save_design(design, source)
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            str(source),
            "--frequency-mhz",
            f"{frequency_hz/1e6:.12g}",
            "--worker-output",
            str(output),
            "--worker-air-margin",
            str(air_margin),
            "--worker-abc-buffer",
            str(abc_buffer),
            "--worker-resolution",
            str(resolution),
            "--angular-step",
            str(angular_step),
            "--solver",
            solver,
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if not output.exists():
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"isolated solve exited {completed.returncode}: {detail}"
            )
        sample = json.loads(output.read_text(encoding="utf-8"))
        if "error" in sample:
            raise RuntimeError(sample["error"])
        return sample


def direction_delta_deg(first: dict, second: dict) -> float:
    vectors = []
    for sample in (first, second):
        theta = math.radians(sample["peak_theta_deg"])
        phi = math.radians(sample["peak_phi_deg"])
        vectors.append(
            np.array(
                (
                    math.sin(theta)*math.cos(phi),
                    math.sin(theta)*math.sin(phi),
                    math.cos(theta),
                )
            )
        )
    cosine = float(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def compare_samples(
    name: str,
    selected_key: str,
    reference_key: str,
    samples: dict[str, dict],
    tolerances: dict[str, float],
) -> dict:
    selected = samples.get(selected_key, {})
    reference = samples.get(reference_key, {})
    if "error" in selected or "error" in reference or not selected or not reference:
        return {
            "name": name,
            "selected": selected_key,
            "reference": reference_key,
            "passed": False,
            "error": "one or both required solves failed",
        }

    s11_selected = complex(selected["s11_real"], selected["s11_imag"])
    s11_reference = complex(reference["s11_real"], reference["s11_imag"])
    deltas = {
        "reflection_magnitude": abs(abs(s11_selected) - abs(s11_reference)),
        "peak_gain_dbi": abs(
            selected["peak_gain_dbi"] - reference["peak_gain_dbi"]
        ),
        "horizon_min_gain_dbi": abs(
            selected["horizon_min_gain_dbi"]
            - reference["horizon_min_gain_dbi"]
        ),
        "horizon_p10_gain_dbi": abs(
            selected["horizon_p10_gain_dbi"]
            - reference["horizon_p10_gain_dbi"]
        ),
        "horizon_mean_gain_dbi": abs(
            selected["horizon_mean_gain_dbi"]
            - reference["horizon_mean_gain_dbi"]
        ),
        "horizon_ripple_db": abs(
            selected["horizon_ripple_db"] - reference["horizon_ripple_db"]
        ),
    }
    checks = {
        metric: value <= tolerances[metric]
        for metric, value in deltas.items()
    }
    return {
        "name": name,
        "selected": selected_key,
        "reference": reference_key,
        "deltas": deltas,
        "peak_direction_delta_deg": direction_delta_deg(selected, reference),
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        help=(
            "optional design or optimizer result; defaults to a generated "
            "wavelength-scaled numerical reference"
        ),
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        help=(
            "solve frequency; inferred from optimizer output, otherwise "
            f"defaults to {REFERENCE_DESIGN_FREQUENCY_HZ/1e6:g} MHz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="certificate path; defaults to a frequency-specific result file",
    )
    parser.add_argument("--air-margins", type=parse_float_list, default=(0.20, 0.25, 0.35))
    parser.add_argument("--abc-buffers", type=parse_float_list, default=(0.75, 1.00, 1.25))
    parser.add_argument("--mesh-resolutions", type=parse_float_list, default=(0.50, 0.33, 0.25))
    parser.add_argument("--selected-air-margin", type=float, default=0.25)
    parser.add_argument("--selected-abc-buffer", type=float, default=1.00)
    parser.add_argument("--selected-resolution", type=float, default=0.33)
    parser.add_argument("--angular-step", type=float, default=4.0)
    parser.add_argument("--solver", choices=SOLVER_CHOICES, default="auto")
    parser.add_argument("--sample-timeout", type=float, default=600.0)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-air-margin", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-abc-buffer", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-resolution", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frequency_mhz is not None and (
        not math.isfinite(args.frequency_mhz) or args.frequency_mhz <= 0
    ):
        parser.error("--frequency-mhz must be finite and positive")
    selected = (
        args.selected_air_margin,
        args.selected_abc_buffer,
        args.selected_resolution,
    )
    if any(not math.isfinite(value) or value <= 0 for value in selected):
        parser.error("selected open-region values must be finite and positive")
    if not 0 < args.angular_step <= 10:
        parser.error("--angular-step must be between zero and 10 degrees")
    if args.sample_timeout <= 0:
        parser.error("--sample-timeout must be positive")
    if args.worker_output is None:
        if max(args.air_margins) <= args.selected_air_margin:
            parser.error("--air-margins must include a value above the selected margin")
        if max(args.abc_buffers) <= args.selected_abc_buffer:
            parser.error("--abc-buffers must include a value above the selected buffer")
        if min(args.mesh_resolutions) >= args.selected_resolution:
            parser.error(
                "--mesh-resolutions must include a value below the selected resolution"
            )
    elif args.result is None:
        parser.error("internal convergence workers require a design file")
    return args


def main() -> None:
    args = parse_args()
    source_frequency = None
    if args.result is not None:
        source_payload = json.loads(args.result.read_text(encoding="utf-8"))
        if isinstance(source_payload, dict):
            source_frequency = source_payload.get("frequency_hz")
    if args.frequency_mhz is not None:
        args.frequency_hz = args.frequency_mhz*1e6
    elif source_frequency is not None:
        args.frequency_hz = float(source_frequency)
    else:
        args.frequency_hz = REFERENCE_DESIGN_FREQUENCY_HZ
    if not math.isfinite(args.frequency_hz) or args.frequency_hz <= 0:
        raise SystemExit("Design metadata contains an invalid frequency_hz")
    args.frequency_mhz = args.frequency_hz/1e6
    if args.output is None:
        args.output = Path("optimization_results")/(
            f"open_region_convergence_{round(args.frequency_hz)}hz.json"
        )
    if args.result is None:
        design = load_reference_design(args.frequency_hz)
        source_label = "generated wavelength-scaled numerical reference"
    else:
        design = load_design(args.result)
        source_label = str(args.result.resolve())
    if args.worker_output is not None:
        try:
            sample = run_sample(
                design,
                args.frequency_hz,
                args.worker_air_margin,
                args.worker_abc_buffer,
                args.worker_resolution,
                args.angular_step,
                args.solver,
            )
        except Exception as error:
            sample = {"error": f"{type(error).__name__}: {error}"}
        write_json(args.worker_output, sample)
        return
    air_margins = tuple(dict.fromkeys((*args.air_margins, args.selected_air_margin)))
    abc_buffers = tuple(dict.fromkeys((*args.abc_buffers, args.selected_abc_buffer)))
    resolutions = tuple(
        dict.fromkeys((*args.mesh_resolutions, args.selected_resolution))
    )
    selected_tuple = (
        args.selected_air_margin,
        args.selected_abc_buffer,
        args.selected_resolution,
    )
    probes = [selected_tuple]
    probes.extend(
        (air, args.selected_abc_buffer, args.selected_resolution)
        for air in air_margins
    )
    probes.extend(
        (args.selected_air_margin, buffer, args.selected_resolution)
        for buffer in abc_buffers
    )
    probes.extend(
        (args.selected_air_margin, args.selected_abc_buffer, resolution)
        for resolution in resolutions
    )
    probes = list(dict.fromkeys(probes))

    print(f"OPEN-REGION CONVERGENCE @ {args.frequency_mhz:g} MHz")
    print(f"Design          : {source_label}")
    print("Boundary        : closed Huygens box + all-face buffered ABC")
    print(f"Solved probes   : {len(probes)}")
    print(f"Certificate     : {args.output.resolve()}\n")

    samples: dict[str, dict] = {}
    for index, (air_margin, abc_buffer, resolution) in enumerate(probes, 1):
        key = sample_key(air_margin, abc_buffer, resolution)
        print(f"[{index}/{len(probes)}] {key} ...", end=" ", flush=True)
        try:
            sample = run_sample_isolated(
                design,
                args.frequency_hz,
                air_margin,
                abc_buffer,
                resolution,
                args.angular_step,
                args.solver,
                args.sample_timeout,
            )
            samples[key] = sample
            print(
                f"H10 {sample['horizon_p10_gain_dbi']:.3f} dBi | "
                f"peak {sample['peak_gain_dbi']:.3f} dBi | "
                f"S11 {sample['s11_db']:.2f} dB | "
                f"{sample['elapsed_seconds']:.1f} s"
            )
        except Exception as error:  # EMerge failures must remain in the report.
            samples[key] = {"error": f"{type(error).__name__}: {error}"}
            print(f"FAILED: {error}")

    selected_key = sample_key(*selected_tuple)
    comparisons = (
        compare_samples(
            "huygens_clearance",
            selected_key,
            sample_key(
                max(air_margins),
                args.selected_abc_buffer,
                args.selected_resolution,
            ),
            samples,
            DEFAULT_TOLERANCES,
        ),
        compare_samples(
            "abc_distance",
            selected_key,
            sample_key(
                args.selected_air_margin,
                max(abc_buffers),
                args.selected_resolution,
            ),
            samples,
            DEFAULT_TOLERANCES,
        ),
        compare_samples(
            "air_mesh",
            selected_key,
            sample_key(
                args.selected_air_margin,
                args.selected_abc_buffer,
                min(resolutions),
            ),
            samples,
            DEFAULT_TOLERANCES,
        ),
    )
    passed = all(comparison["passed"] for comparison in comparisons)
    selected_mesh = replace(
        MeshSettings(),
        air_margin_wavelengths=args.selected_air_margin,
        wavelength_resolution=args.selected_resolution,
    )
    selected_open_region = replace(
        OpenRegionSettings(),
        mode="abc",
        abc_buffer_wavelengths=args.selected_abc_buffer,
    )
    payload = {
        "schema_version": CONVERGENCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "frequency_hz": args.frequency_hz,
        "source": source_label,
        "design_fingerprint": design_fingerprint(design),
        "design": asdict(design),
        "selected_configuration": selected_open_region_configuration(
            selected_mesh,
            selected_open_region,
        ),
        "criteria": DEFAULT_TOLERANCES,
        "samples": samples,
        "comparisons": list(comparisons),
    }
    write_json(args.output, payload)

    print("\nCONVERGENCE COMPARISONS")
    for comparison in comparisons:
        status = "PASS" if comparison["passed"] else "FAIL"
        print(f"{comparison['name']:20s}: {status}")
        for metric, delta in comparison.get("deltas", {}).items():
            print(
                f"  {metric:27s} {delta:.4f} "
                f"(limit {DEFAULT_TOLERANCES[metric]:.4f})"
            )
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    print(f"Report : {args.output.resolve()}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
