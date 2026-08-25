"""Fine-mesh, fine-angle verification at an inferred or requested frequency."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("EMERGE_STD_LOGLEVEL", "WARNING")

import emerge as em
import matplotlib.pyplot as plt
import numpy as np

from emerge_loaded_antenna import (
    REFERENCE_DESIGN_FREQUENCY_HZ,
    SOLVER_CHOICES,
    AntennaDesign,
    FrequencySweep,
    MeshSettings,
    SimulationOptions,
    SimulationResult,
    load_design,
    load_reference_design,
    simulate,
)
from emerge_loaded_antenna.drawing import export_drawing
from emerge_loaded_antenna.formers import export_coil_formers


def gain_db(farfield) -> np.ndarray:
    amplitude = np.abs(np.asarray(farfield.normE)/em.lib.EISO)
    return 20*np.log10(np.maximum(amplitude, 1e-12))


def pattern_target_from_payload(payload: object) -> dict | None:
    """Recover a ring or directional objective from optimizer metadata."""
    if not isinstance(payload, dict):
        return None
    simulation = payload.get("simulation")
    if not isinstance(simulation, dict):
        return None
    objective = simulation.get("objective")
    if not isinstance(objective, dict):
        return None
    mode = objective.get("pattern_mode")
    if mode not in {"ring", "directional"}:
        return None
    raw_thetas = objective.get("target_theta_degrees")
    theta_degrees = (
        tuple(float(value) for value in raw_thetas)
        if isinstance(raw_thetas, (list, tuple)) and raw_thetas
        else (float(objective.get("target_theta_deg", 90.0)),)
    )
    return {
        "mode": mode,
        "theta_deg": theta_degrees[0],
        "theta_degrees": theta_degrees,
        "phi_deg": (
            float(objective.get("target_phi_deg", 0.0))
            if mode == "directional"
            else None
        ),
        "beamwidth_deg": (
            float(objective["target_beamwidth_deg"])
            if objective.get("target_beamwidth_deg") is not None
            else None
        ),
    }


def verification_quality(
    verification: dict,
    source_payload: object,
    *,
    convergence_tolerance_db: float = 0.5,
) -> dict:
    """Assess fine-mesh matching and coarse/fine numerical agreement."""
    objective = {}
    simulation_metadata = None
    if isinstance(source_payload, dict):
        simulation = source_payload.get("simulation", {})
        if isinstance(simulation, dict):
            simulation_metadata = simulation
            candidate = simulation.get("objective", {})
            if isinstance(candidate, dict):
                objective = candidate
    s11_limit_db = float(objective.get("maximum_s11_db", -10.0))
    fine = verification["fine"]
    worst_s11_db = float(fine["worst_s11_db"])
    checks: dict[str, dict] = {
        "fine_worst_s11": {
            "passed": worst_s11_db <= s11_limit_db,
            "observed_db": worst_s11_db,
            "maximum_db": s11_limit_db,
        }
    }
    if simulation_metadata is not None:
        preflight_status = simulation_metadata.get("convergence_status")
        if preflight_status is not None:
            checks["open_region_preflight"] = {
                "passed": preflight_status == "passed",
                "status": preflight_status,
                "warning": simulation_metadata.get("convergence_warning"),
            }
    convergence = verification.get("convergence")
    if isinstance(convergence, dict) and convergence:
        largest_delta = max(abs(float(value)) for value in convergence.values())
        checks["coarse_fine_agreement"] = {
            "passed": largest_delta <= convergence_tolerance_db,
            "largest_absolute_delta_db": largest_delta,
            "maximum_absolute_delta_db": convergence_tolerance_db,
        }

    observations = {}
    if "target_beamwidth_deg" in fine:
        target = float(fine["target_beamwidth_deg"])
        if "ring_beamwidths_deg" in fine:
            widths = [float(value) for value in fine["ring_beamwidths_deg"]]
            observations["beamwidth"] = {
                "target_deg": target,
                "rings_deg": widths,
                "errors_deg": [width-target for width in widths],
                "rms_error_deg": float(
                    np.sqrt(np.mean((np.asarray(widths)-target)**2))
                ),
            }
        elif "ring_beamwidth_deg" in fine:
            observations["beamwidth"] = {
                "target_deg": target,
                "ring_deg": float(fine["ring_beamwidth_deg"]),
                "error_deg": float(fine["ring_beamwidth_deg"])-target,
            }
        elif "elevation_beamwidth_deg" in fine:
            observations["beamwidth"] = {
                "target_deg": target,
                "elevation_deg": float(fine["elevation_beamwidth_deg"]),
                "azimuth_deg": float(fine["azimuth_beamwidth_deg"]),
            }
    passed = all(check["passed"] for check in checks.values())
    return {
        "status": "passed" if passed else "warning",
        "checks": checks,
        "observations": observations,
        "note": (
            "Beamwidth is a soft optimization goal and is reported, not used "
            "as a pass/fail threshold."
        ),
    }


def result_summary(
    result: SimulationResult,
    frequency_hz: float,
    pattern_target: dict | None = None,
) -> dict:
    pattern = result.farfield_metrics
    if pattern is None:
        raise RuntimeError("verification far field was not computed")
    summary = {
        "s11_at_target_db": result.s11_db_at(frequency_hz),
        "worst_s11_db": float(np.max(result.s11_db)),
        "best_s11_db": float(np.min(result.s11_db)),
        "peak_gain_dbi": pattern.peak_gain_dbi,
        "peak_theta_deg": pattern.peak_theta_deg,
        "peak_phi_deg": pattern.peak_phi_deg,
        "peak_elevation_deg": pattern.peak_elevation_deg,
        "horizon_min_gain_dbi": pattern.horizon_min_gain_dbi,
        "horizon_p10_gain_dbi": pattern.horizon_p10_gain_dbi,
        "horizon_mean_gain_dbi": pattern.horizon_mean_gain_dbi,
        "horizon_peak_gain_dbi": pattern.horizon_peak_gain_dbi,
        "horizon_ripple_p90_p10_db": pattern.horizon_ripple_p90_p10_db,
        "horizon_peak_to_null_db": pattern.horizon_peak_to_null_db,
        "antenna_height_m": result.antenna_height,
        "mesh_nodes": result.artifacts.mesh_nodes,
        "mesh_elements": result.artifacts.mesh_elements,
        "volume_elements": result.artifacts.volume_elements,
        "open_region": asdict(result.options.open_region),
        "mesh_settings": asdict(result.options.mesh),
        "huygens_face_count": len(result.artifacts.farfield_selection.tags),
        "termination_face_count": (
            len(result.artifacts.termination_selection.tags)
            if result.artifacts.termination_selection is not None
            else len(result.artifacts.outer_boundary_tags)
        ),
        "frequencies_hz": result.frequencies.tolist(),
        "s11_db": result.s11_db.tolist(),
    }
    if pattern_target is not None:
        theta_degrees = tuple(
            pattern_target.get(
                "theta_degrees",
                (pattern_target["theta_deg"],),
            )
        )
        theta_deg = float(theta_degrees[0])
        summary["target_theta_deg"] = theta_deg
        summary["target_theta_degrees"] = list(theta_degrees)
        if pattern_target["mode"] == "ring":
            rings = [
                result.azimuth_ring_metrics(float(target_theta))
                for target_theta in theta_degrees
            ]
            worst_index = int(
                np.argmin([ring.p10_gain_dbi for ring in rings])
            )
            ring = rings[worst_index]
            ring_summaries = [
                {
                    "target_theta_deg": float(target_theta),
                    "sampled_theta_deg": candidate.sampled_theta_deg,
                    "min_gain_dbi": candidate.min_gain_dbi,
                    "p10_gain_dbi": candidate.p10_gain_dbi,
                    "mean_gain_dbi": candidate.mean_gain_dbi,
                    "p90_gain_dbi": candidate.p90_gain_dbi,
                    "peak_gain_dbi": candidate.peak_gain_dbi,
                    "ripple_p90_p10_db": candidate.ripple_p90_p10_db,
                    "peak_to_null_db": candidate.peak_to_null_db,
                }
                for target_theta, candidate in zip(
                    theta_degrees,
                    rings,
                    strict=True,
                )
            ]
            summary.update(
                target_theta_deg=float(theta_degrees[worst_index]),
                rings=ring_summaries,
                ring_target_count=len(rings),
                ring_worst_target_theta_deg=float(theta_degrees[worst_index]),
                ring_sampled_theta_deg=ring.sampled_theta_deg,
                ring_min_gain_dbi=ring.min_gain_dbi,
                ring_p10_gain_dbi=ring.p10_gain_dbi,
                ring_mean_gain_dbi=ring.mean_gain_dbi,
                ring_p90_gain_dbi=ring.p90_gain_dbi,
                ring_peak_gain_dbi=ring.peak_gain_dbi,
                ring_ripple_p90_p10_db=ring.ripple_p90_p10_db,
                ring_peak_to_null_db=ring.peak_to_null_db,
                ring_minimum_across_targets_dbi=min(
                    candidate.min_gain_dbi for candidate in rings
                ),
                ring_maximum_ripple_across_targets_db=max(
                    candidate.ripple_p90_p10_db for candidate in rings
                ),
            )
            if pattern_target["beamwidth_deg"] is not None:
                widths = [
                    result.azimuth_ring_beamwidth_deg(float(target_theta))
                    for target_theta in theta_degrees
                ]
                for ring_summary, width in zip(
                    ring_summaries,
                    widths,
                    strict=True,
                ):
                    ring_summary["beamwidth_deg"] = width
                summary.update(
                    target_beamwidth_deg=pattern_target["beamwidth_deg"],
                    ring_beamwidth_deg=widths[worst_index],
                    ring_beamwidths_deg=widths,
                    ring_beamwidth_rms_error_deg=float(
                        np.sqrt(
                            np.mean(
                                (
                                    np.asarray(widths)
                                    - pattern_target["beamwidth_deg"]
                                )**2
                            )
                        )
                    ),
                )
        else:
            phi_deg = pattern_target["phi_deg"]
            summary.update(
                target_phi_deg=phi_deg,
                target_gain_dbi=result.gain_db_at(theta_deg, phi_deg),
            )
        if (
            pattern_target["mode"] == "directional"
            and pattern_target["beamwidth_deg"] is not None
        ):
            elevation_width, azimuth_width = result.directional_beamwidths_deg(
                theta_deg,
                phi_deg,
            )
            summary.update(
                target_beamwidth_deg=pattern_target["beamwidth_deg"],
                elevation_beamwidth_deg=elevation_width,
                azimuth_beamwidth_deg=azimuth_width,
            )
    return summary


def print_summary(name: str, summary: dict, frequency_hz: float) -> None:
    print(f"\n{name.upper()} VERIFICATION")
    print(
        f"S11 at {frequency_hz/1e6:g} MHz: "
        f"{summary['s11_at_target_db']:.3f} dB"
    )
    print(f"Worst sweep S11 : {summary['worst_s11_db']:.3f} dB")
    print(f"Peak gain       : {summary['peak_gain_dbi']:.3f} dBi")
    print(
        f"Peak direction  : theta {summary['peak_theta_deg']:.2f} deg, "
        f"phi {summary['peak_phi_deg']:.2f} deg, "
        f"elevation {summary['peak_elevation_deg']:.2f} deg"
    )
    print(f"Horizon P10     : {summary['horizon_p10_gain_dbi']:.3f} dBi")
    print(f"Horizon minimum : {summary['horizon_min_gain_dbi']:.3f} dBi")
    print(f"Horizon mean    : {summary['horizon_mean_gain_dbi']:.3f} dBi")
    print(f"Horizon ripple  : {summary['horizon_ripple_p90_p10_db']:.3f} dB")
    region = summary["open_region"]
    print(
        f"Open region     : {region['mode'].upper()}, "
        f"{summary['huygens_face_count']} closed Huygens faces, "
        f"{summary['termination_face_count']} termination faces"
    )
    if "target_gain_dbi" in summary:
        print(
            f"Target gain     : {summary['target_gain_dbi']:.3f} dBi at "
            f"theta {summary['target_theta_deg']:.2f} deg, "
            f"phi {summary['target_phi_deg']:.2f} deg"
        )
    if "ring_p10_gain_dbi" in summary:
        rings = summary.get("rings", [])
        if len(rings) > 1:
            for ring in rings:
                print(
                    f"Ring {ring['target_theta_deg']:g} deg    : all phi; "
                    f"P10 {ring['p10_gain_dbi']:.3f} dBi, "
                    f"minimum {ring['min_gain_dbi']:.3f} dBi, "
                    f"ripple {ring['ripple_p90_p10_db']:.3f} dB"
                )
            print(
                f"Worst ring P10  : {summary['ring_p10_gain_dbi']:.3f} dBi "
                f"at theta {summary['ring_worst_target_theta_deg']:g} deg"
            )
        else:
            print(
                f"Ring target     : theta {summary['target_theta_deg']:.2f} deg, "
                f"all phi; P10 {summary['ring_p10_gain_dbi']:.3f} dBi, "
                f"minimum {summary['ring_min_gain_dbi']:.3f} dBi, "
                f"ripple {summary['ring_ripple_p90_p10_db']:.3f} dB"
            )
    if "target_beamwidth_deg" in summary:
        if "ring_beamwidth_deg" in summary:
            if len(summary.get("ring_beamwidths_deg", [])) > 1:
                widths = "/".join(
                    f"{float(width):.2f}"
                    for width in summary["ring_beamwidths_deg"]
                )
                print(
                    f"Ring HPBWs      : target "
                    f"{summary['target_beamwidth_deg']:.2f} deg each; measured "
                    f"{widths} deg"
                )
            else:
                print(
                    f"Ring HPBW       : target "
                    f"{summary['target_beamwidth_deg']:.2f} deg; measured "
                    f"{summary['ring_beamwidth_deg']:.2f} deg"
                )
        else:
            print(
                f"Target HPBW     : {summary['target_beamwidth_deg']:.2f} deg; "
                f"measured elevation "
                f"{summary['elevation_beamwidth_deg']:.2f} deg, "
                f"azimuth {summary['azimuth_beamwidth_deg']:.2f} deg"
            )
    print(f"Antenna height  : {summary['antenna_height_m']*1e3:.2f} mm")
    print(
        f"Mesh            : {summary['mesh_nodes']} nodes, "
        f"{summary['volume_elements']} volume elements"
    )


def save_plots(
    result: SimulationResult,
    output: Path,
    frequency_hz: float,
    pattern_target: dict | None = None,
    maximum_s11_db: float = -10.0,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frequency_mhz = result.frequencies/1e6
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frequency_mhz, result.s11_db, marker="o")
    axis.axhline(
        maximum_s11_db,
        color="tab:red",
        linestyle="--",
        label=f"{maximum_s11_db:g} dB limit",
    )
    axis.axvline(frequency_hz/1e6, color="0.5", linestyle=":")
    axis.set(xlabel="Frequency (MHz)", ylabel="S11 (dB)", title="Verified S11")
    axis.grid(True, alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output/"s11_verified.png", dpi=180)
    plt.close(fig)

    farfield = result.farfield_3d
    theta = np.asarray(farfield.theta)
    phi = np.asarray(farfield.phi)
    values = gain_db(farfield)
    horizon_index = int(np.argmin(np.abs(theta[0, :] - np.pi/2)))
    horizon_phi = phi[:-1, horizon_index]
    horizon_gain = values[:-1, horizon_index]
    polar = plt.figure(figsize=(7, 7))
    axis = polar.add_subplot(111, projection="polar")
    axis.plot(horizon_phi, horizon_gain)
    axis.set_title("Verified XY/horizon realized gain (dBi)")
    axis.grid(True, alpha=0.35)
    polar.tight_layout()
    polar.savefig(output/"horizon_gain.png", dpi=180)
    plt.close(polar)

    if pattern_target is not None and pattern_target["mode"] == "ring":
        polar = plt.figure(figsize=(7, 7))
        axis = polar.add_subplot(111, projection="polar")
        sampled_thetas = []
        for target_theta_deg in pattern_target.get(
            "theta_degrees",
            (pattern_target["theta_deg"],),
        ):
            target_theta = np.deg2rad(target_theta_deg)
            target_index = int(np.argmin(np.abs(theta[0, :] - target_theta)))
            sampled_theta = float(np.rad2deg(theta[0, target_index]))
            sampled_thetas.append(sampled_theta)
            axis.plot(
                phi[:-1, target_index],
                values[:-1, target_index],
                label=f"theta {target_theta_deg:g} deg",
            )
        axis.set_title("Verified target-ring realized gain (dBi)")
        if len(sampled_thetas) > 1:
            axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
        axis.grid(True, alpha=0.35)
        polar.tight_layout()
        polar.savefig(output/"target_ring_gain.png", dpi=180)
        plt.close(polar)

    field = result.raw_data.field.find(freq=frequency_hz)
    faces = result.artifacts.farfield_selection
    origin = result.artifacts.farfield_origin
    points = max(361, round(360/result.options.farfield_angular_step_deg) + 1)
    cuts = {
        "X-Z": field.farfield_2d(
            (0, 0, 1),
            (0, 1, 0),
            faces,
            (-180, 180),
            Npoints=points,
            origin=origin,
        ),
        "Y-Z": field.farfield_2d(
            (0, 0, 1),
            (-1, 0, 0),
            faces,
            (-180, 180),
            Npoints=points,
            origin=origin,
        ),
        "X-Y": field.farfield_2d(
            (1, 0, 0),
            (0, 0, 1),
            faces,
            (-180, 180),
            Npoints=points,
            origin=origin,
        ),
    }
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for label, cut in cuts.items():
        axis.plot(np.rad2deg(cut.ang), gain_db(cut), label=label)
    axis.set(
        xlabel="Cut angle (degrees)",
        ylabel="Realized gain (dBi)",
        title=f"Verified principal-plane gain at {frequency_hz/1e6:g} MHz",
    )
    axis.grid(True, alpha=0.35)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output/"principal_plane_gain.png", dpi=180)
    plt.close(fig)


def show_3d(result: SimulationResult) -> None:
    model = result.artifacts.model
    model.display.add_object(result.artifacts.antenna, opacity=0.85)
    model.display.add_object(result.artifacts.ground_system, opacity=0.85)
    model.display.add_farfield3d(
        result.farfield_3d,
        component="normE",
        quantity="abs",
        dB=True,
        dBfloor=-30,
        rmax=0.45*result.antenna_height,
        offset=(0.0, 0.0, result.design.port_height + result.antenna_height/2),
        opacity=0.7,
    )
    model.display.show()


def export_fabrication_artifacts(
    design: AntennaDesign,
    result: SimulationResult,
    output: Path,
    frequency_hz: float,
    *,
    design_sheet: bool,
    jig_models: bool,
    target_ring_thetas_deg: tuple[float, ...] = (),
) -> dict:
    """Generate requested fabrication files from the fine verification result."""
    artifacts: dict[str, object] = {}
    if design_sheet:
        sheet = export_drawing(
            design,
            output / "design_sheet.pdf",
            result=result,
            title=f"{frequency_hz / 1e6:g} MHz Verified Antenna",
            target_ring_thetas_deg=target_ring_thetas_deg,
        )
        artifacts["design_sheet"] = sheet.name
        print(f"Design sheet       : {sheet.resolve()}")
    if jig_models:
        if design.coil_count:
            if design.has_circular_groundplane:
                models = export_coil_formers(
                    design,
                    output / "coil_formers.step",
                    include_radial_gauge=False,
                )
            else:
                models = export_coil_formers(
                    design,
                    output / "coil_formers.step",
                )
            artifacts["jig_models"] = [models.name]
            print(f"Forming tools      : {models.resolve()}")
        else:
            artifacts["jig_models"] = []
            print("Forming tools      : no loading coils")
    return artifacts


def options(
    mesh: MeshSettings,
    angular_step: float,
    points: int,
    solver: str,
    frequency_hz: float,
    sweep_bandwidth_hz: float,
    *,
    show_model: bool = False,
    show_mesh: bool = False,
) -> SimulationOptions:
    return SimulationOptions(
        sweep=FrequencySweep(
            center=frequency_hz,
            span=sweep_bandwidth_hz,
            points=points,
        ),
        mesh=mesh,
        solve=True,
        solver=solver,
        compute_farfield=True,
        farfield_frequency=frequency_hz,
        farfield_angular_step_deg=angular_step,
        show_geometry=show_model,
        show_mesh=show_mesh,
        verbose=True,
    )


def latest_optimizer_result(
    root: Path = Path("optimization_results"),
) -> Path:
    """Return the most recently modified campaign-wide optimizer winner."""
    if not root.is_dir():
        raise FileNotFoundError(
            f"optimizer result directory does not exist: {root.resolve()}"
        )
    candidates = tuple(
        path
        for path in root.rglob("campaign_best.json")
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no campaign_best.json files found under {root.resolve()}"
        )
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, str(path.resolve())),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        help=(
            "design or optimizer result; defaults to a generated reference"
        ),
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help=(
            "use the most recently modified campaign_best.json anywhere "
            "under optimization_results"
        ),
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        help="target frequency; inferred from optimizer output when possible",
    )
    parser.add_argument(
        "--sweep-bandwidth-mhz",
        type=float,
        help="verification sweep span; defaults to wavelength-scaled 30 MHz",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--angular-step", type=float, default=0.5)
    parser.add_argument("--frequency-points", type=int, default=13)
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help="linear solver backend (default: %(default)s)",
    )
    parser.add_argument("--skip-coarse", action="store_true")
    parser.add_argument(
        "--show-model",
        action="store_true",
        help="open the final verification geometry in the Gmsh viewer",
    )
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="open the final verification surface mesh in the Gmsh viewer",
    )
    parser.add_argument("--show-3d", action="store_true")
    parser.add_argument(
        "--design-sheet",
        action="store_true",
        help=(
            "export design_sheet.pdf with dimensions, S11, gain cuts, and "
            "configured target rings"
        ),
    )
    parser.add_argument(
        "--jig-models",
        action="store_true",
        help=(
            "export coil formers and sizing mandrels as STEP, plus a radial "
            "gauge when the design uses radials"
        ),
    )
    args = parser.parse_args()
    if args.latest and args.result is not None:
        parser.error("provide either an explicit result path or --latest, not both")
    if args.latest:
        try:
            args.result = latest_optimizer_result()
        except FileNotFoundError as error:
            parser.error(str(error))
    if args.frequency_mhz is not None and (
        not np.isfinite(args.frequency_mhz) or args.frequency_mhz <= 0
    ):
        parser.error("--frequency-mhz must be finite and positive")
    if args.sweep_bandwidth_mhz is not None and (
        not np.isfinite(args.sweep_bandwidth_mhz)
        or args.sweep_bandwidth_mhz <= 0
    ):
        parser.error("--sweep-bandwidth-mhz must be finite and positive")
    if args.frequency_points < 3 or args.frequency_points % 2 == 0:
        parser.error("--frequency-points must be an odd integer of at least 3")
    if args.output is None:
        if args.result is None:
            args.output = Path("optimization_results/reference_design_verification")
        else:
            args.output = args.result.parent/(args.result.stem + "_verification")
    return args


def main() -> None:
    args = parse_args()
    if args.latest:
        print(f"Latest result     : {args.result.resolve()}")
    source_frequency = None
    pattern_target = None
    source_payload = None
    if args.result is not None:
        source_payload = json.loads(args.result.read_text(encoding="utf-8"))
        if isinstance(source_payload, dict):
            source_frequency = source_payload.get("frequency_hz")
            pattern_target = pattern_target_from_payload(source_payload)
    if args.frequency_mhz is not None:
        frequency_hz = args.frequency_mhz*1e6
    elif source_frequency is not None:
        frequency_hz = float(source_frequency)
    else:
        frequency_hz = REFERENCE_DESIGN_FREQUENCY_HZ
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise SystemExit("Design metadata contains an invalid frequency_hz")
    sweep_bandwidth_hz = (
        args.sweep_bandwidth_mhz*1e6
        if args.sweep_bandwidth_mhz is not None
        else 30e6*frequency_hz/REFERENCE_DESIGN_FREQUENCY_HZ
    )
    if args.result is None:
        design = load_reference_design(frequency_hz)
        source = "generated wavelength-scaled reference"
    else:
        design = load_design(args.result)
        source = str(args.result.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "frequency_hz": frequency_hz,
        "design": asdict(design),
    }
    coarse_result = None

    if not args.skip_coarse:
        coarse_result = simulate(
            design,
            options(
                MeshSettings(),
                2.0,
                args.frequency_points,
                args.solver,
                frequency_hz,
                sweep_bandwidth_hz,
            ),
        )
        payload["coarse"] = result_summary(
            coarse_result,
            frequency_hz,
            pattern_target,
        )
        print_summary("coarse", payload["coarse"], frequency_hz)

    fine_mesh = MeshSettings(
        wire_sections=8,
        antenna_size_factor=2.0,
        radial_size_factor=6.0,
        feed_size_factor=2.0,
        curved_boundary_segments=20,
        wavelength_resolution=0.33,
        air_margin_wavelengths=0.30,
        preview_points_per_turn=30,
    )
    fine_result = simulate(
        design,
        options(
            fine_mesh,
            args.angular_step,
            args.frequency_points,
            args.solver,
            frequency_hz,
            sweep_bandwidth_hz,
            show_model=args.show_model,
            show_mesh=args.show_mesh,
        ),
    )
    payload["fine"] = result_summary(
        fine_result,
        frequency_hz,
        pattern_target,
    )
    print_summary("fine", payload["fine"], frequency_hz)

    if coarse_result is not None:
        coarse = payload["coarse"]
        fine = payload["fine"]
        convergence = {
            "peak_gain_delta_db": fine["peak_gain_dbi"] - coarse["peak_gain_dbi"],
            "horizon_p10_delta_db": (
                fine["horizon_p10_gain_dbi"] - coarse["horizon_p10_gain_dbi"]
            ),
            "s11_target_delta_db": (
                fine["s11_at_target_db"] - coarse["s11_at_target_db"]
            ),
        }
        if "target_gain_dbi" in fine:
            convergence["target_gain_delta_db"] = (
                fine["target_gain_dbi"] - coarse["target_gain_dbi"]
            )
        if "ring_p10_gain_dbi" in fine:
            convergence["ring_p10_delta_db"] = (
                fine["ring_p10_gain_dbi"] - coarse["ring_p10_gain_dbi"]
            )
        payload["convergence"] = convergence
        print("\nMESH CONVERGENCE (fine - coarse)")
        for name, value in convergence.items():
            print(f"{name:24s}: {value:+.3f} dB")
        if max(abs(value) for value in convergence.values()) > 0.5:
            print("WARNING: result changes by more than 0.5 dB; refine again.")

    payload["quality"] = verification_quality(payload, source_payload)
    quality_status = payload["quality"]["status"]
    print(f"\nVERIFICATION QUALITY: {quality_status.upper()}")
    for name, check in payload["quality"]["checks"].items():
        print(f"  {name}: {'PASS' if check['passed'] else 'WARNING'}")

    report_path = args.output/"verification.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_plots(
        fine_result,
        args.output,
        frequency_hz,
        pattern_target,
        maximum_s11_db=float(
            payload["quality"]["checks"]["fine_worst_s11"]["maximum_db"]
        ),
    )
    artifacts = export_fabrication_artifacts(
        design,
        fine_result,
        args.output,
        frequency_hz,
        design_sheet=args.design_sheet,
        jig_models=args.jig_models,
        target_ring_thetas_deg=(
            tuple(pattern_target["theta_degrees"])
            if pattern_target is not None
            and pattern_target["mode"] == "ring"
            else ()
        ),
    )
    if artifacts:
        payload["artifacts"] = artifacts
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nVerification report: {report_path.resolve()}")
    print(f"Plots              : {args.output.resolve()}")
    if args.show_3d:
        show_3d(fine_result)


if __name__ == "__main__":
    main()
