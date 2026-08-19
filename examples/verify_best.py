"""Fine-mesh, fine-angle verification of an optimizer result."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault("EMERGE_STD_LOGLEVEL", "WARNING")

import emerge as em
import matplotlib.pyplot as plt
import numpy as np

from emerge_loaded_antenna import (
    FrequencySweep,
    MeshSettings,
    SOLVER_CHOICES,
    SimulationOptions,
    SimulationResult,
    load_design,
    simulate,
)

FREQUENCY = 868e6


def gain_db(farfield) -> np.ndarray:
    amplitude = np.abs(np.asarray(farfield.normE)/em.lib.EISO)
    return 20*np.log10(np.maximum(amplitude, 1e-12))


def result_summary(result: SimulationResult) -> dict:
    pattern = result.farfield_metrics
    if pattern is None:
        raise RuntimeError("verification far field was not computed")
    return {
        "s11_at_868_db": result.s11_db_at(FREQUENCY),
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
        "frequencies_hz": result.frequencies.tolist(),
        "s11_db": result.s11_db.tolist(),
    }


def print_summary(name: str, summary: dict) -> None:
    print(f"\n{name.upper()} VERIFICATION")
    print(f"S11 at 868 MHz  : {summary['s11_at_868_db']:.3f} dB")
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
    print(f"Antenna height  : {summary['antenna_height_m']*1e3:.2f} mm")
    print(
        f"Mesh            : {summary['mesh_nodes']} nodes, "
        f"{summary['volume_elements']} volume elements"
    )


def save_plots(result: SimulationResult, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frequency_mhz = result.frequencies/1e6
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(frequency_mhz, result.s11_db, marker="o")
    axis.axhline(-10.0, color="tab:red", linestyle="--", label="-10 dB limit")
    axis.axvline(868.0, color="0.5", linestyle=":")
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

    field = result.raw_data.field.find(freq=FREQUENCY)
    faces = result.artifacts.absorbing_selection
    points = max(361, int(round(360/result.options.farfield_angular_step_deg)) + 1)
    cuts = {
        "X-Z": field.farfield_2d(
            (0, 0, 1), (0, 1, 0), faces, (-180, 180), Npoints=points
        ),
        "Y-Z": field.farfield_2d(
            (0, 0, 1), (-1, 0, 0), faces, (-180, 180), Npoints=points
        ),
        "X-Y": field.farfield_2d(
            (1, 0, 0), (0, 0, 1), faces, (-180, 180), Npoints=points
        ),
    }
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for label, cut in cuts.items():
        axis.plot(np.rad2deg(cut.ang), gain_db(cut), label=label)
    axis.set(
        xlabel="Cut angle (degrees)",
        ylabel="Realized gain (dBi)",
        title="Verified principal-plane gain at 868 MHz",
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


def options(
    mesh: MeshSettings,
    angular_step: float,
    points: int,
    solver: str,
) -> SimulationOptions:
    return SimulationOptions(
        sweep=FrequencySweep(center=FREQUENCY, span=30e6, points=points),
        mesh=mesh,
        solve=True,
        solver=solver,
        compute_farfield=True,
        farfield_frequency=FREQUENCY,
        farfield_angular_step_deg=angular_step,
        verbose=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=Path("optimization_results/best_result.json"),
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
    parser.add_argument("--show-3d", action="store_true")
    args = parser.parse_args()
    if args.frequency_points < 3 or args.frequency_points % 2 == 0:
        parser.error("--frequency-points must be an odd integer of at least 3")
    if args.output is None:
        args.output = args.result.parent/(args.result.stem + "_verification")
    return args


def main() -> None:
    args = parse_args()
    design = load_design(args.result)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"source": str(args.result.resolve()), "design": asdict(design)}
    coarse_result = None

    if not args.skip_coarse:
        coarse_result = simulate(
            design,
            options(MeshSettings(), 2.0, args.frequency_points, args.solver),
        )
        payload["coarse"] = result_summary(coarse_result)
        print_summary("coarse", payload["coarse"])

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
        options(fine_mesh, args.angular_step, args.frequency_points, args.solver),
    )
    payload["fine"] = result_summary(fine_result)
    print_summary("fine", payload["fine"])

    if coarse_result is not None:
        coarse = payload["coarse"]
        fine = payload["fine"]
        convergence = {
            "peak_gain_delta_db": fine["peak_gain_dbi"] - coarse["peak_gain_dbi"],
            "horizon_p10_delta_db": (
                fine["horizon_p10_gain_dbi"] - coarse["horizon_p10_gain_dbi"]
            ),
            "s11_868_delta_db": fine["s11_at_868_db"] - coarse["s11_at_868_db"],
        }
        payload["convergence"] = convergence
        print("\nMESH CONVERGENCE (fine - coarse)")
        for name, value in convergence.items():
            print(f"{name:24s}: {value:+.3f} dB")
        if max(abs(value) for value in convergence.values()) > 0.5:
            print("WARNING: result changes by more than 0.5 dB; refine again.")

    report_path = args.output/"verification.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_plots(fine_result, args.output)
    print(f"\nVerification report: {report_path.resolve()}")
    print(f"Plots              : {args.output.resolve()}")
    if args.show_3d:
        show_3d(fine_result)


if __name__ == "__main__":
    main()
