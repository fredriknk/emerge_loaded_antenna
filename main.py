"""Interactive example for the :mod:`emerge_loaded_antenna` library.

Optimizers should import the package directly. This file intentionally keeps
plotting and interactive viewers outside the reusable simulation API.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

import emerge as em
from emerge.plot import plot_ff, plot_sp, smith

from emerge_loaded_antenna import (
    FrequencySweep,
    MeshSettings,
    SOLVER_CHOICES,
    SimulationOptions,
    load_reference_design,
    simulate,
)
from emerge_loaded_antenna.drawing import export_drawing
from emerge_loaded_antenna.formers import export_coil_formers

MHz = 1e6
mm = 1e-3
F0 = 869.5*MHz


# Tracked example geometry evaluated at this script's target frequency.
DESIGN = load_reference_design(F0)
PDF_NAME = "my_manual_antenna.pdf"
FORMER_NAME = "coil_formers.step"


trans = 6*mm
DESIGN = replace(
    DESIGN,
    wire_radius=0.8*mm,
    radial_length=100*mm,
    straight_lengths=np.array([96, 75, 112])*mm,
    coils=(replace(DESIGN.coils[0], transition=trans),
           replace(DESIGN.coils[1], transition=trans),
    )
)
print(DESIGN)

RUN_SOLVER = False
SHOW_GEOMETRY = False
SHOW_COIL_PREVIEW = False
SHOW_MESH = False
SHOW_3D_FARFIELD = False
EXPORT_FORMERS = True
FARFIELD_DB_FLOOR = -30.0

OPTIONS = SimulationOptions(
    sweep=FrequencySweep(center=F0, span=50*MHz, points=20),
    mesh=MeshSettings(
        wire_sections=6,
        antenna_size_factor=3.0,
        radial_size_factor=10.0,
        feed_size_factor=3.0,
        curved_boundary_segments = 12,
        wavelength_resolution=0.33,
        air_margin_wavelengths=0.25,
        preview_points_per_turn=20,
    ),
    solve=RUN_SOLVER,
    compute_farfield=RUN_SOLVER,
    farfield_frequency=F0,
    show_geometry=SHOW_GEOMETRY,
    show_mesh=SHOW_MESH,
    show_coil_preview=SHOW_COIL_PREVIEW,
    verbose=True,
    solver= "cudss"
)


def gain_amplitude(farfield):
    return np.abs(np.asarray(farfield.normE)/em.lib.EISO)


def plane_metrics(name, farfield, orientation):
    """Print peak gain, average, HPBW and front/back ratio for one cut."""
    angles = np.asarray(farfield.ang, dtype=float)*180/np.pi
    amplitude = gain_amplitude(farfield)
    gain_db = 20*np.log10(np.maximum(amplitude, 1e-12))
    order = np.argsort(angles)
    angles = angles[order]
    gain_db = gain_db[order]
    amplitude = amplitude[order]

    if len(angles) > 1 and np.isclose(angles[-1] - angles[0], 360.0):
        angles = angles[:-1]
        gain_db = gain_db[:-1]
        amplitude = amplitude[:-1]

    peak_index = int(np.nanargmax(gain_db))
    peak_angle = float(angles[peak_index])
    peak_gain_db = float(gain_db[peak_index])
    average_gain_db = float(
        10*np.log10(np.maximum(np.mean(amplitude**2), 1e-24))
    )
    opposite_angle = ((peak_angle + 360.0) % 360.0) - 180.0
    opposite_index = int(np.argmin(np.abs(angles - opposite_angle)))
    front_to_back_db = peak_gain_db - float(gain_db[opposite_index])

    threshold = peak_gain_db - 3.0
    above = gain_db >= threshold
    if np.all(above):
        beamwidth = 360.0
    else:
        left = peak_index
        right = peak_index
        while above[(left - 1) % len(above)] and (left - 1) % len(above) != right:
            left = (left - 1) % len(above)
        while above[(right + 1) % len(above)] and (right + 1) % len(above) != left:
            right = (right + 1) % len(above)
        step = float(np.median(np.abs(np.diff(angles))))
        beamwidth = ((right - left) % len(above) + 1)*step

    print(
        f"{name:4s}  peak {peak_gain_db:7.2f} dBi at {peak_angle:7.1f} deg  "
        f"avg {average_gain_db:7.2f} dBi  "
        f"HPBW {beamwidth:6.1f} deg  F/B {front_to_back_db:6.2f} dB"
    )
    print(f"      angle reference: {orientation}")


def report_s11(result) -> None:
    print()
    print("S11 RESULTS")
    print("----------------------------------------------------")
    print("Frequency (MHz)   S11 (dB)")
    print("----------------------------------------------------")
    for frequency, value in zip(result.frequencies, result.s11_db):
        print(f"{frequency/MHz:10.3f}       {value:8.3f}")
    best = int(np.nanargmin(result.s11_db))
    target = result.nearest_index(F0)
    print("----------------------------------------------------")
    print(
        f"Nearest to 868   : {result.frequencies[target]/MHz:.3f} MHz, "
        f"{result.s11_db[target]:.3f} dB"
    )
    print(
        f"Best in sweep    : {result.frequencies[best]/MHz:.3f} MHz, "
        f"{result.s11_db[best]:.3f} dB"
    )

    plot_sp(result.frequencies, result.s11)
    smith(result.s11, f=result.frequencies)


def report_farfield(result) -> None:
    field = result.raw_data.field.find(freq=F0)
    selection = result.artifacts.farfield_selection
    origin = result.artifacts.farfield_origin
    ff_xz = field.farfield_2d(
        (0, 0, 1), (0, 1, 0), selection, (-180, 180), origin=origin
    )
    ff_yz = field.farfield_2d(
        (0, 0, 1), (-1, 0, 0), selection, (-180, 180), origin=origin
    )
    ff_xy = field.farfield_2d(
        (1, 0, 0), (0, 0, 1), selection, (-180, 180), origin=origin
    )

    print()
    print(f"FAR-FIELD GAIN @ {F0/MHz:.1f} MHz")
    print("--------------------------------------------------------------------------")
    plane_metrics("X-Z", ff_xz, "0 deg = +Z, +90 deg = +X")
    plane_metrics("Y-Z", ff_yz, "0 deg = +Z, +90 deg = +Y")
    plane_metrics("X-Y", ff_xy, "0 deg = +X, +90 deg = +Y")
    metrics = result.farfield_metrics
    print(f"3D peak isotropic gain: {result.peak_gain_dbi:.2f} dBi")
    if metrics is not None:
        print(
            f"3D peak direction     : theta {metrics.peak_theta_deg:.1f} deg, "
            f"phi {metrics.peak_phi_deg:.1f} deg, "
            f"elevation {metrics.peak_elevation_deg:.1f} deg"
        )
        print(
            f"Horizon gain          : min {metrics.horizon_min_gain_dbi:.2f}, "
            f"P10 {metrics.horizon_p10_gain_dbi:.2f}, "
            f"mean {metrics.horizon_mean_gain_dbi:.2f} dBi"
        )
        print(
            f"Horizon ripple        : "
            f"{metrics.horizon_ripple_p90_p10_db:.2f} dB (P90-P10)"
        )

    plane_peak_db = max(
        float(np.nanmax(20*np.log10(np.maximum(gain_amplitude(ff), 1e-12))))
        for ff in (ff_xz, ff_yz, ff_xy)
    )
    plot_ff(
        ff_xz.ang*180/np.pi,
        [
            ff_xz.normE/em.lib.EISO,
            ff_yz.normE/em.lib.EISO,
            ff_xy.normE/em.lib.EISO,
        ],
        dB=True,
        labels=["Total gain X-Z", "Total gain Y-Z", "Total gain X-Y"],
        xlabel="Cut angle (degrees)",
        ylabel="Isotropic gain (dBi)",
        ylim=(FARFIELD_DB_FLOOR, max(5.0, np.ceil(plane_peak_db + 1))),
        title=f"Principal-plane gain at {F0/MHz:.1f} MHz",
    )

    if SHOW_3D_FARFIELD:
        model = result.artifacts.model
        z = np.asarray(result.artifacts.path.z)
        antenna_height = z[-1] - DESIGN.port_height
        model.display.add_object(result.artifacts.antenna, opacity=0.85)
        model.display.add_object(result.artifacts.ground_system, opacity=0.85)
        model.display.add_farfield3d(
            result.farfield_3d,
            component="normE",
            quantity="abs",
            dB=True,
            dBfloor=FARFIELD_DB_FLOOR,
            rmax=0.45*antenna_height,
            offset=(0.0, 0.0, DESIGN.port_height + antenna_height/2),
            opacity=0.7,
        )
        model.display.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default=OPTIONS.solver,
        help="linear solver backend (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if EXPORT_FORMERS:
        export_coil_formers(
            DESIGN,
            FORMER_NAME,
            extra_length=5 * mm,
            groove_clearance=0.1 * mm,
            spacing=5 * mm,
        )
        
    print(f"Exported coil formers to {FORMER_NAME}")
    
    result = simulate(DESIGN, replace(OPTIONS, solver=args.solver))
    export_drawing(
        DESIGN,
        PDF_NAME,
        result=result if result.solved else None,
        title=f"{F0/MHz:g} MHz Prototype",
    )
    if not result.solved:
        print("Done (mesh only).")
        return
    report_s11(result)
    report_farfield(result)
    print("Done.")
    


if __name__ == "__main__":
    main()
