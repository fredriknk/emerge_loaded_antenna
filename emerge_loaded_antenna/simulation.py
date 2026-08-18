"""Reusable EMerge model construction and simulation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

# Prefer the MKL runtime from the active virtual environment before EMerge is
# imported. This avoids accidentally loading an inaccessible base-Conda DLL.
_venv_mkl_dir = Path(__file__).resolve().parents[1]/".venv"/"Library"/"bin"
_venv_mkl = next(iter(sorted(_venv_mkl_dir.glob("mkl_rt*.dll"))), None)
if _venv_mkl is not None:
    os.environ.setdefault("EMERGE_PARDISO_PATH", str(_venv_mkl))

import emerge as em  # noqa: E402
import gmsh  # noqa: E402

from .config import AntennaDesign, SimulationOptions
from .geometry import CompositeCurve, AntennaPath, build_centerline

C0 = 299_792_458.0


@dataclass
class ModelArtifacts:
    """EMerge objects retained for optional inspection and far-field work."""

    model: Any
    path: AntennaPath
    antenna: Any
    ground_system: Any
    feed: Any
    feed_polygon: Any
    airbox: Any
    absorbing_selection: Any
    port: Any
    mesh_nodes: int
    mesh_elements: int
    volume_elements: int


@dataclass
class SimulationResult:
    """Numerical result returned to scripts and optimizers."""

    design: AntennaDesign
    options: SimulationOptions
    artifacts: ModelArtifacts
    frequencies: np.ndarray
    s11: np.ndarray
    s11_db: np.ndarray
    peak_gain_dbi: float | None = None
    farfield_3d: Any | None = None
    raw_data: Any | None = None

    @property
    def solved(self) -> bool:
        return bool(self.frequencies.size)

    def nearest_index(self, frequency: float) -> int:
        if not self.solved:
            raise RuntimeError("simulation was built without solving")
        return int(np.argmin(np.abs(self.frequencies - frequency)))

    def s11_db_at(self, frequency: float) -> float:
        return float(self.s11_db[self.nearest_index(frequency)])

    def as_dict(self) -> dict[str, Any]:
        return {
            "frequencies_hz": self.frequencies.tolist(),
            "s11_real": self.s11.real.tolist(),
            "s11_imag": self.s11.imag.tolist(),
            "s11_db": self.s11_db.tolist(),
            "peak_gain_dbi": self.peak_gain_dbi,
            "mesh_nodes": self.artifacts.mesh_nodes,
            "mesh_elements": self.artifacts.mesh_elements,
            "volume_elements": self.artifacts.volume_elements,
        }


def _print_design(design: AntennaDesign, path: AntennaPath) -> None:
    xpts, _, zpts = path.arrays()
    antenna_height = zpts[-1] - design.port_height
    print()
    print("----------------------------------------------------")
    print("ANTENNA")
    print("----------------------------------------------------")
    print(f"CAD path segments  : {len(path.segments)}")
    print(f"Preview samples    : {len(xpts)}")
    print(f"Wire diameter      : {2*design.wire_radius*1e3:.2f} mm")
    print(f"Antenna height     : {antenna_height*1e3:.2f} mm")
    print(f"Overall height     : {zpts[-1]*1e3:.2f} mm")
    print(
        f"Radials            : {design.radial_count} x "
        f"{design.radial_length*1e3:.1f} mm"
    )
    print(
        f"Radial angle       : {design.radial_angle_deg:.1f} deg "
        "below horizontal"
    )
    print("----------------------------------------------------")


def build_model(
    design: AntennaDesign | None = None,
    options: SimulationOptions | None = None,
) -> ModelArtifacts:
    """Build and mesh one antenna model without running the solver."""
    design = design or AntennaDesign()
    options = options or SimulationOptions(solve=False)
    design.validate()
    options.validate()
    mesh = options.mesh
    path = build_centerline(design, mesh)
    xpts, ypts, zpts = path.arrays()
    if options.verbose:
        _print_design(design, path)

    model = em.Simulation(options.model_name)
    model.check_version("2.8.4")

    antenna_curve = CompositeCurve(path.segments, name="AntennaCenterline")
    wire_section = em.geo.XYPolygon.circle(
        design.wire_radius,
        Nsections=mesh.wire_sections,
    )
    antenna_size = mesh.antenna_size_factor*design.wire_radius
    antenna = (
        antenna_curve.pipe(
            wire_section,
            max_mesh_size=antenna_size,
            name="Antenna",
        )
        .set_material(em.lib.MET_COPPER)
        .foreground()
    )
    antenna.max_meshsize = antenna_size

    if options.show_coil_preview:
        model.view(use_gmsh=True)

    feed_polygon = em.geo.XYPolygon.circle(
        design.wire_radius,
        Nsections=mesh.wire_sections,
    )
    feed = (
        feed_polygon.extrude(
            design.port_height,
            em.GCS.displace(xpts[0], ypts[0], 0.0),
            name="Feed",
        ).foreground()
    )
    feed.max_meshsize = mesh.feed_size_factor*design.wire_radius

    hub_radius = 4*design.wire_radius
    hub_height = 3*design.wire_radius
    ground_hub = em.geo.Cylinder(
        hub_radius,
        hub_height,
        cs=em.GCS.displace(0.0, 0.0, -hub_height),
        Nsections=max(24, mesh.wire_sections),
        name="GroundHub",
    )
    ground_hub.max_meshsize = mesh.feed_size_factor*design.wire_radius

    radial_start = 3*design.wire_radius
    if design.radial_length <= radial_start:
        raise ValueError("radial_length must exceed three wire radii")
    radial_tilt = 90.0 + design.radial_angle_deg
    radials = []
    for index in range(design.radial_count):
        radial = em.geo.Cylinder(
            design.wire_radius,
            design.radial_length - radial_start,
            cs=em.GCS.displace(0.0, 0.0, radial_start),
            Nsections=mesh.wire_sections,
            name=f"Radial{index + 1}",
        )
        radial = em.geo.rotate(
            radial,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            radial_tilt,
        )
        radial = em.geo.rotate(
            radial,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            index*360.0/design.radial_count,
        )
        radial = radial.set_material(em.lib.MET_COPPER).foreground()
        radial.max_meshsize = mesh.radial_size_factor*design.wire_radius
        radials.append(radial)

    ground_system = (
        em.geo.unite(ground_hub, *radials)
        .set_material(em.lib.MET_COPPER)
        .foreground()
    )
    ground_system.max_meshsize = mesh.radial_size_factor*design.wire_radius

    wavelength = C0/options.sweep.center
    air_margin = mesh.air_margin_wavelengths*wavelength
    radial_horizontal = design.radial_length*np.cos(
        np.deg2rad(design.radial_angle_deg)
    )
    xmin = min(float(np.min(xpts)), -radial_horizontal) - air_margin
    xmax = max(float(np.max(xpts)), radial_horizontal) + air_margin
    ymin = min(float(np.min(ypts)), -radial_horizontal) - air_margin
    ymax = max(float(np.max(ypts)), radial_horizontal) + air_margin
    radial_lowest_z = -design.radial_length*np.sin(
        np.deg2rad(design.radial_angle_deg)
    )
    zmin = radial_lowest_z - air_margin
    zmax = float(np.max(zpts)) + air_margin
    airbox = em.geo.Box(
        width=xmax - xmin,
        depth=ymax - ymin,
        height=zmax - zmin,
        position=(xmin, ymin, zmin),
        name="Airbox",
    ).background()

    if options.show_geometry:
        model.view(use_gmsh=True)

    model.mw.set_frequency_range(
        options.sweep.start,
        options.sweep.stop,
        options.sweep.points,
    )
    model.set_resolution(mesh.wavelength_resolution)
    model.mesher.set_curved_boundary_meshing(
        mesh.curved_boundary_segments
    )
    if options.verbose:
        print("Generating mesh...")
    model.generate_mesh()

    mesh_nodes = len(gmsh.model.mesh.getNodes()[0])
    mesh_elements = sum(
        len(block) for block in gmsh.model.mesh.getElements()[1]
    )
    volume_elements = sum(
        len(block) for block in gmsh.model.mesh.getElements(3)[1]
    )
    if options.verbose:
        print(f"Mesh nodes          : {mesh_nodes}")
        print(f"Mesh elements       : {mesh_elements}")
        print(f"Volume elements     : {volume_elements}")

    if options.show_mesh:
        model.view(plot_mesh=True, volume_mesh=False)

    absorbing_selection = airbox.boundary(exclude=("bottom",))
    model.mw.bc.AbsorbingBoundary(absorbing_selection)
    port_selection = feed.boundary(exclude=("front", "back"))
    port = model.mw.bc.LumpedPort(
        port_selection,
        1,
        feed_polygon.length,
        design.port_height,
        em.ZAX,
        Z0=design.port_impedance,
    )

    return ModelArtifacts(
        model=model,
        path=path,
        antenna=antenna,
        ground_system=ground_system,
        feed=feed,
        feed_polygon=feed_polygon,
        airbox=airbox,
        absorbing_selection=absorbing_selection,
        port=port,
        mesh_nodes=mesh_nodes,
        mesh_elements=mesh_elements,
        volume_elements=volume_elements,
    )


def simulate(
    design: AntennaDesign | None = None,
    options: SimulationOptions | None = None,
) -> SimulationResult:
    """Build, mesh and optionally solve one antenna design."""
    design = design or AntennaDesign()
    options = options or SimulationOptions()
    artifacts = build_model(design, options)
    empty_real = np.array([], dtype=float)
    empty_complex = np.array([], dtype=complex)
    if not options.solve:
        return SimulationResult(
            design=design,
            options=options,
            artifacts=artifacts,
            frequencies=empty_real,
            s11=empty_complex,
            s11_db=empty_real,
        )

    if options.verbose:
        print(
            f"Running {options.sweep.start/1e6:.1f}-"
            f"{options.sweep.stop/1e6:.1f} MHz sweep..."
        )
    data = artifacts.model.mw.run_sweep()
    scalar = data.scalar.grid
    frequencies = np.asarray(scalar.freq, dtype=float)
    s11 = np.asarray(scalar.S(1, 1), dtype=complex)
    s11_db = 20*np.log10(np.maximum(np.abs(s11), 1e-12))

    peak_gain_dbi = None
    farfield_3d = None
    if options.compute_farfield:
        farfield_frequency = (
            options.farfield_frequency
            if options.farfield_frequency is not None
            else options.sweep.center
        )
        field = data.field.find(freq=farfield_frequency)
        farfield_3d = field.farfield_3d(artifacts.absorbing_selection)
        gain_amplitude = np.abs(np.asarray(farfield_3d.normE)/em.lib.EISO)
        gain_db = 20*np.log10(np.maximum(gain_amplitude, 1e-12))
        peak_gain_dbi = float(np.nanmax(gain_db))

    return SimulationResult(
        design=design,
        options=options,
        artifacts=artifacts,
        frequencies=frequencies,
        s11=s11,
        s11_db=s11_db,
        peak_gain_dbi=peak_gain_dbi,
        farfield_3d=farfield_3d,
        raw_data=data,
    )
