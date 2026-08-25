"""Reusable EMerge model construction and simulation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import gc
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

from .config import AntennaDesign, OpenRegionSettings, SimulationOptions
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
    open_region_volumes: tuple[Any, ...]
    pml_volumes: tuple[Any, ...]
    farfield_selection: Any
    termination_selection: Any | None
    open_region_mode: str
    farfield_origin: tuple[float, float, float]
    outer_boundary_tags: tuple[int, ...]
    port: Any
    mesh_nodes: int
    mesh_elements: int
    volume_elements: int


@dataclass(frozen=True)
class _OpenRegionBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return (
            self.xmax - self.xmin,
            self.ymax - self.ymin,
            self.zmax - self.zmin,
        )

    @property
    def corner(self) -> tuple[float, float, float]:
        return (self.xmin, self.ymin, self.zmin)

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.xmin + self.xmax)/2,
            (self.ymin + self.ymax)/2,
            (self.zmin + self.zmax)/2,
        )


@dataclass(frozen=True)
class FarFieldMetrics:
    """Useful scalar summaries of a sampled 3D realized-gain pattern."""

    frequency_hz: float
    angular_step_deg: float
    peak_gain_dbi: float
    peak_theta_deg: float
    peak_phi_deg: float
    peak_elevation_deg: float
    horizon_min_gain_dbi: float
    horizon_p10_gain_dbi: float
    horizon_mean_gain_dbi: float
    horizon_p90_gain_dbi: float
    horizon_peak_gain_dbi: float
    horizon_ripple_p90_p10_db: float
    horizon_peak_to_null_db: float

    def as_dict(self) -> dict[str, float]:
        return {
            field: float(value)
            for field, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class AzimuthRingMetrics:
    """Gain statistics around one constant-theta azimuth ring."""

    target_theta_deg: float
    sampled_theta_deg: float
    min_gain_dbi: float
    p10_gain_dbi: float
    mean_gain_dbi: float
    p90_gain_dbi: float
    peak_gain_dbi: float
    ripple_p90_p10_db: float
    peak_to_null_db: float

    def as_dict(self) -> dict[str, float]:
        return {
            field: float(value)
            for field, value in self.__dict__.items()
        }


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
    farfield_metrics: FarFieldMetrics | None = None
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

    @property
    def antenna_height(self) -> float:
        return float(self.artifacts.path.z[-1] - self.design.port_height)

    def gain_db_at(self, theta_deg: float, phi_deg: float) -> float:
        """Return gain at the nearest sampled spherical direction."""
        if self.farfield_3d is None:
            raise RuntimeError("far-field gain was not computed")
        theta = np.asarray(self.farfield_3d.theta, dtype=float)
        phi = np.asarray(self.farfield_3d.phi, dtype=float)
        theta_target = np.deg2rad(theta_deg)
        phi_delta = np.angle(np.exp(1j*(phi - np.deg2rad(phi_deg))))
        distance = (theta - theta_target)**2 + phi_delta**2
        index = np.unravel_index(int(np.argmin(distance)), distance.shape)
        amplitude = abs(np.asarray(self.farfield_3d.normE)[index]/em.lib.EISO)
        return float(20*np.log10(max(amplitude, 1e-12)))

    def azimuth_ring_metrics(self, theta_deg: float) -> AzimuthRingMetrics:
        """Return gain statistics across every azimuth at one theta."""
        if self.farfield_3d is None:
            raise RuntimeError("far-field gain was not computed")
        return _azimuth_ring_metrics(self.farfield_3d, theta_deg)

    def azimuth_ring_beamwidth_deg(self, theta_deg: float) -> float:
        """Return elevation HPBW of the azimuthal-P10 ring profile."""
        if self.farfield_3d is None:
            raise RuntimeError("far-field gain was not computed")
        return _azimuth_ring_beamwidth_deg(self.farfield_3d, theta_deg)

    def directional_beamwidths_deg(
        self,
        theta_deg: float,
        phi_deg: float,
    ) -> tuple[float, float]:
        """Return orthogonal elevation/azimuth HPBW through a direction."""
        if self.farfield_3d is None:
            raise RuntimeError("far-field gain was not computed")
        return _directional_beamwidths_deg(
            self.farfield_3d,
            theta_deg,
            phi_deg,
            self.options.farfield_angular_step_deg,
        )

    def as_dict(self) -> dict[str, Any]:
        values = {
            "frequencies_hz": self.frequencies.tolist(),
            "s11_real": self.s11.real.tolist(),
            "s11_imag": self.s11.imag.tolist(),
            "s11_db": self.s11_db.tolist(),
            "peak_gain_dbi": self.peak_gain_dbi,
            "antenna_height_m": self.antenna_height,
            "mesh_nodes": self.artifacts.mesh_nodes,
            "mesh_elements": self.artifacts.mesh_elements,
            "volume_elements": self.artifacts.volume_elements,
            "open_region_mode": self.artifacts.open_region_mode,
            "huygens_face_count": len(
                self.artifacts.farfield_selection.tags
            ),
            "outer_boundary_face_count": len(
                self.artifacts.outer_boundary_tags
            ),
        }
        if self.farfield_metrics is not None:
            values["farfield_metrics"] = self.farfield_metrics.as_dict()
        return values


def _half_power_beamwidth_deg(
    offsets_deg: np.ndarray,
    gain_db: np.ndarray,
) -> float:
    """Measure the periodic -3 dB region containing the zero-degree sample."""
    offsets = np.asarray(offsets_deg, dtype=float)
    gain = np.asarray(gain_db, dtype=float)
    if offsets.ndim != 1 or gain.shape != offsets.shape or offsets.size < 4:
        raise ValueError("beamwidth cuts must be matching one-dimensional arrays")
    center = int(np.argmin(np.abs(offsets)))
    if not np.isclose(offsets[center], 0.0):
        raise ValueError("beamwidth cut must contain zero degrees")
    if not np.isfinite(gain[center]):
        raise RuntimeError("directional target gain was not finite")

    threshold = gain[center] - 3.0
    above = np.isfinite(gain) & (gain >= threshold)
    if np.all(above):
        return 360.0

    count = offsets.size
    right_inside = center
    while above[(right_inside + 1) % count]:
        right_inside = (right_inside + 1) % count
    right_outside = (right_inside + 1) % count

    left_inside = center
    while above[(left_inside - 1) % count]:
        left_inside = (left_inside - 1) % count
    left_outside = (left_inside - 1) % count

    center_angle = offsets[center]
    right_inside_angle = (offsets[right_inside] - center_angle) % 360.0
    right_outside_angle = (offsets[right_outside] - center_angle) % 360.0
    left_inside_angle = -((center_angle - offsets[left_inside]) % 360.0)
    left_outside_angle = -((center_angle - offsets[left_outside]) % 360.0)

    def crossing(
        inside_angle: float,
        outside_angle: float,
        inside_gain: float,
        outside_gain: float,
    ) -> float:
        fraction = (threshold - inside_gain) / (outside_gain - inside_gain)
        return inside_angle + float(np.clip(fraction, 0.0, 1.0)) * (
            outside_angle - inside_angle
        )

    right_crossing = crossing(
        right_inside_angle,
        right_outside_angle,
        gain[right_inside],
        gain[right_outside],
    )
    left_crossing = crossing(
        left_inside_angle,
        left_outside_angle,
        gain[left_inside],
        gain[left_outside],
    )
    return float(np.clip(right_crossing - left_crossing, 0.0, 360.0))


def _directional_beamwidths_deg(
    farfield: Any,
    theta_deg: float,
    phi_deg: float,
    angular_step_deg: float,
) -> tuple[float, float]:
    """Measure orthogonal great-circle HPBW cuts through a target direction."""
    if not np.isfinite(theta_deg) or not 0 <= theta_deg <= 180:
        raise ValueError("theta_deg must be finite and between zero and 180")
    if not np.isfinite(phi_deg):
        raise ValueError("phi_deg must be finite")
    if not np.isfinite(angular_step_deg) or angular_step_deg <= 0:
        raise ValueError("angular_step_deg must be finite and positive")

    gain_amplitude = np.abs(np.asarray(farfield.normE)/em.lib.EISO)
    gain_db = 20*np.log10(np.maximum(gain_amplitude, 1e-12))
    theta_grid = np.asarray(farfield.theta, dtype=float)
    phi_grid = np.asarray(farfield.phi, dtype=float)
    if (
        theta_grid.ndim != 2
        or theta_grid.shape != gain_db.shape
        or phi_grid.shape != gain_db.shape
    ):
        raise RuntimeError("far-field angular grid has an unsupported shape")
    theta_axis = theta_grid[0, :]
    phi_axis = phi_grid[:, 0]

    sample_count = max(4, int(np.ceil(360.0/angular_step_deg)))
    if sample_count % 2:
        sample_count += 1
    offsets_deg = np.linspace(-180.0, 180.0, sample_count, endpoint=False)
    offsets = np.deg2rad(offsets_deg)
    theta_target = np.deg2rad(theta_deg)
    phi_target = np.deg2rad(phi_deg)
    direction = np.asarray(
        (
            np.sin(theta_target)*np.cos(phi_target),
            np.sin(theta_target)*np.sin(phi_target),
            np.cos(theta_target),
        )
    )
    elevation_tangent = np.asarray(
        (
            np.cos(theta_target)*np.cos(phi_target),
            np.cos(theta_target)*np.sin(phi_target),
            -np.sin(theta_target),
        )
    )
    azimuth_tangent = np.asarray(
        (-np.sin(phi_target), np.cos(phi_target), 0.0)
    )

    def cut_gain(tangent: np.ndarray) -> np.ndarray:
        points = (
            np.cos(offsets)[:, None]*direction
            + np.sin(offsets)[:, None]*tangent
        )
        sample_theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
        sample_phi = np.arctan2(points[:, 1], points[:, 0])
        theta_indices = np.argmin(
            np.abs(theta_axis[None, :] - sample_theta[:, None]),
            axis=1,
        )
        phi_delta = np.angle(
            np.exp(1j*(phi_axis[None, :] - sample_phi[:, None]))
        )
        phi_indices = np.argmin(np.abs(phi_delta), axis=1)
        return np.asarray(gain_db[phi_indices, theta_indices], dtype=float)

    elevation_width = _half_power_beamwidth_deg(
        offsets_deg,
        cut_gain(elevation_tangent),
    )
    azimuth_width = _half_power_beamwidth_deg(
        offsets_deg,
        cut_gain(azimuth_tangent),
    )
    return elevation_width, azimuth_width


def _azimuth_ring_data(
    farfield: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return theta samples and gain with the duplicate azimuth removed."""
    theta = np.asarray(farfield.theta, dtype=float)
    phi = np.asarray(farfield.phi, dtype=float)
    gain_amplitude = np.abs(np.asarray(farfield.normE)/em.lib.EISO)
    gain_db = 20*np.log10(np.maximum(gain_amplitude, 1e-12))
    if theta.ndim != 2 or phi.shape != theta.shape or gain_db.shape != theta.shape:
        raise ValueError("far-field theta, phi, and gain grids must have equal shape")
    theta_axis = np.asarray(theta[0, :], dtype=float)
    phi_axis = np.asarray(phi[:, 0], dtype=float)
    if phi_axis.size > 1 and np.isclose(
        np.angle(np.exp(1j*(phi_axis[-1] - phi_axis[0]))),
        0.0,
    ):
        gain_db = gain_db[:-1, :]
    return theta_axis, gain_db


def _azimuth_ring_metrics(
    farfield: Any,
    theta_deg: float,
) -> AzimuthRingMetrics:
    """Measure azimuthal gain statistics at the nearest sampled theta."""
    if not np.isfinite(theta_deg) or not 0 <= theta_deg <= 180:
        raise ValueError("theta_deg must be finite and between 0 and 180")
    theta_axis, gain_db = _azimuth_ring_data(farfield)
    theta_target = np.deg2rad(theta_deg)
    theta_index = int(np.argmin(np.abs(theta_axis - theta_target)))
    ring_gain = np.asarray(gain_db[:, theta_index], dtype=float)
    p10 = float(np.nanpercentile(ring_gain, 10))
    p90 = float(np.nanpercentile(ring_gain, 90))
    mean = float(10*np.log10(np.nanmean(10**(ring_gain/10))))
    minimum = float(np.nanmin(ring_gain))
    peak = float(np.nanmax(ring_gain))
    return AzimuthRingMetrics(
        target_theta_deg=float(theta_deg),
        sampled_theta_deg=float(np.rad2deg(theta_axis[theta_index])),
        min_gain_dbi=minimum,
        p10_gain_dbi=p10,
        mean_gain_dbi=mean,
        p90_gain_dbi=p90,
        peak_gain_dbi=peak,
        ripple_p90_p10_db=p90 - p10,
        peak_to_null_db=peak - minimum,
    )


def _azimuth_ring_beamwidth_deg(farfield: Any, theta_deg: float) -> float:
    """Measure HPBW of the azimuthal-P10 profile around a target theta."""
    if not np.isfinite(theta_deg) or not 0 <= theta_deg <= 180:
        raise ValueError("theta_deg must be finite and between 0 and 180")
    theta_axis_rad, gain_db = _azimuth_ring_data(farfield)
    theta_axis = np.rad2deg(theta_axis_rad)
    profile = np.nanpercentile(gain_db, 10, axis=0)
    center = int(np.argmin(np.abs(theta_axis - theta_deg)))
    if not np.isfinite(profile[center]):
        raise RuntimeError("azimuth ring target gain was not finite")
    threshold = profile[center] - 3.0
    above = np.isfinite(profile) & (profile >= threshold)

    left_inside = center
    while left_inside > 0 and above[left_inside - 1]:
        left_inside -= 1
    right_inside = center
    while right_inside + 1 < above.size and above[right_inside + 1]:
        right_inside += 1

    def crossing(inside: int, outside: int) -> float:
        inside_gain = profile[inside]
        outside_gain = profile[outside]
        if not np.isfinite(outside_gain) or np.isclose(inside_gain, outside_gain):
            return float(theta_axis[inside])
        fraction = float(
            np.clip(
                (threshold - inside_gain)/(outside_gain - inside_gain),
                0.0,
                1.0,
            )
        )
        return float(
            theta_axis[inside]
            + fraction*(theta_axis[outside] - theta_axis[inside])
        )

    left = (
        float(theta_axis[0])
        if left_inside == 0
        else crossing(left_inside, left_inside - 1)
    )
    right = (
        float(theta_axis[-1])
        if right_inside == above.size - 1
        else crossing(right_inside, right_inside + 1)
    )
    return right - left


def _farfield_metrics(
    farfield: Any,
    frequency: float,
    angular_step_deg: float,
) -> FarFieldMetrics:
    gain_amplitude = np.abs(np.asarray(farfield.normE)/em.lib.EISO)
    gain_db = 20*np.log10(np.maximum(gain_amplitude, 1e-12))
    theta = np.asarray(farfield.theta, dtype=float)
    phi = np.asarray(farfield.phi, dtype=float)

    peak_index = np.unravel_index(int(np.nanargmax(gain_db)), gain_db.shape)
    peak_theta = float(np.rad2deg(theta[peak_index]))
    peak_phi = float(np.rad2deg(phi[peak_index]))

    horizon = _azimuth_ring_metrics(farfield, 90.0)

    return FarFieldMetrics(
        frequency_hz=float(frequency),
        angular_step_deg=float(angular_step_deg),
        peak_gain_dbi=float(gain_db[peak_index]),
        peak_theta_deg=peak_theta,
        peak_phi_deg=peak_phi,
        peak_elevation_deg=90.0 - peak_theta,
        horizon_min_gain_dbi=horizon.min_gain_dbi,
        horizon_p10_gain_dbi=horizon.p10_gain_dbi,
        horizon_mean_gain_dbi=horizon.mean_gain_dbi,
        horizon_p90_gain_dbi=horizon.p90_gain_dbi,
        horizon_peak_gain_dbi=horizon.peak_gain_dbi,
        horizon_ripple_p90_p10_db=horizon.ripple_p90_p10_db,
        horizon_peak_to_null_db=horizon.peak_to_null_db,
    )


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
    if design.has_circular_groundplane:
        assert design.groundplane_diameter is not None
        print(
            "Groundplane        : circular, "
            f"{design.groundplane_diameter*1e3:.1f} mm diameter"
        )
    else:
        print(
            f"Radials            : {design.radial_count} x "
            f"{design.radial_length*1e3:.1f} mm"
        )
        print(
            f"Radial angle       : {design.radial_angle_deg:.1f} deg "
            "below horizontal"
        )
    print("----------------------------------------------------")


def _configure_solver(model: em.Simulation, solver_name: str) -> None:
    if solver_name == "auto":
        return
    solver = getattr(em.EMSolver, solver_name.upper())
    try:
        model.set_solver(solver)
    except KeyError as error:
        install_hint = (
            " Run 'emerge install-solver cudss' in the active environment."
            if solver_name == "cudss"
            else ""
        )
        raise RuntimeError(
            f"EMerge solver {solver_name!r} is unavailable.{install_hint}"
        ) from error


def _build_open_region(
    bounds: _OpenRegionBounds,
    wavelength: float,
    settings: OpenRegionSettings,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    """Create the inner air box and its selected six-sided termination."""
    width, depth, height = bounds.dimensions
    if settings.mode == "abc":
        buffer = settings.abc_buffer_wavelengths*wavelength
        airbox = em.geo.Box(
            width=width,
            depth=depth,
            height=height,
            position=bounds.corner,
            name="HuygensAirbox",
        )
        shell = (
            em.geo.Box(
                buffer,
                depth + 2*buffer,
                height + 2*buffer,
                (bounds.xmin - buffer, bounds.ymin - buffer, bounds.zmin - buffer),
                name="ABCBufferLeft",
            ),
            em.geo.Box(
                buffer,
                depth + 2*buffer,
                height + 2*buffer,
                (bounds.xmax, bounds.ymin - buffer, bounds.zmin - buffer),
                name="ABCBufferRight",
            ),
            em.geo.Box(
                width,
                buffer,
                height + 2*buffer,
                (bounds.xmin, bounds.ymin - buffer, bounds.zmin - buffer),
                name="ABCBufferFront",
            ),
            em.geo.Box(
                width,
                buffer,
                height + 2*buffer,
                (bounds.xmin, bounds.ymax, bounds.zmin - buffer),
                name="ABCBufferBack",
            ),
            em.geo.Box(
                width,
                depth,
                buffer,
                (bounds.xmin, bounds.ymin, bounds.zmin - buffer),
                name="ABCBufferBottom",
            ),
            em.geo.Box(
                width,
                depth,
                buffer,
                (bounds.xmin, bounds.ymin, bounds.zmax),
                name="ABCBufferTop",
            ),
        )
        volumes = (airbox, *shell)
        for volume in volumes:
            volume.background()
        return airbox, volumes, ()

    pml_thickness = settings.pml_thickness_wavelengths*wavelength
    volumes = em.geo.pmlbox(
        width=width,
        depth=depth,
        height=height,
        position=bounds.corner,
        material=em.lib.AIR,
        thickness=pml_thickness,
        Nlayers=1,
        N_mesh_layers=settings.pml_mesh_layers,
        exponent=settings.pml_exponent,
        deltamax=settings.pml_delta_max,
        sides="tblrfa",
    )
    for index, volume in enumerate(volumes):
        volume.give_name("Airbox" if index == 0 else f"PML{index:02d}")
        volume.background()
    return volumes[0], tuple(volumes), tuple(volumes[1:])


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

    # EMerge simulations own reference cycles. If a previous optimizer result
    # is collected after the next SimState signs on, its destructor clears the
    # process-global selection interface used by the new model. Collect those
    # stale cycles before creating any new EMerge geometry.
    gc.collect()
    model = em.Simulation(
        options.model_name,
        loglevel="INFO" if options.verbose else "ERROR",
    )
    model.check_version("2.8.4")
    if options.solve:
        _configure_solver(model, options.solver)

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
    if design.has_circular_groundplane:
        assert design.groundplane_diameter is not None
        ground_horizontal = design.groundplane_diameter/2
        ground_lowest_z = 0.0
        ground_system = (
            em.geo.Disc(
                (0.0, 0.0, 0.0),
                ground_horizontal,
                name="CircularGroundPlane",
            )
            .set_material(em.lib.MET_COPPER)
            .foreground()
        )
    else:
        mm = 1e-3
        hub_radius = 1.95*mm/2+design.wire_radius*2#4*design.wire_radius
        hub_height = 6*mm#3*design.wire_radius
        ground_hub = em.geo.Cylinder(
            hub_radius,
            hub_height,
            cs=em.GCS.displace(0.0, 0.0, -hub_height),
            Nsections=mesh.wire_sections,#max(24, mesh.wire_sections),
            name="GroundHub",
        )
        ground_hub.max_meshsize = mesh.feed_size_factor*design.wire_radius

        radial_start =hub_radius-design.wire_radius
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

            em.geo.translate(
                radial,
                dz=-hub_height+design.wire_radius*2,
            )

            radial = radial.set_material(em.lib.MET_COPPER).foreground()
            radial.max_meshsize = mesh.radial_size_factor*design.wire_radius
            radials.append(radial)

        ground_system = (
            em.geo.unite(ground_hub, *radials)
            .set_material(em.lib.MET_COPPER)
            .foreground()
        )
        ground_horizontal = design.radial_length*np.cos(
            np.deg2rad(design.radial_angle_deg)
        )
        ground_lowest_z = -design.radial_length*np.sin(
            np.deg2rad(design.radial_angle_deg)
        )

    ground_system.max_meshsize = mesh.radial_size_factor*design.wire_radius

    wavelength = C0/options.sweep.center
    air_margin = mesh.air_margin_wavelengths*wavelength
    xmin = min(float(np.min(xpts)), -ground_horizontal) - air_margin
    xmax = max(float(np.max(xpts)), ground_horizontal) + air_margin
    ymin = min(float(np.min(ypts)), -ground_horizontal) - air_margin
    ymax = max(float(np.max(ypts)), ground_horizontal) + air_margin
    zmin = ground_lowest_z - air_margin
    zmax = float(np.max(zpts)) + air_margin
    open_region_bounds = _OpenRegionBounds(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
    )
    airbox, open_region_volumes, pml_volumes = _build_open_region(
        open_region_bounds,
        wavelength,
        options.open_region,
    )

    if options.verbose:
        print(f"Open-region mode    : {options.open_region.mode.upper()} (all 6 sides)")
        print(f"Inner air margin    : {mesh.air_margin_wavelengths:.3f} wavelengths")
        if options.open_region.mode == "abc":
            print(
                "ABC air buffer      : "
                f"{options.open_region.abc_buffer_wavelengths:.3f} wavelengths"
            )
        else:
            print(
                "PML thickness       : "
                f"{options.open_region.pml_thickness_wavelengths:.3f} wavelengths, "
                f"{options.open_region.pml_mesh_layers} mesh layers"
            )

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

    farfield_selection = airbox.boundary()
    if len(farfield_selection.tags) != 6:
        raise RuntimeError(
            "The closed Huygens surface must contain exactly six faces; "
            f"EMerge returned {len(farfield_selection.tags)}."
        )
    outer_boundary_tags = tuple(model.mesher.domain_boundary_face_tags)
    termination_selection = None
    if options.open_region.mode == "abc":
        termination_selection = em.FaceSelection(list(outer_boundary_tags))
        if set(farfield_selection.tags) & set(termination_selection.tags):
            raise RuntimeError(
                "The ABC termination unexpectedly touches the Huygens surface."
            )
        model.mw.bc.AbsorbingBoundary(
            termination_selection,
            order=options.open_region.abc_order,
            origin=open_region_bounds.center,
            abctype=options.open_region.abc_type,
        )
    elif set(farfield_selection.tags) & set(outer_boundary_tags):
        raise RuntimeError(
            "The PML Huygens surface unexpectedly touches the exterior boundary."
        )
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
        open_region_volumes=open_region_volumes,
        pml_volumes=pml_volumes,
        farfield_selection=farfield_selection,
        termination_selection=termination_selection,
        open_region_mode=options.open_region.mode,
        farfield_origin=open_region_bounds.center,
        outer_boundary_tags=outer_boundary_tags,
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
    farfield_metrics = None
    farfield_3d = None
    if options.compute_farfield:
        farfield_frequency = (
            options.farfield_frequency
            if options.farfield_frequency is not None
            else options.sweep.center
        )
        field = data.field.find(freq=farfield_frequency)
        angular_step = options.farfield_angular_step_deg
        theta_count = max(2, int(round(180.0/angular_step)) + 1)
        phi_count = max(2, int(round(360.0/angular_step)) + 1)
        thetas = np.linspace(0.0, np.pi, theta_count)
        phis = np.linspace(-np.pi, np.pi, phi_count)
        farfield_3d = field.farfield_3d(
            artifacts.farfield_selection,
            thetas=thetas,
            phis=phis,
            origin=artifacts.farfield_origin,
        )
        farfield_metrics = _farfield_metrics(
            farfield_3d,
            float(field.freq),
            angular_step,
        )
        peak_gain_dbi = farfield_metrics.peak_gain_dbi

    return SimulationResult(
        design=design,
        options=options,
        artifacts=artifacts,
        frequencies=frequencies,
        s11=s11,
        s11_db=s11_db,
        peak_gain_dbi=peak_gain_dbi,
        farfield_metrics=farfield_metrics,
        farfield_3d=farfield_3d,
        raw_data=data,
    )
