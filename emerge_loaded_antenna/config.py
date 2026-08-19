"""Public configuration objects for loaded-antenna simulations.

All dimensions are expressed in metres and all frequencies in hertz. Keeping
the public API unitless makes it straightforward to connect to numerical
optimizers without teaching them about presentation units.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class CoilDesign:
    """Geometry of one helical loading coil."""

    radius: float = 10e-3
    turns: int = 1
    pitch: float = 7e-3
    transition: float = 6e-3
    transition_offset: float = 4.75e-3
    handedness: str = "RH"

    def validate(self) -> None:
        if self.radius <= 0:
            raise ValueError("coil radius must be positive")
        if isinstance(self.turns, bool) or int(self.turns) != self.turns:
            raise ValueError("coil turns must be an integer")
        if self.turns <= 0:
            raise ValueError("coil turns must be positive")
        if self.pitch <= 0 or self.transition <= 0:
            raise ValueError("coil pitch and transition must be positive")
        if not 0 < self.transition_offset < 2*self.radius:
            raise ValueError(
                "coil transition_offset must be between zero and the diameter"
            )
        if self.handedness.upper() not in {"RH", "LH"}:
            raise ValueError("coil handedness must be 'RH' or 'LH'")


@dataclass(frozen=True, init=False)
class AntennaDesign:
    """Complete physical antenna design.

    ``N`` coils are interleaved with exactly ``N + 1`` straight sections.
    Defaults reproduce the two-coil working example in :mod:`main`.
    """

    wire_radius: float = 1e-3
    radial_length: float = 72e-3
    radial_angle_deg: float = 45.0
    radial_count: int = 4
    straight_lengths: tuple[float, ...] = (140e-3, 221e-3, 140e-3)
    coils: tuple[CoilDesign, ...] = field(
        default_factory=lambda: (CoilDesign(), CoilDesign())
    )
    port_height: float = 2e-3
    port_impedance: float = 50.0

    def __init__(
        self,
        wire_radius: float = 1e-3,
        radial_length: float = 72e-3,
        radial_angle_deg: float = 45.0,
        radial_count: int = 4,
        straight_lengths: Sequence[float] | None = None,
        coils: Sequence[CoilDesign] | None = None,
        port_height: float = 2e-3,
        port_impedance: float = 50.0,
        *,
        bottom_length: float | None = None,
        coil1: CoilDesign | None = None,
        middle_length: float | None = None,
        coil2: CoilDesign | None = None,
        top_length: float | None = None,
    ) -> None:
        lengths = tuple(
            (140e-3, 221e-3, 140e-3)
            if straight_lengths is None
            else straight_lengths
        )
        coil_items = tuple(
            (CoilDesign(), CoilDesign()) if coils is None else coils
        )

        legacy_lengths = (bottom_length, middle_length, top_length)
        if any(value is not None for value in legacy_lengths):
            if len(lengths) != 3:
                raise ValueError(
                    "legacy length fields require exactly two configured coils"
                )
            lengths = tuple(
                current if replacement is None else replacement
                for current, replacement in zip(lengths, legacy_lengths)
            )
        legacy_coils = (coil1, coil2)
        if any(coil is not None for coil in legacy_coils):
            if len(coil_items) != 2:
                raise ValueError(
                    "legacy coil fields require exactly two configured coils"
                )
            coil_items = tuple(
                current if replacement is None else replacement
                for current, replacement in zip(coil_items, legacy_coils)
            )

        for name, value in (
            ("wire_radius", wire_radius),
            ("radial_length", radial_length),
            ("radial_angle_deg", radial_angle_deg),
            ("radial_count", radial_count),
            ("straight_lengths", lengths),
            ("coils", coil_items),
            ("port_height", port_height),
            ("port_impedance", port_impedance),
        ):
            object.__setattr__(self, name, value)

    @property
    def coil_count(self) -> int:
        return len(self.coils)

    @property
    def bottom_length(self) -> float:
        return self.straight_lengths[0]

    @property
    def middle_length(self) -> float:
        if self.coil_count != 2:
            raise AttributeError("middle_length is only defined for two coils")
        return self.straight_lengths[1]

    @property
    def top_length(self) -> float:
        return self.straight_lengths[-1]

    @property
    def coil1(self) -> CoilDesign:
        if self.coil_count < 1:
            raise AttributeError("coil1 is not defined for a zero-coil design")
        return self.coils[0]

    @property
    def coil2(self) -> CoilDesign:
        if self.coil_count < 2:
            raise AttributeError("coil2 requires at least two coils")
        return self.coils[1]

    def validate(self) -> None:
        if self.wire_radius <= 0:
            raise ValueError("wire_radius must be positive")
        if self.radial_length <= 0:
            raise ValueError("radial_length must be positive")
        if not 0 < self.radial_angle_deg < 90:
            raise ValueError("radial_angle_deg must be between 0 and 90")
        if isinstance(self.radial_count, bool) or int(self.radial_count) != self.radial_count:
            raise ValueError("radial_count must be an integer")
        if self.radial_count < 2:
            raise ValueError("radial_count must be at least two")
        if len(self.straight_lengths) != self.coil_count + 1:
            raise ValueError(
                "straight_lengths must contain exactly one more item than coils"
            )
        for index, value in enumerate(self.straight_lengths):
            if value <= 0:
                raise ValueError(f"straight_lengths[{index}] must be positive")
        for name, value in (
            ("port_height", self.port_height),
            ("port_impedance", self.port_impedance),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for index, coil in enumerate(self.coils):
            if not isinstance(coil, CoilDesign):
                raise ValueError(f"coils[{index}] must be a CoilDesign")
            coil.validate()


@dataclass(frozen=True)
class FrequencySweep:
    """Frequency range evaluated by EMerge."""

    center: float = 868e6
    span: float = 100e6
    points: int = 5

    @classmethod
    def single(cls, frequency: float) -> FrequencySweep:
        return cls(center=frequency, span=0.0, points=1)

    @property
    def start(self) -> float:
        return self.center - self.span/2

    @property
    def stop(self) -> float:
        return self.center + self.span/2

    def validate(self) -> None:
        if self.center <= 0 or self.span < 0:
            raise ValueError("frequency center must be positive and span non-negative")
        if isinstance(self.points, bool) or int(self.points) != self.points:
            raise ValueError("frequency points must be an integer")
        if self.points <= 0:
            raise ValueError("frequency points must be positive")
        if self.start <= 0:
            raise ValueError("frequency sweep start must be positive")


@dataclass(frozen=True)
class MeshSettings:
    """Meshing controls separated from physical design variables."""

    wire_sections: int = 6
    antenna_size_factor: float = 3.0
    radial_size_factor: float = 10.0
    feed_size_factor: float = 3.0
    curved_boundary_segments: int = 12
    wavelength_resolution: float = 0.5
    air_margin_wavelengths: float = 0.25
    preview_points_per_turn: int = 20

    def validate(self) -> None:
        if self.wire_sections < 6:
            raise ValueError("wire_sections must be at least six")
        if self.curved_boundary_segments < 6:
            raise ValueError("curved_boundary_segments must be at least six")
        for name, value in (
            ("antenna_size_factor", self.antenna_size_factor),
            ("radial_size_factor", self.radial_size_factor),
            ("feed_size_factor", self.feed_size_factor),
            ("wavelength_resolution", self.wavelength_resolution),
            ("air_margin_wavelengths", self.air_margin_wavelengths),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SimulationOptions:
    """Runtime behavior for one model evaluation."""

    sweep: FrequencySweep = field(default_factory=FrequencySweep)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    solve: bool = True
    compute_farfield: bool = False
    farfield_frequency: float | None = None
    farfield_angular_step_deg: float = 2.0
    show_geometry: bool = False
    show_mesh: bool = False
    show_coil_preview: bool = False
    verbose: bool = True
    model_name: str = "SmoothLoadedAntenna"

    def validate(self) -> None:
        self.sweep.validate()
        self.mesh.validate()
        if self.compute_farfield and not self.solve:
            raise ValueError("compute_farfield requires solve=True")
        if self.farfield_frequency is not None and self.farfield_frequency <= 0:
            raise ValueError("farfield_frequency must be positive")
        if not 0 < self.farfield_angular_step_deg <= 10:
            raise ValueError(
                "farfield_angular_step_deg must be between zero and 10 degrees"
            )
