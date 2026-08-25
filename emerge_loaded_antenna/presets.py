"""Frequency-scalable reference geometry for examples and numerical checks."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from .config import AntennaDesign
from .serialization import load_design


REFERENCE_DESIGN_FREQUENCY_HZ = 868e6
REFERENCE_DESIGN_PATH = (
    Path(__file__).resolve().parent/"data"/"reference_design.json"
)


def scale_design(design: AntennaDesign, factor: float) -> AntennaDesign:
    """Scale every physical length while retaining angles and impedance."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("design scale factor must be finite and positive")
    scaled = replace(
        design,
        wire_radius=design.wire_radius*factor,
        groundplane_diameter=(
            None
            if design.groundplane_diameter is None
            else design.groundplane_diameter*factor
        ),
        radial_length=design.radial_length*factor,
        straight_lengths=tuple(value*factor for value in design.straight_lengths),
        coils=tuple(
            replace(
                coil,
                radius=coil.radius*factor,
                pitch=coil.pitch*factor,
                transition=coil.transition*factor,
            )
            for coil in design.coils
        ),
        port_height=design.port_height*factor,
    )
    scaled.validate()
    return scaled


def load_reference_design(
    frequency_hz: float = REFERENCE_DESIGN_FREQUENCY_HZ,
) -> AntennaDesign:
    """Load the reference geometry at any frequency by wavelength scaling."""
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("reference frequency must be finite and positive")
    reference = load_design(REFERENCE_DESIGN_PATH)
    return scale_design(reference, REFERENCE_DESIGN_FREQUENCY_HZ/frequency_hz)
