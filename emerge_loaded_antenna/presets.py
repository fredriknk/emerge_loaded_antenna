"""Tracked antenna designs suitable for examples and optimizer warm starts."""

from __future__ import annotations

from pathlib import Path

from .config import AntennaDesign
from .serialization import load_design


REFERENCE_868MHZ_DESIGN_PATH = (
    Path(__file__).resolve().parent/"data"/"868mhz_reference_design.json"
)


def load_reference_868mhz_design() -> AntennaDesign:
    """Load the known-convergent two-coil 868 MHz reference design."""
    return load_design(REFERENCE_868MHZ_DESIGN_PATH)
