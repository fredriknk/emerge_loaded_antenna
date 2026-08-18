"""JSON-friendly antenna design serialization helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from .config import AntennaDesign, CoilDesign


def design_from_dict(values: Mapping[str, Any]) -> AntennaDesign:
    """Construct and validate a design from an ``asdict``-style mapping."""
    data = dict(values)
    for name in ("coil1", "coil2"):
        coil = data.get(name)
        if isinstance(coil, Mapping):
            data[name] = CoilDesign(**dict(coil))
    design = AntennaDesign(**data)
    design.validate()
    return design


def load_design(path: str | Path) -> AntennaDesign:
    """Load a raw design or an optimizer result containing a ``design`` key."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} does not contain a JSON object")
    values = payload.get("design", payload)
    if not isinstance(values, Mapping):
        raise ValueError(f"{source} has no valid design object")
    return design_from_dict(values)


def save_design(design: AntennaDesign, path: str | Path) -> Path:
    """Write a standalone antenna design as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(design), indent=2),
        encoding="utf-8",
    )
    return destination
