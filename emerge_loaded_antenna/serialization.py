"""JSON-friendly antenna design serialization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from .config import AntennaDesign, CoilDesign


def _coil_from_value(value: Any) -> CoilDesign:
    if isinstance(value, CoilDesign):
        return value
    if isinstance(value, Mapping):
        return CoilDesign(**dict(value))
    raise ValueError("each coil must be a CoilDesign or JSON object")


def design_from_dict(values: Mapping[str, Any]) -> AntennaDesign:
    """Construct and validate a design from an ``asdict``-style mapping."""
    data = dict(values)
    legacy_fields = {
        "bottom_length",
        "middle_length",
        "top_length",
        "coil1",
        "coil2",
    }
    if "straight_lengths" not in data and legacy_fields & data.keys():
        missing = legacy_fields - data.keys()
        if missing:
            raise ValueError(
                "legacy two-coil design is missing: " + ", ".join(sorted(missing))
            )
        data["straight_lengths"] = (
            data.pop("bottom_length"),
            data.pop("middle_length"),
            data.pop("top_length"),
        )
        data["coils"] = (
            _coil_from_value(data.pop("coil1")),
            _coil_from_value(data.pop("coil2")),
        )
    if "straight_lengths" in data:
        lengths = data["straight_lengths"]
        if isinstance(lengths, (str, bytes)) or not isinstance(lengths, Sequence):
            raise ValueError("straight_lengths must be a sequence")
        data["straight_lengths"] = tuple(lengths)
    if "coils" in data:
        coils = data["coils"]
        if isinstance(coils, (str, bytes)) or not isinstance(coils, Sequence):
            raise ValueError("coils must be a sequence")
        data["coils"] = tuple(_coil_from_value(coil) for coil in coils)
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
