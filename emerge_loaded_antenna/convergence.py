"""Open-region convergence certificates used by long optimizer campaigns."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .config import AntennaDesign, MeshSettings, OpenRegionSettings

CONVERGENCE_SCHEMA_VERSION = 2


def design_fingerprint(design: AntennaDesign) -> str:
    """Return a stable fingerprint for one physical antenna design."""
    encoded = json.dumps(
        asdict(design),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_open_region_configuration(
    mesh: MeshSettings,
    open_region: OpenRegionSettings,
) -> dict[str, Any]:
    """Serialize the numerical settings that a certificate covers."""
    values: dict[str, Any] = {
        "mode": open_region.mode,
        "air_margin_wavelengths": mesh.air_margin_wavelengths,
        "wavelength_resolution": mesh.wavelength_resolution,
    }
    if open_region.mode == "abc":
        values.update(
            abc_buffer_wavelengths=open_region.abc_buffer_wavelengths,
            abc_order=open_region.abc_order,
            abc_type=open_region.abc_type,
        )
    else:
        values.update(
            pml_thickness_wavelengths=(
                open_region.pml_thickness_wavelengths
            ),
            pml_mesh_layers=open_region.pml_mesh_layers,
            pml_exponent=open_region.pml_exponent,
            pml_delta_max=open_region.pml_delta_max,
        )
    return values


def load_convergence_certificate(
    path: str | Path,
    require_passed: bool = True,
) -> dict[str, Any]:
    """Load a convergence report, optionally accepting a completed failure."""
    report_path = Path(path)
    if not report_path.exists():
        raise RuntimeError(
            f"Open-region convergence certificate not found: {report_path}"
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read open-region certificate {report_path}: {error}"
        ) from error
    if payload.get("schema_version") != CONVERGENCE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported open-region certificate schema in {report_path}."
        )
    if not isinstance(payload.get("passed"), bool):
        raise RuntimeError(
            f"Open-region certificate {report_path} has no pass/fail result."
        )
    if require_passed and payload.get("passed") is not True:
        raise RuntimeError(
            f"Open-region convergence did not pass in {report_path}."
        )
    if not require_passed:
        return payload
    samples = payload.get("samples")
    if not isinstance(samples, dict) or len(samples) < 4:
        raise RuntimeError(
            f"Open-region certificate {report_path} lacks independent solved samples."
        )
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) < 3:
        raise RuntimeError(
            f"Open-region certificate {report_path} lacks convergence comparisons."
        )
    for comparison in comparisons:
        if not isinstance(comparison, dict) or comparison.get("passed") is not True:
            raise RuntimeError(
                f"Open-region certificate {report_path} has a failed comparison."
            )
        for key_name in ("selected", "reference"):
            key = comparison.get(key_name)
            sample = samples.get(key)
            if not isinstance(sample, dict) or "error" in sample:
                raise RuntimeError(
                    f"Open-region certificate {report_path} references a failed solve."
                )
    return payload


def validate_convergence_certificate(
    path: str | Path,
    reference_design: AntennaDesign,
    mesh: MeshSettings,
    open_region: OpenRegionSettings,
    frequency_hz: float,
    require_passed: bool = True,
) -> dict[str, Any]:
    """Ensure a certificate covers this numerical reference and configuration."""
    payload = load_convergence_certificate(path, require_passed=require_passed)
    if payload.get("design_fingerprint") != design_fingerprint(reference_design):
        raise RuntimeError(
            "The convergence certificate was generated for a different "
            "numerical reference design."
        )
    certified_frequency = payload.get("frequency_hz")
    if not isinstance(certified_frequency, (int, float)) or not math.isclose(
        float(certified_frequency),
        float(frequency_hz),
        rel_tol=1e-9,
        abs_tol=1e-3,
    ):
        raise RuntimeError(
            "The convergence certificate was generated at a different frequency."
        )

    expected = selected_open_region_configuration(mesh, open_region)
    certified = payload.get("selected_configuration", {})
    mismatches = []
    for name, value in expected.items():
        actual = certified.get(name)
        if isinstance(value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                value,
                float(actual),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        else:
            matches = actual == value
        if not matches:
            mismatches.append(f"{name}: expected {value!r}, certificate has {actual!r}")
    if mismatches:
        raise RuntimeError(
            "The optimizer settings do not match the convergence certificate: "
            + "; ".join(mismatches)
        )
    return payload
