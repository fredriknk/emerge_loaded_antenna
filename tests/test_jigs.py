from __future__ import annotations

import json
import struct

import pytest

from emerge_loaded_antenna import (
    AntennaDesign,
    CoilDesign,
    JigModelSettings,
    export_jig_models,
)


def jig_design() -> AntennaDesign:
    return AntennaDesign(
        wire_radius=0.8e-3,
        straight_lengths=(0.10, 0.12, 0.09),
        coils=(
            CoilDesign(radius=10e-3, turns=1, pitch=7e-3, transition=2e-3),
            CoilDesign(
                radius=12e-3,
                turns=2,
                pitch=6e-3,
                transition=2e-3,
                handedness="LH",
            ),
        ),
    )


def test_export_jig_models_writes_print_scale_binary_stls_and_manifest(tmp_path):
    settings = JigModelSettings(
        circumferential_segments=32,
        axial_segments_per_turn=12,
        end_margin_segments=2,
    )

    paths = export_jig_models(jig_design(), tmp_path, settings=settings)

    assert [path.name for path in paths] == [
        "coil_01_winding_jig.stl",
        "coil_02_winding_jig.stl",
    ]
    for path in paths:
        data = path.read_bytes()
        facet_count = struct.unpack("<I", data[80:84])[0]
        first_facet = struct.unpack("<12fH", data[84:134])
        assert data.startswith(b"EMerge loaded antenna coil winding jig")
        assert len(data) == 84 + 50 * facet_count
        assert facet_count > 0
        assert max(abs(value) for value in first_facet[3:12]) > 5.0

    manifest = json.loads((tmp_path / "jig_models.json").read_text())
    assert manifest["format"] == "binary STL"
    assert manifest["units"] == "millimetres"
    assert [model["handedness"] for model in manifest["models"]] == ["RH", "LH"]
    first = manifest["models"][0]
    assert first["formed_wire_centerline_radius_mm"] == pytest.approx(9.85)
    assert first["coil_centerline_radius_mm"] == pytest.approx(10.0)
    assert first["model_length_mm"] == pytest.approx(13.0)


def test_export_jig_models_for_unloaded_design_writes_manifest_only(tmp_path):
    design = AntennaDesign(straight_lengths=(0.25,), coils=())

    paths = export_jig_models(design, tmp_path)

    assert paths == ()
    manifest = json.loads((tmp_path / "jig_models.json").read_text())
    assert manifest["models"] == []


def test_jig_settings_reject_underresolved_mesh():
    with pytest.raises(ValueError, match="at least 24"):
        JigModelSettings(circumferential_segments=12).validate()
