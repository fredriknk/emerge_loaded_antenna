"""Parameterized loaded-antenna simulation library."""

import os
from pathlib import Path

_venv_mkl_dir = Path(__file__).resolve().parents[1]/".venv"/"Library"/"bin"
_venv_mkl = next(iter(sorted(_venv_mkl_dir.glob("mkl_rt*.dll"))), None)
if _venv_mkl is not None:
    os.environ.setdefault("EMERGE_PARDISO_PATH", str(_venv_mkl))

from .config import (
    AntennaDesign,
    ABC_TYPES,
    CoilDesign,
    FrequencySweep,
    MeshSettings,
    OPEN_REGION_MODES,
    OpenRegionSettings,
    SOLVER_CHOICES,
    SimulationOptions,
)
from .convergence import (
    CONVERGENCE_SCHEMA_VERSION,
    design_fingerprint,
    load_convergence_certificate,
    selected_open_region_configuration,
    validate_convergence_certificate,
)
from .geometry import AntennaPath, CompositeCurve, build_centerline
from .optimize import (
    DesignSpace,
    DesignVariable,
    EvaluationCallback,
    EvaluationRecord,
    GainMatchObjective,
    RobustGainObjective,
    S11Objective,
)
from .presets import REFERENCE_868MHZ_DESIGN_PATH, load_reference_868mhz_design
from .simulation import (
    FarFieldMetrics,
    ModelArtifacts,
    SimulationResult,
    build_model,
    simulate,
)
from .serialization import design_from_dict, load_design, save_design

__all__ = [
    "AntennaDesign",
    "ABC_TYPES",
    "AntennaPath",
    "CoilDesign",
    "CompositeCurve",
    "CONVERGENCE_SCHEMA_VERSION",
    "DesignSpace",
    "DesignVariable",
    "EvaluationCallback",
    "EvaluationRecord",
    "FarFieldMetrics",
    "FrequencySweep",
    "GainMatchObjective",
    "MeshSettings",
    "ModelArtifacts",
    "RobustGainObjective",
    "REFERENCE_868MHZ_DESIGN_PATH",
    "OPEN_REGION_MODES",
    "OpenRegionSettings",
    "SOLVER_CHOICES",
    "S11Objective",
    "SimulationOptions",
    "SimulationResult",
    "build_centerline",
    "build_model",
    "design_fingerprint",
    "design_from_dict",
    "load_design",
    "load_reference_868mhz_design",
    "load_convergence_certificate",
    "save_design",
    "selected_open_region_configuration",
    "simulate",
    "validate_convergence_certificate",
]
