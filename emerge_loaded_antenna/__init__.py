"""Parameterized loaded-antenna simulation library."""

import os
from pathlib import Path

_venv_mkl_dir = Path(__file__).resolve().parents[1]/".venv"/"Library"/"bin"
_venv_mkl = next(iter(sorted(_venv_mkl_dir.glob("mkl_rt*.dll"))), None)
if _venv_mkl is not None:
    os.environ.setdefault("EMERGE_PARDISO_PATH", str(_venv_mkl))

from .config import (
    ABC_TYPES,
    OPEN_REGION_MODES,
    SOLVER_CHOICES,
    AntennaDesign,
    CoilDesign,
    FrequencySweep,
    MeshSettings,
    OpenRegionSettings,
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
    ConfirmationCallback,
    DesignSpace,
    DesignVariable,
    EvaluationCallback,
    EvaluationRecord,
    GainMatchObjective,
    IncumbentConfirmation,
    RobustGainObjective,
    S11Objective,
)
from .presets import (
    REFERENCE_DESIGN_FREQUENCY_HZ,
    REFERENCE_DESIGN_PATH,
    load_reference_design,
    scale_design,
)
from .serialization import design_from_dict, load_design, save_design
from .simulation import (
    AzimuthRingMetrics,
    FarFieldMetrics,
    ModelArtifacts,
    SimulationResult,
    build_model,
    simulate,
)

__all__ = [
    "ABC_TYPES",
    "CONVERGENCE_SCHEMA_VERSION",
    "OPEN_REGION_MODES",
    "REFERENCE_DESIGN_FREQUENCY_HZ",
    "REFERENCE_DESIGN_PATH",
    "SOLVER_CHOICES",
    "AntennaDesign",
    "AntennaPath",
    "AzimuthRingMetrics",
    "CoilDesign",
    "CompositeCurve",
    "ConfirmationCallback",
    "DesignSpace",
    "DesignVariable",
    "EvaluationCallback",
    "EvaluationRecord",
    "FarFieldMetrics",
    "FrequencySweep",
    "GainMatchObjective",
    "IncumbentConfirmation",
    "MeshSettings",
    "ModelArtifacts",
    "OpenRegionSettings",
    "RobustGainObjective",
    "S11Objective",
    "SimulationOptions",
    "SimulationResult",
    "build_centerline",
    "build_model",
    "design_fingerprint",
    "design_from_dict",
    "load_convergence_certificate",
    "load_design",
    "load_reference_design",
    "save_design",
    "scale_design",
    "selected_open_region_configuration",
    "simulate",
    "validate_convergence_certificate",
]
