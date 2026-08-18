"""Parameterized loaded-antenna simulation library."""

import os
from pathlib import Path

_venv_mkl_dir = Path(__file__).resolve().parents[1]/".venv"/"Library"/"bin"
_venv_mkl = next(iter(sorted(_venv_mkl_dir.glob("mkl_rt*.dll"))), None)
if _venv_mkl is not None:
    os.environ.setdefault("EMERGE_PARDISO_PATH", str(_venv_mkl))

from .config import (
    AntennaDesign,
    CoilDesign,
    FrequencySweep,
    MeshSettings,
    SimulationOptions,
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
    "AntennaPath",
    "CoilDesign",
    "CompositeCurve",
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
    "S11Objective",
    "SimulationOptions",
    "SimulationResult",
    "build_centerline",
    "build_model",
    "design_from_dict",
    "load_design",
    "save_design",
    "simulate",
]
