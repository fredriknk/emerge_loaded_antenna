"""Adapters that make antenna designs convenient optimizer inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal, Sequence

import numpy as np

from .config import AntennaDesign, FrequencySweep, SimulationOptions
from .simulation import simulate


@dataclass(frozen=True)
class DesignVariable:
    """One bounded field in an :class:`AntennaDesign`.

    Nested dataclass fields use dotted paths, e.g. ``"coil1.pitch"``.
    """

    path: str
    lower: float
    upper: float
    kind: Literal["float", "int"] = "float"

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("variable path cannot be empty")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("variable bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError("variable lower bound must be smaller than upper")
        if self.kind not in {"float", "int"}:
            raise ValueError("variable kind must be 'float' or 'int'")

    def coerce(self, value: float) -> float | int:
        value = float(np.clip(value, self.lower, self.upper))
        return int(round(value)) if self.kind == "int" else value


def _replace_nested(instance, path: Sequence[str], value):
    head, *tail = path
    if not hasattr(instance, head):
        raise ValueError(
            f"{type(instance).__name__} has no optimizer field {head!r}"
        )
    replacement = (
        _replace_nested(getattr(instance, head), tail, value)
        if tail
        else value
    )
    return replace(instance, **{head: replacement})


@dataclass(frozen=True)
class DesignSpace:
    """Map optimizer vectors to immutable antenna designs."""

    base: AntennaDesign
    variables: tuple[DesignVariable, ...]

    def __init__(
        self,
        base: AntennaDesign,
        variables: Sequence[DesignVariable],
    ):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "variables", tuple(variables))
        self.base.validate()
        if not self.variables:
            raise ValueError("design space needs at least one variable")
        if len({variable.path for variable in self.variables}) != len(self.variables):
            raise ValueError("design variable paths must be unique")
        # Exercise every path immediately so typos fail before an expensive run.
        self.decode(self.initial_vector)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(variable.path for variable in self.variables)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((variable.lower, variable.upper) for variable in self.variables)

    @property
    def initial_vector(self) -> np.ndarray:
        values = []
        for variable in self.variables:
            current = self.base
            for part in variable.path.split("."):
                if not hasattr(current, part):
                    raise ValueError(
                        f"invalid design variable path {variable.path!r}"
                    )
                current = getattr(current, part)
            values.append(float(current))
        return np.asarray(values, dtype=float)

    def decode(self, vector: Sequence[float]) -> AntennaDesign:
        if len(vector) != len(self.variables):
            raise ValueError(
                f"expected {len(self.variables)} values, got {len(vector)}"
            )
        design = self.base
        for variable, value in zip(self.variables, vector):
            design = _replace_nested(
                design,
                variable.path.split("."),
                variable.coerce(float(value)),
            )
        design.validate()
        return design

    def normalize(self, vector: Sequence[float]) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        lower = np.asarray([item.lower for item in self.variables])
        upper = np.asarray([item.upper for item in self.variables])
        return (values - lower)/(upper - lower)

    def denormalize(self, normalized: Sequence[float]) -> np.ndarray:
        values = np.clip(np.asarray(normalized, dtype=float), 0.0, 1.0)
        lower = np.asarray([item.lower for item in self.variables])
        upper = np.asarray([item.upper for item in self.variables])
        return lower + values*(upper - lower)


@dataclass(frozen=True)
class EvaluationRecord:
    vector: tuple[float, ...]
    score: float
    s11_db: float | None
    peak_gain_dbi: float | None
    error: str | None = None


EvaluationCallback = Callable[[EvaluationRecord], None]


def _best_record(history: Sequence[EvaluationRecord]) -> EvaluationRecord | None:
    successful = [record for record in history if record.error is None]
    return min(successful, key=lambda record: record.score, default=None)


class S11Objective:
    """Callable scalar objective for minimizers; more-negative S11 is better."""

    def __init__(
        self,
        space: DesignSpace,
        target_frequency: float = 868e6,
        options: SimulationOptions | None = None,
        failure_penalty: float = 100.0,
        on_evaluation: EvaluationCallback | None = None,
    ):
        self.space = space
        self.target_frequency = target_frequency
        self.options = options or SimulationOptions(
            sweep=FrequencySweep.single(target_frequency),
            solve=True,
            compute_farfield=False,
            verbose=False,
        )
        self.options = replace(
            self.options,
            solve=True,
            compute_farfield=False,
            show_geometry=False,
            show_mesh=False,
            show_coil_preview=False,
        )
        self.failure_penalty = float(failure_penalty)
        self.on_evaluation = on_evaluation
        self.history: list[EvaluationRecord] = []

    @property
    def best_record(self) -> EvaluationRecord | None:
        return _best_record(self.history)

    def __call__(self, vector: Sequence[float]) -> float:
        values = tuple(float(value) for value in vector)
        try:
            design = self.space.decode(values)
            result = simulate(design, self.options)
            score = result.s11_db_at(self.target_frequency)
            record = EvaluationRecord(values, score, score, None)
        except Exception as error:
            score = self.failure_penalty
            record = EvaluationRecord(
                values,
                score,
                None,
                None,
                f"{type(error).__name__}: {error}",
            )
        self.history.append(record)
        if self.on_evaluation is not None:
            self.on_evaluation(record)
        return float(score)

class GainMatchObjective:
    """Minimize mismatch penalty while maximizing peak isotropic gain."""

    def __init__(
        self,
        space: DesignSpace,
        target_frequency: float = 868e6,
        maximum_s11_db: float = -10.0,
        mismatch_weight: float = 1.0,
        gain_weight: float = 1.0,
        options: SimulationOptions | None = None,
        failure_penalty: float = 1_000.0,
        on_evaluation: EvaluationCallback | None = None,
    ):
        self.space = space
        self.target_frequency = target_frequency
        self.maximum_s11_db = maximum_s11_db
        self.mismatch_weight = mismatch_weight
        self.gain_weight = gain_weight
        self.options = options or SimulationOptions(
            sweep=FrequencySweep.single(target_frequency),
            solve=True,
            compute_farfield=True,
            farfield_frequency=target_frequency,
            verbose=False,
        )
        self.options = replace(
            self.options,
            solve=True,
            compute_farfield=True,
            farfield_frequency=target_frequency,
            show_geometry=False,
            show_mesh=False,
            show_coil_preview=False,
        )
        self.failure_penalty = float(failure_penalty)
        self.on_evaluation = on_evaluation
        self.history: list[EvaluationRecord] = []

    @property
    def best_record(self) -> EvaluationRecord | None:
        return _best_record(self.history)

    def __call__(self, vector: Sequence[float]) -> float:
        values = tuple(float(value) for value in vector)
        try:
            design = self.space.decode(values)
            result = simulate(design, self.options)
            s11_db = result.s11_db_at(self.target_frequency)
            if result.peak_gain_dbi is None:
                raise RuntimeError("far-field gain was not computed")
            mismatch = max(0.0, s11_db - self.maximum_s11_db)
            score = (
                self.mismatch_weight*mismatch**2
                - self.gain_weight*result.peak_gain_dbi
            )
            record = EvaluationRecord(
                values,
                float(score),
                s11_db,
                result.peak_gain_dbi,
            )
        except Exception as error:
            score = self.failure_penalty
            record = EvaluationRecord(
                values,
                score,
                None,
                None,
                f"{type(error).__name__}: {error}",
            )
        self.history.append(record)
        if self.on_evaluation is not None:
            self.on_evaluation(record)
        return float(score)
