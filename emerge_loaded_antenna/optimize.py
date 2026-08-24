"""Adapters that make antenna designs convenient optimizer inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from .config import AntennaDesign, FrequencySweep, SimulationOptions
from .simulation import simulate


@dataclass(frozen=True)
class DesignVariable:
    """One bounded field in an :class:`AntennaDesign`.

    Nested fields and tuple indexes use dotted paths, e.g.
    ``"coils.0.pitch"`` and ``"straight_lengths.1"``. ``linked_paths``
    applies the same optimizer value to additional fields, while ``label``
    gives that shared variable a meaningful output name.
    """

    path: str
    lower: float
    upper: float
    kind: Literal["float", "int"] = "float"
    linked_paths: tuple[str, ...] = ()
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "linked_paths", tuple(self.linked_paths))
        if not self.path:
            raise ValueError("variable path cannot be empty")
        if any(not path for path in self.linked_paths):
            raise ValueError("linked variable paths cannot be empty")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("linked variable paths must be unique")
        if self.label is not None and not self.label:
            raise ValueError("variable label cannot be empty")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("variable bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError("variable lower bound must be smaller than upper")
        if self.kind not in {"float", "int"}:
            raise ValueError("variable kind must be 'float' or 'int'")

    def coerce(self, value: float) -> float | int:
        value = float(np.clip(value, self.lower, self.upper))
        return int(round(value)) if self.kind == "int" else value

    @property
    def name(self) -> str:
        return self.label or self.path

    @property
    def paths(self) -> tuple[str, ...]:
        return (self.path, *self.linked_paths)


def _tuple_index(instance: tuple, part: str) -> int:
    try:
        index = int(part)
    except ValueError as error:
        raise ValueError(f"tuple optimizer field {part!r} is not an index") from error
    if index < 0 or index >= len(instance):
        raise ValueError(f"tuple optimizer index {index} is out of range")
    return index


def _get_nested(instance, path: Sequence[str]):
    head, *tail = path
    if isinstance(instance, tuple):
        current = instance[_tuple_index(instance, head)]
    else:
        if not hasattr(instance, head):
            raise ValueError(
                f"{type(instance).__name__} has no optimizer field {head!r}"
            )
        current = getattr(instance, head)
    return _get_nested(current, tail) if tail else current


def _replace_nested(instance, path: Sequence[str], value):
    head, *tail = path
    if isinstance(instance, tuple):
        index = _tuple_index(instance, head)
        items = list(instance)
        items[index] = (
            _replace_nested(items[index], tail, value) if tail else value
        )
        return tuple(items)
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
        if len({variable.name for variable in self.variables}) != len(self.variables):
            raise ValueError("design variable names must be unique")
        paths = [path for variable in self.variables for path in variable.paths]
        if len(set(paths)) != len(paths):
            raise ValueError("design variable paths must be unique")
        # Exercise every path immediately so typos fail before an expensive run.
        self.decode(self.initial_vector)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((variable.lower, variable.upper) for variable in self.variables)

    @property
    def initial_vector(self) -> np.ndarray:
        values = []
        for variable in self.variables:
            current = _get_nested(self.base, variable.path.split("."))
            values.append(float(current))
        return np.asarray(values, dtype=float)

    def decode(self, vector: Sequence[float]) -> AntennaDesign:
        if len(vector) != len(self.variables):
            raise ValueError(
                f"expected {len(self.variables)} values, got {len(vector)}"
            )
        design = self.base
        for variable, value in zip(self.variables, vector):
            coerced = variable.coerce(float(value))
            for path in variable.paths:
                design = _replace_nested(
                    design,
                    path.split("."),
                    coerced,
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
    metrics: Mapping[str, float] = field(default_factory=dict)
    confirmation_status: Literal[
        "not_requested",
        "not_needed",
        "confirmed",
        "confirmed_with_outliers",
        "quarantined",
    ] = "not_requested"
    simulation_runs: int = 1


EvaluationCallback = Callable[[EvaluationRecord], None]


@dataclass(frozen=True)
class IncumbentConfirmation:
    """Details from repeating a candidate that appeared to be a new best.

    Confirmation runs are deliberately kept out of the normal evaluation
    history and progress callback.  This report exposes their raw records for
    diagnostics without making one optimizer evaluation look like several.
    """

    vector: tuple[float, ...]
    status: Literal[
        "confirmed",
        "confirmed_with_outliers",
        "quarantined",
    ]
    requested_runs: int
    successful_runs: int
    preliminary_score: float
    confirmed_score: float | None
    score_spread: float | None
    consensus_runs: int
    outlier_runs: int
    incumbent_score: float | None
    records: tuple[EvaluationRecord, ...]
    reason: str | None = None


ConfirmationCallback = Callable[[IncumbentConfirmation], None]


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


class RobustGainObjective:
    """Optimize useful gain with broadband match and pattern constraints.

    ``pattern_mode="horizon"`` maximizes the 10th-percentile azimuth gain at
    zero elevation. ``"directional"`` maximizes one requested theta/phi
    direction and can target the HPBW of both orthogonal cuts through it,
    while ``"peak"`` retains the original unconstrained behavior.
    """

    def __init__(
        self,
        space: DesignSpace,
        target_frequency: float = 868e6,
        pattern_mode: Literal["horizon", "directional", "peak"] = "horizon",
        target_theta_deg: float = 90.0,
        target_phi_deg: float = 0.0,
        target_beamwidth_deg: float | None = None,
        beamwidth_weight: float = 1.0,
        maximum_s11_db: float = -10.0,
        mismatch_weight: float = 2.0,
        s11_margin_target_db: float | None = None,
        s11_margin_weight: float = 0.0,
        gain_weight: float = 1.0,
        maximum_horizon_ripple_db: float = 1.5,
        ripple_weight: float = 0.15,
        minimum_horizon_gain_dbi: float = 2.0,
        null_weight: float = 0.25,
        maximum_height: float = 0.60,
        height_weight: float = 0.10,
        options: SimulationOptions | None = None,
        failure_penalty: float = 1_000.0,
        on_evaluation: EvaluationCallback | None = None,
        confirmation_runs: int = 1,
        confirmation_score_tolerance: float = 1.0,
        on_confirmation: ConfirmationCallback | None = None,
    ):
        if pattern_mode not in {"horizon", "directional", "peak"}:
            raise ValueError("invalid pattern_mode")
        if not 0 <= target_theta_deg <= 180:
            raise ValueError("target_theta_deg must be between zero and 180")
        if not np.isfinite(target_phi_deg):
            raise ValueError("target_phi_deg must be finite")
        if target_beamwidth_deg is not None:
            if pattern_mode != "directional":
                raise ValueError(
                    "target_beamwidth_deg requires directional pattern mode"
                )
            if (
                not np.isfinite(target_beamwidth_deg)
                or not 0 < target_beamwidth_deg <= 360
            ):
                raise ValueError(
                    "target_beamwidth_deg must be finite and between 0 and 360"
                )
        if maximum_height <= 0:
            raise ValueError("maximum_height must be positive")
        weights = (
            mismatch_weight,
            s11_margin_weight,
            gain_weight,
            beamwidth_weight,
            ripple_weight,
            null_weight,
            height_weight,
        )
        if any(not np.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("objective weights must be finite and non-negative")
        if maximum_horizon_ripple_db < 0:
            raise ValueError("maximum_horizon_ripple_db must be non-negative")
        if not np.isfinite(maximum_s11_db):
            raise ValueError("maximum_s11_db must be finite")
        if s11_margin_target_db is not None:
            if not np.isfinite(s11_margin_target_db):
                raise ValueError("s11_margin_target_db must be finite")
            if s11_margin_target_db >= maximum_s11_db:
                raise ValueError(
                    "s11_margin_target_db must be below maximum_s11_db"
                )
        if (
            not isinstance(confirmation_runs, (int, np.integer))
            or isinstance(confirmation_runs, bool)
            or confirmation_runs < 1
            or confirmation_runs % 2 == 0
        ):
            raise ValueError("confirmation_runs must be a positive odd integer")
        if (
            not np.isfinite(confirmation_score_tolerance)
            or confirmation_score_tolerance < 0
        ):
            raise ValueError(
                "confirmation_score_tolerance must be finite and non-negative"
            )
        self.space = space
        self.target_frequency = target_frequency
        self.pattern_mode = pattern_mode
        self.target_theta_deg = target_theta_deg
        self.target_phi_deg = target_phi_deg
        self.target_beamwidth_deg = target_beamwidth_deg
        self.beamwidth_weight = beamwidth_weight
        self.maximum_s11_db = maximum_s11_db
        self.mismatch_weight = mismatch_weight
        self.s11_margin_target_db = s11_margin_target_db
        self.s11_margin_weight = s11_margin_weight
        self.gain_weight = gain_weight
        self.maximum_horizon_ripple_db = maximum_horizon_ripple_db
        self.ripple_weight = ripple_weight
        self.minimum_horizon_gain_dbi = minimum_horizon_gain_dbi
        self.null_weight = null_weight
        self.maximum_height = maximum_height
        self.height_weight = height_weight
        self.options = options or SimulationOptions(
            sweep=FrequencySweep(center=target_frequency, span=10e6, points=3),
            solve=True,
            compute_farfield=True,
            farfield_frequency=target_frequency,
            farfield_angular_step_deg=2.0,
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
        self.confirmation_runs = int(confirmation_runs)
        self.confirmation_score_tolerance = float(
            confirmation_score_tolerance
        )
        self.on_confirmation = on_confirmation
        self.history: list[EvaluationRecord] = []
        self.confirmation_history: list[IncumbentConfirmation] = []
        self.simulation_evaluations = 0

    @property
    def best_record(self) -> EvaluationRecord | None:
        return _best_record(self.history)

    def _is_feasible(self, record: EvaluationRecord) -> bool:
        if record.error is not None or not np.isfinite(record.score):
            return False
        worst_s11 = record.metrics.get("worst_s11_db")
        if (
            worst_s11 is None
            or not np.isfinite(worst_s11)
            or worst_s11 > self.maximum_s11_db
        ):
            return False
        return all(
            record.metrics.get(name, float("inf")) <= 0.0
            for name in (
                "mismatch_penalty",
                "pattern_penalty",
                "height_penalty",
            )
        )

    @property
    def best_feasible_record(self) -> EvaluationRecord | None:
        return min(
            (record for record in self.history if self._is_feasible(record)),
            key=lambda record: record.score,
            default=None,
        )

    def _evaluate_once(self, values: tuple[float, ...]) -> EvaluationRecord:
        self.simulation_evaluations += 1
        try:
            design = self.space.decode(values)
            result = simulate(design, self.options)
            pattern = result.farfield_metrics
            if pattern is None or result.peak_gain_dbi is None:
                raise RuntimeError("far-field metrics were not computed")

            s11_center = result.s11_db_at(self.target_frequency)
            worst_s11 = float(np.max(result.s11_db))
            mismatch = np.maximum(0.0, result.s11_db - self.maximum_s11_db)
            mismatch_penalty = self.mismatch_weight*float(np.mean(mismatch**2))
            beamwidth_penalty = 0.0
            s11_margin_reward = 0.0
            s11_margin_db = 0.0
            if self.s11_margin_target_db is not None:
                available_margin = (
                    self.maximum_s11_db - self.s11_margin_target_db
                )
                s11_margin_db = float(np.clip(
                    self.maximum_s11_db - worst_s11,
                    0.0,
                    available_margin,
                ))
                s11_margin_reward = self.s11_margin_weight*s11_margin_db

            if self.pattern_mode == "horizon":
                useful_gain = pattern.horizon_p10_gain_dbi
                ripple_excess = max(
                    0.0,
                    pattern.horizon_ripple_p90_p10_db
                    - self.maximum_horizon_ripple_db,
                )
                null_deficit = max(
                    0.0,
                    self.minimum_horizon_gain_dbi
                    - pattern.horizon_min_gain_dbi,
                )
                pattern_penalty = (
                    self.ripple_weight*ripple_excess**2
                    + self.null_weight*null_deficit**2
                )
            elif self.pattern_mode == "directional":
                useful_gain = result.gain_db_at(
                    self.target_theta_deg,
                    self.target_phi_deg,
                )
                pattern_penalty = 0.0
                if self.target_beamwidth_deg is not None:
                    elevation_width, azimuth_width = (
                        result.directional_beamwidths_deg(
                            self.target_theta_deg,
                            self.target_phi_deg,
                        )
                    )
                    beamwidth_errors = np.asarray(
                        (
                            elevation_width - self.target_beamwidth_deg,
                            azimuth_width - self.target_beamwidth_deg,
                        ),
                        dtype=float,
                    )
                    beamwidth_error = float(
                        np.sqrt(np.mean(beamwidth_errors**2))
                    )
                    beamwidth_penalty = self.beamwidth_weight*float(
                        np.mean((beamwidth_errors/10.0)**2)
                    )
            else:
                useful_gain = result.peak_gain_dbi
                pattern_penalty = 0.0

            height_excess_cm = max(
                0.0,
                (result.antenna_height - self.maximum_height)/0.01,
            )
            height_penalty = self.height_weight*height_excess_cm**2
            score = (
                mismatch_penalty
                + pattern_penalty
                + beamwidth_penalty
                + height_penalty
                - s11_margin_reward
                - self.gain_weight*useful_gain
            )
            if not np.isfinite(score):
                raise RuntimeError("objective score was not finite")
            metrics = {
                "s11_low_db": float(result.s11_db[0]),
                "center_s11_db": s11_center,
                "s11_high_db": float(result.s11_db[-1]),
                "worst_s11_db": worst_s11,
                "useful_gain_dbi": float(useful_gain),
                "peak_gain_dbi": result.peak_gain_dbi,
                "peak_theta_deg": pattern.peak_theta_deg,
                "peak_phi_deg": pattern.peak_phi_deg,
                "horizon_min_gain_dbi": pattern.horizon_min_gain_dbi,
                "horizon_p10_gain_dbi": pattern.horizon_p10_gain_dbi,
                "horizon_mean_gain_dbi": pattern.horizon_mean_gain_dbi,
                "horizon_ripple_db": pattern.horizon_ripple_p90_p10_db,
                "antenna_height_m": result.antenna_height,
                "mismatch_penalty": mismatch_penalty,
                "s11_margin_db": s11_margin_db,
                "s11_margin_reward": s11_margin_reward,
                "pattern_penalty": pattern_penalty,
                "beamwidth_penalty": beamwidth_penalty,
                "height_penalty": height_penalty,
            }
            if self.pattern_mode == "directional":
                metrics.update(
                    target_theta_deg=self.target_theta_deg,
                    target_phi_deg=self.target_phi_deg,
                    target_gain_dbi=float(useful_gain),
                )
            if self.target_beamwidth_deg is not None:
                metrics.update(
                    target_beamwidth_deg=self.target_beamwidth_deg,
                    elevation_beamwidth_deg=elevation_width,
                    azimuth_beamwidth_deg=azimuth_width,
                    beamwidth_error_deg=beamwidth_error,
                )
            if self.s11_margin_target_db is not None:
                metrics["s11_margin_target_db"] = self.s11_margin_target_db
            metrics.update(
                {
                    f"s11_{frequency/1e6:g}_mhz_db": float(s11_value)
                    for frequency, s11_value in zip(
                        result.frequencies,
                        result.s11_db,
                    )
                }
            )
            record = EvaluationRecord(
                values,
                float(score),
                s11_center,
                result.peak_gain_dbi,
                metrics=metrics,
            )
        except Exception as error:
            record = EvaluationRecord(
                values,
                self.failure_penalty,
                None,
                None,
                f"{type(error).__name__}: {error}",
            )
        return record

    @staticmethod
    def _representative_record(
        records: Sequence[EvaluationRecord],
    ) -> EvaluationRecord:
        """Return the real repeat at the conservative median score."""

        if not records:
            raise ValueError("cannot aggregate an empty confirmation")
        ordered = sorted(records, key=lambda record: record.score)
        # The upper middle is deliberately conservative if failed runs leave
        # an even number of usable repeats.
        return ordered[len(ordered)//2]

    def _confirm_incumbent(
        self,
        preliminary: EvaluationRecord,
        incumbent: EvaluationRecord | None,
    ) -> EvaluationRecord:
        records = [preliminary]
        records.extend(
            self._evaluate_once(preliminary.vector)
            for _ in range(self.confirmation_runs - 1)
        )
        successful = [record for record in records if record.error is None]
        scores = np.asarray([record.score for record in successful], dtype=float)
        combined = self._representative_record(successful)
        confirmed_score = combined.score
        score_spread = float(np.ptp(scores)) if scores.size else None
        failed_runs = len(records) - len(successful)
        consensus_runs = int(
            np.count_nonzero(
                np.abs(scores - confirmed_score)
                <= self.confirmation_score_tolerance
            )
        )
        outlier_runs = len(successful) - consensus_runs
        required_consensus = self.confirmation_runs//2 + 1

        reason = None
        if failed_runs:
            reason = (
                f"{failed_runs} of {self.confirmation_runs} confirmation "
                "simulations failed"
            )
        elif consensus_runs < required_consensus:
            reason = (
                f"only {consensus_runs} of {self.confirmation_runs} scores "
                "agreed with the median within tolerance "
                f"{self.confirmation_score_tolerance:.6g}"
            )

        confirmation_metrics = {
            "confirmation_requested_runs": float(self.confirmation_runs),
            "confirmation_successful_runs": float(len(successful)),
            "confirmation_preliminary_score": preliminary.score,
            "confirmation_confirmed_score": float(confirmed_score),
            "confirmation_score_min": float(np.min(scores)),
            "confirmation_score_max": float(np.max(scores)),
            "confirmation_score_spread": float(score_spread),
            "confirmation_score_tolerance": self.confirmation_score_tolerance,
            "confirmation_consensus_runs": float(consensus_runs),
            "confirmation_outlier_runs": float(outlier_runs),
            "confirmation_consistent": float(reason is None),
            "confirmation_quarantined": float(reason is not None),
        }
        if incumbent is not None:
            confirmation_metrics["confirmation_incumbent_score"] = incumbent.score

        if reason is None:
            status: Literal[
                "confirmed",
                "confirmed_with_outliers",
                "quarantined",
            ] = (
                "confirmed_with_outliers" if outlier_runs else "confirmed"
            )
            record = replace(
                combined,
                metrics={**combined.metrics, **confirmation_metrics},
                confirmation_status=status,
                simulation_runs=self.confirmation_runs,
            )
        else:
            quarantine_score = self.failure_penalty
            if incumbent is not None:
                quarantine_score = max(
                    quarantine_score,
                    float(np.nextafter(incumbent.score, np.inf)),
                )
            confirmation_metrics["confirmation_returned_score"] = quarantine_score
            record = replace(
                combined,
                score=quarantine_score,
                error=f"ConfirmationError: quarantined candidate; {reason}",
                metrics={**combined.metrics, **confirmation_metrics},
                confirmation_status="quarantined",
                simulation_runs=self.confirmation_runs,
            )
            status = "quarantined"

        report = IncumbentConfirmation(
            vector=preliminary.vector,
            status=status,
            requested_runs=self.confirmation_runs,
            successful_runs=len(successful),
            preliminary_score=preliminary.score,
            confirmed_score=confirmed_score,
            score_spread=score_spread,
            consensus_runs=consensus_runs,
            outlier_runs=outlier_runs,
            incumbent_score=None if incumbent is None else incumbent.score,
            records=tuple(records),
            reason=reason,
        )
        self.confirmation_history.append(report)
        if self.on_confirmation is not None:
            self.on_confirmation(report)
        return record

    def __call__(self, vector: Sequence[float]) -> float:
        values = tuple(float(value) for value in vector)
        record = self._evaluate_once(values)
        incumbent = self.best_record
        feasible_incumbent = self.best_feasible_record
        if self.confirmation_runs > 1 and record.error is None:
            is_new_overall = incumbent is None or record.score < incumbent.score
            is_new_feasible = self._is_feasible(record) and (
                feasible_incumbent is None
                or record.score < feasible_incumbent.score
            )
            if is_new_overall or is_new_feasible:
                record = self._confirm_incumbent(record, incumbent)
            else:
                record = replace(record, confirmation_status="not_needed")
        self.history.append(record)
        if self.on_evaluation is not None:
            self.on_evaluation(record)
        return float(record.score)
