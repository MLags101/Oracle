"""Validation mode: did the recommendation do what it said (spec section 5.10).

Every other screen in this tool argues from a model. This one argues from two
flights. The user loaded a parameter file, flew it, and now has a second log --
and the only question that matters is whether the aircraft did what the tool said
it would.

That makes prediction-versus-outcome the single most useful trust signal
available, and it is why this is a first-class screen rather than a footnote. A
tool that predicts a 90 ms rise and then measures 92 ms has earned something no
amount of coherence plots can buy. A tool that predicted 90 and measured 210 has
to say so, loudly, because everything else it says rests on the same model.

Three comparisons, and they answer different questions:

* **Outcome.** Before against after, on quantities read straight off each log --
  tracking error, step shape, D-term noise. Says whether the aircraft got better.
* **Prediction.** The earlier session's *predicted* closed-loop step against the
  new log's *measured* one. Says whether the model was right.
* **Filters.** The earlier session's predicted post-filter spectrum against the
  new log's measured post-filter spectrum. Says whether the filter design did
  what it claimed, which is the half of the recommendation that usually goes
  unchecked.

Nothing here identifies anything. Validation must work on an after-log with no
usable excitation in it at all -- which is most after-logs, because the point of
the flight was to fly the new tune rather than to sweep it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from rotorid.config import Config
from rotorid.core.analysis.deconv import measured_step
from rotorid.core.analysis.noise import measured_dterm_rms_pct, noise_profile
from rotorid.core.preprocess.params import chain_from_bundle, gains_from_bundle
from rotorid.core.types import (
    AXES,
    Axis,
    FloatArray,
    GainSet,
    LogBundle,
    MeasuredStep,
    NoiseProfile,
    Session,
    StepMetrics,
)

__all__ = ["AxisComparison", "ValidationReport", "compare_logs"]

#: How closely a flown gain has to match a recommended one before the after-log
#: counts as flying that recommendation. Ground stations round, and a user who
#: typed 0.152 for a recommended 0.1518 has applied it.
_APPLIED_TOLERANCE = 0.02

#: Rise-time agreement band, as a ratio of measured to predicted. Wider than it
#: looks reasonable to be, and deliberately: the deconvolution's regularizer
#: low-passes the recovered response, so a measurement reads slow even when the
#: model is right. This catches a prediction wrong by a factor, not by a fifth.
_RISE_RATIO_BOUNDS = (0.6, 1.7)

#: Overshoot agreement, in percentage points of the step.
_MAX_OVERSHOOT_DIFF_PCT = 12.0

#: How much a summary number has to move before the change is worth reporting as
#: a change rather than as noise between two flights of the same aircraft.
_MATERIAL_CHANGE = 0.10

#: Band over which the filter prediction is scored. Below the first, the spectrum
#: is aircraft motion rather than noise; above the second, both logs are mostly
#: describing their own anti-aliasing.
_FILTER_SCORE_BAND_HZ = (20.0, 350.0)


@dataclass(frozen=True, slots=True)
class AxisComparison:
    """Before and after on one axis, plus what was predicted for it.

    Every field is optional on purpose. A validation flight that carries no PID
    messages still has a step in it; a before-log analysed without a session
    still has a measurement even though it has no prediction. Refusing the whole
    comparison because one quantity is missing would make the screen unusable on
    exactly the logs people have.

    Attributes:
        predicted_step: What the earlier analysis said the *recommended* tune
            would do. Compared against ``after_step``, never against
            ``before_step`` -- the before log was flown on the old gains, so
            agreement there would be a coincidence and disagreement would be
            correct behaviour.
        applied: Whether the after-log's flown gains match what was recommended.
            Checked first, because a prediction compared against a flight that
            never loaded the parameters is not a failed prediction.
    """

    axis: Axis
    before_step: MeasuredStep | None = None
    after_step: MeasuredStep | None = None
    predicted_step: StepMetrics | None = None
    before_noise: NoiseProfile | None = None
    after_noise: NoiseProfile | None = None
    predicted_psd_f_hz: FloatArray | None = None
    predicted_psd_post: FloatArray | None = None
    before_dterm_pct: float | None = None
    after_dterm_pct: float | None = None
    before_tracking_rms: float | None = None
    after_tracking_rms: float | None = None
    before_gains: GainSet | None = None
    after_gains: GainSet | None = None
    recommended_gains: GainSet | None = None
    applied: bool | None = None

    @property
    def rise_ratio(self) -> float | None:
        """Measured-after rise time over predicted rise time, or ``None``."""
        if self.after_step is None or self.predicted_step is None:
            return None
        predicted = self.predicted_step.rise_time_s
        measured = self.after_step.metrics.rise_time_s
        if predicted <= 0.0 or not np.isfinite(measured):
            return None
        return measured / predicted

    @property
    def prediction_holds(self) -> bool | None:
        """Whether the flown step matches what was predicted for the new tune."""
        ratio = self.rise_ratio
        if ratio is None or self.after_step is None or self.predicted_step is None:
            return None
        low, high = _RISE_RATIO_BOUNDS
        gap = abs(self.after_step.metrics.overshoot_pct - self.predicted_step.overshoot_pct)
        return low <= ratio <= high and gap <= _MAX_OVERSHOOT_DIFF_PCT

    @property
    def tracking_change(self) -> float | None:
        """Fractional change in tracking-error RMS. Negative is better."""
        return _fractional_change(self.before_tracking_rms, self.after_tracking_rms)

    @property
    def dterm_change(self) -> float | None:
        """Fractional change in D-term noise. Negative is better."""
        return _fractional_change(self.before_dterm_pct, self.after_dterm_pct)

    @property
    def filter_prediction_error_db(self) -> float | None:
        """Median error of the predicted post-filter spectrum, in dB.

        Signed: positive means the aircraft is noisier than the filter design
        promised, which is the direction that matters -- an over-delivering
        filter costs phase nobody budgeted for, but an under-delivering one puts
        noise into the motors.
        """
        if self.predicted_psd_f_hz is None or self.predicted_psd_post is None:
            return None
        if self.after_noise is None:
            return None
        return _spectrum_error_db(
            self.predicted_psd_f_hz,
            self.predicted_psd_post,
            self.after_noise.f_hz,
            self.after_noise.psd_post,
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """One before/after pair, compared.

    Attributes:
        axes: Only the axes both logs could say something about. An axis in one
            log and not the other is dropped rather than half-reported.
        predicted_from: Where the predictions came from -- a saved session's
            file name, or ``None`` when the comparison is outcome-only. The
            distinction is the difference between "the tune got better" and "the
            tool was right", and the report must never blur them.
    """

    before: LogBundle
    after: LogBundle
    axes: dict[Axis, AxisComparison]
    tool_version: str
    created_utc: datetime
    predicted_from: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_predictions(self) -> bool:
        """Whether anything in this comparison tests the model rather than the tune."""
        return any(c.predicted_step is not None for c in self.axes.values())


def compare_logs(
    before: LogBundle,
    after: LogBundle,
    config: Config,
    *,
    tool_version: str,
    session: Session | None = None,
    axes: tuple[Axis, ...] = AXES,
) -> ValidationReport:
    """Compare two flights of the same aircraft, before and after a change.

    Args:
        session: The analysis that produced the change, if it was saved. Without
            it the report can still say whether the aircraft improved, but not
            whether the tool was right about why -- so passing one is what turns
            an outcome comparison into a validation.

    Raises:
        ValueError: if the two logs came from different firmware stacks. Every
            parameter name, gain convention and filter structure differs between
            them, so a cross-stack "before and after" is comparing two aircraft.
    """
    if before.stack != after.stack:
        raise ValueError(
            f"cannot compare a {before.stack} log against a {after.stack} one: "
            "the gains, the filters and the parameter names are all different quantities"
        )

    notes: list[str] = []
    comparisons: dict[Axis, AxisComparison] = {}
    for axis in axes:
        comparison = _compare_axis(before, after, axis, config, session)
        if comparison is None:
            notes.append(f"{axis}: not present in both logs, so it is not compared")
            continue
        comparisons[axis] = comparison

    if session is not None and session.log.stack != after.stack:
        raise ValueError(
            f"the saved session is for a {session.log.stack} vehicle and the logs are {after.stack}"
        )

    return ValidationReport(
        before=before,
        after=after,
        axes=comparisons,
        tool_version=tool_version,
        created_utc=datetime.now(UTC),
        predicted_from=session.log.path.name if session is not None else None,
        notes=tuple(notes),
    )


def _compare_axis(
    before: LogBundle,
    after: LogBundle,
    axis: Axis,
    config: Config,
    session: Session | None,
) -> AxisComparison | None:
    """Everything the two logs can say about one axis, side by side."""
    key = f"rate.{axis}.measured"
    if key not in before.signals or key not in after.signals:
        return None

    recommendation = (session.recommendations.get(axis) if session is not None else None) or None
    before_gains = _gains(before, axis)
    after_gains = _gains(after, axis)

    return AxisComparison(
        axis=axis,
        before_step=measured_step(before, axis, config),
        after_step=measured_step(after, axis, config),
        predicted_step=recommendation.predicted_step if recommendation else None,
        before_noise=_noise(before, axis, config),
        after_noise=_noise(after, axis, config),
        predicted_psd_f_hz=recommendation.filters.psd_f_hz if recommendation else None,
        predicted_psd_post=recommendation.filters.predicted_psd_post if recommendation else None,
        before_dterm_pct=_dterm(before, axis, config),
        after_dterm_pct=_dterm(after, axis, config),
        before_tracking_rms=_tracking_rms(before, axis),
        after_tracking_rms=_tracking_rms(after, axis),
        before_gains=before_gains,
        after_gains=after_gains,
        recommended_gains=recommendation.gains if recommendation else None,
        applied=(
            _gains_match(after_gains, recommendation.gains) if recommendation is not None else None
        ),
    )


def _gains(bundle: LogBundle, axis: Axis) -> GainSet | None:
    """The gains the log was flown with, or ``None`` if it did not record them."""
    try:
        return gains_from_bundle(bundle, axis)
    except (KeyError, ValueError):
        return None


def _gains_match(flown: GainSet | None, recommended: GainSet | None) -> bool | None:
    """Whether the after-log was actually flying the recommendation.

    Compared per term and relatively, because the terms differ by three orders of
    magnitude: an absolute tolerance that is sensible for P is meaningless for D.
    """
    if flown is None or recommended is None:
        return None
    for got, wanted in (
        (flown.kp, recommended.kp),
        (flown.ki, recommended.ki),
        (flown.kd, recommended.kd),
    ):
        if wanted == 0.0:
            if abs(got) > 1e-9:
                return False
            continue
        if abs(got - wanted) / abs(wanted) > _APPLIED_TOLERANCE:
            return False
    return True


def _noise(bundle: LogBundle, axis: Axis, config: Config) -> NoiseProfile | None:
    """The gyro spectrum over the whole record.

    The whole record rather than a quiet window, unlike identification. A
    validation flight is compared against another validation flight, and two
    spectra taken over differently-chosen windows are not comparable however
    carefully each was chosen.
    """
    signal = bundle.signals.get(f"rate.{axis}.measured")
    if signal is None or signal.t.size < 2:
        return None
    try:
        return noise_profile(
            bundle,
            axis,
            t_start=float(signal.t[0]),
            t_end=float(signal.t[-1]),
            chain=chain_from_bundle(bundle, axis),
            prominence_db=config.float_("noise", "peak_prominence_db"),
            track_margin_db=config.float_("noise", "rpm_track_margin_db"),
            deconv_floor_db=config.float_("filters", "deconv_floor_db"),
            evidence_ceiling_hz=signal.native_nyquist_hz,
        )
    except (ValueError, KeyError):
        return None


def _dterm(bundle: LogBundle, axis: Axis, config: Config) -> float | None:
    """Measured D-term noise, as a percentage of full motor range."""
    return measured_dterm_rms_pct(
        bundle, axis, above_hz=config.float_("noise", "dterm_measure_above_hz")
    )


def _tracking_rms(bundle: LogBundle, axis: Axis) -> float | None:
    """RMS of setpoint minus measurement, in rad/s.

    The bluntest performance number there is, and the one a pilot recognizes: how
    far the aircraft was from what it was told, averaged over the flight. It says
    nothing about *why*, which is what everything else on the screen is for.
    """
    setpoint = bundle.signals.get(f"rate.{axis}.setpoint")
    measured = bundle.signals.get(f"rate.{axis}.measured")
    if setpoint is None or measured is None:
        return None
    n = min(setpoint.y.size, measured.y.size)
    if n == 0:
        return None
    error = setpoint.y[:n] - measured.y[:n]
    return float(np.sqrt(np.mean(np.square(error))))


def _fractional_change(before: float | None, after: float | None) -> float | None:
    """``(after - before) / before``, or ``None`` when either is missing or zero."""
    if before is None or after is None or before <= 0.0:
        return None
    return (after - before) / before


def _spectrum_error_db(
    predicted_f: FloatArray,
    predicted_psd: FloatArray,
    measured_f: FloatArray,
    measured_psd: FloatArray,
) -> float | None:
    """Median dB error of a predicted spectrum against a measured one.

    Interpolated onto the measured grid rather than the other way round: the
    measurement is the fact, and resampling it to fit the prediction is the wrong
    way round in a comparison whose entire purpose is to let the measurement
    contradict the prediction.
    """
    band = (measured_f >= _FILTER_SCORE_BAND_HZ[0]) & (measured_f <= _FILTER_SCORE_BAND_HZ[1])
    if not band.any() or predicted_f.size < 2:
        return None
    interpolated = np.interp(
        measured_f[band], predicted_f, predicted_psd, left=np.nan, right=np.nan
    )
    measured = measured_psd[band]
    usable = np.isfinite(interpolated) & (interpolated > 0.0) & (measured > 0.0)
    if not usable.any():
        return None
    return float(np.median(10.0 * np.log10(measured[usable] / interpolated[usable])))


def material(change: float | None) -> bool:
    """Whether a fractional change is big enough to be worth calling a change.

    Two flights of the same aircraft on the same tune do not produce the same
    numbers, so a report that announced every difference would announce noise.
    """
    return change is not None and abs(change) >= _MATERIAL_CHANGE
