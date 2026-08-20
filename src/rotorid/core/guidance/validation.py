"""What a before/after pair says, in words (spec sections 5.10 and 8.1).

Separate from :mod:`rotorid.core.guidance.findings` because the evidence is a
different shape. Those checks look at one log and an identification of it; these
look at two flights and a promise made between them, and the questions they can
answer are ones no single log can: did the aircraft improve, and was the tool
right about why.

The order the checks run in is the order a reader needs them. Whether the tune
was applied comes first, because a prediction compared against a flight that
never loaded the parameters is not a failed prediction and reading it as one
would send the user to debug the wrong thing.
"""

from __future__ import annotations

from rotorid.core.analysis.compare import AxisComparison, ValidationReport, material
from rotorid.core.types import Finding

__all__ = ["validation_findings"]

#: A filter prediction is judged against this much error, in dB, over the band
#: the design was scored on. Three decibels is a factor of two in power, which is
#: about where a notch that landed on the wrong line stops being arguable.
_FILTER_TOLERANCE_DB = 3.0


def validation_findings(report: ValidationReport) -> tuple[Finding, ...]:
    """Everything worth saying about one before/after pair, most severe first."""
    findings: list[Finding] = []
    for comparison in report.axes.values():
        findings.extend(_applied(comparison))
        findings.extend(_prediction(comparison, report))
        findings.extend(_filters(comparison))
        findings.extend(_tracking(comparison))
        findings.extend(_dterm(comparison))
    order = {"blocker": 0, "warning": 1, "info": 2, "good": 3}
    return tuple(sorted(findings, key=lambda f: (order[f.severity], f.code, f.title)))


def _applied(c: AxisComparison) -> list[Finding]:
    """Whether the after-log was flying what was recommended.

    Not a criticism when it was not -- staged tuning means the gain file is
    deliberately loaded a flight *after* the filter file, so an after-log flying
    the old gains is the expected outcome of following the plan. What it means is
    that the gain prediction has not been tested yet, and saying so is different
    from saying it failed.
    """
    if c.applied is not False or c.recommended_gains is None or c.after_gains is None:
        return []
    return [
        Finding(
            severity="info",
            code="TUNE_NOT_APPLIED",
            title=f"{c.axis}: the after-flight was not flying the recommended gains",
            detail=(
                f"The recommendation was P {c.recommended_gains.kp:.4f}, "
                f"I {c.recommended_gains.ki:.4f}, D {c.recommended_gains.kd:.5f}; this log "
                f"was flown with P {c.after_gains.kp:.4f}, I {c.after_gains.ki:.4f}, "
                f"D {c.after_gains.kd:.5f}. Whatever else this comparison shows, it does "
                f"not test the gain recommendation, because the gains were never flown."
            ),
            action=(
                "If you are working through the staged plan, this is expected -- the filter "
                "flight comes first. Load the gain file and fly it before reading the step "
                "comparison as a verdict on the tune."
            ),
            evidence={
                "recommended_kp": c.recommended_gains.kp,
                "flown_kp": c.after_gains.kp,
            },
        )
    ]


def _prediction(c: AxisComparison, report: ValidationReport) -> list[Finding]:
    """The measured step of the new tune against what was predicted for it.

    The single most useful thing this tool can tell anyone. Every margin, every
    gain and every filter choice in the earlier report came out of one model, and
    this is the only measurement that puts that model against the aircraft under
    the tune it recommended.
    """
    holds = c.prediction_holds
    if holds is None or c.after_step is None or c.predicted_step is None:
        return []
    if c.applied is False:
        return []

    measured, predicted = c.after_step.metrics, c.predicted_step
    evidence = {
        "measured_rise_ms": measured.rise_time_s * 1000.0,
        "predicted_rise_ms": predicted.rise_time_s * 1000.0,
        "rise_ratio": c.rise_ratio or 0.0,
        "measured_overshoot_pct": measured.overshoot_pct,
        "predicted_overshoot_pct": predicted.overshoot_pct,
        "windows": float(c.after_step.n_windows),
    }
    numbers = (
        f"predicted {predicted.rise_time_s * 1000:.0f} ms rise and "
        f"{predicted.overshoot_pct:.0f}% overshoot; the aircraft flew "
        f"{measured.rise_time_s * 1000:.0f} ms and {measured.overshoot_pct:.0f}%, "
        f"measured from {c.after_step.n_windows} windows of "
        f"{report.after.path.name}"
    )

    if holds:
        return [
            Finding(
                severity="good",
                code="PREDICTION_CONFIRMED",
                title=f"{c.axis}: the aircraft did what the model said it would",
                detail=(
                    f"The earlier analysis {numbers}. The model that produced every margin "
                    f"and every gain in that report has now been checked against the "
                    f"aircraft flying the tune it recommended, and it holds."
                ),
                action="Nothing. This is the evidence the rest of the report was asking for.",
                evidence=evidence,
                plot_hint="step",
            )
        ]

    return [
        Finding(
            severity="warning",
            code="PREDICTION_MISSED",
            title=f"{c.axis}: the aircraft did not do what the model said it would",
            detail=(
                f"The earlier analysis {numbers}. Those are different aircraft. The gains "
                f"that were flown came out of the model that made the prediction, so a "
                f"prediction this far out means the margins quoted alongside it are not "
                f"the margins this vehicle has."
            ),
            action=(
                "Do not push the tune further on the strength of that model. Re-identify "
                "from a fresh tuning flight, and check first for the usual causes: a "
                "filter chain modelled differently from the firmware, motor saturation "
                "during the manoeuvre, or a payload change between the two flights."
            ),
            evidence=evidence,
            plot_hint="step",
        )
    ]


def _filters(c: AxisComparison) -> list[Finding]:
    """The predicted post-filter spectrum against the measured one.

    The half of a recommendation that normally goes unchecked. A gain change
    announces itself in how the aircraft feels; a notch that landed two hertz off
    the motor line does not, and the only way anybody finds out is by measuring
    the spectrum of a flight flown with it.
    """
    error = c.filter_prediction_error_db
    if error is None:
        return []
    evidence = {"median_error_db": error}
    if abs(error) <= _FILTER_TOLERANCE_DB:
        return [
            Finding(
                severity="good",
                code="FILTER_PREDICTION_CONFIRMED",
                title=f"{c.axis}: the filters attenuated what they were designed to",
                detail=(
                    f"The post-filter gyro spectrum in this flight is within "
                    f"{abs(error):.1f} dB of what the filter design predicted, across the "
                    f"band the design was scored over. The chain the tool modelled is the "
                    f"chain the firmware is running."
                ),
                action="Nothing.",
                evidence=evidence,
                plot_hint="spectrum",
            )
        ]

    noisier = error > 0.0
    return [
        Finding(
            severity="warning",
            code="FILTER_PREDICTION_MISSED",
            title=(
                f"{c.axis}: the gyro is {abs(error):.0f} dB "
                f"{'noisier' if noisier else 'quieter'} than the filter design predicted"
            ),
            detail=(
                f"Median error of {error:+.1f} dB between the predicted post-filter "
                f"spectrum and the one measured in this flight. "
                + (
                    "Noisier than predicted means the filters are not attenuating what the "
                    "design assumed they would, so the D term is seeing noise the gain "
                    "design did not budget for."
                    if noisier
                    else "Quieter than predicted is not free: attenuation the design did not "
                    "ask for came with phase the design did not budget for, and the "
                    "margins were computed on the phase it expected."
                )
            ),
            action=(
                "Check that the filter parameters were loaded exactly as exported, and that "
                "the notch is tracking -- a dynamic notch with no RPM or FFT source falls "
                "back to a fixed frequency that is only right at one throttle."
            ),
            evidence=evidence,
            plot_hint="spectrum",
        )
    ]


def _tracking(c: AxisComparison) -> list[Finding]:
    """Whether the aircraft ended up closer to what it was told.

    The bluntest number in the report and the one a pilot recognizes. It is
    reported as an outcome, never as a verdict: tracking error depends on what
    the pilot asked for, and two flights flown differently produce different
    numbers on an unchanged aircraft.
    """
    change = c.tracking_change
    if not material(change) or change is None:
        return []
    better = change < 0.0
    return [
        Finding(
            severity="good" if better else "info",
            code="TRACKING_IMPROVED" if better else "TRACKING_WORSE",
            title=(
                f"{c.axis}: rate tracking error {'fell' if better else 'rose'} "
                f"{abs(change) * 100:.0f}%"
            ),
            detail=(
                f"RMS of setpoint minus measurement went from "
                f"{c.before_tracking_rms:.3f} to {c.after_tracking_rms:.3f} rad/s. "
                f"This depends on what the pilot asked for as much as on the tune, so "
                f"it is worth reading only alongside the step comparison -- two flights "
                f"flown differently move this number on an unchanged aircraft."
            ),
            action=(
                "Nothing."
                if better
                else "Compare the step responses before reading this as a worse tune: a "
                "more aggressive flight produces a larger tracking error on the same "
                "aircraft."
            ),
            evidence={
                "before_rms_rad_s": c.before_tracking_rms or 0.0,
                "after_rms_rad_s": c.after_tracking_rms or 0.0,
            },
        )
    ]


def _dterm(c: AxisComparison) -> list[Finding]:
    """Whether the derivative term is putting less noise into the motors.

    The quantity the filter half of the recommendation exists to move, measured
    rather than predicted. A filter change that widened the disturbance-rejection
    bandwidth and left this unchanged did not do what it was for.
    """
    change = c.dterm_change
    if not material(change) or change is None:
        return []
    better = change < 0.0
    return [
        Finding(
            severity="good" if better else "warning",
            code="DTERM_NOISE_IMPROVED" if better else "DTERM_NOISE_WORSE",
            title=(
                f"{c.axis}: D-term noise {'fell' if better else 'rose'} {abs(change) * 100:.0f}%"
            ),
            detail=(
                f"Measured above the control band, the derivative term's contribution to "
                f"motor output went from {c.before_dterm_pct:.2f}% to "
                f"{c.after_dterm_pct:.2f}% of full range. "
                + (
                    "That is the filter recommendation doing what it was for."
                    if better
                    else "Higher D-term noise heats motors and wastes control authority, and "
                    "it is the cost side of every derivative gain increase."
                )
            ),
            action=(
                "Nothing."
                if better
                else "If the D gain was raised in this change, the noise is its price and the "
                "trade is yours to accept. If it was not, look at the notch: the motors "
                "may have moved out from under it."
            ),
            evidence={
                "before_pct": c.before_dterm_pct or 0.0,
                "after_pct": c.after_dterm_pct or 0.0,
            },
        )
    ]
