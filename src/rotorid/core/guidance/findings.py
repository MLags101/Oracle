"""Everything the tool noticed, with the numbers behind it (spec section 6).

A finding is not a log message. It is a claim about the aircraft, and it carries
four things or it is not worth showing: what was observed, what it means, what to
do about it, and the evidence anyone could use to check the claim. Codes are
stable identifiers -- tests, the report and the flight-plan generator all
reference them -- so they are never reworded once published.

Severity has a specific operational meaning here:

``blocker``
    Something downstream would be wrong. Exports are disabled until the user
    explicitly acknowledges it.
``warning``
    The recommendation stands but is weaker than it looks.
``info``
    Nothing is wrong; the next flight could be more informative.
``good``
    Worth saying so. A user who only ever sees problems learns nothing about what
    a healthy log looks like.

Every check is a small pure function of :class:`GuidanceContext`, registered in
:data:`CHECKS`. Adding a finding means adding a function and a test, and nothing
else in the tool changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rotorid.config import Config
from rotorid.core.analysis.operating_point import OperatingPointSpread
from rotorid.core.analysis.sysid import check_filter_model
from rotorid.core.analysis.vibration import VibrationSummary, vibration_summary
from rotorid.core.design.recommend import AxisAnalysis
from rotorid.core.logkind import capabilities, detect_kind, kind_evidence
from rotorid.core.types import AXES, Axis, Finding, LogBundle, TuneRecommendation

__all__ = ["CHECKS", "GuidanceContext", "collect_findings"]

#: A vehicle whose loop crosses over closer than this to its gyro filter is
#: spending a large part of its phase budget on noise rejection.
_LPF_CROSSOVER_SEPARATION = 4.0

#: Fraction of samples a nonlinear limiter must be active over before it is worth
#: reporting. Below this it is a transient, not a characteristic of the tune.
_LIMITER_DUTY = 0.02

#: How far above the design crossover the log must still carry real information
#: before a loop can be designed from it. Below the crossover there is nothing to
#: shape; a little above it is where the phase that decides stability lives.
_MIN_EVIDENCE_MULTIPLE_OF_CROSSOVER = 2.0

#: Peak CPU load above which the expensive filter options stop being affordable.
_CPU_LOAD_HIGH = 0.8

#: How far the vehicle's own tuner may land from this one before the two are
#: reported as disagreeing. A factor of 1.5 is well outside what two estimators
#: of the same aircraft should produce and well inside what a genuine
#: identification error produces, which is where a threshold wants to sit.
_VENDOR_AGREEMENT_RATIO = 1.5


@dataclass(frozen=True, slots=True)
class GuidanceContext:
    """Everything the checks are allowed to look at.

    Deliberately a plain bundle of already-computed results: a check that had to
    re-run analysis to reach its verdict would be slow, and worse, could reach a
    different verdict than the one the report shows.
    """

    bundle: LogBundle
    analyses: dict[Axis, AxisAnalysis]
    recommendations: dict[Axis, TuneRecommendation]
    config: Config

    def axes(self) -> tuple[Axis, ...]:
        """Axes that were analysed, in canonical order."""
        return tuple(a for a in ("roll", "pitch", "yaw") if a in self.analyses)


Check = Callable[[GuidanceContext], list[Finding]]


_PX4_SUFFIX: dict[Axis, str] = {"roll": "ROLL", "pitch": "PITCH", "yaw": "YAW"}


def _doc(context: GuidanceContext) -> str:
    """The logging-setup document for whichever firmware this log came from."""
    return f"docs/logging-setup-{context.bundle.stack}.md"


def _d_gain_name(context: GuidanceContext) -> str:
    """What the user has to type to change the derivative gain, on their stack."""
    return "ATC_RAT_*_D" if context.bundle.stack == "ardupilot" else "MC_*RATE_D"


def _imax_name(stack: str, axis: Axis) -> str:
    """The integrator-limit parameter, which the two stacks name differently."""
    if stack == "ardupilot":
        return f"ATC_RAT_{_AP_SUFFIX[axis]}_IMAX"
    return f"MC_{_PX4_SUFFIX[axis]}RATE_I_LIM"


def collect_findings(context: GuidanceContext) -> tuple[Finding, ...]:
    """Run every check, most severe first.

    A check that raises is a bug in the check rather than a fact about the
    aircraft, so it is allowed to propagate: a silently swallowed check would
    make the tool quietly stop noticing something.
    """
    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(context))
    order = {"blocker": 0, "warning": 1, "info": 2, "good": 3}
    return tuple(sorted(findings, key=lambda f: (order[f.severity], f.code)))


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #


def check_log_kind(context: GuidanceContext) -> list[Finding]:
    """What this log was flown as, what that unlocks, and whether it looks like it.

    Runs first among the log-level checks because everything after it is read
    differently depending on the answer. A user who does not know their log is
    being analysed as ordinary flight has no way to interpret a medium-confidence
    rating that could not have been anything else.
    """
    bundle = context.bundle
    kind = bundle.kind
    caps = capabilities(kind)
    detected = detect_kind(bundle)
    evidence = kind_evidence(bundle)
    out: list[Finding] = []

    if bundle.kind_was_declared and detected != kind and kind == "tuning":
        out.append(
            Finding(
                severity="warning",
                code="LOG_KIND_MISMATCH",
                title="declared a tuning flight, but no deliberate excitation is in it",
                detail=(
                    "This log was loaded as a tuning flight, so identification looks only "
                    "for an injected sweep or an autotune run. Neither is present, which "
                    "is why the axes below were refused rather than identified from stick "
                    "input."
                ),
                action=(
                    "Load it as a general flight log to identify it from what the pilot "
                    f"flew, or fly a tuning flight (see {_doc(context)}) and analyse that."
                ),
                doc_link=_doc(context),
            )
        )
    elif bundle.kind_was_declared and detected != kind and kind == "general":
        out.append(
            Finding(
                severity="warning",
                code="LOG_KIND_MISMATCH",
                title="deliberate excitation in a log loaded as a general flight",
                detail=(
                    "This log carries "
                    + "; ".join(evidence)
                    + ". It was loaded as a general flight, so that excitation is not "
                    "being used and the model comes from ordinary stick input instead -- "
                    "much the weaker of the two."
                ),
                action="Load it again as a tuning flight log to identify from the sweep.",
                doc_link=_doc(context),
            )
        )

    if kind == "general":
        out.append(
            Finding(
                severity="info",
                code="GENERAL_FLIGHT_LOG",
                title="analysed as a general flight",
                detail=(caps.summary + " Because of that: " + " ".join(caps.limits)),
                action=(
                    "Nothing, if that is the flight you flew. For a wider identification "
                    f"band and a tune designed with less held back, fly a sweep or an "
                    f"autotune (see {_doc(context)}) and load it as a tuning flight."
                ),
                evidence={"conservatism_floor": caps.conservatism_floor},
                doc_link=_doc(context),
            )
        )
    else:
        out.append(
            Finding(
                severity="good",
                code="TUNING_FLIGHT_LOG",
                title="analysed as a tuning flight",
                detail=(
                    "Identification used deliberate excitation: "
                    + ("; ".join(evidence) if evidence else "a recorded sweep")
                    + ". That is the evidence a wide-band model and a high confidence "
                    "rating require."
                ),
                action="Nothing. This is the log the analysis is designed around.",
            )
        )
    return out


def check_excitation(context: GuidanceContext) -> list[Finding]:
    """Whether the excitation that was found was as good as its class allows.

    Silent on a general flight: that every segment is stick input is the
    definition of the thing the user declared, and repeating it once per axis
    would bury the findings that are actually about this aircraft.
    ``GENERAL_FLIGHT_LOG`` states it once instead.
    """
    out: list[Finding] = []
    if context.bundle.kind == "general":
        return out
    for axis in context.axes():
        analysis = context.analyses[axis]
        best = max(s.confidence for s in analysis.segments)
        if best >= 1.0:
            continue
        out.append(
            Finding(
                severity="warning",
                code="WEAK_EXCITATION",
                title=f"{axis}: no deliberate frequency sweep",
                detail=(
                    f"The {axis} identification used {len(analysis.segments)} "
                    f"{analysis.segments[0].kind} segment(s) rather than a SYSTEMID sweep. "
                    f"An autotune twitch is a step, not a sweep, so it excites a band "
                    f"nobody chose and the model is fitted where the data happens to be "
                    f"rather than where the loop is designed."
                ),
                action=(
                    (
                        "Fly a SYSTEMID sweep on this axis"
                        if context.bundle.stack == "ardupilot"
                        else "Run the multicopter autotune on this axis, or fly a "
                        "deliberate slow-to-fast stick sweep"
                    )
                    + f" (see {_doc(context)}) and re-run."
                ),
                evidence={"segment_confidence": float(best)},
                doc_link=_doc(context),
            )
        )
    return out


def _identification_windows(context: GuidanceContext) -> tuple[tuple[float, float], ...] | None:
    """The stretches of flight the model was actually fitted to.

    ``None`` when nothing was analysed, which means the whole log. A log that was
    refused still deserves a vibration verdict -- quite often the vibration is why.
    """
    windows = tuple(
        (segment.t_start, segment.t_end)
        for axis in context.axes()
        for segment in context.analyses[axis].segments
    )
    return windows or None


def _vibration(context: GuidanceContext) -> VibrationSummary:
    return vibration_summary(context.bundle, _identification_windows(context))


def check_vibration(context: GuidanceContext) -> list[Finding]:
    """Whether the frame was still enough for its own sensors to measure it."""
    summary = _vibration(context)
    if not summary.measured:
        return [
            Finding(
                severity="info",
                code="VIBRATION_NOT_LOGGED",
                title="No vibration data in this log",
                detail=(
                    "The log carries no vibration message, so the tool cannot tell whether "
                    "the gyro trace it identified from is the aircraft's motion or the "
                    "frame's. That is an absence of evidence rather than a clean bill of "
                    "health: high vibration is the most common reason an identification is "
                    "confidently wrong."
                ),
                action=(
                    "Enable the IMU log group (ArduPilot LOG_BITMASK bit 2, VIBE) and re-fly."
                    if context.bundle.stack == "ardupilot"
                    else "Raise SDLOG_PROFILE to include vehicle_imu_status and re-fly."
                ),
                doc_link=_doc(context),
            )
        ]

    warn = context.config.float_("vibration", "warn_m_s2")
    blocker = context.config.float_("vibration", "blocker_m_s2")
    where = f"IMU {summary.worst_imu} on {summary.worst_component}"
    evidence = {
        "level_m_s2": summary.level_m_s2,
        "peak_m_s2": summary.peak_m_s2,
        "warn_m_s2": warn,
        "blocker_m_s2": blocker,
        "imu": float(summary.worst_imu),
    }
    # Stated whenever the flight had excursions well above what it sat at, so a
    # log that is calm on average and violent in two places does not read as calm.
    excursion = (
        f" It reached {summary.peak_m_s2:.0f} m/s^2 briefly."
        if summary.peak_m_s2 > 2.0 * max(summary.level_m_s2, 1.0)
        else ""
    )

    # Two thresholds against two different statistics, and neither substitutes for
    # the other. The blocker is set on the *sustained* level, because a frame that
    # is unusable is unusable throughout and a single sample cannot establish that
    # -- the counters arrive here splined, and a spline overshoots. The warning is
    # set on either, because a flight that touched the limit twice is a flight
    # with a mechanical problem that happened not to be excited the rest of the time.
    if summary.level_m_s2 < warn and summary.peak_m_s2 < warn:
        return [
            Finding(
                severity="good",
                code="VIBRATION_LOW",
                title=f"Vibration sits at {summary.level_m_s2:.1f} m/s^2",
                detail=(
                    f"The worst vibration over the identified windows was {where}, and it "
                    f"stayed under the {warn:.0f} m/s^2 the tuning guide treats as the "
                    f"limit throughout. The gyro trace the model was fitted to is the "
                    f"aircraft's motion."
                ),
                action="Nothing to do.",
                evidence=evidence,
            )
        ]

    severe = summary.level_m_s2 >= blocker
    brief = summary.level_m_s2 < warn
    return [
        Finding(
            severity="blocker" if severe else "warning",
            code="VIBRATION_HIGH",
            title=(
                f"Vibration reaches {summary.peak_m_s2:.0f} m/s^2 on {where}"
                if brief
                else f"Vibration sits at {summary.level_m_s2:.1f} m/s^2 on {where}"
            ),
            detail=(
                f"Sustained vibration over the identified windows was "
                f"{summary.level_m_s2:.1f} m/s^2, against {warn:.0f} for a healthy frame "
                f"and {blocker:.0f} where the accelerometers stop being usable.{excursion} "
                + (
                    "At this level the gyro is measuring the frame's resonance as much as "
                    "the aircraft's motion, so the identified model is not of the aircraft "
                    "and no gain derived from it can be trusted."
                    if severe
                    else "The frame is calm for most of the flight, so whatever shakes it "
                    "is not excited the whole time -- which makes it a mechanical fault "
                    "waiting for the right manoeuvre rather than a flight that went badly."
                    if brief
                    else "The identification still stands, but part of what it fitted is "
                    "frame motion rather than airframe response, and the noise it implies "
                    "will push the filter design harder than a clean frame needs."
                )
                + " Vibration is a mechanical problem; no gain or filter fixes it."
            ),
            action=(
                "Balance the propellers, check for damaged or bent props and loose motor "
                "mounts, and check the flight controller's vibration isolation before "
                "tuning anything. Re-fly and confirm the level has come down."
            ),
            evidence=evidence,
            doc_link=_doc(context),
        )
    ]


def check_oscillation(context: GuidanceContext) -> list[Finding]:
    """Whether the aircraft was already oscillating when the log was recorded.

    A blocker, and the reason is worth stating precisely. An aircraft in a limit
    cycle has a real loop with unity gain and inverted phase somewhere. A model
    fitted to that flight can still come back with comfortable margins, because
    the frequency response it was fitted over is dominated by the excitation
    rather than by the limit cycle. Without this check the tool would look at an
    oscillating vehicle, see a healthy model, and recommend more gain.
    """
    out: list[Finding] = []
    for axis in context.axes():
        found = context.analyses[axis].oscillation
        if found is None:
            continue

        # Two quite different situations, and they need different words. Either
        # the model agrees the flown loop was at its limit -- in which case the
        # identification is sound and the flown tune was simply too hot -- or the
        # model says there was margin left, which means the model is wrong.
        optimistic = found.model_optimism_db > 1.0
        out.append(
            Finding(
                severity="blocker",
                code="OSCILLATION_DETECTED",
                title=(
                    f"{axis}: oscillating at {found.f_hz:.0f} Hz for {found.duty:.0%} of the flight"
                ),
                detail=(
                    f"A tone at {found.f_hz:.1f} Hz stands {found.excess_db:.0f} dB above the "
                    f"local noise floor in the measured {axis} rate, accounts for "
                    f"{found.amplitude_frac:.0%} of the aircraft's total {axis} motion, and is "
                    f"present over {found.duty:.0%} of the record. The loop answers a command "
                    f"at that frequency with {found.amplification_db:.0f} dB more output than "
                    f"input, so the aircraft is not tracking there -- it is ringing. "
                    + (
                        f"This model claims {found.model_optimism_db:.0f} dB of gain margin at "
                        f"that frequency, so the aircraft is demonstrating that the model is "
                        f"wrong by at least that much. The design has been made to hold that "
                        f"margin back, but a model contradicted by the aircraft is a model to "
                        f"be suspicious of everywhere, not only here."
                        if optimistic
                        else "This model agrees the flown loop had no margin left at that "
                        "frequency, so the identification and the aircraft are telling the "
                        "same story: the gains that were flown were too high."
                    )
                ),
                action=(
                    f"Reduce the flown {axis} gains -- ArduPilot's guidance is to halve them -- "
                    f"and re-fly before using anything from this log. If the tone is near a "
                    f"motor or frame frequency, check the notch configuration first: a "
                    f"filter chasing the wrong line leaves the loop with less phase than the "
                    f"design assumed."
                ),
                evidence={
                    "f_hz": found.f_hz,
                    "excess_db": found.excess_db,
                    "duty": found.duty,
                    "amplitude_rad_s": found.amplitude_rad_s,
                    "amplitude_frac": found.amplitude_frac,
                    "amplification_db": found.amplification_db,
                    "model_optimism_db": found.model_optimism_db,
                },
                plot_hint="spectrum",
            )
        )
    return out


def check_step_response(context: GuidanceContext) -> list[Finding]:
    """Whether the model reproduces the step the aircraft actually flew.

    The only check in the tool that closes the loop back onto flown data. Every
    other one asks whether the identification was well-conditioned; this asks
    whether it was *right*, by predicting what the flown gains should have done
    and comparing that against what the log says they did. A model that is wrong
    about the aircraft can still be perfectly coherent, well fitted and narrowly
    banded, and it will fail here.

    Deliberately a coarse instrument. The deconvolution's regularizer low-passes
    the recovered response, so a measured rise time reads slow even when the model
    is right, and the thresholds have to leave room for that. It catches a model
    that is wrong by a factor, not one that is wrong by a fifth.
    """
    low, high = context.config.pair("validate", "rise_ratio_bounds")
    max_overshoot = context.config.float_("validate", "max_overshoot_diff_pct")

    out: list[Finding] = []
    for axis in context.axes():
        analysis = context.analyses[axis]
        measured, predicted = analysis.measured, analysis.flown_prediction
        if measured is None or predicted is None:
            continue
        if not np.isfinite(measured.metrics.rise_time_s) or predicted.rise_time_s <= 0.0:
            continue

        ratio = measured.metrics.rise_time_s / predicted.rise_time_s
        overshoot_gap = measured.metrics.overshoot_pct - predicted.overshoot_pct
        evidence = {
            "measured_rise_ms": measured.metrics.rise_time_s * 1000.0,
            "predicted_rise_ms": predicted.rise_time_s * 1000.0,
            "rise_ratio": ratio,
            "measured_overshoot_pct": measured.metrics.overshoot_pct,
            "predicted_overshoot_pct": predicted.overshoot_pct,
            "windows": float(measured.n_windows),
            "explained": measured.explained,
        }

        if low <= ratio <= high and abs(overshoot_gap) <= max_overshoot:
            out.append(
                Finding(
                    severity="good",
                    code="STEP_RESPONSE_AGREES",
                    title=f"{axis}: the model reproduces the flown step response",
                    detail=(
                        f"The step deconvolved from {measured.n_windows} windows of this "
                        f"flight rises in {measured.metrics.rise_time_s * 1000:.0f} ms with "
                        f"{measured.metrics.overshoot_pct:.0f}% overshoot; the identified "
                        f"model, driven by the gains the aircraft was flying, predicts "
                        f"{predicted.rise_time_s * 1000:.0f} ms and "
                        f"{predicted.overshoot_pct:.0f}%. The identification is not just "
                        f"well-conditioned, it describes this aircraft."
                    ),
                    action="Nothing to do.",
                    evidence=evidence,
                    plot_hint="step",
                )
            )
            continue

        out.append(
            Finding(
                severity="warning",
                code="STEP_RESPONSE_DISAGREES",
                title=f"{axis}: the model does not reproduce the flown step response",
                detail=(
                    f"Measured from the log: {measured.metrics.rise_time_s * 1000:.0f} ms "
                    f"rise, {measured.metrics.overshoot_pct:.0f}% overshoot, over "
                    f"{measured.n_windows} windows. Predicted by the identified model under "
                    f"the gains that were flown: {predicted.rise_time_s * 1000:.0f} ms and "
                    f"{predicted.overshoot_pct:.0f}%. These describe different aircraft. "
                    f"Every margin and every recommended gain in this report was computed "
                    f"against the prediction, so if the measurement is the true one, they "
                    f"are all computed against a vehicle that does not exist."
                ),
                action=(
                    "Compare the two curves on the Validate stage before using these gains. "
                    "The usual causes are a filter chain the tool has modelled differently "
                    "from the firmware, a gain change part-way through the flight, or a "
                    "manoeuvre that saturated the motors -- none of which the frequency "
                    "response alone can see."
                ),
                evidence=evidence,
                plot_hint="step",
            )
        )
    return out


def check_clipping(context: GuidanceContext) -> list[Finding]:
    """Whether an accelerometer saturated while the model was being measured.

    Separate from :func:`check_vibration` and always a blocker, because clipping
    is not a degree of badness. A clipped sample is not a poor measurement of the
    aircraft; it is the absence of one, and averaging more of them does not bring
    it back.
    """
    summary = _vibration(context)
    if not summary.clip_measured or not summary.clipped:
        return []
    imus = ", ".join(str(i) for i in summary.clipping_imus)
    return [
        Finding(
            severity="blocker",
            code="ACCEL_CLIPPING",
            title=f"Accelerometer clipping on IMU {imus}",
            detail=(
                f"The clip counters rose by {summary.clip_count} inside the windows the "
                f"model was identified from, which means the accelerometer hit the end of "
                f"its measurement range. Beyond that point the sensor reports its limit "
                f"rather than the aircraft, and the estimator has no way to tell the "
                f"difference -- so the identification is fitted partly to a flat line that "
                f"the aircraft never flew."
            ),
            action=(
                "Fix the vibration first: balance props, check motor mounts and the "
                "controller's isolation. Clipping must read zero throughout before any "
                "tune from this aircraft means anything."
            ),
            evidence={
                "clip_count": float(summary.clip_count),
                "level_m_s2": summary.level_m_s2,
            },
            doc_link=_doc(context),
        )
    ]


def check_identification_band(context: GuidanceContext) -> list[Finding]:
    """Whether the coherent band is wide enough to design a loop inside."""
    minimum = context.config.float_("coherence", "min_valid_octaves")
    out: list[Finding] = []
    for axis in context.axes():
        low, high = context.analyses[axis].deconvolved.valid_band_hz
        octaves = float(np.log2(high / low)) if low > 0.0 else 0.0
        if octaves >= minimum:
            continue
        out.append(
            Finding(
                severity="blocker",
                code="COHERENCE_NARROW_BAND",
                title=f"{axis}: identified over only {octaves:.1f} octaves",
                detail=(
                    f"Coherence held from {low:.2f} to {high:.1f} Hz, which is narrower than "
                    f"the {minimum:.0f} octaves needed to pin the model's shape. A model fitted "
                    f"this narrowly can match the data closely and still be the wrong "
                    f"aircraft outside the band -- including where the loop crosses over."
                ),
                action=(
                    "Raise SID_MAGNITUDE until the response clearly exceeds the noise, and "
                    "widen SID_F_START_HZ/SID_F_STOP_HZ to bracket the crossover."
                ),
                evidence={"octaves": octaves, "f_low_hz": low, "f_high_hz": high},
                plot_hint="coherence",
            )
        )
    return out


def check_log_rate(context: GuidanceContext) -> list[Finding]:
    """Whether the controller messages were written to the card fast enough.

    This is the one condition in the whole tool that is invisible from every
    other angle. ``SCHED_LOOP_RATE`` says 400 Hz, the gyro says 2 kHz, the
    analysis grid gets built at 800 Hz -- and none of that is a claim about how
    often ``RATE`` reached the SD card. On ArduPilot that is a separate decision:
    with ``LOG_BITMASK`` bit 0 (ATTITUDE_FAST) clear, the attitude and rate
    messages go out on the 10 Hz medium-rate schedule instead.

    Nothing downstream notices. The resampler splines 10 Hz onto the grid,
    coherence stays high because both signals were splined by the same
    interpolator, and a confident airframe model comes back fitted to the shape
    of a cubic. So the rate is checked here against what the design actually
    needs to see: a loop crossing over near 10 Hz needs evidence to several times
    that, and a message logged at 10 Hz carries none above 5.
    """
    bundle = context.bundle
    ceiling = context.config.float_("margins", "crossover_frac_of_loop") * bundle.loop_rate_hz / 2.0
    # Deliberately *not* the crossover the design landed on. That number is
    # itself a consequence of the evidence band, so a starved log produces a low
    # crossover, which then makes the starved log look adequate. The honest
    # yardstick is the crossover this vehicle's loop rate would have allowed.
    crossover = ceiling
    needed_hz = _MIN_EVIDENCE_MULTIPLE_OF_CROSSOVER * crossover

    # Every axis the log carries, not only the ones that got as far as a
    # recommendation. A log too slow to identify anything fails on all three
    # axes at once, and that is exactly when the user most needs to be told the
    # reason rather than three copies of "no usable excitation".
    starved: list[tuple[Axis, str, float]] = []
    for axis in AXES:
        rates: list[tuple[str, float]] = [
            (key, native)
            for key in (f"rate.{axis}.measured", f"rate.{axis}.output")
            if key in bundle.signals
            for native in (bundle.signals[key].native_rate_hz,)
            if native is not None
        ]
        if not rates:
            continue
        # The slowest contributing message sets the ceiling: an FRF is only as
        # informative as the worse of its two signals.
        key, rate = min(rates, key=lambda kv: kv[1])
        if 0.5 * rate < needed_hz:
            starved.append((axis, bundle.signals[key].source_msg, rate))
    if not starved:
        return []

    # One finding, not three. All three axes ride the same logging schedule, so
    # three copies of it would be three copies of one decision the user made
    # once and will undo once.
    rate = min(r for _, _, r in starved)
    nyquist = 0.5 * rate
    sources = ", ".join(sorted({source for _, source, _ in starved}))
    axes = ", ".join(axis for axis, _, _ in starved)
    ardupilot = bundle.stack == "ardupilot"
    return [
        Finding(
            severity="blocker",
            code="LOG_RATE_TOO_LOW",
            title=(
                f"Rate loop logged at {rate:.0f} Hz, not the "
                f"{bundle.loop_rate_hz:.0f} Hz it runs at"
            ),
            detail=(
                f"{sources} ({axes}) reached the card {rate:.0f} times a second, so the "
                f"log carries no information above {nyquist:.1f} Hz. This vehicle's loop "
                f"rate would allow a crossover as high as {crossover:.0f} Hz, and "
                f"designing for that needs evidence to roughly {needed_hz:.0f} Hz. "
                f"Everything the analysis would show above {nyquist:.1f} Hz is "
                f"interpolation rather than measurement -- and it looks entirely "
                f"convincing, because coherence between two signals put on the grid by "
                f"the same spline stays high whether or not either still means anything."
            ),
            action=(
                "Set LOG_BITMASK bit 0 (ATTITUDE_FAST) so RATE and PID messages log at "
                "the loop rate, and re-fly. Bit 18 (IMU_FAST) is worth adding at the "
                "same time."
                if ardupilot
                else "Add the high-rate profile to SDLOG_PROFILE so "
                "vehicle_angular_velocity and vehicle_torque_setpoint log at full "
                "rate, and re-fly."
            ),
            evidence={
                "native_rate_hz": float(rate),
                "loop_rate_hz": float(bundle.loop_rate_hz),
                "usable_to_hz": nyquist,
                "needed_to_hz": needed_hz,
            },
            doc_link=_doc(context),
        )
    ]


def check_estimator(context: GuidanceContext) -> list[Finding]:
    """Whether the loop was divided out of the identification, or assumed away.

    Every flight log is closed-loop data. The mixer command is the controller's
    own output, so it carries the gyro noise fed back through the controller, and
    the ordinary estimate of the plant from it is biased towards the inverse of
    that controller -- an answer that looks like a measurement and is not one.
    Dividing the loop out needs an exogenous signal: the injected chirp, or
    failing that the pilot's commanded attitude.

    Two different things can go wrong, and they want different responses. Having
    no such signal at all is a blocker, because the number is wrong by an unknown
    amount. Having one, and finding that it disagrees sharply with the direct
    estimate, is a warning: the right answer was used, and the size of the
    disagreement is worth seeing because it says how much of the mixer command
    was the controller chasing its own noise.
    """
    warn_db = context.config.float_("estimator", "bias_warn_db")
    warn_deg = context.config.float_("estimator", "bias_warn_deg")
    ardupilot = context.bundle.stack == "ardupilot"
    out: list[Finding] = []
    for axis in context.axes():
        plant = context.analyses[axis].effective
        if not plant.unbiased:
            out.append(
                Finding(
                    severity="blocker",
                    code="ESTIMATOR_BIASED",
                    title=f"{axis}: nothing in this log is independent of the gyro",
                    detail=(
                        "The aircraft was identified from the mixer command alone. That "
                        "command is what the controller produced from the gyro, so "
                        "whatever noise was on the gyro is in both signals, and the "
                        "estimate is pulled towards the inverse of the controller rather "
                        "than towards the aircraft. It is not a weaker measurement of the "
                        "right thing; it is a measurement of something else, and how far "
                        "off it lands depends on how noisy the flight was."
                    ),
                    action=(
                        "Log the ATT message so the pilot's commanded attitude is "
                        "available as an independent reference -- add the ATTITUDE bits "
                        "to LOG_BITMASK -- or fly a SYSTEMID sweep, which is better still."
                        if ardupilot
                        else "Add the high-rate profile to SDLOG_PROFILE so "
                        "vehicle_attitude_setpoint is logged, and re-fly."
                    ),
                    evidence={"axis_index": float(AXES.index(axis))},
                    doc_link=_doc(context),
                )
            )
            continue
        if abs(plant.bias_db) < warn_db and abs(plant.bias_deg) < warn_deg:
            continue
        out.append(
            Finding(
                severity="warning",
                code="ESTIMATOR_BIAS_LARGE",
                title=f"{axis}: feedback moved the answer by {plant.bias_db:+.1f} dB",
                detail=(
                    f"Identified against {plant.instrument}, which is independent of the "
                    f"gyro, so the number used is the unbiased one. Reading the same "
                    f"flight the naive way -- straight from the mixer command -- gives an "
                    f"aircraft {plant.bias_db:+.1f} dB and {plant.bias_deg:+.1f} deg "
                    f"different. That gap is the controller reacting to its own noise, and "
                    f"a gap this size means the excitation was weak next to the noise, "
                    f"which limits how much of the band this flight can speak for."
                ),
                action=(
                    "Nothing is wrong with the result. To narrow the gap on the next "
                    "flight, excite the axis harder relative to the noise: a larger "
                    "SID_MAGNITUDE, or firmer and more deliberate stick movement."
                    if ardupilot
                    else "Nothing is wrong with the result. To narrow the gap, excite the "
                    "axis harder relative to the noise on the next flight."
                ),
                evidence={"bias_db": plant.bias_db, "bias_deg": plant.bias_deg},
                plot_hint="bode",
            )
        )
    return out


def check_prefilter_data(context: GuidanceContext) -> list[Finding]:
    """Whether the log lets the filter model be checked against the aircraft."""
    bundle = context.bundle
    has_pre = bundle.batch is not None and bundle.batch.has_pre_filter
    has_pre = has_pre or any(k.endswith(".prefilter") for k in bundle.signals)
    if has_pre:
        return [
            Finding(
                severity="good",
                code="RAW_IMU_DATA_PRESENT",
                title="Pre-filter gyro is logged",
                detail=(
                    "The log contains gyro samples from before the filter chain, so the "
                    "modelled filters were checked against this aircraft rather than "
                    "against arithmetic."
                ),
                action="Nothing to do. Turn batch logging off once tuning is finished.",
            )
        ]
    return [
        Finding(
            severity="info",
            code="NO_RAW_IMU_DATA",
            title="No pre-filter gyro in this log",
            detail=(
                "Only the filtered gyro was logged, so the noise the filters removed had to "
                "be reconstructed by dividing the modelled chain out. That reconstruction "
                "cannot see inside a deep notch, which is why an existing notch is kept "
                "rather than redesigned."
            ),
            action=(
                "Set INS_LOG_BAT_MASK = 1 and INS_LOG_BAT_OPT = 4 (or INS_RAW_LOG_OPT = 9 "
                "on an H7 board) and re-fly to get pre- and post-filter gyro."
                if context.bundle.stack == "ardupilot"
                else "Add the raw-IMU bit to SDLOG_PROFILE and re-fly, so sensor_gyro_fifo "
                "gives the pre-filter gyro directly."
            ),
            doc_link=_doc(context),
        )
    ]


def check_esc_telemetry(context: GuidanceContext) -> list[Finding]:
    """Whether measured motor speed is available and actually being used."""
    bundle = context.bundle
    has_rpm = any(k.startswith("motor.") and k.endswith(".rpm") for k in bundle.signals)
    if not has_rpm:
        return []

    ardupilot = bundle.stack == "ardupilot"
    name = "INS_HNTCH_MODE" if ardupilot else "IMU_GYRO_DNF_EN"
    setting = bundle.param(name, 0.0) or 0.0
    # ArduPilot MODE 3 is ESC telemetry; PX4 sets bit 0 of DNF_EN for the same thing.
    already_tracking = setting == 3.0 if ardupilot else bool(int(setting) & 1)
    if already_tracking:
        return []
    return [
        Finding(
            severity="warning",
            code="ESC_TELEM_AVAILABLE_UNUSED",
            title="ESC telemetry is logged but the notch is not using it",
            detail=(
                f"The log carries per-motor RPM, which is the most direct measurement of "
                f"where the motor noise is, but {name} is {setting:g}. A notch that is not "
                f"following measured motor speed is in the right place only when the "
                f"throttle happens to be where it was configured for."
            ),
            action=(
                "Set INS_HNTCH_MODE = 3 (ESC telemetry) and INS_HNTCH_REF = 1."
                if ardupilot
                else "Set bit 0 of IMU_GYRO_DNF_EN so the dynamic notch tracks ESC RPM."
            ),
            evidence={name: float(setting)},
        )
    ]


def check_cpu_headroom(context: GuidanceContext) -> list[Finding]:
    """Whether the board can afford the filtering it is being asked to run."""
    if "cpu.load" not in context.bundle.signals:
        return []
    peak = float(np.max(context.bundle.signals["cpu.load"].y))
    if peak < _CPU_LOAD_HIGH:
        return []
    return [
        Finding(
            severity="warning",
            code="CPU_HEADROOM_LOW",
            title=f"Peak CPU load {peak * 100:.0f}%",
            detail=(
                "The scheduler was already close to its limit on this flight. Adding notch "
                "harmonics, per-motor tracking or loop-rate notch updates costs CPU, and a "
                "board that misses its loop deadline does not degrade gracefully."
            ),
            action=(
                "Do not enable the expensive notch options (OPTS bits 1, 2 and 3). If the "
                "recommendation needs them, reduce SCHED_LOOP_RATE or the logging rate first."
            ),
            evidence={"peak_cpu_load": peak},
        )
    ]


def check_pid_messages(context: GuidanceContext) -> list[Finding]:
    """Whether term-level diagnosis is possible at all.

    Not the same question on the two stacks. ArduPilot can log every controller
    term and simply was not asked to, so the finding is an instruction. PX4 does
    not log them at all, so the finding is a statement about what this analysis
    cannot see -- telling a PX4 user to set LOG_BITMASK would be advice they
    cannot act on, which is worse than saying nothing.
    """
    if any(k.endswith(".p_term") for k in context.bundle.signals):
        return []
    ardupilot = context.bundle.stack == "ardupilot"
    return [
        Finding(
            severity="info",
            code="LOG_MISSING_PID",
            title=(
                "No PID messages in this log"
                if ardupilot
                else "Controller terms are not logged on this stack"
            ),
            detail=(
                "Without the individual controller terms, slew limiting, D-term noise "
                "and integrator windup cannot be diagnosed -- only inferred from the "
                "model and the measured spectrum."
            ),
            action=(
                "Add the PID bit to LOG_BITMASK and re-fly."
                if ardupilot
                else "Nothing to change: PX4 does not log the separate P, I and D "
                "contributions, so those checks are inferred rather than measured here."
            ),
            doc_link=_doc(context),
        )
    ]


# --------------------------------------------------------------------------- #
# Filters and noise
# --------------------------------------------------------------------------- #


def check_filter_model_agreement(context: GuidanceContext) -> list[Finding]:
    """Whether the modelled chain matches the shape the log actually shows."""
    threshold = context.config.float_("filters", "model_mismatch_depth_db")
    out: list[Finding] = []
    for axis in context.axes():
        analysis = context.analyses[axis]
        result = check_filter_model(
            analysis.effective,
            analysis.chain,
            analysis.airframe,
            op=analysis.operating_point,
        )
        if not result.checkable or result.max_magnitude_error_db <= threshold:
            continue
        out.append(
            Finding(
                severity="warning",
                code="FILTER_MODEL_MISMATCH",
                title=f"{axis}: the log does not match the modelled filter chain",
                detail=(
                    f"Measured and modelled response differ by up to "
                    f"{result.max_magnitude_error_db:.1f} dB over "
                    f"{result.n_bins_compared} coherent bins. Every gain here was designed "
                    f"after dividing that model out, so if the model is wrong the gains "
                    f"are wrong by the same amount."
                ),
                action=(
                    (
                        "Check INS_HNTCH_REF, INS_HNTCH_FM_RAT and the firmware version."
                        if context.bundle.stack == "ardupilot"
                        else "Check IMU_GYRO_DNF_MIN, IMU_GYRO_DNF_BW and the firmware version."
                    )
                    + " If the notch never tracked, the flown chain is not the configured one."
                ),
                evidence={
                    "max_error_db": result.max_magnitude_error_db,
                    "bins": float(result.n_bins_compared),
                },
                plot_hint="filter_model",
            )
        )
    return out


def check_structural_resonance(context: GuidanceContext) -> list[Finding]:
    """Fixed-frequency peaks: a mechanical problem wearing a filter-shaped hat."""
    out: list[Finding] = []
    for axis in context.axes():
        noise = context.analyses[axis].noise
        if noise is None:
            continue
        for peak in noise.peaks:
            if peak.kind != "structural":
                continue
            out.append(
                Finding(
                    severity="warning",
                    code="STRUCTURAL_RESONANCE",
                    title=f"{axis}: fixed {peak.f_hz:.0f} Hz resonance",
                    detail=(
                        f"A peak {peak.magnitude_db:.0f} dB above the noise floor sits at "
                        f"{peak.f_hz:.0f} Hz and does not move with motor speed. That is the "
                        f"airframe ringing -- a soft mount, a loose arm, a flexing plate -- "
                        f"not motor noise. A tracking notch will chase the motors away from "
                        f"it and remove nothing."
                    ),
                    action=(
                        "Find it mechanically: check arm bolts, FC mounting and any cable "
                        "under tension. A static notch is a stopgap, and it costs phase in "
                        "the loop for as long as it is there."
                    ),
                    evidence={"f_hz": peak.f_hz, "above_floor_db": peak.magnitude_db},
                    plot_hint="noise_spectrum",
                )
            )
    return out


def check_gyro_lpf_separation(context: GuidanceContext) -> list[Finding]:
    """Whether the gyro filter is crowding the control bandwidth."""
    out: list[Finding] = []
    for axis in context.axes():
        rec = context.recommendations.get(axis)
        if rec is None:
            continue
        cutoff = rec.filters.chain.gyro_lpf_hz
        crossover = rec.margins.crossover_hz
        if cutoff is None or crossover <= 0.0:
            continue
        ratio = cutoff / crossover
        if ratio >= _LPF_CROSSOVER_SEPARATION:
            continue
        out.append(
            Finding(
                severity="warning",
                code="GYRO_LPF_TOO_LOW",
                title=f"{axis}: gyro filter only {ratio:.1f}x above the crossover",
                detail=(
                    f"The {cutoff:.0f} Hz gyro low-pass sits close to the {crossover:.2f} Hz "
                    f"crossover, so a large share of the loop's phase budget is being spent "
                    f"on noise rejection rather than on control."
                ),
                action=(
                    "Reduce the noise at its source -- balance props, check motor bearings, "
                    "soften the FC mount -- and the filter can be opened up, which buys "
                    "bandwidth back directly."
                ),
                evidence={"gyro_lpf_hz": float(cutoff), "crossover_hz": crossover},
                plot_hint="latency_budget",
            )
        )
    return out


def check_dterm_noise(context: GuidanceContext) -> list[Finding]:
    """Whether the derivative term is pushing noise into the motors."""
    limit = context.config.float_("noise", "dterm_output_rms_limit_pct")
    out: list[Finding] = []
    for axis in context.axes():
        rec = context.recommendations.get(axis)
        if rec is None or not np.isfinite(rec.dterm_noise_rms_pct):
            continue
        if rec.dterm_noise_rms_pct <= limit:
            continue
        out.append(
            Finding(
                severity="warning",
                code="DTERM_NOISE_HIGH",
                title=f"{axis}: D term drives {rec.dterm_noise_rms_pct:.1f}% output noise",
                detail=(
                    f"With the recommended filters and D gain, gyro noise reaches the motors "
                    f"at {rec.dterm_noise_rms_pct:.1f}% of full scale RMS, above the "
                    f"{limit:.0f}% limit. That is heat in the motors and ESCs for no control "
                    f"benefit, and it is what makes an over-D'd vehicle feel harsh."
                ),
                action=(
                    f"Lower {_d_gain_name(context)}, or fix the noise mechanically. "
                    f"Filtering it away instead costs phase and buys back less than it "
                    f"removes."
                ),
                evidence={"dterm_rms_pct": rec.dterm_noise_rms_pct, "limit_pct": limit},
            )
        )
    return out


def check_measured_dterm_noise(context: GuidanceContext) -> list[Finding]:
    """What the derivative term was measured doing, not what it is predicted to do.

    Separate from :func:`check_dterm_noise` and deliberately so. That one is about
    the tune being recommended and is only as right as the noise model behind it;
    this one is about the aircraft that flew, and needs no model at all. When they
    disagree, the measurement is the one to believe.
    """
    limit = context.config.float_("noise", "dterm_output_rms_limit_pct")
    above = context.config.float_("noise", "dterm_measure_above_hz")
    out: list[Finding] = []
    for axis in context.axes():
        measured = context.analyses[axis].dterm_measured_pct
        if measured is None or not np.isfinite(measured) or measured <= limit:
            continue
        rec = context.recommendations.get(axis)
        predicted = rec.dterm_noise_rms_pct if rec is not None else float("nan")
        out.append(
            Finding(
                severity="warning",
                code="DTERM_NOISE_MEASURED",
                title=f"{axis}: the flown D term put {measured:.1f}% noise into the motors",
                detail=(
                    f"Above {above:.0f} Hz the logged {axis} derivative term has an RMS of "
                    f"{measured:.1f}% of full motor range, against a {limit:.0f}% limit. The "
                    f"loop has no authority up there, so none of that is control -- it is "
                    f"heat in the motors and ESCs, and it is what makes an over-filtered or "
                    f"over-D'd vehicle feel harsh. This is measured from PIDR rather than "
                    f"predicted, so it stands whatever the noise model says."
                    + (
                        ""
                        if not np.isfinite(predicted)
                        else f" The recommended tune is predicted to produce {predicted:.1f}%."
                    )
                ),
                action=(
                    f"Fix the noise mechanically first -- balance, mounts, isolation. If it "
                    f"is already as clean as it will get, lower {_d_gain_name(context)}. "
                    f"Filtering it away costs phase and buys back less than it removes."
                ),
                evidence={
                    "measured_pct": measured,
                    "limit_pct": limit,
                    "above_hz": above,
                },
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Nonlinear controller behaviour, which the LTI model cannot see
# --------------------------------------------------------------------------- #


def check_slew_limiter(context: GuidanceContext) -> list[Finding]:
    """``SMAX`` scaling P and D down is invisible to every margin in this report."""
    out: list[Finding] = []
    for axis in context.axes():
        key = f"rate.{axis}.dmod"
        if key not in context.bundle.signals:
            continue
        dmod = context.bundle.signals[key].y
        duty = float(np.mean(dmod < 0.99)) if dmod.size else 0.0
        if duty <= _LIMITER_DUTY:
            continue
        out.append(
            Finding(
                severity="warning",
                code="SLEW_LIMITER_ACTIVE",
                title=f"{axis}: slew limiter active {duty * 100:.0f}% of the flight",
                detail=(
                    f"SMAX scaled P and D down over {duty * 100:.0f}% of the record, with a "
                    f"minimum modifier of {float(np.min(dmod)):.2f}. The vehicle was "
                    f"therefore not flying the gains it was configured with, and none of the "
                    f"margins in this report describe what it actually did."
                ),
                action=(
                    "This usually means the gains are above what the airframe can carry. "
                    "Apply the recommendation, re-fly, and check the limiter stays idle."
                ),
                evidence={"duty": duty, "min_dmod": float(np.min(dmod))},
            )
        )
    return out


def check_integrator_windup(context: GuidanceContext) -> list[Finding]:
    """An integrator sitting on its limit is not integrating."""
    out: list[Finding] = []
    for axis in context.axes():
        key = f"rate.{axis}.i_term"
        imax = context.bundle.param(_imax_name(context.bundle.stack, axis))
        if key not in context.bundle.signals or not imax:
            continue
        i_term = np.abs(context.bundle.signals[key].y)
        duty = float(np.mean(i_term >= 0.98 * imax)) if i_term.size else 0.0
        if duty <= _LIMITER_DUTY:
            continue
        out.append(
            Finding(
                severity="warning",
                code="INTEGRATOR_WINDUP",
                title=f"{axis}: integrator saturated {duty * 100:.0f}% of the flight",
                detail=(
                    f"The I term sat at its IMAX limit of {imax:g} for {duty * 100:.0f}% of "
                    f"the record. A saturated integrator cannot correct anything, and it "
                    f"unwinds slowly afterwards -- which is felt as a delayed, wallowing "
                    f"response rather than as an obvious fault."
                ),
                action=(
                    "Look for a persistent trim offset first: a bent arm, a heavy battery "
                    "off centre, or a mis-levelled FC. Raising IMAX hides the cause."
                ),
                evidence={"duty": duty, "imax": float(imax)},
            )
        )
    return out


# --------------------------------------------------------------------------- #
# The recommendation itself
# --------------------------------------------------------------------------- #


def check_gain_step_size(context: GuidanceContext) -> list[Finding]:
    """A large jump is not wrong, but it should never arrive unannounced."""
    limit = context.config.float_("design", "max_gain_step_ratio")
    out: list[Finding] = []
    for axis in context.axes():
        rec = context.recommendations.get(axis)
        if rec is None:
            continue
        ratios = {
            "P": _ratio(rec.gains.kp, rec.baseline_gains.kp),
            "I": _ratio(rec.gains.ki, rec.baseline_gains.ki),
            "D": _ratio(rec.gains.kd, rec.baseline_gains.kd),
        }
        worst = max(ratios.values())
        if worst <= limit:
            continue
        biggest = max(ratios, key=lambda k: ratios[k])
        out.append(
            Finding(
                severity="warning",
                code="GAINS_FAR_FROM_CURRENT",
                title=f"{axis}: {biggest} changes by {worst:.1f}x",
                detail=(
                    f"The recommended {biggest} gain is {worst:.1f} times away from what was "
                    f"flown. The design says the margins hold, but the model behind it was "
                    f"identified at one operating point on one flight, and a step this large "
                    f"leaves no room for it to be slightly wrong."
                ),
                action=(
                    "Move roughly half way, fly, and confirm the response before applying "
                    "the rest. Watch for oscillation on sharp stick inputs."
                ),
                evidence={f"{k}_ratio": v for k, v in ratios.items()},
            )
        )
    return out


def check_operating_point(context: GuidanceContext) -> list[Finding]:
    """Whether the vehicle the model describes was the same vehicle throughout.

    A gain spread is not a fit problem and must not read as one. The
    identification can be excellent and the aircraft still be a different plant
    at 70% throttle than at hover, and the response to that is compensation --
    thrust linearization, battery scaling -- rather than a different tune.
    """
    out: list[Finding] = []
    warn = context.config.float_("operating_point", "warn_spread_pct")
    severe = context.config.float_("operating_point", "severe_spread_pct")
    for axis in context.axes():
        spread = context.analyses[axis].spread
        if spread is None:
            continue
        evidence = {
            "spread_pct": round(spread.spread_pct, 1),
            "n_operating_points": float(len(spread.samples)),
        }
        if spread.throttle_r is not None:
            evidence["throttle_r"] = round(spread.throttle_r, 2)
        if spread.voltage_r is not None:
            evidence["voltage_r"] = round(spread.voltage_r, 2)

        if spread.spread_pct <= warn:
            out.append(
                Finding(
                    severity="good",
                    code="OPERATING_POINT_STABLE",
                    title=f"{axis}: gain holds across the envelope",
                    detail=(
                        f"The {axis} airframe gain varied {spread.spread_pct:.0f}% across "
                        f"{len(spread.samples)} operating points in this flight. A tune "
                        f"designed at one of them is a tune that holds at the others."
                    ),
                    action="Nothing.",
                    evidence=evidence,
                    plot_hint="operating_point",
                )
            )
            continue

        if spread.attributed_to_throttle and spread.attributed_to_voltage:
            detail_tail = (
                f"It tracks throttle (r = {spread.throttle_r:+.2f}) and pack voltage "
                f"(r = {spread.voltage_r:+.2f}) about equally well, and in this flight "
                f"those two moved together -- so which of them is the cause cannot be "
                f"read off this log."
            )
            action = (
                "Fly again holding one of the two roughly constant: a flight at steady "
                "throttle separates the voltage effect, and a flight on a fresh pack "
                "separates the thrust curve."
            )
        elif spread.attributed_to_throttle:
            detail_tail = (
                f"It moves with throttle (r = {spread.throttle_r:+.2f}), which is what an "
                f"uncompensated or mis-set thrust curve looks like: the same stick "
                f"deflection buys a different angular acceleration at different power."
            )
            action = (
                "Check MOT_THST_EXPO against your propeller and motor combination"
                if context.bundle.stack == "ardupilot"
                else "Check THR_MDL_FAC against your propeller and motor combination"
            ) + ", then re-fly. Until it is right, no single set of gains is correct."
        elif spread.attributed_to_voltage:
            detail_tail = (
                f"It moves with pack voltage (r = {spread.voltage_r:+.2f}), so the vehicle "
                f"is a different plant at the end of the flight than at the start."
            )
            action = (
                "Enable battery voltage compensation (MOT_BAT_VOLT_MAX / MOT_BAT_VOLT_MIN) "
                if context.bundle.stack == "ardupilot"
                else "Enable battery-scaled thrust compensation "
            ) + "and re-fly, so one tune covers the whole pack."
        else:
            detail_tail = (
                "Nothing this log measured explains it -- not throttle, not pack voltage. "
                "Payload changes, a damaged propeller or a loose arm all produce this."
            )
            action = (
                "Fly again with the vehicle in one configuration throughout. If the spread "
                "persists, look for something mechanical before trusting any tune."
            )

        out.append(
            Finding(
                severity="warning",
                code=_spread_code(spread),
                title=(f"{axis}: airframe gain moved {spread.spread_pct:.0f}% across the flight"),
                detail=(
                    f"Measured over {len(spread.samples)} segments at different points in "
                    f"the envelope, holding the identified shape fixed and letting only "
                    f"the gain move. {detail_tail}"
                    + (
                        " That is past the point where a single-operating-point tune means "
                        "much, so the confidence on this axis is capped at low."
                        if spread.spread_pct > severe
                        else " The design holds extra margin back to cover it."
                    )
                ),
                action=action,
                evidence=evidence,
                plot_hint="operating_point",
                doc_link=_doc(context),
            )
        )
    return out


def _spread_code(spread: OperatingPointSpread) -> str:
    """Which finding a gain spread is, by what the log can actually attribute it to.

    Two strong correlations do not make a stronger claim than one -- they make a
    weaker one, because the log cannot separate them. So the specific codes are
    reserved for the case where exactly one variable explains the spread, and
    everything else is the honest generic.
    """
    if spread.attributed_to_throttle and spread.attributed_to_voltage:
        return "OPERATING_POINT_SPREAD"
    if spread.attributed_to_throttle:
        return "THRUST_LINEARIZATION_SUSPECT"
    if spread.attributed_to_voltage:
        return "BATTERY_SAG_LARGE"
    return "OPERATING_POINT_SPREAD"


def check_vendor_tune(context: GuidanceContext) -> list[Finding]:
    """Whether the vehicle's own tuner reached the same answer (spec 6.2).

    PX4's autotune identifies an ARX model in flight and derives gains from it.
    That is a second opinion about the same aircraft, produced by different code
    from different data, and it is the only external check this tool ever gets.

    Agreement is worth reporting because it is worth something: two independent
    estimates landing in the same place is evidence neither one can provide
    alone. Disagreement is worth more, because one of them is wrong about the
    vehicle the user is about to fly and no amount of internal consistency will
    reveal which.
    """
    out: list[Finding] = []
    for vendor in context.bundle.vendor_tunes:
        recommendation = context.recommendations.get(vendor.axis)
        if recommendation is None:
            continue
        ours, theirs = recommendation.gains, vendor.gains
        ratio = max(_ratio(theirs.kp, ours.kp), _ratio(theirs.kd, ours.kd))
        evidence = {
            "their_kp": theirs.kp,
            "our_kp": ours.kp,
            "their_kd": theirs.kd,
            "our_kd": ours.kd,
            "worst_ratio": ratio,
        }
        if vendor.fitness is not None:
            evidence["their_fitness"] = vendor.fitness

        if ratio <= _VENDOR_AGREEMENT_RATIO:
            out.append(
                Finding(
                    severity="good",
                    code="VENDOR_TUNE_AGREES",
                    title=f"{vendor.axis}: PX4's own autotune reached similar gains",
                    detail=(
                        f"PX4's autotune, running on this flight, arrived at P "
                        f"{theirs.kp:.4f} and D {theirs.kd:.5f}; this analysis recommends "
                        f"P {ours.kp:.4f} and D {ours.kd:.5f}. Two estimators with nothing "
                        f"in common but the aircraft agreeing to within "
                        f"{(ratio - 1.0) * 100:.0f}% is evidence neither of them can "
                        f"produce on its own."
                    ),
                    action="Nothing.",
                    evidence=evidence,
                )
            )
            continue

        out.append(
            Finding(
                severity="warning",
                code="VENDOR_TUNE_DISAGREES",
                title=(
                    f"{vendor.axis}: PX4's own autotune reached gains {ratio:.1f}x away from these"
                ),
                detail=(
                    f"PX4's autotune arrived at P {theirs.kp:.4f} and D {theirs.kd:.5f} on "
                    f"this flight; this analysis recommends P {ours.kp:.4f} and D "
                    f"{ours.kd:.5f}. Both were derived from the same aircraft, so one of "
                    f"them describes a vehicle that does not exist -- and no amount of "
                    f"internal consistency in either will say which."
                    + (
                        f" PX4 reports its own fit quality as {vendor.fitness:.3g}."
                        if vendor.fitness is not None
                        else ""
                    )
                ),
                action=(
                    "Fly the more conservative of the two first, at altitude, and compare "
                    "the result against the prediction on the Validate stage. That is the "
                    "only evidence that settles it."
                ),
                evidence=evidence,
            )
        )
    return out


def check_confidence(context: GuidanceContext) -> list[Finding]:
    """A low-confidence recommendation must be acknowledged, not merely noticed."""
    out: list[Finding] = []
    for axis in context.axes():
        rec = context.recommendations.get(axis)
        if rec is None or rec.confidence != "low":
            continue
        model = rec.model
        out.append(
            Finding(
                severity="blocker",
                code="LOW_CONFIDENCE_MODEL",
                title=f"{axis}: the identification is weak",
                detail=(
                    f"The model fitted to {model.fit_rms_db:.1f} dB and "
                    f"{model.fit_rms_deg:.0f} degrees RMS over "
                    f"{model.valid_band_hz[0]:.2f}-{model.valid_band_hz[1]:.1f} Hz at mean "
                    f"coherence {model.coherence_mean:.2f}. Gains designed against it may be "
                    f"right; there is not enough evidence in this log to say so."
                ),
                action=(
                    "Fly a proper SYSTEMID sweep before using these numbers. To export "
                    "anyway, acknowledge this finding explicitly -- it is recorded in the "
                    "export header."
                ),
                evidence={
                    "fit_rms_db": model.fit_rms_db,
                    "fit_rms_deg": model.fit_rms_deg,
                    "coherence_mean": model.coherence_mean,
                },
            )
        )
    return out


_AP_SUFFIX: dict[Axis, str] = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}


def _ratio(new: float, old: float) -> float:
    """Fold change, always >= 1, and 1.0 where the comparison is meaningless."""
    if old <= 0.0 or new <= 0.0:
        return 1.0
    return max(new / old, old / new)


#: Every check, in the order they are run. Order does not affect the output --
#: findings are sorted by severity -- but keeping related checks adjacent makes
#: the module readable.
CHECKS: tuple[Check, ...] = (
    check_log_kind,
    check_vibration,
    check_clipping,
    check_oscillation,
    check_step_response,
    check_excitation,
    check_identification_band,
    check_log_rate,
    check_estimator,
    check_prefilter_data,
    check_esc_telemetry,
    check_cpu_headroom,
    check_pid_messages,
    check_filter_model_agreement,
    check_structural_resonance,
    check_gyro_lpf_separation,
    check_dterm_noise,
    check_measured_dterm_noise,
    check_slew_limiter,
    check_integrator_windup,
    check_gain_step_size,
    check_operating_point,
    check_vendor_tune,
    check_confidence,
)
