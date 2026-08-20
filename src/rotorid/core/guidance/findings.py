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
from rotorid.core.analysis.sysid import check_filter_model
from rotorid.core.design.recommend import AxisAnalysis
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


def check_excitation(context: GuidanceContext) -> list[Finding]:
    """Whether anything in the log was worth identifying from."""
    out: list[Finding] = []
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
                    f"The {axis} identification used {len(analysis.segments)} segment(s) of "
                    f"ordinary flight rather than a SYSTEMID sweep. Pilot inputs excite a "
                    f"narrow, uneven band, so the model is fitted where the data happens to "
                    f"be rather than where the loop is designed."
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
    check_slew_limiter,
    check_integrator_windup,
    check_gain_step_size,
    check_confidence,
)
