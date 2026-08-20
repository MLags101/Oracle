"""Log in, recommendation out (spec sections 5.3 through 5.8, assembled).

This is the walking skeleton the whole tool hangs off: one function that takes a
log and an axis and returns a fully traceable
:class:`~rotorid.core.types.TuneRecommendation`. Everything it calls has its own
tests; what this module is responsible for is the *order*, and one specific
piece of order matters more than the rest:

    measure the effective plant -> divide the chain out -> fit -> design against
    the chain multiplied back in

Do those in any other order and filter phase is counted twice or not at all.

Filters and gains are designed together (:mod:`rotorid.core.design.joint`), from
the noise the same log recorded. Where a log carries no usable noise evidence the
flown chain is kept and labelled as a deliberate decision, never as a silent
no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotorid.config import Config
from rotorid.core.analysis.margins import LoopDelay, design_grid, loop_delay
from rotorid.core.analysis.noise import MotorTrack, motor_track, noise_profile
from rotorid.core.analysis.spectra import choose_nperseg, combine, estimate_frf
from rotorid.core.analysis.step import step_metrics, step_response
from rotorid.core.analysis.sysid import DeconvolvedPlant, deconvolve, fit_airframe
from rotorid.core.design.controller import controller_for
from rotorid.core.design.joint import JointResult, optimize_jointly
from rotorid.core.design.objectives import DesignResult, DesignTargets
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.filters.latency import actuator_latency_ms, build_budget
from rotorid.core.preprocess.params import (
    chain_from_bundle,
    gains_from_bundle,
    hover_operating_point,
)
from rotorid.core.preprocess.segment import propose_segments
from rotorid.core.types import (
    AirframeModel,
    Axis,
    Confidence,
    EffectivePlant,
    ExcitationSegment,
    FloatArray,
    LogBundle,
    NoiseProfile,
    TuneRecommendation,
)

__all__ = [
    "AxisAnalysis",
    "analyze_axis",
    "evidence_ceiling_hz",
    "identify_axis",
    "recommend_from",
]


def evidence_ceiling_hz(bundle: LogBundle, *keys: str) -> float | None:
    """Highest frequency the *logged* signals can say anything about.

    The analysis grid is chosen from the vehicle's gyro and loop rates, which is
    right for modelling the filters but says nothing about how often the
    controller's own messages were written to the card. On ArduPilot those are
    two different schedules: with ``LOG_BITMASK`` bit 0 (ATTITUDE_FAST) clear,
    ``RATE`` goes to the card at 10 Hz while ``SCHED_LOOP_RATE`` still reads 400,
    and the resampler will happily spline that onto an 800 Hz grid. Coherence
    does not catch it -- both signals are interpolated by the same spline, so
    they stay beautifully correlated all the way up, and the fit comes back
    confident and wrong.

    So the identification band is capped here at the slowest contributing
    message's own Nyquist. Above it there is nothing but interpolation.

    Returns:
        The cap in Hz, or ``None`` if no named signal knows its own native rate.
    """
    nyquists = [
        n
        for key in keys
        if key in bundle.signals
        for n in (bundle.signals[key].native_nyquist_hz,)
        if n is not None
    ]
    return min(nyquists) if nyquists else None


#: Shortest gap between excitation segments worth measuring noise over. Below
#: this the spectrum is too coarse to separate the motor lines.
_MIN_NOISE_WINDOW_S = 5.0

#: Peak logged CPU load above which the expensive notch options are not offered.
_CPU_HEADROOM_FOR_PER_MOTOR = 0.5


@dataclass(frozen=True, slots=True)
class AxisAnalysis:
    """Everything the identification produced, before any design decision.

    Kept separate from the recommendation so the GUI can show the identification
    and let the user argue with it before any gains are proposed -- and so that
    moving a design slider never re-runs the expensive part.
    """

    axis: Axis
    segments: tuple[ExcitationSegment, ...]
    effective: EffectivePlant
    deconvolved: DeconvolvedPlant
    airframe: AirframeModel
    chain: FilterChain
    operating_point: OperatingPoint
    delay: LoopDelay
    noise: NoiseProfile | None
    track: MotorTrack | None


def identify_axis(bundle: LogBundle, axis: Axis, config: Config) -> AxisAnalysis:
    """Measure the effective plant on one axis and recover the airframe from it.

    Every usable segment on the axis is estimated separately and merged by summing
    spectra, which is ArduPilot's own published methodology for repeated sweeps
    and is strictly better than analysing the longest segment alone.

    Raises:
        ValueError: if the log has no usable excitation on this axis, or if the
            required signals are missing. Both are conditions the user can fix by
            flying differently, and saying so beats returning a weak answer.
    """
    segments = tuple(s for s in propose_segments(bundle) if s.axis == axis)
    if not segments:
        how = (
            "Fly an ArduPilot SYSTEMID sweep on this axis (see docs/logging-setup-ardupilot.md)."
            if bundle.stack == "ardupilot"
            else (
                "PX4 has no SYSTEMID mode, so the excitation has to come from "
                "somewhere else: run the multicopter autotune on this axis, or fly "
                "deliberate single-axis stick sweeps from slow to fast with the "
                "other two axes held still."
            )
        )
        raise ValueError(f"no usable excitation found on {axis}. {how}")

    measured_key = f"rate.{axis}.measured"
    excite_key = f"excite.{axis}"
    output_key = f"rate.{axis}.output"
    if measured_key not in bundle.signals:
        raise ValueError(f"{measured_key} is not in the log; nothing to identify against")

    # The injected chirp is a far better reference than the mixer command, which
    # also contains the controller reacting to the vehicle's own motion.
    if excite_key in bundle.signals:
        input_key, source = excite_key, "injected_chirp"
    elif output_key in bundle.signals:
        input_key, source = output_key, "mixer_cmd"
    else:
        raise ValueError(f"neither {excite_key} nor {output_key} is in the log")

    chain = chain_from_bundle(bundle, axis)
    fs = bundle.sample_rate_hz
    f_lowest = min((s.f_start_hz or 0.5) for s in segments)
    min_averages = config.int_("spectra", "min_averages")

    estimates = []
    for segment in segments:
        u, y = _windowed(bundle, input_key, measured_key, segment)
        try:
            nperseg = choose_nperseg(u.size, fs, f_lowest_hz=f_lowest, min_averages=min_averages)
        except ValueError:
            continue
        estimates.append(
            estimate_frf(
                u, y, fs, nperseg=nperseg, input_signal=input_key, output_signal=measured_key
            )
        )
    if not estimates:
        raise ValueError(
            f"every {axis} segment is too short to resolve {f_lowest:g} Hz; "
            "lengthen SID_T_REC or start the sweep higher"
        )

    f_stop = max((s.f_stop_hz or fs / 4.0) for s in segments)
    ceiling = evidence_ceiling_hz(bundle, input_key, measured_key)
    if ceiling is not None:
        f_stop = min(f_stop, ceiling)
    if f_stop <= f_lowest:
        raise ValueError(
            f"{axis}: the excitation starts at {f_lowest:g} Hz but the log only carries "
            f"information to {ceiling:g} Hz "
            f"({bundle.signal(measured_key).source_msg} was logged at "
            f"{bundle.signal(measured_key).native_rate_hz:.0f} Hz). "
            "There is no band left to identify over."
        )
    band = (f_lowest, f_stop)
    frf = combine(estimates).to_frf(
        coherence_threshold=config.float_("coherence", "threshold"), band_hz=band
    )
    effective = EffectivePlant(
        axis=axis,
        frf=frf,
        filters_included=True,
        source=source,  # type: ignore[arg-type]
    )

    op = hover_operating_point(bundle, segments[0].t_start, segments[-1].t_end)
    plant = deconvolve(
        effective, chain, op=op, floor_db=config.float_("filters", "deconv_floor_db")
    )
    airframe = fit_airframe(
        plant,
        wn_bounds_hz=_pair(config, "fit", "wn_bounds_hz"),
        zeta_bounds=_pair(config, "fit", "zeta_bounds"),
        tau_bounds_ms=_pair(config, "fit", "tau_bounds_ms"),
    )

    noise, track = _noise_for(bundle, axis, segments, chain, op, config, ceiling)

    return AxisAnalysis(
        axis=axis,
        segments=segments,
        effective=effective,
        deconvolved=plant,
        airframe=airframe,
        chain=chain,
        operating_point=op,
        delay=_delay_for(bundle, config),
        noise=noise,
        track=track,
    )


def analyze_axis(
    bundle: LogBundle, axis: Axis, config: Config, *, conservatism: float = 0.5
) -> TuneRecommendation:
    """Identify one axis and design gains for it against its own filter chain."""
    analysis = identify_axis(bundle, axis, config)
    return recommend_from(analysis, bundle, config, conservatism=conservatism)


def recommend_from(
    analysis: AxisAnalysis,
    bundle: LogBundle,
    config: Config,
    *,
    conservatism: float = 0.5,
    chain_override: FilterChain | None = None,
) -> TuneRecommendation:
    """Design against an identification that has already been done.

    Split out from :func:`analyze_axis` because the sandbox re-solves this part on
    every slider movement and must not re-run the identification to do it.

    Args:
        chain_override: A filter chain chosen by hand. Skips filter design and
            designs the gains against exactly that chain, so what the sandbox
            shows is what the aircraft would do -- including when the hand-built
            chain is worse than the recommended one.
    """
    targets = DesignTargets(
        pm_min_deg=config.float_("margins", "pm_min_deg"),
        gm_min_db=config.float_("margins", "gm_min_db"),
        ms_max_db=config.float_("margins", "ms_max_db"),
        pm_floor_deg=config.float_("margins", "pm_floor_deg"),
        crossover_frac_of_loop=config.float_("margins", "crossover_frac_of_loop"),
        conservatism=conservatism,
    )
    band = analysis.deconvolved.valid_band_hz
    f_grid = design_grid(min(band[0], 0.1), max(band[1] * 4.0, 100.0))

    joint = optimize_jointly(
        analysis.airframe,
        analysis.chain,
        analysis.noise,
        config,
        stack=bundle.stack,
        delay=analysis.delay,
        targets=targets,
        f_grid=f_grid,
        op=analysis.operating_point,
        axis=analysis.axis,
        track=analysis.track,
        hover_thrust=bundle.param("MOT_THST_HOVER"),
        fft_available=bool(bundle.param("FFT_ENABLE", 0.0)),
        per_motor_capable=_per_motor_capable(bundle),
        chain_override=chain_override,
    )
    result = joint.design
    designed_chain = joint.filters.chain

    controller = controller_for(bundle.stack, result.gains, designed_chain)
    t, y = step_response(
        controller, analysis.airframe, delay=analysis.delay, op=analysis.operating_point
    )
    budget = build_budget(
        result.margins.crossover_hz,
        chain=designed_chain,
        airframe_tau_s=float(analysis.airframe.params.get("tau", 0.0)),
        actuator_ms=analysis.delay.actuator_s * 1000.0,
        zoh_loops=config.float_("design", "zoh_delay_loops"),
        compute_loops=config.float_("design", "compute_delay_loops"),
        op=analysis.operating_point,
    )

    return TuneRecommendation(
        axis=analysis.axis,
        gains=result.gains,
        baseline_gains=gains_from_bundle(bundle, analysis.axis),
        filters=joint.filters,
        model=analysis.airframe,
        margins=result.margins,
        latency=budget,
        predicted_step=step_metrics(t, y),
        dterm_noise_rms_pct=joint.dterm_noise_rms * 100.0,
        rationale=_rationale(analysis, joint),
        confidence=_confidence(analysis, config),
        conservatism=conservatism,
        binding_constraint=result.binding_constraint,
    )


def _pair(config: Config, section: str, key: str) -> tuple[float, float]:
    """A config list that must be exactly a low/high bound.

    Raises:
        ValueError: if the list is not a pair. A three-element "range" would
            otherwise be silently truncated to its first two entries.
    """
    values = config.floats(section, key)
    if len(values) != 2:
        raise ValueError(f"[{section}].{key} must be a [low, high] pair, got {list(values)}")
    return values[0], values[1]


def _windowed(
    bundle: LogBundle, input_key: str, output_key: str, segment: ExcitationSegment
) -> tuple[FloatArray, FloatArray]:
    u_sig, y_sig = bundle.signal(input_key), bundle.signal(output_key)
    window = (u_sig.t >= segment.t_start) & (u_sig.t <= segment.t_end)
    return u_sig.y[window], y_sig.y[window]


def _delay_for(bundle: LogBundle, config: Config) -> LoopDelay:
    latency = config.section("design")["actuator_latency_ms"]
    if not isinstance(latency, dict):
        raise ValueError("[design.actuator_latency_ms] must be a table of protocol to ms")
    table = {str(k): float(v) for k, v in latency.items()}
    return loop_delay(
        loop_rate_hz=bundle.loop_rate_hz,
        actuator_ms=actuator_latency_ms(bundle.params, table),
        zoh_loops=config.float_("design", "zoh_delay_loops"),
        compute_loops=config.float_("design", "compute_delay_loops"),
    )


def _noise_for(
    bundle: LogBundle,
    axis: Axis,
    segments: tuple[ExcitationSegment, ...],
    chain: FilterChain,
    op: OperatingPoint,
    config: Config,
    evidence_ceiling_hz: float | None = None,
) -> tuple[NoiseProfile | None, MotorTrack | None]:
    """Characterize gyro noise, preferring the part of the flight that is not a sweep.

    During a SYSTEMID sweep the vehicle is being deliberately shaken, and that
    motion is signal rather than noise. Where the log has quiet flight outside the
    segments, the noise is measured there instead. Failure here is not fatal: an
    axis with no usable noise measurement still gets gains, and the filter
    recommendation says why it is absent.
    """
    start, end = _quiet_window(bundle, axis, segments)
    try:
        profile = noise_profile(
            bundle,
            axis,
            t_start=start,
            t_end=end,
            chain=chain,
            op=op,
            prominence_db=config.float_("noise", "peak_prominence_db"),
            track_margin_db=config.float_("noise", "rpm_track_margin_db"),
            deconv_floor_db=config.float_("filters", "deconv_floor_db"),
            evidence_ceiling_hz=evidence_ceiling_hz,
        )
    except ValueError:
        return None, None
    return profile, motor_track(bundle, start, end)


def _quiet_window(
    bundle: LogBundle, axis: Axis, segments: tuple[ExcitationSegment, ...]
) -> tuple[float, float]:
    """The longest stretch of the log outside any excitation segment.

    Falls back to the whole record when the sweeps leave nothing behind them --
    the noise peaks are still there under the excitation, just harder to see.
    """
    signal = bundle.signals[f"rate.{axis}.measured"]
    t0, t1 = float(signal.t[0]), float(signal.t[-1])
    if not segments:
        return t0, t1

    edges = [t0, *[t for s in segments for t in (s.t_start, s.t_end)], t1]
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    best = max(gaps, key=lambda g: g[1] - g[0], default=(t0, t1))
    if best[1] - best[0] < _MIN_NOISE_WINDOW_S:
        return t0, t1
    return best


def _per_motor_capable(bundle: LogBundle) -> bool:
    """Whether per-motor notch tracking is both possible and affordable.

    Needs measured per-motor RPM, and enough CPU headroom that several times the
    notch count will not push the loop over. Where the log does not say what the
    load was, the answer is no: a board that cannot keep up does not fail
    gracefully.
    """
    motors = sum(1 for i in range(1, 13) if f"motor.{i}.rpm" in bundle.signals)
    if motors < 2 or "cpu.load" not in bundle.signals:
        return False
    return float(np.max(bundle.signals["cpu.load"].y)) < _CPU_HEADROOM_FOR_PER_MOTOR


def _rationale(analysis: AxisAnalysis, joint: JointResult) -> str:
    result: DesignResult = joint.design
    model = analysis.airframe
    band = model.valid_band_hz
    filters = (
        f"Filters redesigned: {joint.filters.chain.describe()}, costing "
        f"{joint.filters.phase_cost_deg:.1f} deg at crossover."
        if joint.filters_changed
        else f"Chain: {analysis.chain.describe()}."
    )
    return (
        f"Identified {model.structure} over {band[0]:.2f}-{band[1]:.1f} Hz from "
        f"{len(analysis.segments)} segment(s) at mean coherence "
        f"{model.coherence_mean:.2f}, with the filter chain divided out "
        f"({model.filter_deconvolution}). Fit residual {model.fit_rms_db:.2f} dB / "
        f"{model.fit_rms_deg:.1f} deg. Gains maximize disturbance-rejection "
        f"bandwidth ({result.margins.disturbance_rejection_bw_hz:.2f} Hz) subject to "
        f"the margin constraints; {result.binding_constraint} is what stops them "
        f"going higher. {filters}"
    )


def _confidence(analysis: AxisAnalysis, config: Config) -> Confidence:
    """How much the identification deserves to be trusted.

    Driven by the evidence rather than by the fit residual alone: a model can fit
    a narrow, weakly excited band beautifully and still describe the aircraft
    badly.
    """
    excitation = max(s.confidence for s in analysis.segments)
    band = analysis.deconvolved.valid_band_hz
    octaves = float(np.log2(band[1] / band[0])) if band[0] > 0.0 else 0.0
    model = analysis.airframe

    poor_fit = model.fit_rms_db > config.float_("fit", "max_rms_db") or (
        model.fit_rms_deg > config.float_("fit", "max_rms_deg")
    )
    if poor_fit or excitation < 0.5 or octaves < config.float_("coherence", "min_valid_octaves"):
        return "low"
    if excitation < 1.0 or model.coherence_mean < 0.8:
        return "medium"
    return "high"
