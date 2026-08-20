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

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from rotorid.config import Config
from rotorid.core.analysis.deconv import MeasuredStep, measured_step
from rotorid.core.analysis.instrument import Rung, choose_instrument, windowed_signals
from rotorid.core.analysis.margins import LoopDelay, design_grid, loop_delay
from rotorid.core.analysis.noise import (
    MotorTrack,
    measured_dterm_rms_pct,
    motor_track,
    noise_profile,
)
from rotorid.core.analysis.operating_point import OperatingPointSpread, gain_spread
from rotorid.core.analysis.oscillation import Oscillation, detect_oscillation, model_optimism_db
from rotorid.core.analysis.spectra import (
    InstrumentedEstimate,
    SpectralEstimate,
    choose_nperseg,
    combine,
    combine_iv,
    estimate_frf,
    estimate_frf_iv,
)
from rotorid.core.analysis.step import step_metrics, step_response
from rotorid.core.analysis.sysid import DeconvolvedPlant, deconvolve, fit_airframe
from rotorid.core.design.controller import controller_for
from rotorid.core.design.joint import JointResult, optimize_jointly
from rotorid.core.design.objectives import DesignResult, DesignTargets
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.filters.latency import actuator_latency_ms, build_budget
from rotorid.core.logkind import capabilities
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
    FrequencyResponse,
    LogBundle,
    LogKind,
    NoiseProfile,
    StepMetrics,
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
    #: The step the aircraft actually flew, deconvolved from the log. ``None``
    #: when the log could not support one -- which is a different statement from
    #: a flat step, and the two are never allowed to look alike.
    measured: MeasuredStep | None = None
    #: RMS of the logged derivative term above the control band, as a percentage
    #: of full motor range. Measured, where ``TuneRecommendation.dterm_noise_rms_pct``
    #: is predicted for a tune nobody has flown. ``None`` when the log carries no
    #: PID messages, which is not the same as a quiet D term.
    dterm_measured_pct: float | None = None
    #: A sustained tone the aircraft produced and nobody commanded. Carries how
    #: much gain margin this model wrongly claims at that frequency, which is what
    #: the design is made to hold back.
    oscillation: Oscillation | None = None
    #: How far the airframe gain moved between segments, and what it moved with
    #: (spec 5.9). ``None`` on a log whose kind cannot support the measurement --
    #: which is a different statement from a vehicle that held still, and the two
    #: are never allowed to look alike.
    spread: OperatingPointSpread | None = None
    #: What this model predicts the step would have been *under the gains the log
    #: was flown with*, over exactly the window ``measured`` covers. Not the
    #: recommended gains: comparing the flown response against a prediction for a
    #: different tune would only measure that the recommendation changed something.
    #: The pair is an end-to-end check of the identification against flown data --
    #: the only one in the tool that does not go through the model twice.
    flown_prediction: StepMetrics | None = None


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
        raise ValueError(_nothing_to_identify(bundle, axis))

    measured_key = f"rate.{axis}.measured"
    output_key = f"rate.{axis}.output"
    if measured_key not in bundle.signals:
        raise ValueError(f"{measured_key} is not in the log; nothing to identify against")

    instrument_key, rung = choose_instrument(bundle, axis)
    chain = chain_from_bundle(bundle, axis)
    fs = bundle.sample_rate_hz
    f_lowest = min((s.f_start_hz or 0.5) for s in segments)
    min_averages = config.int_("spectra", "min_averages")

    # Both estimators are computed on every segment. The instrument-variable one
    # is the answer; the direct one is kept so the two can be compared, which is
    # both the bias diagnostic and the check on the plant-input assembly.
    iv_estimates: list[InstrumentedEstimate] = []
    direct_estimates: list[SpectralEstimate] = []
    # Kept per segment as well as summed. The sum is the identification; these
    # are what says whether the aircraft the sum describes was the same aircraft
    # throughout (spec 5.9).
    by_segment: dict[ExcitationSegment, SpectralEstimate | InstrumentedEstimate] = {}
    rungs: set[Rung] = set()
    summed = False
    for segment in segments:
        cut = windowed_signals(bundle, axis, segment, instrument_key=instrument_key, rung=rung)
        try:
            nperseg = choose_nperseg(
                cut.plant_input.size, fs, f_lowest_hz=f_lowest, min_averages=min_averages
            )
        except ValueError:
            continue
        rungs.add(cut.rung)
        summed = summed or cut.summed_injection
        direct = estimate_frf(
            cut.plant_input,
            cut.response,
            fs,
            nperseg=nperseg,
            input_signal=cut.input_key,
            output_signal=cut.output_key,
        )
        direct_estimates.append(direct)
        by_segment[segment] = direct
        if cut.instrument is not None and cut.instrument_key is not None:
            instrumented = estimate_frf_iv(
                cut.instrument,
                cut.plant_input,
                cut.response,
                fs,
                nperseg=nperseg,
                instrument_signal=cut.instrument_key,
                input_signal=cut.input_key,
                output_signal=cut.output_key,
            )
            iv_estimates.append(instrumented)
            by_segment[segment] = instrumented
    if not direct_estimates:
        raise ValueError(
            f"every {axis} segment is too short to resolve {f_lowest:g} Hz; "
            "lengthen SID_T_REC or start the sweep higher"
        )
    # A segment where the stick sat still demotes only itself; mixing an
    # instrumented and an uninstrumented segment into one estimate would be
    # averaging an unbiased number with a biased one.
    if len(iv_estimates) != len(direct_estimates):
        iv_estimates = []
    effective_rung: Rung = rung if iv_estimates else "none"

    f_stop = max((s.f_stop_hz or fs / 4.0) for s in segments)
    ceiling = evidence_ceiling_hz(bundle, output_key, measured_key)
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
    threshold = config.float_("coherence", "threshold")
    direct_frf = combine(direct_estimates).to_frf(coherence_threshold=threshold, band_hz=band)

    if iv_estimates:
        frf = combine_iv(iv_estimates).to_frf(coherence_threshold=threshold, band_hz=band)
        estimator: Literal["instrument_variable", "direct_h1"] = "instrument_variable"
        bias_db, bias_deg = _estimator_disagreement(frf, direct_frf)
    else:
        frf, estimator = direct_frf, "direct_h1"
        bias_db, bias_deg = 0.0, 0.0

    effective = EffectivePlant(
        axis=axis,
        frf=frf,
        filters_included=True,
        source="injected_chirp" if effective_rung == "injected_chirp" else "mixer_cmd",
        estimator=estimator,
        instrument=instrument_key if iv_estimates else None,
        bias_db=bias_db,
        bias_deg=bias_deg,
    )

    op = hover_operating_point(bundle, segments[0].t_start, segments[-1].t_end)
    plant = deconvolve(
        effective, chain, op=op, floor_db=config.float_("filters", "deconv_floor_db")
    )
    airframe = fit_airframe(
        plant,
        wn_bounds_hz=config.pair("fit", "wn_bounds_hz"),
        zeta_bounds=config.pair("fit", "zeta_bounds"),
        tau_bounds_ms=config.pair("fit", "tau_bounds_ms"),
    )
    # The fit knows how the filters were removed but not how the loop was. Both
    # halves of that provenance travel with the model, because everything that
    # renders a model -- report, screen, explanation -- has to be able to say
    # whether it is describing the aircraft or its controller.
    airframe = replace(airframe, estimator=estimator, instrument=effective.instrument)

    spread = _operating_point_spread(bundle, airframe, by_segment, chain, config, band, threshold)
    if spread is not None:
        airframe = replace(airframe, gain_spread_pct=spread.spread_pct)

    noise, track = _noise_for(bundle, axis, segments, chain, op, config, ceiling)

    # Over the whole record, not the identification segments. Identification wants
    # the stretches that made a good frequency response; this wants every stick
    # input the pilot ever made, and a flight with one usable sweep can easily
    # have two hundred usable steps outside it.
    measured = measured_step(bundle, axis, config)
    delay = _delay_for(bundle, config)
    oscillation = _oscillation_for(bundle, axis, config, airframe, chain, delay, op, noise)
    dterm_measured = measured_dterm_rms_pct(
        bundle, axis, above_hz=config.float_("noise", "dterm_measure_above_hz")
    )
    flown_prediction = (
        _flown_prediction(bundle, axis, airframe, chain, delay, op, measured)
        if measured is not None
        else None
    )

    return AxisAnalysis(
        axis=axis,
        segments=segments,
        effective=effective,
        deconvolved=plant,
        airframe=airframe,
        chain=chain,
        operating_point=op,
        delay=delay,
        noise=noise,
        track=track,
        measured=measured,
        spread=spread,
        flown_prediction=flown_prediction,
        oscillation=oscillation,
        dterm_measured_pct=dterm_measured,
    )


def _spread_holdback(spread: OperatingPointSpread | None, config: Config) -> float:
    """Extra conservatism bought by a vehicle whose gain moves (spec 5.9).

    Ramped rather than stepped, because the underlying quantity is continuous and
    a threshold would make two nearly identical flights produce visibly different
    tunes. Zero below the warning level: every vehicle's gain moves a little, and
    charging for the ordinary case would just be a quieter default.
    """
    if spread is None:
        return 0.0
    warn = config.float_("operating_point", "warn_spread_pct")
    severe = config.float_("operating_point", "severe_spread_pct")
    bonus = config.float_("operating_point", "max_conservatism_bonus")
    if spread.spread_pct <= warn or severe <= warn:
        return 0.0
    return bonus * min((spread.spread_pct - warn) / (severe - warn), 1.0)


def _operating_point_spread(
    bundle: LogBundle,
    airframe: AirframeModel,
    by_segment: dict[ExcitationSegment, SpectralEstimate | InstrumentedEstimate],
    chain: FilterChain,
    config: Config,
    band: tuple[float, float],
    coherence_threshold: float,
) -> OperatingPointSpread | None:
    """How far the airframe gain moved between segments (spec 5.9).

    Only for a log whose kind offers it. A sweep is flown at one throttle on one
    battery state, so its "spread" would be three measurements of the same
    operating point differing by fit noise -- a number that looks like physics
    and is not.

    Each segment is deconvolved at *its own* operating point, so a notch that
    tracked throttle across the flight is divided out where it actually sat
    rather than where the hover average puts it.
    """
    if not capabilities(bundle.kind).allows("operating_point"):
        return None

    plants: dict[ExcitationSegment, DeconvolvedPlant] = {}
    throttle: dict[ExcitationSegment, float] = {}
    voltage: dict[ExcitationSegment, float] = {}
    floor_db = config.float_("filters", "deconv_floor_db")
    for segment, estimate in by_segment.items():
        op = hover_operating_point(bundle, segment.t_start, segment.t_end)
        frf = estimate.to_frf(coherence_threshold=coherence_threshold, band_hz=band)
        try:
            plants[segment] = deconvolve(
                EffectivePlant(
                    axis=segment.axis,
                    frf=frf,
                    filters_included=True,
                    source="mixer_cmd",
                ),
                chain,
                op=op,
                floor_db=floor_db,
            )
        except ValueError:
            continue
        if op.throttle is not None:
            throttle[segment] = op.throttle
        mean_voltage = _mean_over(bundle, "batt.voltage", segment.t_start, segment.t_end)
        if mean_voltage is not None:
            voltage[segment] = mean_voltage

    return gain_spread(airframe, plants, throttle=throttle, voltage=voltage)


def _mean_over(bundle: LogBundle, key: str, t_start: float, t_end: float) -> float | None:
    """Mean of one signal over a window, or ``None`` if the log has no such signal."""
    signal = bundle.signals.get(key)
    if signal is None:
        return None
    window = (signal.t >= t_start) & (signal.t <= t_end)
    if not window.any():
        return None
    return float(np.mean(signal.y[window]))


def _nothing_to_identify(bundle: LogBundle, axis: Axis) -> str:
    """Why this axis produced no segment, phrased against what the user declared.

    The two kinds fail for opposite reasons and the fixes are opposite too, so a
    single "no usable excitation" message would send half the users to change the
    wrong thing. A tuning flight with nothing in it means the sweep did not
    happen; a general flight with nothing in it means the pilot never moved that
    axis on its own.
    """
    if bundle.kind == "tuning":
        how = (
            "Fly an ArduPilot SYSTEMID sweep on this axis (see docs/logging-setup-ardupilot.md), "
            "or run the autotune."
            if bundle.stack == "ardupilot"
            else (
                "PX4 has no SYSTEMID mode, so the excitation has to come from the "
                "multicopter autotune (MC_AT_EN = 1, MC_AT_APPLY = 0)."
            )
        )
        how_it_got_here = (
            "was loaded as a tuning flight"
            if bundle.kind_was_declared
            else "carries deliberate excitation on another axis, so it is being read as a "
            "tuning flight"
        )
        return (
            f"{axis}: this log {how_it_got_here}, but nothing was deliberately excited on "
            f"{axis} itself. {how} If it was an ordinary flight, load it as a general "
            "flight log instead and it will be identified from the stick input it has."
        )
    return (
        f"{axis}: no stretch of this flight excites {axis} on its own for long enough to "
        "identify from. Ordinary flight only identifies an axis the pilot moved "
        "deliberately, for at least five seconds, while holding the other two still. "
        "Fly a tuning flight for a model that does not depend on what the pilot happened "
        "to do."
    )


def _oscillation_for(
    bundle: LogBundle,
    axis: Axis,
    config: Config,
    airframe: AirframeModel,
    chain: FilterChain,
    delay: LoopDelay,
    op: OperatingPoint,
    noise: NoiseProfile | None,
) -> Oscillation | None:
    """Detect a sustained oscillation and price the model's error against it.

    The optimism figure is computed against the gains the aircraft was *flying*,
    because that is the loop that produced the oscillation. Measuring it against
    the recommended gains would be asking how wrong a tune nobody has flown is.
    """
    found = detect_oscillation(bundle, axis, config, noise=noise, gyro_lpf_hz=chain.gyro_lpf_hz)
    if found is None:
        return None
    try:
        flown = gains_from_bundle(bundle, axis)
    except (KeyError, ValueError):
        return found
    optimism = model_optimism_db(
        found.f_hz,
        controller_for(bundle.stack, flown, chain),
        airframe,
        delay=delay,
        op=op,
    )
    return replace(found, model_optimism_db=optimism)


def _flown_prediction(
    bundle: LogBundle,
    axis: Axis,
    airframe: AirframeModel,
    chain: FilterChain,
    delay: LoopDelay,
    op: OperatingPoint,
    measured: MeasuredStep,
) -> StepMetrics | None:
    """Predicted step for the gains the log was flown with, over the measured window.

    Same duration and same sample rate as the measurement, because rise time and
    overshoot are both read off a finite record: a prediction computed over four
    seconds and a measurement covering half a second would disagree about settling
    for reasons that have nothing to do with the aircraft.
    """
    try:
        flown = gains_from_bundle(bundle, axis)
    except (KeyError, ValueError):
        return None
    t, y = step_response(
        controller_for(bundle.stack, flown, chain),
        airframe,
        delay=delay,
        op=op,
        duration_s=float(measured.t.size) / bundle.sample_rate_hz,
        sample_rate_hz=bundle.sample_rate_hz,
    )
    return step_metrics(t, y)


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
        conservatism: 0 aggressive, 1 docile. Raised to the floor the log's kind
            imposes (:mod:`rotorid.core.logkind`) if it is below it, so a slider
            at zero on a general flight still designs a general flight's tune.
        chain_override: A filter chain chosen by hand. Skips filter design and
            designs the gains against exactly that chain, so what the sandbox
            shows is what the aircraft would do -- including when the hand-built
            chain is worse than the recommended one.
    """
    # A general flight identifies a narrower band than a sweep does, so the
    # designer is not allowed to be as bold with it however good the fit looks.
    # Raised here rather than at the call site because the sandbox, the CLI and
    # the report all call this and would each have to remember.
    conservatism = max(conservatism, capabilities(bundle.kind).conservatism_floor)
    conservatism = min(1.0, conservatism + _spread_holdback(analysis.spread, config))

    targets = DesignTargets(
        pm_min_deg=config.float_("margins", "pm_min_deg"),
        gm_min_db=config.float_("margins", "gm_min_db"),
        ms_max_db=config.float_("margins", "ms_max_db"),
        pm_floor_deg=config.float_("margins", "pm_floor_deg"),
        crossover_frac_of_loop=config.float_("margins", "crossover_frac_of_loop"),
        conservatism=conservatism,
        # A measured oscillation is evidence that this model is optimistic by a
        # known number of decibels at a known frequency. Requiring that much more
        # gain margin is the only response that follows from the evidence: a tool
        # that measures an oscillating aircraft and then recommends more gain has
        # not understood what it measured. Capped, because past the cap the model
        # is not describing the aircraft at all and the blocker is the right
        # answer rather than a softer tune.
        gm_holdback_db=min(
            analysis.oscillation.model_optimism_db if analysis.oscillation else 0.0,
            config.float_("oscillation", "max_gm_holdback_db"),
        ),
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
        controller,
        analysis.airframe,
        delay=analysis.delay,
        op=analysis.operating_point,
        # Longer than the plot the Design stage draws, and for a different reason.
        # Rise and overshoot happen in the first fraction of a second either way,
        # but settling time and steady-state error are measured against where the
        # response ended up, and an integrator with a one-second time constant has
        # not finished in the second and a half a pilot judges the feel by. Cutting
        # it there would publish a steady-state error the tune does not have.
        duration_s=4.0,
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
        confidence=_confidence(analysis, config, bundle.kind),
        conservatism=conservatism,
        binding_constraint=result.binding_constraint,
    )


def _estimator_disagreement(
    unbiased: FrequencyResponse, direct: FrequencyResponse
) -> tuple[float, float]:
    """How far the direct estimate sits from the unbiased one, in dB and degrees.

    Median rather than mean, over the bins both estimates call valid: a couple of
    bins where the direct estimate diverges wildly say less about the flight than
    a consistent offset across the band does.

    This number does two jobs. Large values are a finding -- the log had enough
    feedback in it to move the answer, and the user should see by how much. And
    on a flight where the chirp dominates, the two *must* agree, which is what
    tests the plant-input assembly: if we reconstructed the wrong plant input for
    a mixer-injected sweep, this is where it shows up.
    """
    both = unbiased.valid_mask & direct.valid_mask
    if not both.any():
        return 0.0, 0.0
    a, b = unbiased.H[both], direct.H[both]
    usable = (np.abs(a) > 0.0) & (np.abs(b) > 0.0)
    if not usable.any():
        return 0.0, 0.0
    ratio = b[usable] / a[usable]
    return (
        float(np.median(20.0 * np.log10(np.abs(ratio)))),
        float(np.median(np.degrees(np.angle(ratio)))),
    )


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


def _confidence(analysis: AxisAnalysis, config: Config, kind: LogKind) -> Confidence:
    """How much the identification deserves to be trusted.

    Driven by the evidence rather than by the fit residual alone: a model can fit
    a narrow, weakly excited band beautifully and still describe the aircraft
    badly. The log's declared kind sets a ceiling over the top of that, because
    what the flight *was* bounds what any residual can prove about it.
    """
    ceiling = capabilities(kind).max_confidence
    excitation = max(s.confidence for s in analysis.segments)
    band = analysis.deconvolved.valid_band_hz
    octaves = float(np.log2(band[1] / band[0])) if band[0] > 0.0 else 0.0
    model = analysis.airframe

    poor_fit = model.fit_rms_db > config.float_("fit", "max_rms_db") or (
        model.fit_rms_deg > config.float_("fit", "max_rms_deg")
    )
    # A vehicle whose gain moves across the envelope has not been identified
    # badly -- it has been identified at one point, and the tune is about all of
    # them. That is a statement about confidence, not about fit quality.
    if analysis.spread is not None and analysis.spread.spread_pct > config.float_(
        "operating_point", "severe_spread_pct"
    ):
        return "low"

    if poor_fit or excitation < 0.5 or octaves < config.float_("coherence", "min_valid_octaves"):
        rating: Confidence = "low"
    elif excitation < 1.0 or model.coherence_mean < 0.8:
        rating = "medium"
    else:
        rating = "high"
    return _capped(rating, ceiling)


def _capped(rating: Confidence, ceiling: Confidence) -> Confidence:
    """The lower of two ratings.

    A ceiling, never a floor. What a log *is* can only take confidence away: a
    general flight that fits beautifully is still a general flight, and a tuning
    flight that fits badly does not become trustworthy because a sweep was flown.
    """
    order: tuple[Confidence, ...] = ("low", "medium", "high")
    return order[min(order.index(rating), order.index(ceiling))]
