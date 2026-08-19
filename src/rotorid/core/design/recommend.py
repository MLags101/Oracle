"""Log in, recommendation out (spec sections 5.3 through 5.8, assembled).

This is the walking skeleton the whole tool hangs off: one function that takes a
log and an axis and returns a fully traceable
:class:`~rotorid.core.types.TuneRecommendation`. Everything it calls has its own
tests; what this module is responsible for is the *order*, and one specific
piece of order matters more than the rest:

    measure the effective plant -> divide the chain out -> fit -> design against
    the chain multiplied back in

Do those in any other order and filter phase is counted twice or not at all.

Filters are not yet redesigned here: milestone M1 recommends gains against the
chain the vehicle is already flying, and says so. The
:class:`~rotorid.core.types.FilterRecommendation` it returns is therefore the
current chain with an explicit rationale, not a silent no-op -- so that when the
joint optimizer arrives the shape of the output does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotorid.config import Config
from rotorid.core.analysis.margins import LoopDelay, design_grid, loop_delay
from rotorid.core.analysis.spectra import choose_nperseg, combine, estimate_frf
from rotorid.core.analysis.step import step_metrics, step_response
from rotorid.core.analysis.sysid import DeconvolvedPlant, deconvolve, fit_airframe
from rotorid.core.design.controller import controller_for
from rotorid.core.design.objectives import DesignResult, DesignTargets, design_gains
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
    FilterRecommendation,
    FloatArray,
    LogBundle,
    TuneRecommendation,
)

__all__ = ["AxisAnalysis", "analyze_axis", "identify_axis"]


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
        raise ValueError(
            f"no usable excitation found on {axis}. Fly an ArduPilot SYSTEMID "
            f"sweep on this axis (see docs/logging-setup-ardupilot.md)."
        )

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

    band = (f_lowest, max((s.f_stop_hz or fs / 4.0) for s in segments))
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

    return AxisAnalysis(
        axis=axis,
        segments=segments,
        effective=effective,
        deconvolved=plant,
        airframe=airframe,
        chain=chain,
        operating_point=op,
        delay=_delay_for(bundle, config),
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
) -> TuneRecommendation:
    """Design against an identification that has already been done.

    Split out from :func:`analyze_axis` because the sandbox re-solves this part on
    every slider movement and must not re-run the identification to do it.
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

    result = design_gains(
        analysis.airframe,
        analysis.chain,
        stack=bundle.stack,
        delay=analysis.delay,
        targets=targets,
        f_grid=f_grid,
        op=analysis.operating_point,
        axis=analysis.axis,
    )

    controller = controller_for(bundle.stack, result.gains, analysis.chain)
    t, y = step_response(
        controller, analysis.airframe, delay=analysis.delay, op=analysis.operating_point
    )
    budget = build_budget(
        result.margins.crossover_hz,
        chain=analysis.chain,
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
        filters=_unchanged_filters(analysis.chain, bundle.stack),
        model=analysis.airframe,
        margins=result.margins,
        latency=budget,
        predicted_step=step_metrics(t, y),
        dterm_noise_rms_pct=float("nan"),
        rationale=_rationale(analysis, result),
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


def _unchanged_filters(chain: FilterChain, stack: str) -> FilterRecommendation:
    """The current chain, presented as a deliberate decision rather than a gap."""
    return FilterRecommendation(
        stack=stack,  # type: ignore[arg-type]
        chain=chain,
        baseline_chain=chain,
        params={},
        phase_cost_deg=0.0,
        cpu_cost_rel=chain.cpu_cost(),
        rationale=(
            "Filters left as flown. Gains are designed against the chain this log "
            "was recorded through, so they are valid only while that chain is "
            "unchanged. Joint filter and gain design arrives with milestone M4."
        ),
    )


def _rationale(analysis: AxisAnalysis, result: DesignResult) -> str:
    model = analysis.airframe
    band = model.valid_band_hz
    return (
        f"Identified {model.structure} over {band[0]:.2f}-{band[1]:.1f} Hz from "
        f"{len(analysis.segments)} segment(s) at mean coherence "
        f"{model.coherence_mean:.2f}, with the filter chain divided out "
        f"({model.filter_deconvolution}). Fit residual {model.fit_rms_db:.2f} dB / "
        f"{model.fit_rms_deg:.1f} deg. Gains maximize disturbance-rejection "
        f"bandwidth ({result.margins.disturbance_rejection_bw_hz:.2f} Hz) subject to "
        f"the margin constraints; {result.binding_constraint} is what stops them "
        f"going higher. Chain: {analysis.chain.describe()}."
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
