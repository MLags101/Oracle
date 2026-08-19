"""Filter design: notches and low-passes, chosen from evidence (spec section 5.7).

On most real multirotors it is the filter configuration, not the gain arithmetic,
that decides how much bandwidth the vehicle can have. Every filter buys
attenuation and pays phase lag at the crossover, out of one shared budget, so
this module never asks "what is the best notch?" -- it asks "what is the cheapest
set of filters that brings the measured peaks to the target floor?".

The search is a **deterministic ladder**, not an open optimization:

1. pick the tracking source from what the log proves is available;
2. include a harmonic only if a peak that actually tracks RPM sits on it;
3. pick the narrowest bandwidth and the least attenuation that reach the floor;
4. back off -- harmonics first, then bandwidth -- until the phase budget holds;
5. pick the gyro and D-term cutoffs from the residual noise that is left.

An enumerable ladder can be explained; a numerical search over the same space
cannot, and the tool has to justify every number it prints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from rotorid.config import Config
from rotorid.core.analysis.noise import MotorTrack, dterm_noise_rms
from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.filters.harmonic import HarmonicNotch, NotchOption
from rotorid.core.types import FilterRecommendation, FloatArray, NoiseProfile, SpectralPeak

__all__ = [
    "GYRO_LPF_LADDER",
    "NotchSource",
    "choose_notch_source",
    "recommend_filters",
]

#: Candidate gyro low-pass cutoffs, Hz. Spaced roughly a third of an octave apart
#: -- finer than this is below the resolution at which the choice matters, and
#: these are the values users recognize from the wiki.
GYRO_LPF_LADDER: tuple[float, ...] = (20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0)

#: ``INS_HNTCH_ATT`` candidates. Attenuation costs phase, so the ladder is walked
#: upward and stops at the first value that reaches the floor.
_ATT_LADDER: tuple[float, ...] = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0)

#: ``FLTD`` prior, as a fraction of the gyro cutoff. Community guidance, used as a
#: starting point the optimizer must explain itself if it departs from.
_FLTD_FRACTION = {"roll": 0.5, "pitch": 0.5, "yaw": 0.25}

#: ``FLTD`` must never sit above this fraction of the gyro cutoff: past it the
#: D-term filter stops doing anything the gyro filter has not already done, while
#: still costing its own phase.
_FLTD_MAX_FRACTION = 0.75

#: ``FM_RAT``. ArduPilot guidance is 0.7-1.0; the lower end tracks further down
#: toward idle at the cost of a wider effective notch.
_FREQ_MIN_RATIO = 0.7

#: Full-scale normalized motor command. The D-term RMS limit is expressed as a
#: percentage of this.
_FULL_SCALE = 1.0


@dataclass(frozen=True, slots=True)
class NotchSource:
    """Which tracking source the harmonic notch should use, and why.

    Attributes:
        mode: ``INS_HNTCH_MODE``. 0 static, 1 throttle, 2 RPM sensor, 3 ESC
            telemetry, 4 in-flight FFT, 5 second RPM sensor.
        ref: ``INS_HNTCH_REF``. Hover *thrust* for throttle mode, 1.0 for the
            measured-frequency modes.
        freq_hz: ``INS_HNTCH_FREQ``. The hover fundamental for throttle mode, the
            lowest frequency worth tracking for the measured modes.
    """

    mode: int
    ref: float
    freq_hz: float
    rationale: str
    rejected: tuple[tuple[str, str], ...] = ()


def choose_notch_source(
    *,
    track: MotorTrack,
    hover_thrust: float | None,
    fft_available: bool,
    fundamental_hz: float,
) -> NotchSource:
    """Pick the tracking source, best available first.

    The order is fixed by how directly each source measures the thing the notch
    has to follow: ESC telemetry *is* motor speed; an RPM sensor is motor speed
    with one sensor between; the in-flight FFT measures the noise itself but lags
    behind fast throttle changes; throttle is a model, and only as good as
    ``MOT_THST_HOVER``. Static is the last resort and tracks nothing at all.

    Args:
        fundamental_hz: Motor fundamental measured at the operating point, Hz.

    Raises:
        ValueError: if there is no fundamental frequency to build a notch around.
    """
    if fundamental_hz <= 0.0:
        raise ValueError(
            "no motor fundamental frequency was measured; a harmonic notch cannot "
            "be centred without one. Log ESC telemetry, or hover for 20-30 s so "
            "the fundamental can be read off the gyro spectrum."
        )

    rejected: list[tuple[str, str]] = []
    if track.source == "esc_telemetry":
        # FREQ in a measured mode is the lowest frequency worth tracking, not the
        # hover frequency: below it the notch fades out rather than chasing idle.
        return NotchSource(
            mode=3,
            ref=1.0,
            freq_hz=round(fundamental_hz * _FREQ_MIN_RATIO, 1),
            rationale=(
                f"ESC telemetry is in the log, so the notch tracks measured motor "
                f"speed (MODE 3) rather than inferring it from throttle. "
                f"Fundamental at this operating point is {fundamental_hz:.0f} Hz."
            ),
            rejected=(
                ("throttle tracking (MODE 1)", "measured RPM is available and is strictly better"),
            ),
        )

    rejected.append(("ESC telemetry (MODE 3)", "no ESC RPM in the log"))
    if fft_available:
        return NotchSource(
            mode=4,
            ref=1.0,
            freq_hz=round(fundamental_hz * _FREQ_MIN_RATIO, 1),
            rationale=(
                "In-flight FFT is enabled, so the notch tracks the measured noise "
                "peak (MODE 4). It lags fast throttle changes, so prefer ESC "
                "telemetry if the ESCs can provide it."
            ),
            rejected=tuple(rejected),
        )

    rejected.append(("in-flight FFT (MODE 4)", "FFT_ENABLE is off"))
    if hover_thrust and hover_thrust > 0.0:
        return NotchSource(
            mode=1,
            ref=round(hover_thrust, 3),
            freq_hz=round(fundamental_hz, 1),
            rationale=(
                f"No measured motor speed, so the notch tracks throttle (MODE 1) "
                f"with REF = MOT_THST_HOVER = {hover_thrust:.3f} and FREQ = the "
                f"{fundamental_hz:.0f} Hz fundamental measured at hover. This is "
                f"only as accurate as MOT_THST_HOVER; check it before flying."
            ),
            rejected=tuple(rejected),
        )

    rejected.append(("throttle tracking (MODE 1)", "MOT_THST_HOVER is not in the log"))
    return NotchSource(
        mode=0,
        ref=0.0,
        freq_hz=round(fundamental_hz, 1),
        rationale=(
            f"Nothing in the log measures motor speed, so the notch is static "
            f"(MODE 0) at {fundamental_hz:.0f} Hz. It will be in the wrong place "
            f"whenever the throttle is not at the value this log was flown at."
        ),
        rejected=tuple(rejected),
    )


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One rung of the ladder, kept with the evidence for why it was chosen."""

    notch: HarmonicNotch | None
    gyro_lpf_hz: float
    dterm_lpf_hz: float
    notes: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)


def recommend_filters(
    noise: NoiseProfile,
    baseline: FilterChain,
    config: Config,
    *,
    track: MotorTrack,
    op: OperatingPoint,
    crossover_hz: float,
    kd: float,
    hover_thrust: float | None = None,
    fft_available: bool = False,
    per_motor_capable: bool = False,
) -> FilterRecommendation:
    """Design a filter chain for one axis from its measured noise.

    Args:
        noise: The measured profile. Its ``psd_pre`` is required -- candidate
            chains are evaluated against the pre-filter spectrum, because
            evaluating them against the post-filter one would count the flown
            filters a second time.
        baseline: The chain the log was flown with, kept for the diff.
        crossover_hz: Design crossover. The frequency the phase budget is spent at.
        kd: Effective derivative gain, for the D-term noise constraint.
        per_motor_capable: Whether ESC telemetry and CPU headroom allow one notch
            set per motor.

    Returns:
        The recommendation, always including the current chain as its baseline.

    Raises:
        NotImplementedError: for PX4, whose parameter surface lands with M9.
        ValueError: if the profile has no pre-filter spectrum.
    """
    if baseline.stack != "ardupilot":
        raise NotImplementedError("PX4 filter recommendation lands with milestone M9")
    if noise.psd_pre is None:
        raise ValueError(
            "filter design needs a pre-filter spectrum; build the NoiseProfile with "
            "the flown chain so it can be divided out, or log pre-filter gyro"
        )

    target_floor = config.float_("filters", "target_noise_floor_db")
    phase_budget = config.float_("filters", "phase_budget_deg")
    max_harmonics = config.int_("filters", "max_harmonics")
    ratio = config.float_(
        "filters", "freq_bw_ratio_per_motor" if per_motor_capable else "freq_bw_ratio_default"
    )
    att_bounds = (config.float_("filters", "att_min_db"), config.float_("filters", "att_max_db"))
    noise_limit = config.float_("noise", "dterm_output_rms_limit_pct") / 100.0 * _FULL_SCALE

    tracked = [p for p in noise.peaks if p.tracks_rpm]
    structural = [p for p in noise.peaks if p.kind == "structural"]
    fundamental = _fundamental_hz(tracked, track, noise)

    rejected: list[tuple[str, str]] = []
    notes: list[str] = []

    notch: HarmonicNotch | None = None
    source: NotchSource | None = None
    if tracked and fundamental > 0.0:
        source = choose_notch_source(
            track=track,
            hover_thrust=hover_thrust,
            fft_available=fft_available,
            fundamental_hz=fundamental,
        )
        rejected.extend(source.rejected)
        notch, notch_notes = _design_notch(
            tracked,
            fundamental=fundamental,
            source=source,
            sample_rate_hz=baseline.sample_rate_hz,
            target_floor_db=target_floor,
            max_harmonics=max_harmonics,
            ratio=ratio,
            att_bounds=att_bounds,
            per_motor=per_motor_capable,
            phase_budget_deg=phase_budget,
            crossover_hz=crossover_hz,
            op=op,
            rejected=rejected,
        )
        notes.extend(notch_notes)
    else:
        rejected.append(
            (
                "harmonic notch",
                "no peak in the gyro spectrum tracks motor speed, so a tracking "
                "notch would chase nothing",
            )
        )

    for peak in structural:
        if peak.magnitude_db >= abs(target_floor) * 0.0 + config.float_(
            "noise", "peak_prominence_db"
        ):
            notes.append(
                f"Fixed-frequency peak at {peak.f_hz:.0f} Hz, {peak.magnitude_db:.0f} dB above "
                f"the floor, does not track motor speed. That is a structural resonance -- "
                f"a soft mount, a loose arm or a flexing frame. A tracking notch will not "
                f"help it and a static notch only masks it; fix the mechanics."
            )

    gyro_lpf, dterm_lpf, lpf_notes, lpf_rejected = _choose_lowpasses(
        noise,
        baseline,
        notch=notch,
        axis=noise.axis,
        kd=kd,
        op=op,
        noise_limit=noise_limit,
        crossover_hz=crossover_hz,
    )
    notes.extend(lpf_notes)
    rejected.extend(lpf_rejected)

    chain = FilterChain(
        stack="ardupilot",
        sample_rate_hz=baseline.sample_rate_hz,
        loop_rate_hz=baseline.loop_rate_hz,
        gyro_lpf_hz=gyro_lpf,
        notches=(notch,) if notch is not None else (),
        notch_ref=source.ref if source is not None else baseline.notch_ref,
        dterm_lpf_hz=dterm_lpf,
        error_lpf_hz=baseline.error_lpf_hz,
        target_lpf_hz=baseline.target_lpf_hz,
        all_imus=baseline.all_imus,
    )

    phase_cost = float(chain.phase_deg(np.array([crossover_hz]), op)[0])
    predicted = np.asarray(
        noise.psd_pre * np.abs(chain.sensor_response(noise.f_hz, op)) ** 2, dtype=np.float64
    )

    return FilterRecommendation(
        stack="ardupilot",
        chain=chain,
        baseline_chain=baseline,
        params=_ardupilot_params(chain, notch, source, noise.axis),
        phase_cost_deg=phase_cost,
        cpu_cost_rel=chain.cpu_cost(op),
        rationale=" ".join(
            [
                (source.rationale if source is not None else ""),
                *notes,
                f"Chain costs {phase_cost:.1f} deg of phase at the "
                f"{crossover_hz:.2f} Hz crossover, against a "
                f"{phase_budget:.0f} deg budget.",
            ]
        ).strip(),
        predicted_psd_post=predicted,
        attenuation_at_peaks_db=_attenuation_at_peaks(chain, noise.peaks, op),
        rejected=tuple(rejected),
    )


# --------------------------------------------------------------------------- #
# Notch
# --------------------------------------------------------------------------- #


def _fundamental_hz(
    tracked: list[SpectralPeak], track: MotorTrack, noise: NoiseProfile
) -> float:
    """Motor fundamental, measured if possible and inferred from the peaks if not."""
    if track.is_measured and track.mean_hz() > 0.0:
        return track.mean_hz()
    firsts = [p.f_hz for p in tracked if p.harmonic_index == 1]
    if firsts:
        return float(min(firsts))
    if tracked:
        # No harmonic labelling available: the lowest tracking peak is the best
        # available guess at the fundamental.
        return float(min(p.f_hz for p in tracked))
    return 0.0


def _design_notch(
    tracked: list[SpectralPeak],
    *,
    fundamental: float,
    source: NotchSource,
    sample_rate_hz: float,
    target_floor_db: float,
    max_harmonics: int,
    ratio: float,
    att_bounds: tuple[float, float],
    per_motor: bool,
    phase_budget_deg: float,
    crossover_hz: float,
    op: OperatingPoint,
    rejected: list[tuple[str, str]],
) -> tuple[HarmonicNotch | None, list[str]]:
    """Choose harmonics, bandwidth and attenuation, then back off to fit the budget."""
    notes: list[str] = []

    # 1. Harmonics: only where a tracking peak actually sits.
    present: dict[int, SpectralPeak] = {}
    for peak in tracked:
        index = peak.harmonic_index or int(round(peak.f_hz / fundamental))
        if index >= 1 and index not in present:
            present[index] = peak
    harmonics = tuple(sorted(present)[:max_harmonics])
    if not harmonics:
        return None, notes
    for index in sorted(present):
        if index not in harmonics:
            rejected.append(
                (
                    f"harmonic {index} ({present[index].f_hz:.0f} Hz)",
                    f"beyond the {max_harmonics}-harmonic cap; each further notch costs "
                    f"phase at the crossover for progressively less noise",
                )
            )

    # 2. Bandwidth: wide enough to cover the measured line plus the jitter in the
    #    tracking, floored at the firmware convention so it is not absurdly narrow.
    measured_bw = max(present[h].width_hz / max(h, 1) for h in harmonics)
    bandwidth = max(fundamental / ratio, 2.0 * measured_bw)
    bandwidth = min(bandwidth, fundamental)  # BW >= 2*FREQ makes Q non-physical

    # 3. Attenuation: the least that brings the worst peak to the target floor.
    worst_excess = max(present[h].magnitude_db for h in harmonics)
    needed = worst_excess - abs(target_floor_db) + abs(target_floor_db)  # excess above floor
    needed = worst_excess
    attenuation = next(
        (a for a in _ATT_LADDER if a >= needed and a >= att_bounds[0]),
        att_bounds[1],
    )
    attenuation = float(np.clip(attenuation, *att_bounds))
    if attenuation < needed:
        notes.append(
            f"The worst peak stands {worst_excess:.0f} dB above the floor, more than the "
            f"{att_bounds[1]:.0f} dB attenuation cap. The notch takes it down as far as it "
            f"can; the rest is a mechanical problem -- balance props and check mounts."
        )

    opts = NotchOption.TRIPLE_NOTCH if bandwidth > fundamental / 2.0 else 0
    if opts:
        notes.append(
            "The peak is wide enough to need a composite notch, so OPTS bit 4 (triple) "
            "is set rather than bit 0 (double): three narrow notches cover the same "
            "width for less phase than two wide ones."
        )
    if per_motor:
        opts |= NotchOption.MULTI_SOURCE

    # 4. Back off until the phase budget holds. Harmonics go first: the highest
    #    harmonic is the least valuable and, being furthest from the crossover, is
    #    not where the phase is coming from -- so dropping it is checked, but the
    #    bandwidth reduction usually does the work.
    while True:
        notch = HarmonicNotch(
            freq_hz=source.freq_hz,
            bandwidth_hz=round(bandwidth, 1),
            attenuation_db=attenuation,
            harmonics=harmonics,
            sample_rate_hz=sample_rate_hz,
            freq_min_ratio=_FREQ_MIN_RATIO if source.mode != 0 else 1.0,
            opts=opts,
        )
        phase = _notch_phase_deg(notch, crossover_hz, sample_rate_hz, op, fundamental)
        if phase <= phase_budget_deg:
            if notes and phase > 0.5 * phase_budget_deg:
                notes.append(
                    f"The notch stack spends {phase:.1f} deg of the "
                    f"{phase_budget_deg:.0f} deg budget."
                )
            return notch, notes
        if len(harmonics) > 1:
            dropped = harmonics[-1]
            harmonics = harmonics[:-1]
            rejected.append(
                (
                    f"harmonic {dropped} ({present[dropped].f_hz:.0f} Hz)",
                    f"dropped to stay inside the {phase_budget_deg:.0f} deg phase budget",
                )
            )
            continue
        if bandwidth > fundamental / 4.0:
            bandwidth *= 0.75
            continue
        notes.append(
            f"Even a single narrow notch costs {phase:.1f} deg at the crossover, over "
            f"the {phase_budget_deg:.0f} deg budget. The noise peak is too close to the "
            f"control bandwidth to filter cheaply; this is a mechanical fix, not a "
            f"filter one."
        )
        return notch, notes


def _notch_phase_deg(
    notch: HarmonicNotch,
    crossover_hz: float,
    sample_rate_hz: float,
    op: OperatingPoint,
    fundamental: float,
) -> float:
    """Phase lag the notch stack alone contributes at the crossover."""
    probe = FilterChain(
        stack="ardupilot",
        sample_rate_hz=sample_rate_hz,
        loop_rate_hz=sample_rate_hz,
        notches=(notch,),
        notch_ref=1.0,
    )
    centered = op if op.has_measured_frequency else OperatingPoint(motor_hz=(fundamental,))
    return float(probe.phase_deg(np.array([crossover_hz]), centered)[0])


# --------------------------------------------------------------------------- #
# Low-pass filters
# --------------------------------------------------------------------------- #


def _choose_lowpasses(
    noise: NoiseProfile,
    baseline: FilterChain,
    *,
    notch: HarmonicNotch | None,
    axis: str,
    kd: float,
    op: OperatingPoint,
    noise_limit: float,
    crossover_hz: float,
) -> tuple[float, float, list[str], list[tuple[str, str]]]:
    """Pick the highest gyro and D-term cutoffs that hold the D-term noise limit.

    Highest, not lowest: a low-pass is pure phase lag in the loop, so the correct
    cutoff is the least filtering that meets the noise constraint, and the
    constraint is on what the D term does to the motors rather than on how the
    spectrum looks.
    """
    notes: list[str] = []
    rejected: list[tuple[str, str]] = []
    psd_pre = noise.psd_pre
    assert psd_pre is not None  # guarded by the caller

    fraction = _FLTD_FRACTION.get(axis, 0.5)
    best: tuple[float, float] | None = None
    for gyro in sorted(GYRO_LPF_LADDER, reverse=True):
        dterm = min(gyro * fraction, gyro * _FLTD_MAX_FRACTION)
        candidate = FilterChain(
            stack="ardupilot",
            sample_rate_hz=baseline.sample_rate_hz,
            loop_rate_hz=baseline.loop_rate_hz,
            gyro_lpf_hz=gyro,
            notches=(notch,) if notch is not None else (),
            notch_ref=baseline.notch_ref,
            dterm_lpf_hz=dterm,
        )
        rms = dterm_noise_rms(noise.f_hz, psd_pre, candidate, kd=kd, op=op)
        if rms <= noise_limit:
            best = (gyro, dterm)
            notes.append(
                f"Gyro LPF {gyro:.0f} Hz with FLTD {dterm:.0f} Hz is the least filtering "
                f"that keeps D-term output noise at {rms / _FULL_SCALE * 100.0:.1f}% of "
                f"full scale, under the {noise_limit * 100.0:.0f}% limit."
            )
            break
        rejected.append(
            (
                f"gyro LPF {gyro:.0f} Hz",
                f"leaves {rms / _FULL_SCALE * 100.0:.1f}% D-term output noise, over the "
                f"{noise_limit * 100.0:.0f}% limit",
            )
        )

    if best is None:
        gyro = min(GYRO_LPF_LADDER)
        dterm = gyro * fraction
        notes.append(
            f"Even a {gyro:.0f} Hz gyro filter does not bring D-term output noise under "
            f"the limit. The vehicle is too noisy to carry this much D; reduce D, or fix "
            f"the noise at its source."
        )
        best = (gyro, dterm)

    gyro, dterm = best
    if gyro < 4.0 * crossover_hz:
        notes.append(
            f"The {gyro:.0f} Hz gyro filter sits only {gyro / max(crossover_hz, 1e-6):.1f}x "
            f"above the {crossover_hz:.2f} Hz crossover, so a large part of the loop's "
            f"phase budget is being spent on noise rejection."
        )
    if baseline.gyro_lpf_hz and abs(gyro - baseline.gyro_lpf_hz) < 1e-6:
        notes.append("The gyro filter is already at the right value; leave it as it is.")
    return gyro, dterm, notes, rejected


# --------------------------------------------------------------------------- #
# Export shape
# --------------------------------------------------------------------------- #


def _attenuation_at_peaks(
    chain: FilterChain, peaks: tuple[SpectralPeak, ...], op: OperatingPoint
) -> dict[float, float]:
    """How much the candidate chain removes at each measured peak, in dB."""
    if not peaks:
        return {}
    f = np.array([p.f_hz for p in peaks], dtype=np.float64)
    magnitude = np.abs(chain.sensor_response(f, op))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    return {float(freq): float(value) for freq, value in zip(f, db, strict=True)}


_AXIS_PARAM: dict[str, str] = {"roll": "RLL", "pitch": "PIT", "yaw": "YAW"}


def _ardupilot_params(
    chain: FilterChain,
    notch: HarmonicNotch | None,
    source: NotchSource | None,
    axis: str,
) -> dict[str, float]:
    """The parameter set to write, exactly as the vehicle names it."""
    params: dict[str, float] = {}
    if chain.gyro_lpf_hz:
        params["INS_GYRO_FILTER"] = float(chain.gyro_lpf_hz)
    if chain.dterm_lpf_hz:
        params[f"ATC_RAT_{_AXIS_PARAM[axis]}_FLTD"] = float(chain.dterm_lpf_hz)
    if notch is not None and source is not None:
        params.update(
            {
                "INS_HNTCH_ENABLE": 1.0,
                "INS_HNTCH_MODE": float(source.mode),
                "INS_HNTCH_FREQ": float(notch.freq_hz),
                "INS_HNTCH_BW": float(notch.bandwidth_hz),
                "INS_HNTCH_ATT": float(notch.attenuation_db),
                "INS_HNTCH_HMNCS": float(sum(1 << (h - 1) for h in notch.harmonics)),
                "INS_HNTCH_REF": float(source.ref),
                "INS_HNTCH_FM_RAT": float(notch.freq_min_ratio),
                "INS_HNTCH_OPTS": float(notch.opts),
            }
        )
    return params


PeakKind = Literal["motor_fundamental", "motor_harmonic", "structural", "broadband", "unknown"]
