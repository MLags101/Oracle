"""Gyro noise characterization and spectral-peak classification (spec section 5.6).

A notch is only worth its phase lag if it sits on a peak that is really there and
really behaves the way the notch assumes. So this module does not merely find
peaks -- it *classifies* them by how they move over the flight:

* a peak whose frequency follows motor speed wants a **tracking** notch;
* a peak that sits at the same frequency whatever the motors do is a **structural**
  resonance, and a tracking notch will chase it uselessly. The fix is mechanical;
  a static notch is a stopgap;
* energy with no peak at all is **broadband**, and only a low-pass or a mechanical
  change touches it.

Getting this wrong is expensive in a specific way: a tracking notch aimed at a
frame resonance costs its full phase lag at the crossover and removes nothing.

**Pre- versus post-filter spectra.** The gyro trace both stacks log is
post-filter, so a PSD computed from it has the current chain baked in. Designing
a *candidate* chain against it would count the current filters twice -- the same
error :mod:`rotorid.core.analysis.sysid` guards against on the transfer-function
side. :func:`prefilter_psd` is the one place that division happens, and
:func:`NoiseProfile.psd_pre` is where its result is carried.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import find_peaks, peak_widths, spectrogram

from rotorid.core.filters.chain import FilterChain, OperatingPoint
from rotorid.core.types import (
    Axis,
    FloatArray,
    LogBundle,
    NoiseProfile,
    Signal,
    SpectralPeak,
)

__all__ = [
    "MotorTrack",
    "classify_peaks",
    "dterm_noise_rms",
    "find_spectral_peaks",
    "measured_dterm_rms_pct",
    "motor_track",
    "noise_floor_db",
    "noise_profile",
    "prefilter_psd",
    "steady_window",
]

#: Median filter half-width, in log-frequency decades, used to estimate the floor
#: under the peaks. Wide enough to step over a notch-width peak, narrow enough to
#: follow the real 1/f-ish shape of a gyro spectrum.
_FLOOR_HALF_WIDTH_DECADES = 0.15

#: Spectrogram frames per second. Motor speed changes on the order of a second, so
#: this resolves the tracking without making each frame too short to see the peak.
_FRAMES_PER_SECOND = 4.0

#: A peak wider than this fraction of its own centre frequency is not a line --
#: it is a hump, and a notch will not remove it.
_BROADBAND_WIDTH_FRACTION = 0.5

#: Fractional frequency wander below which a peak counts as stationary in the
#: RPM-free fallback classification.
_STATIONARY_WANDER = 0.02

#: Motor frequency must move at least this much across the flight before tracking
#: correlation means anything. A hover at constant RPM correlates with everything.
_MIN_TRACK_SPAN = 0.05

_HARMONIC_TOLERANCE = 0.12

#: Fractional motor-speed variation allowed inside the window the spectrum is
#: measured over. A tone that moves during the record is smeared across the
#: average, and a smeared tone is not a peak.
_STEADY_TOLERANCE = 0.05

#: Shortest window worth taking a spectrum over. Below this the resolution is too
#: coarse to separate a line from its neighbours.
_STEADY_MIN_SECONDS = 4.0


@dataclass(frozen=True, slots=True)
class MotorTrack:
    """Motor fundamental frequency over time, and where the number came from.

    Attributes:
        source: Provenance, in descending order of trust. ``"throttle_model"`` is
            a *shape* only -- its absolute scale depends on ``MOT_THST_HOVER``
            being right -- so it may be used to classify tracking but never to set
            a notch centre directly.
        f_hz: Fundamental frequency per sample of :attr:`t`. For
            ``throttle_model`` this is ``sqrt(throttle)`` in arbitrary units.
        per_motor_hz: One trace per motor where the log has them, for per-motor
            notch tracking.
    """

    t: FloatArray
    f_hz: FloatArray
    source: Literal["esc_telemetry", "rpm_sensor", "throttle_model", "none"]
    per_motor_hz: tuple[FloatArray, ...] = ()

    @property
    def is_measured(self) -> bool:
        """Whether the frequency is a measurement rather than a throttle proxy."""
        return self.source in ("esc_telemetry", "rpm_sensor")

    @property
    def span_fraction(self) -> float:
        """Peak-to-peak variation as a fraction of the mean, over the window.

        Near zero means the flight never changed motor speed, and no correlation
        computed against this trace can distinguish tracking from coincidence.
        """
        if self.f_hz.size == 0:
            return 0.0
        mean = float(np.mean(self.f_hz))
        if mean <= 0.0:
            return 0.0
        return float(np.ptp(self.f_hz) / mean)

    def mean_hz(self) -> float:
        """Mean fundamental over the window, or 0.0 if there is no trace."""
        return float(np.mean(self.f_hz)) if self.f_hz.size else 0.0

    def resample(self, t: FloatArray) -> FloatArray:
        """Interpolate onto another time base, held flat outside the trace."""
        if self.f_hz.size == 0:
            return np.zeros_like(t)
        return np.asarray(np.interp(t, self.t, self.f_hz), dtype=np.float64)


def motor_track(bundle: LogBundle, t_start: float, t_end: float) -> MotorTrack:
    """Recover motor fundamental frequency over one window.

    Prefers measured RPM over the throttle model, because the throttle model is
    only as good as ``MOT_THST_HOVER`` and that parameter is the one most often
    wrong on an untuned vehicle.
    """
    per_motor: list[FloatArray] = []
    times: FloatArray | None = None
    for index in range(1, 13):
        key = f"motor.{index}.rpm"
        if key not in bundle.signals:
            continue
        signal = bundle.signals[key]
        window = (signal.t >= t_start) & (signal.t <= t_end)
        if not window.any():
            continue
        if times is None:
            times = signal.t[window]
        per_motor.append(signal.y[window] / 60.0)

    if per_motor and times is not None:
        stacked = np.vstack([p[: times.size] for p in per_motor])
        return MotorTrack(
            t=times,
            f_hz=np.asarray(np.mean(stacked, axis=0), dtype=np.float64),
            source="esc_telemetry",
            per_motor_hz=tuple(np.asarray(p, dtype=np.float64) for p in per_motor),
        )

    throttle = _throttle_trace(bundle, t_start, t_end)
    if throttle is not None:
        t, value = throttle
        return MotorTrack(t=t, f_hz=np.sqrt(np.clip(value, 0.0, None)), source="throttle_model")

    return MotorTrack(
        t=np.zeros(0, dtype=np.float64), f_hz=np.zeros(0, dtype=np.float64), source="none"
    )


def _throttle_trace(
    bundle: LogBundle, t_start: float, t_end: float
) -> tuple[FloatArray, FloatArray] | None:
    """Mean normalized motor output over the window, or ``None`` if not logged."""
    outputs = [
        bundle.signals[f"motor.{i}.output"]
        for i in range(1, 13)
        if f"motor.{i}.output" in bundle.signals
    ]
    if not outputs:
        return None
    reference = outputs[0]
    window = (reference.t >= t_start) & (reference.t <= t_end)
    if not window.any():
        return None
    t = reference.t[window]
    stacked = np.vstack([np.interp(t, s.t, s.y) for s in outputs])
    return t, np.asarray(np.mean(stacked, axis=0), dtype=np.float64)


def steady_window(
    track: MotorTrack,
    t_start: float,
    t_end: float,
    *,
    tolerance: float = _STEADY_TOLERANCE,
    min_duration_s: float = _STEADY_MIN_SECONDS,
) -> tuple[float, float]:
    """The longest stretch over which motor speed barely moved.

    Spectra are measured here rather than over the whole record. A tone that
    sweeps with the throttle is smeared across a full-record average -- a 25%
    speed swing turns a 180 Hz third harmonic into a 45 Hz-wide hump, which is not
    a peak and cannot be measured as one. Tracking classification still uses the
    whole record, because it needs the motion this window deliberately excludes.

    Falls back to the full window when there is no motor trace, or when nothing in
    the flight was steady for long enough.
    """
    if track.f_hz.size < 2 or track.t.size != track.f_hz.size:
        return t_start, t_end

    f = track.f_hz
    t = track.t
    best = (t_start, t_end)
    best_duration = 0.0
    j = 0
    for i in range(f.size):
        j = max(j, i + 1)
        while j < f.size:
            window = f[i : j + 1]
            mean = float(np.mean(window))
            if mean <= 0.0 or float(np.ptp(window)) / mean > tolerance:
                break
            j += 1
        duration = float(t[j - 1] - t[i])
        if duration > best_duration:
            best_duration, best = duration, (float(t[i]), float(t[j - 1]))

    if best_duration < min_duration_s:
        return t_start, t_end
    return best


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #


def noise_floor_db(f_hz: FloatArray, psd_db: FloatArray) -> FloatArray:
    """Running median of the spectrum in log-frequency: the floor under the peaks.

    A single scalar floor is wrong for a gyro spectrum, which slopes; a running
    median follows the slope while stepping over narrow lines, because a line
    occupies a small fraction of the window it sits in.
    """
    positive = f_hz > 0.0
    floor = np.full_like(psd_db, np.min(psd_db) if psd_db.size else 0.0)
    if not positive.any():
        return floor
    log_f = np.full_like(f_hz, -np.inf)
    log_f[positive] = np.log10(f_hz[positive])
    for i in np.flatnonzero(positive):
        window = positive & (np.abs(log_f - log_f[i]) <= _FLOOR_HALF_WIDTH_DECADES)
        floor[i] = float(np.median(psd_db[window]))
    return floor


def find_spectral_peaks(
    f_hz: FloatArray,
    psd: FloatArray,
    *,
    prominence_db: float,
    f_min_hz: float = 5.0,
) -> tuple[tuple[float, float, float], ...]:
    """Locate peaks and measure them.

    Args:
        psd: Power spectral density, linear.
        prominence_db: ``[noise].peak_prominence_db``. How far a peak must stand
            above the local floor to be worth a notch.
        f_min_hz: Ignore everything below this. The airframe's own rate response
            lives down there and is signal, not noise.

    Returns:
        ``(frequency_hz, height_above_floor_db, width_hz)`` per peak, strongest first.
    """
    with np.errstate(divide="ignore"):
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))
    band = f_hz >= f_min_hz
    if band.sum() < 8:
        return ()

    f = f_hz[band]
    db = psd_db[band]
    floor = noise_floor_db(f, db)
    excess = db - floor

    indices, properties = find_peaks(excess, prominence=prominence_db)
    if indices.size == 0:
        return ()

    # -3 dB relative to each peak's own prominence, so the width reported is the
    # half-power width of the line rather than of the hump it sits on.
    prominences = np.asarray(properties["prominences"], dtype=np.float64)
    rel_height = np.clip(3.0 / np.maximum(prominences, 1e-9), 0.0, 1.0)
    widths_bins = np.array(
        [
            float(peak_widths(excess, [int(idx)], rel_height=float(rel))[0][0])
            for idx, rel in zip(indices, rel_height, strict=True)
        ]
    )
    df = float(np.median(np.diff(f))) if f.size > 1 else 0.0

    peaks = [
        (float(f[idx]), float(excess[idx]), float(width * df))
        for idx, width in zip(indices, widths_bins, strict=True)
    ]
    return tuple(sorted(peaks, key=lambda p: -p[1]))


def classify_peaks(
    signal: Signal,
    peaks: tuple[tuple[float, float, float], ...],
    track: MotorTrack,
    *,
    track_margin_db: float,
    t_start: float,
    t_end: float,
    fundamental_hz: float = 0.0,
) -> tuple[SpectralPeak, ...]:
    """Decide what each peak is by testing it against the motor speed directly.

    For each peak the spectrogram is read along two loci: the frequency the peak
    would follow if it tracked the motors, and the constant frequency it would
    keep if it did not. Whichever locus carries more energy is what the peak is
    doing. Testing the hypothesis this way rather than correlating a ridge
    position matters when peaks are close together -- a ridge tracker inside a
    band around the second harmonic will happily lock onto a strong frame
    resonance 6 Hz away and report that the harmonic is stationary.

    Where the motors never changed speed there is nothing to test against, and
    the fallback is whether the peak moved at all: one that stayed exactly put
    over the whole record is structural.

    Args:
        fundamental_hz: Motor fundamental *at the operating point the spectrum was
            measured at*, for labelling harmonics. Not the mean over the whole
            window, which is a different number whenever the throttle moved.
        track_margin_db: How much more energy the tracking locus must carry before
            the peak counts as following the motors.
    """
    window = (signal.t >= t_start) & (signal.t <= t_end)
    y = signal.y[window]
    t = signal.t[window]
    fs = signal.rate_hz
    nperseg = int(2 ** np.round(np.log2(max(fs / _FRAMES_PER_SECOND, 64.0))))
    if y.size < 2 * nperseg:
        nperseg = int(2 ** np.floor(np.log2(max(y.size // 4, 64))))

    out: list[SpectralPeak] = []
    if y.size < 2 * nperseg:
        # Too short to watch anything move; report the peaks without a verdict.
        return tuple(
            SpectralPeak(f_hz=f, magnitude_db=db, width_hz=w, kind="unknown") for f, db, w in peaks
        )

    f_grid, t_frames, sxx = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    frame_times = t[0] + t_frames
    track_frames = track.resample(frame_times)
    track_usable = track.f_hz.size > 0 and track.span_fraction >= _MIN_TRACK_SPAN

    fundamental = fundamental_hz if fundamental_hz > 0.0 else 0.0

    reference = fundamental_hz if fundamental_hz > 0.0 else track.mean_hz()

    for f_peak, db, width in peaks:
        ridge = _ridge(f_grid, sxx, f_peak, width)
        wander = float(np.ptp(ridge) / f_peak) if f_peak > 0.0 else 0.0
        usable = track_usable and reference > 0.0
        advantage = (
            _tracking_advantage_db(f_grid, sxx, f_peak, track_frames, reference)
            if usable
            else float("nan")
        )
        tracks = bool(usable and advantage >= track_margin_db)

        kind: Literal["motor_fundamental", "motor_harmonic", "structural", "broadband", "unknown"]
        harmonic: int | None = None
        if width > _BROADBAND_WIDTH_FRACTION * f_peak:
            kind = "broadband"
        elif tracks:
            harmonic = _harmonic_index(f_peak, fundamental) if fundamental > 0.0 else None
            kind = "motor_fundamental" if harmonic == 1 else "motor_harmonic"
        elif track_usable or wander < _STATIONARY_WANDER:
            # Either we could have seen it track and it did not, or nothing in the
            # flight moved and the peak still stayed exactly put.
            kind = "structural"
        else:
            kind = "unknown"

        out.append(
            SpectralPeak(
                f_hz=f_peak,
                magnitude_db=db,
                width_hz=width,
                kind=kind,
                tracks_rpm=tracks,
                harmonic_index=harmonic,
            )
        )
    return tuple(out)


def _ridge(f_grid: FloatArray, sxx: FloatArray, f_peak: float, width: float) -> FloatArray:
    """Frequency of the strongest bin near ``f_peak``, per spectrogram frame."""
    half = max(width * 2.0, 0.15 * f_peak)
    band = (f_grid >= f_peak - half) & (f_grid <= f_peak + half)
    if band.sum() < 2:
        return np.full(sxx.shape[1], f_peak)
    sub_f = f_grid[band]
    return np.asarray(sub_f[np.argmax(sxx[band, :], axis=0)], dtype=np.float64)


def _tracking_advantage_db(
    f_grid: FloatArray,
    sxx: FloatArray,
    f_peak: float,
    track_frames: FloatArray,
    reference: float,
) -> float:
    """How much more energy sits on the tracking locus than on a fixed frequency.

    The locus is scaled so it passes through ``f_peak`` at the operating point the
    spectrum was measured at, which is what makes this a fair comparison: both
    loci agree there, and they diverge only as the motors change speed.
    """
    if reference <= 0.0 or sxx.shape[1] != track_frames.size:
        return float("nan")
    locus = f_peak * track_frames / reference
    moving = np.array(
        [np.interp(locus[k], f_grid, sxx[:, k]) for k in range(sxx.shape[1])], dtype=np.float64
    )
    fixed = np.array(
        [np.interp(f_peak, f_grid, sxx[:, k]) for k in range(sxx.shape[1])], dtype=np.float64
    )
    total_moving = float(np.mean(moving))
    total_fixed = float(np.mean(fixed))
    if total_fixed <= 0.0 or total_moving <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(total_moving / total_fixed))


def _harmonic_index(f_peak: float, fundamental: float) -> int | None:
    """Which harmonic of the motor fundamental this is, if it is close to one."""
    if fundamental <= 0.0:
        return None
    ratio = f_peak / fundamental
    nearest = round(ratio)
    if nearest < 1 or abs(ratio - nearest) > _HARMONIC_TOLERANCE * max(nearest, 1):
        return None
    return nearest


# --------------------------------------------------------------------------- #
# Undoing the current chain
# --------------------------------------------------------------------------- #


def prefilter_psd(
    f_hz: FloatArray,
    psd_post: FloatArray,
    chain: FilterChain,
    *,
    op: OperatingPoint | None,
    floor_db: float,
) -> FloatArray:
    """Recover the pre-filter gyro PSD from a post-filter one.

    The only place a measured spectrum is divided by the current chain. Design
    code evaluates candidate chains against the *result*, never against the
    post-filter spectrum, or the filters already flown would be counted twice.

    Where the chain attenuates below ``floor_db`` the division is
    ill-conditioned -- inside a deep notch the post-filter spectrum carries no
    information about what was there -- so the divisor is clamped, which makes
    the reconstruction conservative (an under-estimate) rather than explosive.
    """
    magnitude = np.abs(chain.sensor_response(f_hz, op))
    floor = 10.0 ** (floor_db / 20.0)
    return np.asarray(psd_post / np.maximum(magnitude, floor) ** 2, dtype=np.float64)


def dterm_noise_rms(
    f_hz: FloatArray,
    psd_pre: FloatArray,
    chain: FilterChain,
    *,
    kd: float,
    op: OperatingPoint | None = None,
    f_max_hz: float | None = None,
) -> float:
    """RMS of the D-term output driven by gyro noise, in normalized motor units.

    This is the number that actually limits how much D a vehicle can carry: the
    derivative differentiates, so noise the gyro LPF leaves at 100 Hz arrives at
    the motors multiplied by ``kd * 2*pi*100``. It is evaluated by propagating the
    measured pre-filter spectrum through the candidate sensor chain and the
    derivative branch, so it answers the question the optimizer needs -- what
    would this chain do -- rather than reporting what the flown chain did.

    Args:
        psd_pre: Pre-filter gyro PSD, in ``(rad/s)^2/Hz``.
        kd: Effective derivative gain.
        f_max_hz: Integrate no higher than this. Defaults to the loop Nyquist,
            above which the motors cannot respond anyway.

    Returns:
        RMS in normalized motor-command units (the units ``rate.*.output`` uses).
    """
    limit = f_max_hz if f_max_hz is not None else chain.loop_rate_hz / 2.0
    band = (f_hz > 0.0) & (f_hz <= limit)
    if band.sum() < 2:
        return 0.0

    f = f_hz[band]
    sensor = np.abs(chain.sensor_response(f, op))
    dterm = np.abs(chain.dterm_lpf_response(f))
    derivative = 2.0 * np.pi * f
    gain = abs(kd) * derivative * sensor * dterm
    return float(np.sqrt(np.trapezoid(psd_pre[band] * gain**2, f)))


def measured_dterm_rms_pct(bundle: LogBundle, axis: Axis, *, above_hz: float) -> float | None:
    """RMS of the *logged* derivative term above ``above_hz``, as a percentage.

    :func:`dterm_noise_rms` predicts what a candidate chain and D gain would do to
    the motors. This measures what the flown one actually did, from ``PIDR.D``,
    and the two answer different questions: the prediction is about a tune nobody
    has flown, and this is about the aircraft in front of you.

    Only the content above ``above_hz`` is counted. A multirotor rate loop crosses
    over somewhere between two and five hertz, so below that the derivative term
    is doing control work and its magnitude is not a complaint. Above it the loop
    has no authority left, and everything the D term sends to the motors up there
    is noise being converted into heat.

    Returns:
        Percent of full motor range, RMS, or ``None`` if the log has no PID
        messages -- which is a different statement from "the D term is quiet".
    """
    from rotorid.core.analysis.spectra import power_spectrum

    key = f"rate.{axis}.d_term"
    if key not in bundle.signals:
        return None
    signal = bundle.signals[key]
    if signal.y.size < 256:
        return None

    nperseg = int(min(2 ** np.floor(np.log2(signal.y.size / 4.0)), 4096))
    f_hz, psd = power_spectrum(signal.y, signal.rate_hz, nperseg=max(nperseg, 64))

    ceiling = signal.native_nyquist_hz or f_hz[-1]
    band = (f_hz >= above_hz) & (f_hz <= ceiling)
    if band.sum() < 2:
        return None
    return float(np.sqrt(np.trapezoid(psd[band], f_hz[band]))) * 100.0


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _fundamental_at(track: MotorTrack, t_start: float, t_end: float) -> float:
    """Mean measured motor frequency over the window the spectrum was taken from."""
    if not track.is_measured or track.f_hz.size == 0:
        return 0.0
    window = (track.t >= t_start) & (track.t <= t_end)
    if not window.any():
        return track.mean_hz()
    return float(np.mean(track.f_hz[window]))


#: Below this ceiling a spectrum cannot contain a multirotor motor fundamental,
#: so there is nothing a notch could be aimed at. Small quads run 150-400 Hz and
#: large ones 50-100 Hz; 50 Hz is the bottom of the range that exists at all.
_MIN_NOISE_CEILING_HZ = 50.0


def noise_profile(
    bundle: LogBundle,
    axis: Axis,
    *,
    t_start: float,
    t_end: float,
    chain: FilterChain | None = None,
    op: OperatingPoint | None = None,
    prominence_db: float = 6.0,
    track_margin_db: float = 3.0,
    deconv_floor_db: float = -20.0,
    nperseg: int | None = None,
    evidence_ceiling_hz: float | None = None,
) -> NoiseProfile:
    """Characterize the gyro noise on one axis over one window.

    Prefers genuinely pre-filter data when the log has it (batch logging), and
    otherwise reconstructs the pre-filter spectrum by dividing the modeled chain
    out of the post-filter one. Which route was taken is visible in
    :attr:`~rotorid.core.types.NoiseProfile.has_pre_filter` and matters: the
    reconstruction is only as good as the filter model.

    Args:
        evidence_ceiling_hz: Highest frequency the gyro message was actually
            logged fast enough to describe. Everything above it in the spectrum
            is interpolation, and interpolation of a jittered 10 Hz message
            produces a forest of evenly-spaced lines that look exactly like frame
            resonances. The spectrum is truncated here rather than filtered
            afterwards, because a peak that is an artefact should never reach the
            classifier at all.

    Raises:
        ValueError: if the axis has no rate measurement in the log, or if the
            evidence ceiling is too low for any motor line to be inside it. No
            noise profile at all is the right answer there: a filter recommended
            from a spectrum that cannot contain the motors is worse than none.
    """
    from rotorid.core.analysis.spectra import power_spectrum

    key = f"rate.{axis}.measured"
    if key not in bundle.signals:
        raise ValueError(f"{key} is not in the log; no noise analysis is possible for {axis}")
    signal = bundle.signals[key]
    track = motor_track(bundle, t_start, t_end)
    spectrum_start, spectrum_end = steady_window(track, t_start, t_end)
    window = (signal.t >= spectrum_start) & (signal.t <= spectrum_end)
    y = signal.y[window]
    if y.size < 64:
        raise ValueError(f"{axis}: only {y.size} samples in the window; too short for a spectrum")

    fs = signal.rate_hz
    if nperseg is None:
        nperseg = int(min(2 ** np.floor(np.log2(y.size / 4.0)), 4096))
    f_hz, psd_post = power_spectrum(y, fs, nperseg=max(nperseg, 64))

    if evidence_ceiling_hz is not None:
        if evidence_ceiling_hz < _MIN_NOISE_CEILING_HZ:
            raise ValueError(
                f"{axis}: {key} was logged fast enough to describe frequencies only up to "
                f"{evidence_ceiling_hz:.1f} Hz, which is below anywhere a multirotor's motors "
                f"put their fundamental. There is no noise spectrum here to design a filter from."
            )
        inside = f_hz <= evidence_ceiling_hz
        f_hz, psd_post = f_hz[inside], psd_post[inside]

    prefilter_key = f"gyro.{axis}.prefilter"
    pre_source: Literal["measured", "reconstructed", "none"] = "none"
    if prefilter_key in bundle.signals:
        pre_signal = bundle.signals[prefilter_key]
        pre_window = (pre_signal.t >= spectrum_start) & (pre_signal.t <= spectrum_end)
        f_pre, psd_pre_raw = power_spectrum(
            pre_signal.y[pre_window], pre_signal.rate_hz, nperseg=max(nperseg, 64)
        )
        # Batch-logged gyro runs at the sensor rate, not the analysis grid rate, so
        # its spectrum lands on a different grid and has to be brought onto ours.
        psd_pre = np.asarray(np.interp(f_hz, f_pre, psd_pre_raw), dtype=np.float64)
        pre_source = "measured"
    elif chain is not None:
        psd_pre = prefilter_psd(f_hz, psd_post, chain, op=op, floor_db=deconv_floor_db)
        pre_source = "reconstructed"
    else:
        psd_pre = None

    # Peaks are found on the pre-filter spectrum where one exists: a notch that
    # already works has removed its own evidence from the post-filter trace, and
    # a recommendation built on that would remove the notch and reintroduce the peak.
    search_psd = psd_pre if psd_pre is not None else psd_post
    raw_peaks = find_spectral_peaks(f_hz, search_psd, prominence_db=prominence_db)
    peaks = classify_peaks(
        signal,
        raw_peaks,
        track,
        track_margin_db=track_margin_db,
        t_start=t_start,
        t_end=t_end,
        fundamental_hz=_fundamental_at(track, spectrum_start, spectrum_end),
    )

    with np.errstate(divide="ignore"):
        post_db = 10.0 * np.log10(np.maximum(psd_post, 1e-30))
    band = f_hz >= 5.0
    floor = float(np.median(post_db[band])) if band.any() else float("nan")

    return NoiseProfile(
        axis=axis,
        f_hz=f_hz,
        psd_post=psd_post,
        noise_floor_db=floor,
        peaks=peaks,
        psd_pre=psd_pre,
        pre_filter_source=pre_source,
        motor_fundamental_track=track.f_hz if track.f_hz.size else None,
    )
