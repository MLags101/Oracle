"""Oscillation the aircraft was actually doing (plan phase 2.3).

Until now every statement this tool made about oscillation was a *prediction*:
the model was fitted, the loop closed on paper, and the margins reported said how
close to instability the design would be. Nothing looked at whether the aircraft
in the log was already oscillating.

That gap has one specific consequence, and it is the worst kind. A vehicle sitting
in a limit cycle is a vehicle whose real loop has unity gain and inverted phase
somewhere; a model fitted to that flight will still come back with comfortable
margins, because the frequency response it was fitted over is dominated by the
excitation rather than by the limit cycle. So the tool would look at an
oscillating aircraft, see a healthy model, and recommend *more* gain.

What counts as an oscillation here:

* a narrow peak in the measured rate, above the low-frequency band where the
  pilot's own inputs live;
* amplified far beyond the command -- the loop is answering a small input at
  that frequency with a large output, which is what resonance *is*;
* **not** tracking the motors, which makes it noise rather than a control problem
  and is already handled by the notch designer;
* present over a substantial fraction of the flight, not once after a hard
  manoeuvre.

Each of those is a separate gate and each can be reported, because "no
oscillation" and "something narrow but only for two seconds" deserve different
words.

**Why amplification rather than absence from the command.** The obvious test is
whether the tone appears in the rate setpoint: if it does, the pilot asked for it.
That test works at 20 Hz and fails where it matters most. An attitude loop feeds a
5 Hz oscillation straight back into the rate setpoint, so a genuine limit cycle at
5 Hz shows up in the command almost as strongly as in the measurement, and gets
dismissed. Measured on the closed-loop simulator, a loop driven to 13 degrees of
phase margin was rejected by that test with 5.6 dB to spare.

The ratio of measured power to commanded power in the band does not have that
problem, because it is the closed-loop magnitude and a resonance raises both terms
of it together. On the same fixture it reads +2 dB for a healthy tune and +11 to
+20 dB for every marginal one -- and it is the same quantity the designer already
bounds as peak sensitivity, so the threshold means something rather than being
fitted to the fixture.

**What it does about it.** Detection alone would be an annotation. The measured
frequency is cross-referenced against the *flown* loop as this tool models it: if
the model claims 9 dB of gain margin at a frequency the aircraft demonstrably
sustains oscillation at, then the model is optimistic by 9 dB, and that number is
handed to the designer as extra gain margin to hold back. It is measured rather
than chosen, and it is the one piece of evidence in the tool that corrects the
model using the aircraft's own misbehaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import spectrogram

from rotorid.config import Config
from rotorid.core.analysis.margins import LoopDelay, plant_path
from rotorid.core.analysis.noise import noise_floor_db
from rotorid.core.design.controller import RateController
from rotorid.core.filters.chain import OperatingPoint
from rotorid.core.types import AirframeModel, Axis, FloatArray, LogBundle, NoiseProfile

__all__ = ["Oscillation", "detect_oscillation", "model_optimism_db"]

#: Frequency resolution wanted at the bottom of the search band, as a fraction
#: of it. The noise module's four-frames-per-second geometry is right for motor
#: lines above 100 Hz and useless here: at a 400 Hz log rate it gives 6 Hz bins,
#: which cannot see a 6 Hz oscillation at all. Frame length is derived from this
#: instead, which makes frames longer -- and that is fine, because a limit cycle
#: is steady over seconds while a motor line moves within one.
_RESOLUTION_FRACTION = 0.25

#: Fractional half-width of the band summed around a candidate frequency. A limit
#: cycle wanders a little as the operating point moves.
_BAND_FRACTION = 0.06

#: How close a candidate has to sit to a classified motor line before it is
#: treated as that line rather than as an oscillation, as a fraction of frequency.
_MOTOR_TOLERANCE = 0.08

#: In-frame prominence above which the tone counts as present in that frame.
_PRESENT_DB = 8.0

#: The skirt the per-frame prominence is measured against, in multiples of the
#: band half-width. Starts outside the tone's own shoulders and runs far enough
#: to average many bins, which is what keeps the measurement quiet.
_SKIRT_INNER = 2.0
_SKIRT_OUTER = 8.0


@dataclass(frozen=True, slots=True)
class Oscillation:
    """A persistent narrow tone the aircraft produced and nobody commanded.

    Attributes:
        f_hz: Centre frequency.
        excess_db: How far above the local noise floor the line stands, averaged
            over the frames it was present in.
        duty: Fraction of the record it was present over. The difference between
            a tune that oscillates and a moment that did.
        amplitude_rad_s: RMS of the rate in the band around ``f_hz``.
        amplitude_frac: That amplitude as a fraction of the total rate RMS -- what
            share of the aircraft's motion is this rather than flying.
        amplification_db: Measured rate power in the band over commanded rate
            power in the same band -- the closed loop's magnitude at this
            frequency. Around 0 dB the aircraft is tracking its command. Large and
            positive means it is answering a small input with a large output,
            which is the definition of the resonance being looked for.
        model_optimism_db: How much gain margin the tool's model of the flown loop
            claims at ``f_hz``. The aircraft sustaining a tone there is a
            measurement that the model is wrong by at least this much, and it is
            what the design is made to hold back.
    """

    axis: Axis
    f_hz: float
    excess_db: float
    duty: float
    amplitude_rad_s: float
    amplitude_frac: float
    amplification_db: float
    model_optimism_db: float = 0.0


def detect_oscillation(
    bundle: LogBundle,
    axis: Axis,
    config: Config,
    *,
    noise: NoiseProfile | None = None,
    gyro_lpf_hz: float | None = None,
) -> Oscillation | None:
    """Find a sustained, uncommanded tone on one axis.

    Args:
        noise: The already-computed noise profile, used only to exclude peaks it
            classified as motor lines. Passed in rather than recomputed so that
            the two modules cannot disagree about what a peak is.
        gyro_lpf_hz: Search no higher than half of this. Above the gyro low-pass
            the loop has almost no authority, so a tone up there is sensor or
            structural rather than control.

    Returns:
        The strongest qualifying oscillation, or ``None``. ``None`` is a real
        result -- most flights have none -- but it is not proof of stability: an
        aircraft can be one gain step from a limit cycle without having entered
        one.
    """
    key = f"rate.{axis}.measured"
    if key not in bundle.signals:
        return None
    signal = bundle.signals[key]
    fs = signal.rate_hz

    f_min = config.float_("oscillation", "f_min_hz")
    f_max = fs / 2.0
    if gyro_lpf_hz:
        f_max = min(f_max, gyro_lpf_hz / 2.0)
    if signal.native_nyquist_hz is not None:
        f_max = min(f_max, signal.native_nyquist_hz)
    if f_max <= f_min * 1.5:
        return None

    nperseg = _frame_length(fs, f_min)
    if signal.y.size < 4 * nperseg:
        return None

    f_grid, _, sxx = spectrogram(
        signal.y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear"
    )
    band = (f_grid >= f_min) & (f_grid <= f_max)
    if band.sum() < 8:
        return None

    mean_psd = np.asarray(np.mean(sxx, axis=1), dtype=np.float64)
    candidate = _strongest_line(f_grid[band], mean_psd[band], noise)
    if candidate is None:
        return None
    f_osc, excess_db = candidate

    if excess_db < config.float_("oscillation", "min_excess_db"):
        return None

    duty = _duty(f_grid, sxx, f_osc)
    if duty < config.float_("oscillation", "min_duty"):
        return None

    amplitude = _band_rms(f_grid, mean_psd, f_osc)
    total = float(np.std(signal.y))
    fraction = amplitude / total if total > 0.0 else 0.0
    if fraction < config.float_("oscillation", "min_amplitude_frac"):
        return None

    amplification = _amplification_db(bundle, axis, f_osc, nperseg)
    if amplification < config.float_("oscillation", "min_amplification_db"):
        # The aircraft is tracking rather than ringing. Whatever the tone is, the
        # loop is not making it bigger, so it is not a stability problem.
        return None

    return Oscillation(
        axis=axis,
        f_hz=f_osc,
        excess_db=excess_db,
        duty=duty,
        amplitude_rad_s=amplitude,
        amplitude_frac=fraction,
        amplification_db=amplification,
    )


def model_optimism_db(
    f_hz: float,
    controller: RateController,
    airframe: AirframeModel,
    *,
    delay: LoopDelay,
    op: OperatingPoint | None = None,
) -> float:
    """How much gain margin the model claims at a frequency the aircraft oscillates at.

    A sustained limit cycle is a measurement: at that frequency the real loop has
    unity gain and enough phase lag to close the circle. If the model says the
    loop gain there is -9 dB, the model is wrong by 9 dB, and the honest response
    is to require that much more margin from anything designed against it.

    Returns:
        Decibels, never negative. Zero means the model already agrees the aircraft
        is at its limit there, which needs no correction -- only attention.
    """
    grid = np.asarray([f_hz], dtype=np.float64)
    loop = controller.feedback_response(grid) * plant_path(
        grid, controller, airframe, delay=delay, op=op
    )
    magnitude = float(np.abs(loop[0]))
    if magnitude <= 0.0:
        return 0.0
    return float(max(0.0, -20.0 * np.log10(magnitude)))


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #


def _frame_length(fs: float, f_min_hz: float) -> int:
    """Samples per spectrogram frame, from the resolution the bottom of the band needs."""
    wanted = fs / max(_RESOLUTION_FRACTION * f_min_hz, 1e-6)
    return int(2 ** np.ceil(np.log2(max(wanted, 64.0))))


def _band(f_hz: FloatArray, f_osc: float) -> np.ndarray:
    """Bins within the tone's band, never fewer than a few.

    The fractional half-width alone can be narrower than one bin at the bottom of
    the search range, which silently selects nothing and reports no oscillation.
    """
    df = float(np.median(np.diff(f_hz))) if f_hz.size > 1 else 0.0
    half = max(_BAND_FRACTION * f_osc, 2.0 * df)
    return np.asarray(np.abs(f_hz - f_osc) <= half)


def _strongest_line(
    f_hz: FloatArray, psd: FloatArray, noise: NoiseProfile | None
) -> tuple[float, float] | None:
    """Highest point above the local floor that is not a known motor line."""
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(psd, 1e-30))
    excess = db - noise_floor_db(f_hz, db)

    motors = tuple(
        peak.f_hz
        for peak in (noise.peaks if noise is not None else ())
        if peak.kind in ("motor_fundamental", "motor_harmonic")
    )
    allowed = np.ones(f_hz.shape, dtype=np.bool_)
    for f_motor in motors:
        allowed &= np.abs(f_hz - f_motor) > _MOTOR_TOLERANCE * f_motor
    if not allowed.any():
        return None

    index = int(np.argmax(np.where(allowed, excess, -np.inf)))
    return float(f_hz[index]), float(excess[index])


def _duty(f_grid: FloatArray, sxx: FloatArray, f_osc: float) -> float:
    """Fraction of frames in which the tone stands above its own surroundings.

    Two ways of doing this were tried and both failed, which is why the working
    one is spelt out. Measuring each frame against a median-filtered noise floor
    fires on noise about a third of the time: a single frame's periodogram is
    chi-squared with two degrees of freedom, so its bin-to-bin scatter is several
    decibels, and a five-second burst comes back as a thirty-percent duty cycle.
    Measuring each frame against the *loudest* frame fails the other way: a
    lightly damped loop rings hardest just after a stick input, so its median
    frame sits twenty-five decibels below its best one and a real oscillation
    reads as almost absent.

    What works is a prominence measured inside each frame, against a wide skirt
    either side of the tone. Averaging over many bins gives it many degrees of
    freedom and therefore little variance, and being relative to its own
    neighbourhood makes it independent of how hard the aircraft happened to be
    excited in that frame.
    """
    frames = int(sxx.shape[1])
    if frames == 0:
        return 0.0

    df = float(np.median(np.diff(f_grid))) if f_grid.size > 1 else 0.0
    half = max(_BAND_FRACTION * f_osc, 2.0 * df)
    offset = np.abs(f_grid - f_osc)
    inside = offset <= half
    skirt = (offset > _SKIRT_INNER * half) & (offset <= _SKIRT_OUTER * half)
    if inside.sum() < 1 or skirt.sum() < 4:
        return 0.0

    tone = np.mean(sxx[inside, :], axis=0)
    around = np.median(sxx[skirt, :], axis=0)
    with np.errstate(divide="ignore"):
        excess_db = 10.0 * np.log10(np.maximum(tone, 1e-30) / np.maximum(around, 1e-30))
    return float(np.count_nonzero(excess_db >= _PRESENT_DB)) / frames


def _band_rms(f_hz: FloatArray, psd: FloatArray, f_osc: float) -> float:
    """RMS of whatever sits in the band around ``f_osc``, in the signal's units."""
    near = _band(f_hz, f_osc)
    if near.sum() < 2:
        return 0.0
    return float(np.sqrt(np.trapezoid(psd[near], f_hz[near])))


def _amplification_db(bundle: LogBundle, axis: Axis, f_osc: float, nperseg: int) -> float:
    """Measured rate power over commanded rate power, in the band around the tone.

    The closed loop's magnitude at this frequency, measured rather than modelled.
    A loop tracking its command reads near 0 dB; one ringing reads far above it.

    Returns:
        Decibels, or positive infinity when the command carries no power in the
        band at all -- the aircraft is moving at a frequency nothing asked for,
        which is as clear a case as there is. Negative infinity when there is no
        setpoint signal to compare against, so a log without one cannot claim an
        oscillation it has no way to attribute.
    """
    from rotorid.core.analysis.spectra import power_spectrum

    key = f"rate.{axis}.setpoint"
    if key not in bundle.signals:
        return float("-inf")
    setpoint = bundle.signals[key]
    measured = bundle.signals[f"rate.{axis}.measured"]
    if setpoint.y.size < 4 * nperseg:
        return float("-inf")

    f_r, psd_r = power_spectrum(setpoint.y, setpoint.rate_hz, nperseg=nperseg)
    f_y, psd_y = power_spectrum(measured.y, measured.rate_hz, nperseg=nperseg)
    commanded = _band_power(f_r, psd_r, f_osc)
    produced = _band_power(f_y, psd_y, f_osc)
    if produced <= 0.0:
        return float("-inf")
    if commanded <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(produced / commanded))


def _band_power(f_hz: FloatArray, psd: FloatArray, f_osc: float) -> float:
    near = _band(f_hz, f_osc)
    if near.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[near], f_hz[near]))
