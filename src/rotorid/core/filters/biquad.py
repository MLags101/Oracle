"""Firmware-exact digital filter primitives.

Every formula here is transcribed from autopilot source, not from a textbook, and
every filter is evaluated in **discrete time** at the rate the firmware runs it.
Analog approximations get the phase wrong at exactly the frequencies that decide
achievable crossover, which is the whole point of modeling filters at all
(spec section 0, rule 5).

Sources:
    ArduPilot ``libraries/Filter/NotchFilter.cpp``      -- ``calculate_A_and_Q``,
                                                          ``init_with_A_and_Q``
    ArduPilot ``libraries/Filter/LowPassFilter2p.cpp``  -- ``compute_params``
    ArduPilot ``libraries/AC_PID/AC_PID.cpp``           -- 1-pole target/error/derivative
                                                          filters (``get_filt_*_alpha``)
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PASSTHROUGH",
    "BiquadCoeffs",
    "biquad_response",
    "cascade_response",
    "lpf2p_biquad",
    "notch_A_Q",
    "notch_biquad",
    "onepole_alpha",
    "onepole_response",
    "phase_lag_deg",
    "px4_lpf2p_biquad",
    "px4_notch_A_Q",
]

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

#: ArduPilot clamps a 2-pole low-pass cutoff to 40% of the sample rate.
#: ``LowPassFilter2p.cpp``: ``ret.cutoff_freq = MIN(cutoff_freq, sample_freq * 0.4)``.
LPF2P_MAX_CUTOFF_FRACTION: Final = 0.4

_COS_PI_4: Final = float(np.cos(np.pi / 4.0))


class BiquadCoeffs:
    """Normalized biquad coefficients, ``a0 == 1``.

    Difference equation::

        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    """

    __slots__ = ("a", "b", "sample_rate_hz")

    def __init__(
        self, b: tuple[float, float, float], a: tuple[float, float, float], sample_rate_hz: float
    ) -> None:
        """Store coefficients, normalizing by ``a[0]``.

        Raises:
            ValueError: if ``a[0]`` is zero or the sample rate is not positive.
        """
        if a[0] == 0.0:
            raise ValueError("biquad a0 must be non-zero")
        if sample_rate_hz <= 0.0:
            raise ValueError(f"sample rate must be positive, got {sample_rate_hz}")
        a0 = a[0]
        self.b: tuple[float, float, float] = (b[0] / a0, b[1] / a0, b[2] / a0)
        self.a: tuple[float, float, float] = (1.0, a[1] / a0, a[2] / a0)
        self.sample_rate_hz = sample_rate_hz

    def response(self, f_hz: FloatArray | float) -> ComplexArray:
        """Complex frequency response at ``f_hz``."""
        return biquad_response(self.b, self.a, f_hz, self.sample_rate_hz)

    def __repr__(self) -> str:
        return f"BiquadCoeffs(b={self.b}, a={self.a}, fs={self.sample_rate_hz})"


#: A filter that does nothing. Used where the firmware disables a stage.
PASSTHROUGH: Final = BiquadCoeffs((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)


def notch_A_Q(center_hz: float, bandwidth_hz: float, attenuation_db: float) -> tuple[float, float]:
    """Attenuation factor and quality factor for an ArduPilot notch.

    Transcribed from ``NotchFilter::calculate_A_and_Q``::

        A       = 10^(-attenuation_dB / 40)
        octaves = 2 * log2( f0 / (f0 - BW/2) )
        Q       = sqrt(2^octaves) / (2^octaves - 1)

    The firmware disables the notch when ``center_freq <= bandwidth/2``, because
    the octave expression is undefined there; we mirror that by returning ``Q = 0``,
    which :func:`notch_biquad` turns into a pass-through.

    Args:
        center_hz: Notch centre frequency.
        bandwidth_hz: Bandwidth between the -3 dB points.
        attenuation_db: Depth at the centre, in dB (positive number).

    Returns:
        ``(A, Q)``. ``Q == 0.0`` means "disabled".
    """
    A = 10.0 ** (-attenuation_db / 40.0)
    if center_hz <= 0.5 * bandwidth_hz or center_hz <= 0.0:
        return A, 0.0
    octaves = 2.0 * np.log2(center_hz / (center_hz - bandwidth_hz / 2.0))
    two_oct = 2.0**octaves
    Q = float(np.sqrt(two_oct) / (two_oct - 1.0))
    return A, Q


def px4_notch_A_Q(center_hz: float, bandwidth_hz: float) -> tuple[float, float]:
    """Attenuation factor and quality factor for a PX4 notch.

    Transcribed from ``mathlib/math/filter/NotchFilter.hpp``::

        Q     = notch_freq / bandwidth
        alpha = sin(2*pi*f0/fs) / (2*Q)

    Two differences from ArduPilot matter and are the reason this is a separate
    function rather than a parameter:

    * PX4 has **no attenuation setting**. Its notch is a true null -- ``A = 0`` --
      so depth is not a design variable there; bandwidth alone sets the shape.
    * ``Q`` is the plain ``f0/BW`` ratio rather than ArduPilot's octave-based
      expression, which gives a measurably different skirt for the same numbers.

    Returns:
        ``(A, Q)`` in the same shape :func:`notch_biquad` consumes. ``Q == 0.0``
        means "disabled", matching the ArduPilot path.
    """
    if center_hz <= 0.0 or bandwidth_hz <= 0.0:
        return 0.0, 0.0
    return 0.0, float(center_hz / bandwidth_hz)


def px4_lpf2p_biquad(cutoff_hz: float, sample_rate_hz: float) -> BiquadCoeffs:
    """PX4 2-pole low-pass (``IMU_GYRO_CUTOFF``, ``IMU_DGYRO_CUTOFF``).

    Transcribed from ``mathlib/math/filter/LowPassFilter2p.hpp``. It is the same
    Butterworth design ArduPilot uses, written differently::

        fr  = fs / cutoff
        ohm = tan(pi / fr)
        c   = 1 + 2*cos(pi/4)*ohm + ohm^2

    PX4 disables the filter when the cutoff is at or above the Nyquist rate, and
    unlike ArduPilot it applies no 0.4*fs clamp -- so a badly chosen
    ``IMU_GYRO_CUTOFF`` behaves differently on the two stacks, and this function
    has to behave differently with it.
    """
    if cutoff_hz <= 0.0 or cutoff_hz >= 0.5 * sample_rate_hz:
        return BiquadCoeffs((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), sample_rate_hz)
    ohm = float(np.tan(np.pi * cutoff_hz / sample_rate_hz))
    ohm2 = ohm * ohm
    c = 1.0 + 2.0 * _COS_PI_4 * ohm + ohm2
    b0 = ohm2 / c
    return BiquadCoeffs(
        b=(b0, 2.0 * b0, b0),
        a=(1.0, 2.0 * (ohm2 - 1.0) / c, (1.0 - 2.0 * _COS_PI_4 * ohm + ohm2) / c),
        sample_rate_hz=sample_rate_hz,
    )


def notch_biquad(center_hz: float, A: float, Q: float, sample_rate_hz: float) -> BiquadCoeffs:
    """One ArduPilot notch biquad.

    Transcribed from ``NotchFilter::init_with_A_and_Q``::

        w     = 2*pi*f0/fs
        alpha = sin(w) / (2Q)
        b = [1 + alpha*A^2, -2cos(w), 1 - alpha*A^2]
        a = [1 + alpha,     -2cos(w), 1 - alpha]

    Harmonics reuse the ``A`` and ``Q`` computed at the fundamental -- the firmware
    computes them once in ``HarmonicNotchFilter::init`` and only multiplies the
    centre frequency. That makes the notch constant-Q, so a harmonic's absolute
    bandwidth grows in proportion to its centre frequency.

    Args:
        center_hz: Centre frequency of this notch.
        A: Attenuation factor from :func:`notch_A_Q`.
        Q: Quality factor from :func:`notch_A_Q`. Zero disables the notch.
        sample_rate_hz: Rate the filter runs at in the firmware.

    Returns:
        Normalized coefficients, or :data:`PASSTHROUGH` semantics when disabled.
    """
    if Q <= 0.0 or center_hz <= 0.0 or center_hz >= 0.5 * sample_rate_hz:
        return BiquadCoeffs((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), sample_rate_hz)
    w = 2.0 * np.pi * center_hz / sample_rate_hz
    alpha = float(np.sin(w)) / (2.0 * Q)
    cos_w = float(np.cos(w))
    a_sq = A * A
    return BiquadCoeffs(
        b=(1.0 + alpha * a_sq, -2.0 * cos_w, 1.0 - alpha * a_sq),
        a=(1.0 + alpha, -2.0 * cos_w, 1.0 - alpha),
        sample_rate_hz=sample_rate_hz,
    )


def lpf2p_biquad(cutoff_hz: float, sample_rate_hz: float) -> BiquadCoeffs:
    """ArduPilot 2-pole low-pass (``INS_GYRO_FILTER``, ``INS_ACCEL_FILTER``).

    Transcribed from ``DigitalBiquadFilter::compute_params``::

        cutoff = min(cutoff, 0.4 * fs)
        fr     = fs / cutoff
        ohm    = tan(pi / fr)
        c      = 1 + 2*cos(pi/4)*ohm + ohm^2
        b0 = ohm^2 / c ;  b1 = 2*b0 ;  b2 = b0
        a1 = 2*(ohm^2 - 1)/c
        a2 = (1 - 2*cos(pi/4)*ohm + ohm^2) / c

    A non-positive cutoff is a pass-through in the firmware, and here.

    Args:
        cutoff_hz: Requested cutoff. Clamped to ``0.4 * sample_rate_hz``.
        sample_rate_hz: Rate the filter runs at.
    """
    if cutoff_hz <= 0.0:
        return BiquadCoeffs((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), sample_rate_hz)
    cutoff = min(cutoff_hz, sample_rate_hz * LPF2P_MAX_CUTOFF_FRACTION)
    fr = sample_rate_hz / cutoff
    ohm = float(np.tan(np.pi / fr))
    ohm2 = ohm * ohm
    c = 1.0 + 2.0 * _COS_PI_4 * ohm + ohm2
    b0 = ohm2 / c
    return BiquadCoeffs(
        b=(b0, 2.0 * b0, b0),
        a=(1.0, 2.0 * (ohm2 - 1.0) / c, (1.0 - 2.0 * _COS_PI_4 * ohm + ohm2) / c),
        sample_rate_hz=sample_rate_hz,
    )


def onepole_alpha(cutoff_hz: float, dt: float) -> float:
    """Smoothing factor of ArduPilot's 1-pole IIR filter.

    ``AC_PID`` uses this form for ``FLTT`` (target), ``FLTE`` (error) and ``FLTD``
    (derivative), all running at the loop rate::

        alpha = dt / (dt + 1/(2*pi*fc))
        y += alpha * (x - y)

    A non-positive cutoff disables the filter (``alpha = 1``, pass-through).

    Args:
        cutoff_hz: Corner frequency.
        dt: Loop period in seconds.

    Returns:
        ``alpha`` in ``(0, 1]``.

    Raises:
        ValueError: if ``dt`` is not positive.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if cutoff_hz <= 0.0:
        return 1.0
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(dt / (dt + rc))


def onepole_response(alpha: float, f_hz: FloatArray | float, sample_rate_hz: float) -> ComplexArray:
    """Complex response of the 1-pole IIR ``y[n] = y[n-1] + alpha*(x[n] - y[n-1])``.

    Transfer function ``H(z) = alpha / (1 - (1-alpha) z^-1)``.
    """
    f = np.asarray(f_hz, dtype=np.float64)
    z_inv = np.exp(-2j * np.pi * f / sample_rate_hz)
    return np.asarray(alpha / (1.0 - (1.0 - alpha) * z_inv), dtype=np.complex128)


def biquad_response(
    b: tuple[float, float, float],
    a: tuple[float, float, float],
    f_hz: FloatArray | float,
    sample_rate_hz: float,
) -> ComplexArray:
    """Complex response ``H(e^{jwT})`` of a normalized biquad."""
    f = np.asarray(f_hz, dtype=np.float64)
    z_inv = np.exp(-2j * np.pi * f / sample_rate_hz)
    z_inv2 = z_inv * z_inv
    num = b[0] + b[1] * z_inv + b[2] * z_inv2
    den = a[0] + a[1] * z_inv + a[2] * z_inv2
    return np.asarray(num / den, dtype=np.complex128)


def cascade_response(stages: list[BiquadCoeffs], f_hz: FloatArray | float) -> ComplexArray:
    """Product of the responses of a cascade of biquads.

    Raises:
        ValueError: if the stages disagree about the sample rate, which would
            silently produce a wrong phase.
    """
    f = np.asarray(f_hz, dtype=np.float64)
    total = np.ones_like(f, dtype=np.complex128)
    if not stages:
        return total
    rate = stages[0].sample_rate_hz
    for stage in stages:
        if stage.sample_rate_hz != rate:
            raise ValueError(
                "cascade mixes sample rates "
                f"({rate} vs {stage.sample_rate_hz}); build one cascade per rate"
            )
        total = total * stage.response(f)
    return total


def phase_lag_deg(response: ComplexArray) -> FloatArray:
    """Phase lag in degrees, positive for lag, unwrapped.

    The tool talks about filter cost in "degrees of lag at the crossover", so this
    flips the sign of the usual phase convention to keep that reading natural.
    """
    return np.asarray(-np.degrees(np.unwrap(np.angle(response))), dtype=np.float64)
