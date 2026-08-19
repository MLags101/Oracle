"""Canonical units and every conversion between them.

Canonical form, enforced at the IO boundary (spec section 3):

    angular rate    rad/s
    angle           rad
    time            s
    frequency       Hz for anything user-facing, rad/s inside ``control`` objects
    actuator output normalized, [-1, 1]

This module owns every conversion. No ad-hoc ``* np.pi / 180`` anywhere else in
the codebase -- if you need a conversion that is not here, add it here.
"""

from __future__ import annotations

from typing import Final, TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CANONICAL_UNITS",
    "TWO_PI",
    "assert_canonical",
    "deg_to_rad",
    "hz_to_rads",
    "normalize_pwm",
    "rad_to_deg",
    "rads_to_hz",
    "rpm_to_hz",
]

TWO_PI: Final = 2.0 * np.pi

#: Canonical unit string for each kind of signal the tool handles.
CANONICAL_UNITS: Final[frozenset[str]] = frozenset(
    {"rad/s", "rad/s^2", "rad", "normalized", "s", "Hz", "V", "A", "rev/min", "fraction"}
)

_Num = TypeVar("_Num", float, NDArray[np.float64])


def deg_to_rad(x: _Num) -> _Num:
    """Degrees to radians. Works for angles and for rates (deg/s to rad/s)."""
    return x * (np.pi / 180.0)


def rad_to_deg(x: _Num) -> _Num:
    """Radians to degrees. Works for angles and for rates."""
    return x * (180.0 / np.pi)


def hz_to_rads(f_hz: _Num) -> _Num:
    """Frequency in Hz to angular frequency in rad/s.

    Used at every boundary into ``control`` / complex-response code, which works
    in rad/s. The UI never shows rad/s (spec section 10.6).
    """
    return f_hz * TWO_PI


def rads_to_hz(w_rads: _Num) -> _Num:
    """Angular frequency in rad/s to frequency in Hz, for display."""
    return w_rads / TWO_PI


def rpm_to_hz(rpm: _Num) -> _Num:
    """Motor RPM to its fundamental noise frequency in Hz.

    This is the shaft rotation rate, which is the harmonic-notch fundamental on
    both stacks. Blade-pass frequencies are integer multiples handled by the
    harmonic stack, not by this conversion.
    """
    return rpm / 60.0


def normalize_pwm(pwm_us: _Num, pwm_min: float, pwm_max: float) -> _Num:
    """Raw PWM microseconds to normalized [0, 1] motor output.

    Raises:
        ValueError: if the PWM range is degenerate.
    """
    span = pwm_max - pwm_min
    if span <= 0.0:
        raise ValueError(f"degenerate PWM range: min={pwm_min}, max={pwm_max}")
    return (pwm_us - pwm_min) / span


def assert_canonical(units: str, name: str = "signal") -> None:
    """Raise if ``units`` is not one of the canonical unit strings.

    Called by the IO layer on every :class:`~rotorid.core.types.Signal` it builds,
    so a reader that forgets a conversion fails at ingestion rather than producing
    a plausible-looking but wrong recommendation.

    Raises:
        ValueError: if the unit string is not canonical.
    """
    if units not in CANONICAL_UNITS:
        raise ValueError(
            f"{name}: non-canonical units {units!r}; "
            f"expected one of {sorted(CANONICAL_UNITS)}. Convert in the reader, not downstream."
        )
