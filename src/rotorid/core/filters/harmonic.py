"""ArduPilot harmonic notch stack, reproduced exactly.

Transcribed from ``libraries/Filter/HarmonicNotchFilter.cpp``. The behaviour that
matters and is easy to get wrong:

* ``A`` and ``Q`` are computed **once**, at the fundamental, from
  ``bandwidth / composite_notches``. Harmonics reuse them and only scale the centre
  frequency, which makes the stack constant-Q -- harmonic *n* has roughly *n* times
  the absolute bandwidth of the fundamental.
* Composite notches are *not* symmetric about an implied centre: two notches sit at
  ``1 -/+ spread`` with **no** centre notch, while three sit at ``1.0`` and
  ``1 -/+ spread``.
* Below the minimum tracking frequency the firmware fades the attenuation out
  toward unity rather than switching the notch off, and disables it entirely below
  25% of the minimum.

Each of these changes the phase the loop actually sees, which is what the tool
trades against attenuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from rotorid.core.filters.biquad import (
    BiquadCoeffs,
    notch_A_Q,
    notch_biquad,
    px4_notch_A_Q,
)

__all__ = [
    "HARMONIC_NYQUIST_CUTOFF",
    "NOTCHFILTER_ATTENUATION_CUTOFF",
    "HarmonicNotch",
    "NotchOption",
    "composite_count",
    "harmonics_from_bitmask",
]

#: ``HarmonicNotchFilter.cpp``: notches at or above this fraction of the sample
#: rate are disabled.
HARMONIC_NYQUIST_CUTOFF: Final = 0.48

#: ``HarmonicNotchFilter.cpp``: below this fraction of the minimum tracking
#: frequency the notch is disabled outright.
NOTCHFILTER_ATTENUATION_CUTOFF: Final = 0.25

#: ``init()`` constrains the fundamental used for the A/Q calculation to at least
#: this multiple of the bandwidth.
_BANDWIDTH_LIMIT_FACTOR: Final = 0.52


class NotchOption:
    """Bit values of ``INS_HNTCH_OPTS`` / ``INS_HNTC2_OPTS``."""

    DOUBLE_NOTCH: Final = 1 << 0
    MULTI_SOURCE: Final = 1 << 1  # one notch set per motor / per FFT peak
    LOOP_RATE_UPDATE: Final = 1 << 2
    ALL_IMUS: Final = 1 << 3
    TRIPLE_NOTCH: Final = 1 << 4
    TREAT_LOW_AS_MIN: Final = 1 << 5


def harmonics_from_bitmask(hmncs: int) -> tuple[int, ...]:
    """Decode ``INS_HNTCH_HMNCS`` into harmonic multipliers.

    Bit 0 is the fundamental, bit 1 the second harmonic, and so on, up to 16x.

    Args:
        hmncs: The parameter value.

    Returns:
        Harmonic multipliers in ascending order, e.g. ``(1, 2, 3)`` for ``0b111``.
    """
    return tuple(n + 1 for n in range(16) if hmncs & (1 << n))


def composite_count(opts: int) -> Literal[1, 2, 3]:
    """Number of sub-notches per harmonic, from the ``OPTS`` bitmask.

    Triple wins if both bits are set: upstream guidance is to pick one, and triple
    is the preferred option.
    """
    if opts & NotchOption.TRIPLE_NOTCH:
        return 3
    if opts & NotchOption.DOUBLE_NOTCH:
        return 2
    return 1


@dataclass(frozen=True, slots=True)
class HarmonicNotch:
    """One configured harmonic notch stack (``INS_HNTCH_*`` or ``INS_HNTC2_*``).

    Attributes:
        freq_hz: ``FREQ``. Its meaning depends on ``mode``: the static centre, the
            hover-throttle centre for throttle tracking, or the lowest frequency
            worth tracking for RPM/ESC modes.
        freq_min_ratio: ``FM_RAT``. Sets the lowest frequency the notch will track
            down to, as a fraction of ``freq_hz``.
        sample_rate_hz: The rate the firmware runs this filter at (gyro rate).
    """

    freq_hz: float
    bandwidth_hz: float
    attenuation_db: float
    harmonics: tuple[int, ...]
    sample_rate_hz: float
    freq_min_ratio: float = 1.0
    opts: int = 0
    flavor: Literal["ardupilot", "px4"] = "ardupilot"

    @property
    def composite_notches(self) -> int:
        """Sub-notches per harmonic: 1, 2 (double) or 3 (triple).

        Always 1 on PX4, which has no composite-notch option at all. Reading
        ArduPilot's ``OPTS`` bits into a PX4 chain would silently triple the
        phase cost of every notch.
        """
        if self.flavor == "px4":
            return 1
        return composite_count(self.opts)

    @property
    def per_motor(self) -> bool:
        """Whether each source (motor / FFT peak) gets its own notch set."""
        return bool(self.opts & NotchOption.MULTI_SOURCE)

    @property
    def treat_low_as_min(self) -> bool:
        """Whether low frequencies clamp to the minimum instead of fading out.

        Always true on PX4: ``IMU_GYRO_DNF_MIN`` is a floor the tracked frequency
        stops at, not the start of a fade-out. ArduPilot's fade is a deliberate
        behaviour of its own and would be wrong to model here.
        """
        if self.flavor == "px4":
            return True
        return bool(self.opts & NotchOption.TREAT_LOW_AS_MIN)

    @property
    def minimum_freq_hz(self) -> float:
        """Lowest frequency the fundamental notch will track down to."""
        return self.freq_hz * self.freq_min_ratio

    def _shaping(self) -> tuple[float, float, float]:
        """Return ``(A, Q, notch_spread)`` as ``init()`` computes them.

        ``A`` and ``Q`` come from the *constrained* fundamental and from
        ``bandwidth / composite_notches``, once, for the whole stack.
        """
        if self.flavor == "px4":
            # No composite notches, no bandwidth constraint on the fundamental,
            # and no attenuation setting: PX4's notch is a true null shaped by
            # bandwidth alone.
            A, Q = px4_notch_A_Q(self.freq_hz, self.bandwidth_hz)
            return A, Q, 0.0

        nyquist_limit = self.sample_rate_hz * HARMONIC_NYQUIST_CUTOFF
        bandwidth_limit = self.bandwidth_hz * _BANDWIDTH_LIMIT_FACTOR
        center = float(np.clip(self.freq_hz, bandwidth_limit, nyquist_limit))
        spread = self.bandwidth_hz / (32.0 * center) if center > 0.0 else 0.0
        A, Q = notch_A_Q(center, self.bandwidth_hz / self.composite_notches, self.attenuation_db)
        return A, Q, spread

    def stages(self, centers_hz: float | list[float] | tuple[float, ...]) -> list[BiquadCoeffs]:
        """Build every enabled sub-notch for the given tracked centre frequencies.

        Args:
            centers_hz: The current tracked fundamental(s). One value for a single
                source; several for ``MULTI_SOURCE`` (one per motor / FFT peak).

        Returns:
            The biquads to cascade. Disabled notches are simply omitted.
        """
        centers = (
            [float(centers_hz)]
            if isinstance(centers_hz, (int, float))
            else [float(c) for c in centers_hz]
        )
        A_base, Q, spread = self._shaping()
        if Q <= 0.0:
            return []

        nyquist_limit = self.sample_rate_hz * HARMONIC_NYQUIST_CUTOFF
        composite = self.composite_notches
        spread_muls: tuple[float, ...]
        if composite == 2:
            spread_muls = (1.0 - spread, 1.0 + spread)
        elif composite == 3:
            spread_muls = (1.0, 1.0 - spread, 1.0 + spread)
        else:
            spread_muls = (1.0,)

        out: list[BiquadCoeffs] = []
        for center_raw in centers:
            center = float(np.clip(center_raw, 0.0, nyquist_limit))
            for harmonic_mul in self.harmonics:
                stage = self._harmonic_stages(
                    center, harmonic_mul, A_base, Q, spread_muls, nyquist_limit
                )
                out.extend(stage)
        return out

    def _harmonic_stages(
        self,
        center: float,
        harmonic_mul: int,
        A_base: float,
        Q: float,
        spread_muls: tuple[float, ...],
        nyquist_limit: float,
    ) -> list[BiquadCoeffs]:
        """Sub-notches for one harmonic of one source, mirroring ``set_center_frequency``."""
        notch_center = center * harmonic_mul
        if notch_center >= nyquist_limit:
            return []

        harmonic_min_freq = self.minimum_freq_hz
        A = A_base
        if self.treat_low_as_min:
            harmonic_min_freq *= harmonic_mul
        else:
            disable_freq = harmonic_min_freq * NOTCHFILTER_ATTENUATION_CUTOFF
            if notch_center < disable_freq:
                return []
            if notch_center < harmonic_min_freq:
                # Fade attenuation toward 1.0 (no attenuation) as we drop below the
                # minimum, so there is no discontinuity at the disable point.
                span = harmonic_min_freq - disable_freq
                frac = 0.0 if span <= 0.0 else (harmonic_min_freq - notch_center) / span
                A = A_base + (1.0 - A_base) * float(np.clip(frac, 0.0, 1.0))

        notch_center = max(notch_center, harmonic_min_freq)
        return [notch_biquad(notch_center * mul, A, Q, self.sample_rate_hz) for mul in spread_muls]

    def tracked_center_hz(
        self,
        *,
        throttle: float | None = None,
        ref: float | None = None,
        motor_hz: float | list[float] | None = None,
    ) -> list[float]:
        """Resolve the tracked centre frequency at one operating point.

        Throttle tracking scales as the square root of thrust, which is what makes
        ``REF`` a *thrust* reference rather than a frequency one::

            f = FREQ * sqrt(throttle / REF)

        Args:
            throttle: Normalized throttle, for throttle-based tracking.
            ref: ``INS_HNTCH_REF``. For throttle mode this is hover thrust.
            motor_hz: Measured motor frequency (RPM / ESC / FFT modes). A list
                gives one centre per motor for ``MULTI_SOURCE``.

        Returns:
            One or more centre frequencies, already floored at the minimum.

        Raises:
            ValueError: if neither a measured frequency nor a throttle+ref pair is given.
        """
        if motor_hz is not None:
            values = [motor_hz] if isinstance(motor_hz, (int, float)) else list(motor_hz)
            return [max(float(v), self.minimum_freq_hz) for v in values]

        if throttle is None or ref is None or ref <= 0.0:
            raise ValueError(
                "need either motor_hz, or throttle and a positive ref, to resolve the notch centre"
            )
        scaled = self.freq_hz * float(np.sqrt(max(throttle, 0.0) / ref))
        return [max(scaled, self.minimum_freq_hz)]
