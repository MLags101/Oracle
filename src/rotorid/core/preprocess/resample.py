"""Asynchronous log messages onto one uniform grid (spec section 5.1).

Log messages arrive jittered and at different rates, and every spectral method
downstream assumes uniform sampling. Getting this wrong is invisible: an
interpolation that quietly smooths a signal shows up much later as an airframe
that appears to have more lag than it does.

Two rules:

* The grid rate is derived from what has to be represented, never assumed.
* Jitter is measured and reported rather than interpolated away in silence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt

from rotorid.core.io.base import native_rate_hz
from rotorid.core.types import FloatArray, Signal

__all__ = ["JitterStats", "grid_rate_hz", "measure_jitter", "resample_to_grid", "uniform_grid"]


@dataclass(frozen=True, slots=True)
class JitterStats:
    """How uniform a message stream actually was."""

    median_gap_s: float
    p99_gap_s: float
    max_gap_s: float
    n_duplicate_timestamps: int

    @property
    def ratio(self) -> float:
        """p99 gap over median gap. 1.0 is perfect."""
        return self.p99_gap_s / self.median_gap_s if self.median_gap_s > 0.0 else np.inf

    def is_irregular(self, warn_ratio: float) -> bool:
        """Whether this stream should raise ``LOG_RATE_IRREGULAR``."""
        return self.ratio > warn_ratio


def measure_jitter(t: FloatArray) -> JitterStats:
    """Gap statistics for one message stream.

    Raises:
        ValueError: if there are fewer than two samples to measure between.
    """
    if t.size < 2:
        raise ValueError("need at least two samples to measure jitter")
    gaps = np.diff(np.sort(t))
    return JitterStats(
        median_gap_s=float(np.median(gaps)),
        p99_gap_s=float(np.percentile(gaps, 99)),
        max_gap_s=float(np.max(gaps)),
        n_duplicate_timestamps=int(np.sum(gaps <= 0.0)),
    )


def grid_rate_hz(
    *,
    gyro_sample_rate_hz: float,
    loop_rate_hz: float,
    highest_modeled_notch_hz: float,
    min_oversample_of_highest_notch: float,
) -> float:
    """Choose the analysis grid rate.

    Two floors and one ceiling. The grid must be at least twice the loop rate, so
    the controller's own bandwidth is represented; and at least
    ``min_oversample_of_highest_notch`` times the highest notch the filter model
    has to reproduce, because a discrete notch evaluated on too coarse a grid has
    the wrong depth and the wrong phase. It is never raised above the gyro rate --
    there is no information up there to recover.

    Raises:
        ValueError: if the gyro rate cannot satisfy the notch requirement, which
            is a real finding about the vehicle's configuration rather than
            something to round away.
    """
    needed_for_notch = min_oversample_of_highest_notch * highest_modeled_notch_hz
    needed = max(2.0 * loop_rate_hz, needed_for_notch)
    if needed_for_notch > gyro_sample_rate_hz:
        raise ValueError(
            f"a notch at {highest_modeled_notch_hz:g} Hz needs a "
            f"{needed_for_notch:g} Hz grid, but the gyro runs at {gyro_sample_rate_hz:g} Hz"
        )
    return float(min(needed, gyro_sample_rate_hz))


def uniform_grid(t_start: float, t_end: float, rate_hz: float) -> FloatArray:
    """Uniform time base covering ``[t_start, t_end]``.

    Raises:
        ValueError: on a non-positive span or rate.
    """
    if t_end <= t_start:
        raise ValueError(f"empty time span {t_start}..{t_end}")
    if rate_hz <= 0.0:
        raise ValueError(f"invalid grid rate {rate_hz}")
    n = int(np.floor((t_end - t_start) * rate_hz)) + 1
    return np.asarray(t_start + np.arange(n, dtype=np.float64) / rate_hz, dtype=np.float64)


def resample_to_grid(signal: Signal, grid: FloatArray) -> Signal:
    """Put one signal on the uniform grid by cubic interpolation.

    Duplicate timestamps are dropped first, keeping the first sample: a repeated
    timestamp is a logging artefact, and a spline through it is undefined.

    Extrapolation is deliberately refused -- values outside the signal's own span
    are filled with the nearest end sample rather than with a spline's opinion,
    which for a cubic can diverge spectacularly within a few samples.

    A signal logged *faster* than the grid is low-passed at the grid's Nyquist
    first. Splining it directly would fold everything above that back down as a
    mirror image, and there is no way to tell afterwards: the folded lines look
    exactly like real ones, at plausible frequencies, and land in the same band
    the notch designer works in. Raw gyro logged at 1.6 kHz onto an 800 Hz grid
    would put its 500 Hz content at 300 Hz and call it a frame resonance.

    Raises:
        ValueError: if fewer than four unique samples remain, which is below what
            a cubic spline can fit.
    """
    order = np.argsort(signal.t, kind="stable")
    t_sorted = signal.t[order]
    y_sorted = signal.y[order]

    keep = np.concatenate([[True], np.diff(t_sorted) > 0.0])
    t_unique = t_sorted[keep]
    y_unique = y_sorted[keep]
    if t_unique.size < 4:
        raise ValueError(
            f"{signal.name}: {t_unique.size} unique samples is too few to resample; "
            "the message is present but effectively empty"
        )

    source_rate = signal.native_rate_hz or native_rate_hz(t_unique)
    grid_rate = 1.0 / float(grid[1] - grid[0]) if grid.size > 1 else None
    if source_rate is not None and grid_rate is not None and source_rate > grid_rate * 1.05:
        y_unique = _antialias(y_unique, source_rate, grid_rate)
        source_rate = grid_rate

    spline = CubicSpline(t_unique, y_unique, extrapolate=False)
    y = np.asarray(spline(grid), dtype=np.float64)
    y[grid < t_unique[0]] = y_unique[0]
    y[grid > t_unique[-1]] = y_unique[-1]
    _fill_gaps_linearly(grid, y, t_unique, y_unique)

    return Signal(
        name=signal.name,
        t=grid,
        y=y,
        units=signal.units,
        source_msg=signal.source_msg,
        filtered=signal.filtered,
        # On the grid every signal reports the grid rate, and the one number that
        # says how much of that is real is the rate it arrived at -- losing it here
        # is how a 10 Hz message comes to look like an 800 Hz one. Carried
        # unchanged, except where the anti-alias filter above has just made the
        # claim untrue: after decimation the signal genuinely does not describe
        # anything past the grid's Nyquist, and reporting the rate it arrived at
        # would let the evidence ceiling promise a band that is now empty by
        # construction.
        native_rate_hz=source_rate,
    )


#: How many times the median sample spacing a hole has to be before the spline is
#: not trusted across it. A message that misses a sample or two is still a smooth
#: record; one that stops for a second is two records with a hole between them.
_GAP_MULTIPLE = 8.0


def _fill_gaps_linearly(
    grid: FloatArray, y: FloatArray, t_unique: FloatArray, y_unique: FloatArray
) -> None:
    """Replace spline values inside long dropouts with a straight line. In place.

    A cubic spline is only well behaved where it has neighbours. Across a hole --
    a dropped SD write, a message that stopped for a second, a log that was
    briefly too busy -- it is fitting a third-order polynomial to two distant
    points and the two beyond them, and on a noisy kilohertz signal it can leave
    the data's range by three orders of magnitude. Raw gyro logged with dropouts
    came back with excursions of forty thousand radians per second, which is not
    a value anything downstream is equipped to disbelieve.

    Linear rather than a hold, because a hold introduces a step and a step has a
    spectrum. Neither one *adds* information; the point is only that the
    interpolation must not invent any.
    """
    if t_unique.size < 2:
        return
    gaps = np.diff(t_unique)
    threshold = _GAP_MULTIPLE * float(np.median(gaps))
    if not np.any(gaps > threshold):
        return
    left = np.searchsorted(t_unique, grid, side="right") - 1
    inside = (left >= 0) & (left < gaps.size)
    suspect = np.zeros(grid.shape, dtype=np.bool_)
    suspect[inside] = gaps[left[inside]] > threshold
    if suspect.any():
        y[suspect] = np.interp(grid[suspect], t_unique, y_unique)


def _antialias(y: FloatArray, source_rate_hz: float, grid_rate_hz: float) -> FloatArray:
    """Low-pass just under the grid's Nyquist, zero-phase.

    ``sosfiltfilt`` rather than a one-way filter: this runs before the samples are
    placed on the grid, and a filter that shifted them in time would move every
    feature of the spectrum relative to every signal that was not decimated.
    """
    nyquist = 0.5 * source_rate_hz
    corner = min(0.45 * grid_rate_hz / nyquist, 0.99)
    sos = butter(8, corner, btype="lowpass", output="sos")
    return np.asarray(sosfiltfilt(sos, y), dtype=np.float64)
