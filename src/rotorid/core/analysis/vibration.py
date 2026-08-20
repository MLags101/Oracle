"""What the accelerometers say about the airframe (spec section 5).

Vibration is the precondition nothing else in the tool can work around. A
recommendation is a statement about the aircraft's rigid-body response, and a
frame that is shaking hard enough to move its own sensors is not producing a
measurement of that response -- it is producing a measurement of the shaking. So
this is read first, and it is allowed to stop everything below it.

Two separate quantities, deliberately not merged:

**Vibration level.** ArduPilot's ``VIBE`` message carries a per-IMU, per-axis
vibration figure in m/s^2. It is already a level rather than a raw signal, so the
statistic taken over a window is a high percentile of it -- the question the
tuning guide asks is how high the peaks go, not what the average was, and an
average over a long hover would hide the one aggressive manoeuvre where the frame
lost its composure.

**Clipping.** A running count of how often the accelerometer hit its measurement
range. This is categorical, not graded: a clipped sample is not a bad
measurement, it is an absent one, and no amount of averaging recovers it. One
clip inside an identification window is enough to say so.

The clip counters are read as ``max - min`` over the window rather than
``last - first``, with a half-count tolerance, because the counters reach this
module after cubic resampling onto the analysis grid and a spline through a step
rings on both sides of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rotorid.core.types import BoolArray, LogBundle

__all__ = ["VibrationSummary", "vibration_summary"]

#: Percentile of the windowed vibration level taken as "the" level. Not the max:
#: one sample of a 10 Hz message can be a write glitch, and the resampler's spline
#: overshoots a step. Not the mean either -- see the module docstring.
_LEVEL_PERCENTILE = 95.0

#: Percentile taken as "the highest it got". Not the maximum, for the same reason
#: the level is not the mean: one sample cannot establish an excursion, and the
#: counters arrive here splined onto the analysis grid. On a two-minute flight
#: this is still under a second, so a real excursion clears it easily and a lone
#: bad sample does not.
_PEAK_PERCENTILE = 99.5

#: A clip counter must rise by more than this across a window to count as having
#: clipped. Half a count, because the counter is an integer and the only thing
#: between zero and one is resampling ringing.
_CLIP_TOLERANCE = 0.5

_VIBE_KEY = re.compile(r"^imu\.(\d+)\.vibe\.([xyz])$")
_CLIP_KEY = re.compile(r"^imu\.(\d+)\.clip$")


@dataclass(frozen=True, slots=True)
class VibrationSummary:
    """Vibration and clipping over one set of time windows.

    Attributes:
        measured: Whether the log carried any vibration message at all. False is
            not "no vibration"; it is "no evidence either way", and the two must
            never be reported the same way.
        level_m_s2: The sustained level -- a high percentile, over every IMU and
            axis, of which the worst is taken. The worst rather than the average,
            because a single shaking sensor is enough to corrupt the measurement
            that sensor feeds.
        peak_m_s2: The highest the level got over the same windows -- a very high
            percentile rather than the maximum, so a single sample cannot claim
            an excursion the flight did not have. Reported
            alongside ``level_m_s2`` and never used to set severity: a flight can
            hit 26 m/s^2 for a moment and sit at 6 for the rest of it, and calling
            that a 26 m/s^2 aircraft is as wrong as calling it a 6 m/s^2 one.
        worst_imu: Which IMU produced ``level_m_s2``.
        worst_component: Which of x/y/z it was on.
        per_imu_m_s2: Worst level per IMU, for the display that shows whether the
            problem is one sensor or the whole frame.
        clip_measured: Whether any clip counter was present.
        clip_count: Total rise in the clip counters across every IMU, rounded.
        clipping_imus: The IMUs whose counters rose.
    """

    measured: bool
    level_m_s2: float
    peak_m_s2: float
    worst_imu: int
    worst_component: str
    per_imu_m_s2: dict[int, float]
    clip_measured: bool
    clip_count: int
    clipping_imus: tuple[int, ...]

    @property
    def clipped(self) -> bool:
        """Whether any accelerometer saturated inside the windows."""
        return bool(self.clipping_imus)


def vibration_summary(
    bundle: LogBundle,
    windows: tuple[tuple[float, float], ...] | None = None,
) -> VibrationSummary:
    """Summarize vibration and clipping, optionally only inside ``windows``.

    Args:
        bundle: The log.
        windows: ``(t_start, t_end)`` pairs in the log's own time base, normally
            the identification segments. ``None`` means the whole flight.
            Windows matter: a frame that shook on the ground and flew smoothly
            should not have its identification blocked, and a frame that only
            shook during the sweep should not have that hidden by a long calm
            hover either.

    Returns:
        A summary whose ``measured`` flag distinguishes "clean" from "unknown".
    """
    mask = _window_mask(bundle, windows)

    per_imu: dict[int, float] = {}
    level, peak, worst_imu, worst_component = 0.0, 0.0, -1, ""
    for key, signal in bundle.signals.items():
        match = _VIBE_KEY.match(key)
        if match is None:
            continue
        imu, component = int(match.group(1)), match.group(2)
        values = signal.y[mask]
        if values.size == 0:
            continue
        # Clamped at zero: a vibration level is a magnitude, and the only way one
        # arrives negative is the resampler's spline undershooting a fast drop.
        clamped = np.maximum(values, 0.0)
        windowed = float(np.percentile(clamped, _LEVEL_PERCENTILE))
        peak = max(peak, float(np.percentile(clamped, _PEAK_PERCENTILE)))
        per_imu[imu] = max(per_imu.get(imu, 0.0), windowed)
        if windowed > level:
            level, worst_imu, worst_component = windowed, imu, component

    clip_measured = False
    clip_total = 0.0
    clipping: list[int] = []
    for key, signal in bundle.signals.items():
        match = _CLIP_KEY.match(key)
        if match is None:
            continue
        values = signal.y[mask]
        if values.size == 0:
            continue
        clip_measured = True
        rise = float(np.max(values) - np.min(values))
        if rise > _CLIP_TOLERANCE:
            clipping.append(int(match.group(1)))
            clip_total += rise

    return VibrationSummary(
        measured=bool(per_imu),
        level_m_s2=level,
        peak_m_s2=peak,
        worst_imu=worst_imu,
        worst_component=worst_component,
        per_imu_m_s2=per_imu,
        clip_measured=clip_measured,
        clip_count=round(clip_total),
        clipping_imus=tuple(sorted(clipping)),
    )


def _window_mask(
    bundle: LogBundle, windows: tuple[tuple[float, float], ...] | None
) -> BoolArray | slice:
    """Boolean mask over the analysis grid selecting the requested windows."""
    if windows is None:
        return slice(None)
    grid = next(iter(bundle.signals.values())).t
    mask = np.zeros(grid.shape, dtype=np.bool_)
    for t_start, t_end in windows:
        mask |= (grid >= t_start) & (grid <= t_end)
    if not mask.any():
        # An empty selection would silently report a clean aircraft. Every signal
        # is on one shared grid, so this means the windows fell outside the log,
        # which is a bug in the caller rather than a fact about the vehicle.
        raise ValueError(f"windows {windows} select no samples of the {grid.size}-sample grid")
    return mask
