"""The reader interface both stacks implement (spec section 6).

Everything above the IO layer is stack-agnostic, which only works if both readers
produce identical canonical keys in identical units. That is enforced here rather
than by convention: :func:`canonical_signal` is the only way a reader is meant to
construct a :class:`~rotorid.core.types.Signal`, and it rejects a key or a unit it
does not recognize.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from rotorid.core.types import AXES, FloatArray, LogBundle, Signal

__all__ = [
    "CANONICAL_KEYS",
    "LogReader",
    "ProgressCallback",
    "canonical_signal",
    "gate_signal",
    "native_rate_hz",
    "signal_units",
]

#: ``(fraction_complete, message)``. Readers call this so a GUI worker can drive
#: a progress bar without knowing anything about log formats.
ProgressCallback = Callable[[float, str], None]


def _per_axis(template: str, units: str) -> dict[str, str]:
    return {template.format(axis=axis): units for axis in AXES}


#: Canonical key to canonical unit. The single source of truth for section 6.3.
CANONICAL_KEYS: dict[str, str] = {
    **_per_axis("rate.{axis}.setpoint", "rad/s"),
    **_per_axis("rate.{axis}.measured", "rad/s"),
    **_per_axis("rate.{axis}.output", "normalized"),
    **_per_axis("rate.{axis}.accel", "rad/s^2"),
    **_per_axis("rate.{axis}.p_term", "normalized"),
    **_per_axis("rate.{axis}.i_term", "normalized"),
    **_per_axis("rate.{axis}.d_term", "normalized"),
    **_per_axis("rate.{axis}.ff_term", "normalized"),
    **_per_axis("rate.{axis}.dmod", "normalized"),
    **_per_axis("att.{axis}.setpoint", "rad"),
    **_per_axis("att.{axis}.measured", "rad"),
    **_per_axis("gyro.{axis}.prefilter", "rad/s"),
    **_per_axis("excite.{axis}", "normalized"),
    "batt.voltage": "V",
    "batt.current": "A",
    "cpu.load": "normalized",
    # A gate rather than a measurement: 1 while the firmware's own autotune was
    # running, 0 otherwise. Both stacks announce it, in completely different ways
    # (ArduPilot in the event log, PX4 in a status topic), and every consumer
    # wants the same question answered -- was the aircraft being deliberately
    # excited here -- so the difference is resolved in the readers.
    "mode.autotune": "normalized",
}

#: Keys that take an index, e.g. ``motor.3.rpm`` or ``imu.1.vibe.z``.
_INDEXED_KEYS: dict[str, str] = {
    "motor.{n}.output": "normalized",
    "motor.{n}.rpm": "rev/min",
    # Vibration is per-IMU rather than aggregated, because that is how both stacks
    # report it and because which IMU is shaking is diagnostic: one noisy sensor is
    # a mounting problem, three are an airframe problem.
    "imu.{n}.vibe.x": "m/s^2",
    "imu.{n}.vibe.y": "m/s^2",
    "imu.{n}.vibe.z": "m/s^2",
    # A running total, not a rate. Only its increase across a window means anything.
    "imu.{n}.clip": "count",
}


def signal_units(key: str) -> str:
    """Canonical units for a key.

    Raises:
        KeyError: on an unknown key. Readers must not invent keys -- a signal
            nobody downstream looks for is dead weight, and a misspelt one is a
            silently missing input.
    """
    if key in CANONICAL_KEYS:
        return CANONICAL_KEYS[key]
    parts = key.split(".")
    for position, part in enumerate(parts):
        if not part.isdigit():
            continue
        template = ".".join([*parts[:position], "{n}", *parts[position + 1 :]])
        if template in _INDEXED_KEYS:
            return _INDEXED_KEYS[template]
        break
    raise KeyError(f"{key!r} is not a canonical signal key (spec 6.3)")


def canonical_signal(
    key: str,
    t: FloatArray,
    y: FloatArray,
    *,
    source_msg: str,
    filtered: bool | None = None,
) -> Signal:
    """Build a signal, checking the key exists and stamping its canonical unit.

    Raises:
        KeyError: if the key is not canonical.
        ValueError: if the time and value arrays disagree in length.
    """
    units = signal_units(key)
    if t.shape != y.shape:
        raise ValueError(f"{key}: {t.shape} timestamps for {y.shape} values")
    return Signal(
        name=key,
        t=t,
        y=y,
        units=units,
        source_msg=source_msg,
        filtered=filtered,
        native_rate_hz=native_rate_hz(t),
    )


def gate_signal(
    key: str,
    grid: FloatArray,
    windows: Sequence[tuple[float, float]],
    *,
    source_msg: str,
) -> Signal:
    """A 0/1 signal on the grid, 1 inside each of ``windows``.

    Built directly rather than resampled. A gate is not a sampled quantity: a
    cubic spline through a step rings, so a hold-based construction is the only
    one whose output means what the name says at every sample.

    The native rate is the grid rate by construction -- the windows are exact,
    not sampled -- which keeps the evidence ceiling from being pulled down by a
    flag that was written twice in a five-minute flight.
    """
    y = np.zeros_like(grid)
    for start, end in windows:
        y[(grid >= start) & (grid <= end)] = 1.0
    units = signal_units(key)
    rate = float(1.0 / (grid[1] - grid[0])) if grid.size > 1 else None
    return Signal(
        name=key,
        t=grid,
        y=y,
        units=units,
        source_msg=source_msg,
        native_rate_hz=rate,
    )


def native_rate_hz(t: FloatArray) -> float | None:
    """The rate a message was logged at, from its own timestamps.

    The *median* interval, not the mean: a log with a handful of gaps -- a
    dropped SD write, a mode change, a moment the scheduler overran -- would have
    its mean interval dragged out by those few, and the answer we want is the
    rate the message was scheduled at, which the bulk of the samples agree on.

    Returns:
        The rate in Hz, or ``None`` if there are too few samples, or if every
        interval is zero (which some logs produce for a burst-written message).
    """
    if t.size < 3:
        return None
    dt = np.diff(np.sort(t))
    dt = dt[dt > 0.0]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


class LogReader(ABC):
    """One flight log, read in two passes.

    The two passes exist for the GUI: :meth:`index` must return fast enough to
    populate a "what is in this log" panel while the user is still looking at the
    file dialog, and only then does the expensive extraction run.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @abstractmethod
    def index(self) -> dict[str, int]:
        """Message type to count. Cheap first pass."""

    @abstractmethod
    def read(self, progress: ProgressCallback | None = None) -> LogBundle:
        """Full extraction onto the uniform grid."""
