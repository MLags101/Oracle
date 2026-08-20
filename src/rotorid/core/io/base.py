"""The reader interface both stacks implement (spec section 6).

Everything above the IO layer is stack-agnostic, which only works if both readers
produce identical canonical keys in identical units. That is enforced here rather
than by convention: :func:`canonical_signal` is the only way a reader is meant to
construct a :class:`~rotorid.core.types.Signal`, and it rejects a key or a unit it
does not recognize.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rotorid.core.types import AXES, FloatArray, LogBundle, Signal

__all__ = [
    "CANONICAL_KEYS",
    "LogReader",
    "ProgressCallback",
    "canonical_signal",
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
}

#: Keys that take an index, e.g. ``motor.3.rpm``.
_INDEXED_KEYS: dict[str, str] = {
    "motor.{n}.output": "normalized",
    "motor.{n}.rpm": "rev/min",
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
    if len(parts) == 3 and parts[0] == "motor" and parts[1].isdigit():
        template = f"motor.{{n}}.{parts[2]}"
        if template in _INDEXED_KEYS:
            return _INDEXED_KEYS[template]
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
