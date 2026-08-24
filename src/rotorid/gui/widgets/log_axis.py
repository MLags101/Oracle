"""A readable frequency axis (spec section 10.6).

Every frequency plot in this tool is log-x, because a linear axis from 0 to
500 Hz hides everything below 20 -- which is where the aircraft is. pyqtgraph's
own log axis, though, draws its labels on top of each other, and the reason is
worth writing down because it is not obvious from the symptom.

In log mode pyqtgraph returns *every* tick -- 1 through 9 in each decade -- as a
single tick level. Its crowding protection then declines to help, because that
protection begins at the second level: the first is always drawn, on the
assumption that a first level is sparse. Over three decades that is twenty-odd
labels stacked into the width of one axis, and the result reads as
``0.70 9 0.01 2 3 4 5 6 7 8 910`` -- not a scale so much as a smear.

So this splits the ticks into the three levels they should have been:

* **Labels that always fit.** The densest ladder -- decades, or 1-2-5, or
  1-2-3-5-7 -- whose labels genuinely fit the axis at its current width, measured
  with the font actually in use rather than guessed at.
* **Labels if there is room.** The next ladder up, offered as a second level, so
  pyqtgraph's crowding logic can take it away when the plot is narrow and put it
  back when the user widens the window.
* **Tick marks with no labels.** Everything else, so the decade structure is
  still visible as marks without competing for space as text.

The strings are plain Hz as well. pyqtgraph writes ``2·10¹`` where this
application's users read ``20``, and every number that reaches this screen is a
frequency somebody is about to type into a ground station.
"""

from __future__ import annotations

import math

import pyqtgraph as pg
from PySide6.QtGui import QFontMetricsF

__all__ = ["LogAxis"]

#: The ladders, sparse first. Each is the set of mantissas labelled within every
#: decade; the axis uses the densest one that fits and offers the next as a
#: second level. Stopping at 1-2-3-5-7 is deliberate -- a decade with all nine
#: mantissas labelled has never fitted on a plot this size, and offering it only
#: gives pyqtgraph something to reject.
_LADDERS: tuple[tuple[int, ...], ...] = (
    (1,),
    (1, 2, 5),
    (1, 2, 3, 5, 7),
)

#: Every mantissa, for the unlabelled marks that show where the decade divides.
_ALL = tuple(range(1, 10))

#: Clear air between one label's edge and the next, in pixels. Two labels that
#: merely touch are two labels a reader has to separate by eye.
_GAP = 8.0


# pyqtgraph ships no type information, so mypy sees ``AxisItem`` as ``Any`` and
# objects to a class deriving from it. Subclassing the axis is the interface
# pyqtgraph documents for exactly this, so the objection is about the library's
# stubs rather than about this code.
class LogAxis(pg.AxisItem):  # type: ignore[misc]
    """A log-scale axis whose labels do not collide, in Hz rather than in powers."""

    def __init__(self, orientation: str = "bottom", **kwargs: object) -> None:
        super().__init__(orientation, **kwargs)
        # Levels 0 and 1 carry text; everything below is marks only. pyqtgraph
        # labels up to level 2 by default, which would put text back on the
        # minor ticks this class exists to strip it from.
        self.setStyle(maxTextLevel=1)

    # ----------------------------------------------------------------- #

    def logTickValues(
        self,
        minVal: float,
        maxVal: float,
        size: float,
        stdTicks: list[tuple[float | None, list[float]]],
    ) -> list[tuple[float | None, list[float]]]:
        lo, hi = sorted((float(minVal), float(maxVal)))
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo or size <= 0:
            return self._fallback(minVal, maxVal, size, stdTicks)

        every = self._points(lo, hi, _ALL)
        if len(every) < 2:
            # Zoomed inside a single decade, where a per-decade ladder has
            # nothing to say. pyqtgraph's own behaviour is right here, and it
            # does not overlap because there is little to draw.
            return self._fallback(minVal, maxVal, size, stdTicks)

        labelled, offered = self._choose(lo, hi, size)
        marks = [v for v in every if v not in labelled and v not in offered]
        return [(None, labelled), (None, offered), (None, marks)]

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        if not self.logMode:  # pragma: no cover - this axis is only used log-scaled
            linear: list[str] = super().tickStrings(values, scale, spacing)
            return linear
        return [_hz(10.0**value) for value in values]

    def _fallback(
        self,
        minVal: float,
        maxVal: float,
        size: float,
        stdTicks: list[tuple[float | None, list[float]]],
    ) -> list[tuple[float | None, list[float]]]:
        """Let pyqtgraph decide, for the ranges where its answer is already fine."""
        levels: list[tuple[float | None, list[float]]] = super().logTickValues(
            minVal, maxVal, size, stdTicks
        )
        return levels

    # ----------------------------------------------------------------- #

    def _choose(self, lo: float, hi: float, size: float) -> tuple[list[float], list[float]]:
        """The labels that fit, and the next ladder up to offer beneath them."""
        fitting = _LADDERS[0]
        for ladder in _LADDERS:
            if self._fits(self._points(lo, hi, ladder), lo, hi, size):
                fitting = ladder
            else:
                break

        labelled = self._points(lo, hi, fitting)
        # Even one label per decade does not fit across six decades on a narrow
        # plot. Dropping whole decades keeps the ones that remain legible, which
        # is worth more than a complete but unreadable scale.
        while len(labelled) > 2 and not self._fits(labelled, lo, hi, size):
            labelled = labelled[::2]

        denser = _LADDERS[min(_LADDERS.index(fitting) + 1, len(_LADDERS) - 1)]
        chosen = set(labelled)
        offered = [v for v in self._points(lo, hi, denser) if v not in chosen]

        # A window that spans no whole decade -- 12 Hz to 79 Hz, say -- has no
        # decade tick in it at all. Rather than leave the always-drawn level
        # empty and let the crowding rule take the only labels there are, the
        # offered level is promoted.
        if not labelled:
            return offered, []
        return labelled, offered

    def _fits(self, values: list[float], lo: float, hi: float, size: float) -> bool:
        """Whether every neighbouring pair of these labels clears the next.

        Pairwise, not a sum of widths against the axis length. On a log scale the
        ticks are not evenly spaced -- 5 and 7 sit far closer together than 1 and
        2 -- so a set whose labels total less than the axis width can still have
        two of them on top of each other at the crowded end of a decade. The
        question is only ever about neighbours.
        """
        if len(values) < 2:
            return True
        span = hi - lo
        if span <= 0:
            return True
        metrics = QFontMetricsF(self.font())
        widths = [metrics.horizontalAdvance(_hz(10.0**value)) for value in values]
        centres = [(value - lo) / span * size for value in values]
        return all(
            centres[i + 1] - centres[i] >= (widths[i] + widths[i + 1]) / 2 + _GAP
            for i in range(len(values) - 1)
        )

    @staticmethod
    def _points(lo: float, hi: float, mantissas: tuple[int, ...]) -> list[float]:
        """Tick positions, in log10 space, for these mantissas across the range."""
        values: list[float] = []
        for decade in range(math.floor(lo), math.ceil(hi) + 1):
            for mantissa in mantissas:
                value = decade + math.log10(mantissa)
                if lo <= value <= hi:
                    values.append(value)
        return values


def _hz(value: float) -> str:
    """A frequency as somebody would say it, never as a power of ten.

    Trailing zeros are dropped rather than padded to a fixed precision: an axis
    reading 1, 2, 5, 10, 20 is scanned at a glance, and one reading 1.00, 2.00,
    5.00 is read.
    """
    if value >= 1000.0:
        thousands = value / 1000.0
        return f"{thousands:.0f}k" if thousands >= 10 else f"{thousands:g}k"
    if value >= 1.0:
        return f"{value:.0f}" if value >= 9.95 else f"{value:g}"
    # Below 1 Hz two significant figures is the most anybody reads off an axis,
    # and %g would otherwise reach for scientific notation around 0.0001.
    return f"{value:.{max(0, 1 - math.floor(math.log10(value)))}f}".rstrip("0").rstrip(".")
