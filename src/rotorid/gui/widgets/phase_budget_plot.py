"""Where the phase goes at crossover (spec section 10.4).

The single most instructive graphic in the tool. Phase margin is a budget, every
element in the loop spends some of it, and until you see the spending itemised
the cost of a filter is an abstraction. One stacked bar, one segment per
contributor, with what is left over marked as the margin.

It is drawn at the design crossover and nowhere else on purpose: phase lag is a
function of frequency, so "this notch costs 12 degrees" is only true somewhere,
and the only frequency where it matters is the one the loop crosses over at.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from rotorid.core.types import LatencyBudget, MarginReport
from rotorid.gui.theme import SERIES, Palette, palette

__all__ = ["PhaseBudgetPlot"]

_EXPLANATION = (
    "Everything that delays the loop, converted to the phase it costs at your "
    "crossover frequency, stacked up.\n\n"
    "The total is fixed by physics and by your hardware. What is left after the "
    "spending is your phase margin, which is what stops the aircraft "
    "oscillating. Every bar segment you shrink -- a narrower notch, a higher "
    "gyro cutoff, a faster loop rate -- becomes margin, and margin becomes gain.\n\n"
    "This is why filters and gains cannot be chosen separately: they are two "
    "ends of one budget."
)

#: Ordered outermost-cost first, so the segments a user can actually change sit
#: at the bottom of the bar where they are easiest to compare.
_ITEMS: tuple[tuple[str, str], ...] = (
    ("gyro_lpf_deg", "Gyro low-pass"),
    ("notches_deg", "Notches"),
    ("dterm_lpf_deg", "D-term low-pass"),
    ("error_lpf_deg", "Error low-pass"),
    ("zoh_deg", "Sample and hold"),
    ("compute_deg", "Compute"),
    ("actuator_deg", "ESC and motors"),
    ("airframe_tau_deg", "Airframe lag"),
)


class PhaseBudgetPlot(QWidget):
    """One horizontal stacked bar: the phase budget, itemised."""

    def __init__(self, theme: Palette | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme if theme is not None else palette("light")
        pg.setConfigOption("background", self._theme.surface)
        pg.setConfigOption("foreground", self._theme.text)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Phase spent at crossover", units="deg")
        self.plot.showGrid(x=True, y=False, alpha=0.25)
        self.plot.getPlotItem().hideAxis("left")
        self.plot.addLegend(offset=(-10, 10))
        layout.addWidget(self.plot)
        self.explanation = _EXPLANATION

    def show_budget(self, budget: LatencyBudget, margins: MarginReport | None = None) -> None:
        self.plot.clear()
        start = 0.0
        for index, (field, label) in enumerate(_ITEMS):
            value = float(getattr(budget, field))
            if value <= 0.0:
                continue
            bar = pg.BarGraphItem(
                x0=[start],
                width=[value],
                y=[0],
                height=0.6,
                brush=SERIES[index % len(SERIES)],
                name=f"{label} ({value:.1f} deg)",
            )
            self.plot.addItem(bar)
            start += value

        if margins is not None:
            remaining = pg.BarGraphItem(
                x0=[start],
                width=[max(margins.phase_margin_deg, 0.0)],
                y=[0],
                height=0.6,
                brush=self._theme.grid,
                name=f"Phase margin left ({margins.phase_margin_deg:.0f} deg)",
            )
            self.plot.addItem(remaining)
            self.plot.setTitle(
                f"At {margins.crossover_hz:.2f} Hz: {start:.0f} deg spent, "
                f"{margins.phase_margin_deg:.0f} deg of margin left"
            )
